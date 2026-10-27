"""Anthropic Messages API 兼容层。

客户端（Claude SDK、Claude Code 等）请求的是 `POST /v1/messages`，而本网关
内部只有 OpenAI 协议。这里做一层无状态翻译：

    Anthropic 请求  →  OpenAI /v1/chat/completions  →  上游 Notion
    OpenAI 响应     →  Anthropic message / SSE 事件

工具调用（借鉴 sticture/notion2api 的做法）：
    Anthropic tools[{name,input_schema}]  →  OpenAI tools[{function}]
    assistant.tool_use                    →  OpenAI assistant.tool_calls
    user.tool_result                      →  OpenAI role=tool
    上游 tool_calls                       →  Anthropic tool_use 块 + stop_reason=tool_use
    流式                                   →  input_json_delta + content_block_stop

前提：网关侧要开 `TOOL_BRIDGE=1`（`TOOL_BRIDGE_FRAME=planner` 更稳），
否则内部 `/v1/chat/completions` 会把 tools 丢掉，这里就永远拿不到 tool_use。

鉴权仍由全局 API Key 中间件负责（同时接受 `Authorization: Bearer` 与 `x-api-key`）。
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Tuple

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import API_KEY
from app.reasoning import resolve_reasoning_effort
from app.logger import logger
from app.tool_bridge import merge_stream_tool_calls

router = APIRouter()

# 内部回环地址：复用既有 OpenAI 链路，避免复制一份推理逻辑
def _internal_base(request: Any = None) -> str:
    """内部回环地址。

    优先用环境变量覆盖；否则跟随**实际请求**的 host/port——
    写死端口时，只要 uvicorn 的 --port 和 .env 的 PORT 不一致，
    这两个兼容层就会连到隔壁进程上去（表现为 401 或 502）。
    """
    override = os.getenv("INTERNAL_OPENAI_BASE", "").strip()
    if override:
        return override.rstrip("/")
    if request is not None:
        return str(request.base_url).rstrip("/") + "/v1"
    return f"http://127.0.0.1:{os.getenv('PORT', '8000')}/v1"


DEFAULT_MODEL = os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-opus5.5")


def _image_policy() -> str:
    """图片策略：vision（本地视觉转写，默认）/ omit / reject。"""
    from app.vision_bridge import image_mode

    return image_mode()


def count_images(node: Any) -> int:
    """递归数一下请求里有多少 image 块（含 tool_result 里的）。"""
    if isinstance(node, dict):
        if node.get("type") in ("image", "input_image"):
            return 1
        return sum(count_images(value) for value in node.values())
    if isinstance(node, list):
        return sum(count_images(item) for item in node)
    return 0


def replace_images_with_descriptions(node: Any) -> int:
    """把 image 块就地换成「本地视觉模型转写」的文本块，返回替换数量。

    在压缩前跑，图片不会占用上游输入预算（它们本来也进不去）。
    """
    from app.vision_bridge import describe_block_source, render_description

    replaced = 0
    if isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, dict) and item.get("type") == "image":
                description = describe_block_source(item.get("source"))
                node[index] = {
                    "type": "text",
                    "text": render_description(description)
                    if description
                    else "[图片未能转写：本地视觉模型不可用或图片格式不支持]",
                }
                replaced += 1
            else:
                replaced += replace_images_with_descriptions(item)
    elif isinstance(node, dict):
        for value in node.values():
            replaced += replace_images_with_descriptions(value)
    return replaced


# --------------------------------------------------------------------------- #
# Anthropic → OpenAI
# --------------------------------------------------------------------------- #
def _flatten(content: Any) -> str:
    """Anthropic 的 content 可能是字符串，也可能是 [{type:text,text:...}]。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("type") == "image":
                    # 上游只吃纯文本，图片占位避免静默丢失
                    parts.append("[image omitted]")
        return "\n".join(part for part in parts if part)
    return ""


def _openai_tools(raw: Any) -> List[Dict[str, Any]]:
    """Anthropic tools[{name,description,input_schema}] → OpenAI tools。"""
    tools: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return tools
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        schema = item.get("input_schema")
        if not isinstance(schema, dict) or not schema:
            schema = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description") or ""),
                    "parameters": schema,
                },
            }
        )
    return tools


def _openai_tool_choice(raw: Any) -> Any:
    """Anthropic tool_choice → OpenAI tool_choice。"""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and raw.get("name"):
        return {"type": "function", "function": {"name": raw["name"]}}
    return None


def _convert_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 Anthropic messages 翻成 OpenAI messages，保留 tool_use / tool_result。"""
    messages: List[Dict[str, Any]] = []
    system = body.get("system")
    if system:
        text = _flatten(system).strip()
        if text:
            messages.append({"role": "system", "content": text})

    for item in body.get("messages") or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        blocks = item.get("content")
        if isinstance(blocks, str):
            messages.append({"role": role, "content": blocks})
            continue

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text") or ""))
            elif kind == "image":
                text_parts.append("[image omitted]")
            elif kind == "tool_use":
                tool_calls.append(
                    {
                        "id": str(block.get("id") or f"call_{uuid.uuid4().hex[:20]}"),
                        "type": "function",
                        "function": {
                            "name": str(block.get("name") or ""),
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            elif kind == "tool_result":
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": _flatten(block.get("content")),
                    }
                )

        text = "\n".join(part for part in text_parts if part)
        if role == "assistant" and tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": tool_calls,
                }
            )
        elif text:
            messages.append({"role": role, "content": text})
        # tool_result 必须在 assistant(tool_use) 之后
        messages.extend(tool_results)
    return messages


def _to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = _convert_messages(body)
    payload: Dict[str, Any] = {
        "model": body.get("model") or DEFAULT_MODEL,
        "messages": messages,
        "stream": bool(body.get("stream")),
    }

    tools = _openai_tools(body.get("tools"))
    if tools:
        payload["tools"] = tools
        choice = _openai_tool_choice(body.get("tool_choice"))
        if choice is not None:
            payload["tool_choice"] = choice

    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    temperature = body.get("temperature")
    if isinstance(temperature, (int, float)):
        payload["temperature"] = float(temperature)
    stop_sequences = body.get("stop_sequences")
    if isinstance(stop_sequences, list) and stop_sequences:
        payload["stop"] = stop_sequences
    payload["reasoning_effort"] = resolve_reasoning_effort(
        body.get("reasoning_effort"), body.get("thinking")
    )
    if payload["stream"]:
        # 让内部网关在末帧回 usage，Anthropic 的 message_delta 需要它
        payload["stream_options"] = {"include_usage": True}
    return payload


# --------------------------------------------------------------------------- #
# OpenAI → Anthropic
# --------------------------------------------------------------------------- #
def _tool_use_blocks(calls: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if not isinstance(calls, list):
        return blocks
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_args = fn.get("arguments")
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else {}
        except (ValueError, TypeError):
            parsed = {"_raw": raw_args}
        if not isinstance(parsed, dict):
            parsed = {"_value": parsed}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}"),
                "name": str(fn.get("name") or ""),
                "input": parsed,
            }
        )
    return blocks


def _assistant_content(message: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """返回 (content blocks, stop_reason)。"""
    text = str(message.get("content") or "")
    blocks: List[Dict[str, Any]] = []
    if text.strip():
        blocks.append({"type": "text", "text": text})
    tool_blocks = _tool_use_blocks(message.get("tool_calls"))
    blocks.extend(tool_blocks)
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    return blocks, ("tool_use" if tool_blocks else "end_turn")


def _usage_from(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def _error_payload(message: str, status: int) -> Dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": "api_error" if status >= 500 else "invalid_request_error",
            "message": message or "upstream error",
        },
    }


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _upstream_status_error(response: httpx.Response) -> Dict[str, Any]:
    detail = ""
    try:
        raw = await response.aread()
        parsed = json.loads(raw.decode("utf-8", "replace"))
        err = parsed.get("error")
        if isinstance(err, dict):
            detail = str(err.get("message") or "")
        elif isinstance(err, str):
            detail = err
    except Exception:  # noqa: BLE001
        detail = ""
    return _error_payload(detail, response.status_code)


@router.post("/messages")
async def create_message(request: Request):
    """Anthropic 风格的 /v1/messages。支持流式、非流式与工具调用。"""
    try:
        raw_body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content=_error_payload("invalid JSON body", 400))

    raw_body = raw_body if isinstance(raw_body, dict) else {}
    images = count_images(raw_body.get("messages"))
    if images:
        mode = _image_policy()
        if mode == "reject":
            return JSONResponse(
                status_code=415,
                content=_error_payload(
                    "图片策略为 reject：上游 runInferenceTranscript 收不到图片，"
                    "而本地视觉转写被显式关闭。改用 NOTION_IMAGE_MODE=vision 或 omit。",
                    415,
                ),
            )
        if mode == "vision":
            replaced = await asyncio.to_thread(
                replace_images_with_descriptions, raw_body.get("messages")
            )
            logger.info(
                "Anthropic images transcribed locally",
                extra={"request_info": {"event": "anthropic_images_transcribed",
                                        "images": images, "replaced": replaced}},
            )
        else:
            logger.warning(
                "Anthropic images omitted",
                extra={"request_info": {"event": "anthropic_images_omitted",
                                        "images": images}},
            )

    openai_body = _to_openai(raw_body)
    if not openai_body["messages"]:
        return JSONResponse(
            status_code=400,
            content=_error_payload("messages 不能为空", 400),
        )

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    message_id = "msg_" + uuid.uuid4().hex[:24]
    model = str(openai_body["model"])

    logger.info(
        "Anthropic compat request",
        extra={
            "request_info": {
                "event": "anthropic_compat",
                "model": model,
                "stream": openai_body["stream"],
                "messages": len(openai_body["messages"]),
                "tools": len(openai_body.get("tools") or []),
            }
        },
    )

    if not openai_body["stream"]:
        try:
            async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                resp = await client.post(
                    f"{_internal_base(request)}/chat/completions", json=openai_body, headers=headers
                )
        except Exception as exc:  # noqa: BLE001
            return JSONResponse(
                status_code=502,
                content=_error_payload(f"internal gateway unreachable: {exc}", 502),
            )
        if resp.status_code >= 400:
            return JSONResponse(
                status_code=resp.status_code,
                content=await _upstream_status_error(resp),
            )
        data = resp.json()
        choice = (data.get("choices") or [{}])[0] or {}
        blocks, stop_reason = _assistant_content(choice.get("message") or {})
        return {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _usage_from(data),
        }

    async def event_stream():
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        text_started = False
        tool_calls: List[Dict[str, Any]] = []
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        try:
            async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                async with client.stream(
                    "POST", f"{_internal_base(request)}/chat/completions",
                    json=openai_body, headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        payload = await _upstream_status_error(resp)
                        yield _sse("error", payload)
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk_text = line[5:].strip()
                        if chunk_text == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_text)
                        except ValueError:
                            continue
                        if chunk.get("usage"):
                            usage = {
                                "input_tokens": int(chunk["usage"].get("prompt_tokens") or 0),
                                "output_tokens": int(chunk["usage"].get("completion_tokens") or 0),
                            }
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("tool_calls"):
                                tool_calls.extend(delta["tool_calls"])
                            piece = delta.get("content")
                            if not piece:
                                continue
                            if not text_started:
                                text_started = True
                                yield _sse("content_block_start", {
                                    "type": "content_block_start", "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                })
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta", "text": piece},
                            })
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", _error_payload(f"stream failed: {exc}", 502))
            return

        if text_started:
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

        # 工具调用在流末尾统一补块：content_block_start → input_json_delta → stop
        base_index = 1 if text_started else 0
        for offset, call in enumerate(merge_stream_tool_calls(tool_calls)):
            index = base_index + offset
            arguments = call.get("arguments") or "{}"
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:20]}",
                    "name": call.get("name") or "",
                    "input": {},
                },
            })
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": arguments},
            })
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": index})

        stop_reason = "tool_use" if tool_calls else "end_turn"
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        })
        yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
