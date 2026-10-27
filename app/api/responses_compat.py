"""OpenAI Responses API 兼容层（POST /v1/responses）。

新版客户端与不少指纹/评测工具走的是 Responses 协议而不是 chat/completions，
只实现 chat 会导致它们全部 404。这里做无状态翻译：

    Responses 请求  →  OpenAI /v1/chat/completions  →  上游
    chat 响应       →  Responses 对象 / response.* 事件流

字段映射：
    input                 → messages（字符串或 [{role, content}]）
    instructions          → system 消息
    max_output_tokens     → max_tokens
    tools[function]       → OpenAI tools
    input.function_call   → assistant.tool_calls
    input.function_call_output → role=tool
    上游 tool_calls       → output[].type="function_call" + arguments 事件

前提：网关侧要开 `TOOL_BRIDGE=1`（`TOOL_BRIDGE_FRAME=planner` 更稳），
否则内部 `/v1/chat/completions` 会把 tools 丢掉，这里永远拿不到 function_call。

未实现：`type="custom"`（如 apply_patch 的自定义工具）与图片输出。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import API_KEY
from app.logger import logger
from app.reasoning import resolve_reasoning_effort
from app.tool_bridge import merge_stream_tool_calls
from app.api.anthropic_compat import count_images

router = APIRouter()

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


DEFAULT_MODEL = os.getenv("RESPONSES_DEFAULT_MODEL", "claude-opus5.5")


# --------------------------------------------------------------------------- #
# Responses → OpenAI
# --------------------------------------------------------------------------- #
def _image_mode() -> str:
    from app.vision_bridge import image_mode

    return image_mode()


def replace_images_with_descriptions(node: Any) -> int:
    """把 input_image 就地换成视觉转写文本（data URL 才处理）。"""
    from app.vision_bridge import describe_data_url, render_description

    replaced = 0
    if isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, dict) and item.get("type") == "input_image":
                raw = item.get("image_url")
                url = raw.get("url") if isinstance(raw, dict) else raw
                description = (
                    describe_data_url(url)
                    if isinstance(url, str) and url.startswith("data:")
                    else None
                )
                node[index] = {
                    "type": "input_text",
                    "text": render_description(description)
                    if description
                    else "[图片未能转写：本地视觉模型不可用，或不是 base64 data URL]",
                }
                replaced += 1
            else:
                replaced += replace_images_with_descriptions(item)
    elif isinstance(node, dict):
        for value in node.values():
            replaced += replace_images_with_descriptions(value)
    return replaced


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind in ("input_text", "output_text", "text"):
                    parts.append(str(item.get("text") or ""))
                elif kind == "image_url":
                    parts.append("[image omitted]")
        return "\n".join(p for p in parts if p)
    return ""


def _openai_tools(raw: Any) -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return tools
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        if kind == "custom":
            # 自由文本工具（apply_patch 这类）：上游只会写文本，所以建模成
            # 只有一个 input 字符串参数的函数，模型把整段原文放进 input。
            description = str(item.get("description") or "").strip()
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (
                            (description + "\n\n" if description else "")
                            + "This is a freeform tool. Put the complete raw input in "
                              "the `input` field as one plain-text string."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": "Raw tool input, verbatim.",
                                }
                            },
                            "required": ["input"],
                        },
                    },
                }
            )
            continue
        if kind not in (None, "function"):
            continue
        params = item.get("parameters")
        if not isinstance(params, dict) or not params:
            params = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(item.get("description") or ""),
                    "parameters": params,
                },
            }
        )
    return tools


def _custom_tool_names(raw: Any) -> set[str]:
    """请求里声明为 type="custom" 的工具名，用于把输出还原成 custom_tool_call。"""
    names: set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("type") == "custom":
                name = str(item.get("name") or "").strip()
                if name:
                    names.add(name)
    return names


def _openai_tool_choice(raw: Any) -> Any:
    if raw in ("auto", "required", "none"):
        return raw
    if isinstance(raw, dict) and raw.get("type") == "function" and raw.get("name"):
        return {"type": "function", "function": {"name": raw["name"]}}
    return None


def _call_from_item(item: Dict[str, Any], *, custom: bool = False) -> Dict[str, Any]:
    if custom:
        # custom 工具的 input 是自由文本；内部用 {"input": "..."} 承载
        arguments = json.dumps({"input": str(item.get("input") or "")}, ensure_ascii=False)
    else:
        arguments = str(item.get("arguments") or "{}")
    return {
        "id": str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:20]}"),
        "type": "function",
        "function": {
            "name": str(item.get("name") or ""),
            "arguments": arguments,
        },
    }


def _convert_input(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions.strip()})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
        return messages
    if not isinstance(raw_input, list):
        return messages

    for item in raw_input:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        kind = item.get("type")
        if kind in ("function_call", "custom_tool_call"):
            call = _call_from_item(item, custom=(kind == "custom_tool_call"))
            last = messages[-1] if messages else None
            if last and last.get("role") == "assistant":
                last.setdefault("tool_calls", []).append(call)
                if last.get("content") is None:
                    last["content"] = None
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            continue
        if kind in ("function_call_output", "custom_tool_call_output"):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": _text_of(item.get("output")),
                }
            )
            continue

        # 普通消息：{role, content} 或 {type:"message", role, content}
        role = item.get("role") or "user"
        if role not in ("user", "assistant", "system"):
            role = "user"
        text = _text_of(item.get("content"))
        if text:
            messages.append({"role": role, "content": text})
    return messages


def _to_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    messages = _convert_input(body)
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

    limit = body.get("max_output_tokens") or body.get("max_tokens")
    if isinstance(limit, (int, float)) and limit > 0:
        payload["max_tokens"] = int(limit)
    if isinstance(body.get("temperature"), (int, float)):
        payload["temperature"] = float(body["temperature"])
    reasoning = body.get("reasoning")
    effort = resolve_reasoning_effort(
        body.get("reasoning_effort"),
        reasoning if isinstance(reasoning, dict) else None,
    )
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        effort = resolve_reasoning_effort(reasoning.get("effort"))
    payload["reasoning_effort"] = effort
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}
    return payload


# --------------------------------------------------------------------------- #
# OpenAI → Responses
# --------------------------------------------------------------------------- #
def _custom_input(arguments: Any) -> str:
    """从内部 arguments 里取出 custom 工具的自由文本 input。"""
    raw = arguments if isinstance(arguments, str) else str(arguments or "")
    try:
        parsed = json.loads(raw) if raw else {}
    except (ValueError, TypeError):
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    return raw


def _function_call_items(
    calls: Any, custom_names: set[str] | None = None
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    custom_names = custom_names or set()
    if not isinstance(calls, list):
        return items
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:20]}")
        if name in custom_names:
            items.append(
                {
                    "id": "ctc_" + uuid.uuid4().hex[:24],
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "input": _custom_input(fn.get("arguments")),
                }
            )
            continue
        items.append(
            {
                "id": "fc_" + uuid.uuid4().hex[:24],
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": str(fn.get("arguments") or "{}"),
            }
        )
    return items


def _message_item(text: str) -> Dict[str, Any]:
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _usage_from(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _response_object(
    response_id: str,
    model: str,
    output: List[Dict[str, Any]],
    text: str,
    created_at: int,
    usage: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": text,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }


def _error(message: str, status: int) -> Dict[str, Any]:
    return {
        "error": {
            "message": message or "upstream error",
            "type": "server_error" if status >= 500 else "invalid_request_error",
            "code": "upstream_error" if status >= 500 else "invalid_request",
        }
    }


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/responses")
async def create_response(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=400, content=_error("invalid JSON body", 400))

    raw_body = body if isinstance(body, dict) else {}
    images = count_images(raw_body.get("input"))
    if images:
        mode = _image_mode()
        if mode == "reject":
            return JSONResponse(
                status_code=415,
                content=_error(
                    "图片策略为 reject：上游 runInferenceTranscript 收不到图片，"
                    "而本地视觉转写被显式关闭。改用 NOTION_IMAGE_MODE=vision 或 omit。",
                    415,
                ),
            )
        if mode == "vision":
            replaced = await asyncio.to_thread(
                replace_images_with_descriptions, raw_body.get("input")
            )
            logger.info(
                "Responses images transcribed locally",
                extra={"request_info": {"event": "responses_images_transcribed",
                                        "images": images, "replaced": replaced}},
            )
        else:
            logger.warning(
                "Responses images omitted",
                extra={"request_info": {"event": "responses_images_omitted",
                                        "images": images}},
            )

    openai_body = _to_openai(raw_body)
    custom_names = _custom_tool_names(raw_body.get("tools"))
    if not openai_body["messages"]:
        return JSONResponse(status_code=400, content=_error("input 不能为空", 400))

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response_id = "resp_" + uuid.uuid4().hex[:24]
    model = str(openai_body["model"])
    created_at = int(time.time())

    logger.info(
        "Responses compat request",
        extra={
            "request_info": {
                "event": "responses_compat",
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
            return JSONResponse(status_code=502, content=_error(f"gateway unreachable: {exc}", 502))
        if resp.status_code >= 400:
            detail = ""
            try:
                err = resp.json().get("error")
                detail = err.get("message") if isinstance(err, dict) else str(err or "")
            except Exception:  # noqa: BLE001
                detail = resp.text[:200]
            return JSONResponse(status_code=resp.status_code, content=_error(detail, resp.status_code))
        data = resp.json()
        choice = (data.get("choices") or [{}])[0] or {}
        message = choice.get("message") or {}
        text = str(message.get("content") or "")
        output: List[Dict[str, Any]] = []
        if text.strip():
            output.append(_message_item(text))
        output.extend(_function_call_items(message.get("tool_calls"), custom_names))
        return _response_object(
            response_id, model, output, text, created_at, _usage_from(data)
        )

    async def event_stream():
        yield _sse("response.created", {
            "type": "response.created",
            "response": {"id": response_id, "object": "response", "status": "in_progress",
                         "model": model, "created_at": created_at, "output": []},
        })

        text_started = False
        text_id = "msg_" + uuid.uuid4().hex[:20]
        collected: List[str] = []
        raw_calls: List[Any] = []
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        try:
            async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
                async with client.stream(
                    "POST", f"{_internal_base(request)}/chat/completions",
                    json=openai_body, headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        yield _sse("error", _error("upstream failed", resp.status_code))
                        return
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        if chunk.get("usage"):
                            usage = {
                                "input_tokens": int(chunk["usage"].get("prompt_tokens") or 0),
                                "output_tokens": int(chunk["usage"].get("completion_tokens") or 0),
                                "total_tokens": int(chunk["usage"].get("total_tokens") or 0),
                            }
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            if delta.get("tool_calls"):
                                raw_calls.extend(delta["tool_calls"])
                            piece = delta.get("content")
                            if not piece:
                                continue
                            if not text_started:
                                text_started = True
                                yield _sse("response.output_item.added", {
                                    "type": "response.output_item.added",
                                    "output_index": 0,
                                    "item": {"id": text_id, "type": "message",
                                             "role": "assistant", "status": "in_progress",
                                             "content": []},
                                })
                                yield _sse("response.content_part.added", {
                                    "type": "response.content_part.added",
                                    "item_id": text_id, "output_index": 0, "content_index": 0,
                                    "part": {"type": "output_text", "text": "", "annotations": []},
                                })
                            collected.append(piece)
                            yield _sse("response.output_text.delta", {
                                "type": "response.output_text.delta",
                                "item_id": text_id, "output_index": 0,
                                "content_index": 0, "delta": piece,
                            })
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", _error(f"stream failed: {exc}", 502))
            return

        text = "".join(collected)
        output: List[Dict[str, Any]] = []
        if text_started:
            yield _sse("response.output_text.done", {
                "type": "response.output_text.done", "item_id": text_id,
                "output_index": 0, "content_index": 0, "text": text,
            })
            item = _message_item(text)
            output.append(item)
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done", "output_index": 0, "item": item,
            })

        calls = merge_stream_tool_calls(raw_calls)
        for offset, call in enumerate(calls):
            index = len(output)
            name = str(call.get("name") or "")
            call_id = str(call.get("id") or f"call_{uuid.uuid4().hex[:20]}")
            if name in custom_names:
                # custom 工具：流里发 input 文本增量，而不是 JSON 参数
                item = {
                    "id": "ctc_" + uuid.uuid4().hex[:24],
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "input": "",
                }
                yield _sse("response.output_item.added", {
                    "type": "response.output_item.added", "output_index": index, "item": item,
                })
                text_input = _custom_input(call.get("arguments"))
                if text_input:
                    yield _sse("response.custom_tool_call_input.delta", {
                        "type": "response.custom_tool_call_input.delta",
                        "item_id": item["id"], "output_index": index, "delta": text_input,
                    })
                yield _sse("response.custom_tool_call_input.done", {
                    "type": "response.custom_tool_call_input.done",
                    "item_id": item["id"], "output_index": index, "input": text_input,
                })
                item = dict(item, status="completed", input=text_input)
                output.append(item)
                yield _sse("response.output_item.done", {
                    "type": "response.output_item.done", "output_index": index, "item": item,
                })
                continue

            item = {
                "id": "fc_" + uuid.uuid4().hex[:24],
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "arguments": "",
            }
            yield _sse("response.output_item.added", {
                "type": "response.output_item.added", "output_index": index, "item": item,
            })
            arguments = str(call.get("arguments") or "{}")
            yield _sse("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": item["id"], "output_index": index, "delta": arguments,
            })
            yield _sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": item["id"], "output_index": index, "arguments": arguments,
            })
            item = dict(item, status="completed", arguments=arguments)
            output.append(item)
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done", "output_index": index, "item": item,
            })

        yield _sse("response.completed", {
            "type": "response.completed",
            "response": _response_object(
                response_id, model, output, text, created_at, usage
            ),
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
