import asyncio
import contextvars
import json
import os
import queue
import re
import threading
import time
import uuid
from difflib import SequenceMatcher
from typing import Any, Dict, Generator, Iterable, List, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import is_lite_mode
from app.context_budget import InputTooLarge, enforce_budget
from app.reasoning import resolve_reasoning_effort
from app.conversation import (
    build_lite_transcript,
    compress_round_if_needed,
    compress_sliding_window_round,
)
from app.logger import logger
from app.model_registry import is_supported_model, list_available_models
from app.notion_client import NotionUpstreamError
from app.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponseChoice,
)

router = APIRouter()


# ─── 结构化错误响应 ─────────────────────────────────────────────
def _classify_upstream_error(exc: NotionUpstreamError) -> dict[str, Any]:
    """将 NotionUpstreamError 分类为结构化错误信息，前端可直接展示。"""
    sc = exc.status_code

    if sc == 401:
        return {
            "code": "NOTION_401",
            "type": "upstream_auth_error",
            "message": "Notion 认证失败 (HTTP 401)，token 可能已过期",
            "suggestion": "请重新获取 token_v2 并更新配置",
        }
    if sc == 403:
        return {
            "code": "NOTION_403",
            "type": "upstream_forbidden",
            "message": "Notion 拒绝访问 (HTTP 403)，可能被 Cloudflare 拦截或账号受限",
            "suggestion": "检查服务器网络环境，或稍后重试",
        }
    if sc == 429:
        return {
            "code": "NOTION_429",
            "type": "upstream_rate_limit",
            "message": "Notion 请求频率过高 (HTTP 429)",
            "suggestion": "等待几秒后重试，或配置多个账号分散请求",
        }
    if sc and sc >= 500:
        return {
            "code": f"NOTION_{sc}",
            "type": "upstream_server_error",
            "message": f"Notion 服务暂时不可用 (HTTP {sc})",
            "suggestion": "Notion 服务端故障，请稍后重试",
        }
    if "timed out" in str(exc).lower():
        return {
            "code": "NETWORK_TIMEOUT",
            "type": "network_timeout",
            "message": "连接 Notion 超时",
            "suggestion": "检查服务器到 notion.so 的网络连通性",
        }
    if "failed" in str(exc).lower() and not sc:
        return {
            "code": "NETWORK_ERROR",
            "type": "network_error",
            "message": "无法连接 Notion 服务",
            "suggestion": "检查服务器网络和 DNS 配置",
        }
    if "empty" in str(exc).lower():
        return {
            "code": "NOTION_EMPTY",
            "type": "upstream_empty_response",
            "message": "Notion 返回了空内容",
            "suggestion": "请重新发送消息",
        }
    # 兜底
    return {
        "code": "UPSTREAM_UNKNOWN",
        "type": "upstream_error",
        "message": str(exc),
        "suggestion": "请稍后重试",
    }


def _build_error_response(
    status_code: int,
    *,
    code: str,
    message: str,
    error_type: str = "server_error",
    suggestion: str = "",
    detail: str = "",
) -> JSONResponse:
    """构建统一格式的错误 JSON 响应，前端可解析展示。"""
    content: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }
    if suggestion:
        content["error"]["suggestion"] = suggestion
    if detail:
        content["error"]["detail"] = detail
    return JSONResponse(status_code=status_code, content=content)


def _upstream_error_response(exc: NotionUpstreamError) -> JSONResponse:
    """将 NotionUpstreamError 转为统一的 503 JSON 响应。"""
    info = _classify_upstream_error(exc)
    return _build_error_response(
        503,
        code=info["code"],
        message=info["message"],
        error_type=info["type"],
        suggestion=info.get("suggestion", ""),
        detail=exc.response_excerpt or "",
    )


RECALL_INTENT_KEYWORDS = [
    "之前",
    "上次",
    "以前",
    "你还记得",
    "我们之前",
    "earlier",
    "before",
    "recall",
    "remember",
    "之前说过",
    "历史记录",
    "找一下",
    "搜索记忆",
]


def _build_stream_chunk(
    response_id: str,
    model: str,
    *,
    content: str = "",
    thinking: str = "",
    role: str = "",
    finish_reason=None,
    tool_calls: Any = None,
) -> str:
    delta: Dict[str, Any] = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    if thinking:
        delta["reasoning_content"] = thinking
    if tool_calls:
        delta["tool_calls"] = tool_calls

    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _build_usage_chunk(
    response_id: str, model: str, usage: Dict[str, int]
) -> str:
    """OpenAI 流式协议里的末帧 usage：choices 为空、顶层带 usage。"""
    payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _wants_stream_usage(req_body: ChatCompletionRequest) -> bool:
    """客户端是否要 `stream_options.include_usage` 的末帧 usage。"""
    options = getattr(req_body, "stream_options", None)
    if isinstance(options, dict):
        return bool(options.get("include_usage"))
    return False


def _prompt_tokens_of(messages: Iterable[Dict[str, Any]]) -> int:
    from app.usage_meter import estimate_tokens, messages_text

    return estimate_tokens(messages_text(messages))


def _heartbeat_seconds() -> float:
    raw = os.getenv("SSE_HEARTBEAT_SECONDS", "15").strip()
    try:
        value = float(raw)
    except ValueError:
        return 15.0
    return max(0.0, value)


def _with_heartbeat(
    items: Iterable[str], interval: float | None = None
) -> Generator[str, None, None]:
    """上游长时间不吐字时补 SSE 注释行，避免客户端/中间层判空闲超时。

    Notion 的思考阶段可以静默几十秒（账号池里见过 40–51s 的首字延迟），
    严格客户端会直接断开。注释行 `: keepalive` 不进入正文，也不改变协议。
    """
    seconds = _heartbeat_seconds() if interval is None else interval
    if seconds <= 0:
        yield from items
        return

    pending: "queue.Queue[Any]" = queue.Queue(maxsize=512)
    sentinel = object()
    stop = threading.Event()

    def _pump() -> None:
        try:
            for chunk in items:
                if stop.is_set():
                    return
                # 用带超时的 put：客户端断开后 stop 会被置位，
                # 队列满也不会把上游生成器永久挂住。
                while not stop.is_set():
                    try:
                        pending.put(chunk, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except (GeneratorExit, SystemExit):
            pass
        except BaseException as exc:  # noqa: BLE001
            try:
                pending.put_nowait(exc)
            except queue.Full:
                pass
        finally:
            stop.set()
            # 生成器必须由迭代它的线程关闭：跨线程 close() 会在 native
            # 传输（curl_cffi）上触发堆损坏，直接把进程带走。
            close = getattr(items, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                pending.put_nowait(sentinel)
            except queue.Full:
                pass

    # 生成器在这个线程里被迭代，lease.mark_success/failed 也就在这里触发；
    # 必须带上 contextvar（模型名），否则调用明细的 model 是空的。
    _ctx = contextvars.copy_context()
    worker = threading.Thread(target=_ctx.run, args=(_pump,), name="sse-heartbeat", daemon=True)
    worker.start()
    try:
        while True:
            try:
                item = pending.get(timeout=seconds)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # 只置停止位，不做跨线程 close（见上）
        stop.set()


def _build_local_ui_chunk(
    response_id: str,
    model: str,
    event_type: str,
    **payload_fields: Any,
) -> str:
    payload: Dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "type": event_type,
    }
    payload.update(payload_fields)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _format_search_results_md(search_data: dict[str, Any]) -> str:
    """将搜索数据格式化为 Markdown 引用块，以便标准客户端显示。"""
    lines = []
    queries = search_data.get("queries", [])
    if queries:
        lines.append(f"> 🔍 **已搜索:** {', '.join(queries)}")

    sources = search_data.get("sources", [])
    if sources:
        lines.append("> 🌐 **来源:**")
        for i, src in enumerate(sources[:5], 1):  # 最多显示5个来源，避免刷屏
            title = src.get("title") or src.get("url") or "未知来源"
            url = src.get("url")
            if url:
                lines.append(f"> {i}. [{title}]({url})")
            else:
                lines.append(f"> {i}. {title}")

    if lines:
        # 前后都留空行：这个块会被直接拼到正文前后，不加分隔会出现
        # "8> 🌐 **来源:**" 这种粘连
        return "\n\n" + "\n".join(lines) + "\n\n"
    return ""


def _normalize_stream_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"type": "content", "text": item}

    if isinstance(item, dict):
        item_type = str(item.get("type", "") or "").lower()
        if item_type == "content":
            return {"type": "content", "text": str(item.get("text", "") or "")}
        if item_type == "search":
            payload = item.get("data")
            return {
                "type": "search",
                "data": payload if isinstance(payload, dict) else {},
            }
        if item_type == "thinking":
            return {"type": "thinking", "text": str(item.get("text", "") or "")}
        if item_type == "final_content":
            return {
                "type": "final_content",
                "text": str(item.get("text", "") or ""),
                "source_type": str(item.get("source_type", "") or ""),
                "source_length": item.get("source_length"),
            }

    return {"type": "unknown"}


def _iter_stream_items(
    first_item: Any, stream_gen: Iterable[Any]
) -> Generator[Any, None, None]:
    if first_item is not None:
        yield first_item
    for item in stream_gen:
        yield item


def _compute_missing_suffix(current_text: str, final_text: str) -> str:
    if not final_text:
        return ""
    if not current_text:
        return final_text
    if final_text.startswith(current_text):
        return final_text[len(current_text) :]
    return ""


def _select_best_final_reply(
    streamed_text: str,
    final_text: str,
    final_source_type: str,
) -> tuple[str, str]:
    streamed = streamed_text or ""
    final = final_text or ""
    streamed_stripped = streamed.strip()
    final_stripped = final.strip()
    source = (final_source_type or "").strip().lower()

    if not final_stripped:
        return streamed, "streamed_only"
    if not streamed_stripped:
        return final, "final_only"
    if final.startswith(streamed):
        return final, "final_extends_streamed"
    if streamed.startswith(final):
        if source == "title" or len(final_stripped) <= max(
            32, int(len(streamed_stripped) * 0.35)
        ):
            return streamed, "streamed_beats_short_final"
        return final, "final_prefix_of_streamed"

    # Diverged content: usually prefer richer non-title final content.
    if source == "title" and len(final_stripped) < max(
        48, int(len(streamed_stripped) * 0.6)
    ):
        return streamed, "streamed_beats_title"
    if len(final_stripped) >= max(48, int(len(streamed_stripped) * 0.6)):
        return final, "final_diverged_preferred"
    return streamed, "streamed_diverged_preferred"


def _normalize_overlap_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return ""
    normalized = re.sub(r"```.*?```", " ", normalized, flags=re.DOTALL)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _trim_redundant_thinking(
    thinking_text: str, final_reply: str
) -> tuple[str, str, float]:
    thinking = str(thinking_text or "").strip()
    final = str(final_reply or "").strip()
    if not thinking or not final:
        return thinking, "missing_text", 0.0

    normalized_thinking = _normalize_overlap_text(thinking)
    normalized_final = _normalize_overlap_text(final)
    if not normalized_thinking or not normalized_final:
        return thinking, "missing_normalized_text", 0.0

    overlap_ratio = SequenceMatcher(None, normalized_thinking, normalized_final).ratio()
    if normalized_thinking == normalized_final:
        return "", "identical", overlap_ratio

    if thinking.endswith(final):
        prefix = thinking[: -len(final)].rstrip()
        if len(_normalize_overlap_text(prefix)) >= 10:
            return prefix, "suffix_trimmed", overlap_ratio
        return "", "suffix_cleared", overlap_ratio

    if overlap_ratio >= 0.92 and (
        normalized_thinking in normalized_final
        or normalized_final in normalized_thinking
    ):
        return "", "high_overlap_cleared", overlap_ratio

    return thinking, "kept", overlap_ratio


def _build_thinking_replacement(
    streamed_content_text: str,
    thinking_text: str,
    final_reply: str,
    final_source_type: str,
) -> dict[str, Any] | None:
    source = str(final_source_type or "").strip().lower()

    # Relax constraint: Allow replacement for more source types to fix Sonnet thinking leakage
    # But still require minimal validation for non-inference sources
    if source not in ("agent-inference", "text", "markdown-chat", ""):
        # Only skip for clearly non-thinking source types
        return None

    normalized_final = _normalize_overlap_text(final_reply)
    normalized_streamed = _normalize_overlap_text(streamed_content_text)

    # Require at least some thinking content to process
    if not _normalize_overlap_text(thinking_text):
        return None

    # For non-agent-inference sources, be more conservative but still check for obvious duplication
    if source != "agent-inference":
        # Only process if there's clear overlap or thinking is redundant
        if not normalized_final:
            return None

        # Check for obvious duplication (thinking appears in final reply)
        if thinking_text.strip() in final_reply or final_reply in thinking_text:
            # Clear case of duplication - trim it
            replacement, decision, overlap_ratio = _trim_redundant_thinking(
                thinking_text, final_reply
            )
            if replacement != str(thinking_text or "").strip():
                logger.debug(
                    "Non-agent-inference thinking replacement applied",
                    extra={
                        "request_info": {
                            "event": "thinking_replacement_non_agent",
                            "source_type": source,
                            "overlap_ratio": round(overlap_ratio, 4),
                            "decision": f"{decision}_non_agent_inference",
                        }
                    },
                )
                return {
                    "thinking": replacement,
                    "decision": f"{decision}_non_agent_inference",
                    "overlap_ratio": round(overlap_ratio, 4),
                    "source_type": source,
                }
        return None

    # Original agent-inference logic continues
    if not normalized_final:
        return None

    # 只在几乎没有真实正文增量时做裁决，避免误伤复杂推理场景。
    if normalized_streamed and len(normalized_streamed) >= max(
        10, int(len(normalized_final) * 0.35)
    ):
        return None

    replacement, decision, overlap_ratio = _trim_redundant_thinking(
        thinking_text, final_reply
    )
    if replacement == str(thinking_text or "").strip():
        return None

    return {
        "thinking": replacement,
        "decision": decision,
        "overlap_ratio": round(overlap_ratio, 4),
        "source_type": source,
    }


def _contains_recall_intent(text: str) -> bool:
    lowered = text.lower()
    for keyword in RECALL_INTENT_KEYWORDS:
        if keyword.isascii():
            if keyword.lower() in lowered:
                return True
            continue
        if keyword in text:
            return True
    return False


def _extract_recall_query(text: str) -> str:
    cleaned = text
    for keyword in RECALL_INTENT_KEYWORDS:
        if keyword.isascii():
            cleaned = re.sub(
                rf"\b{re.escape(keyword)}\b", " ", cleaned, flags=re.IGNORECASE
            )
        else:
            cleaned = cleaned.replace(keyword, " ")
    cleaned = re.sub(r"[\s，。！？、,.!?;:：]+", " ", cleaned).strip()
    return cleaned or text.strip()


def _tools_on() -> bool:
    """tools 是否参与 prompt 注入 / tool_calls 解析。由 TOOL_BRIDGE 控制。"""
    from app.tool_bridge import tools_enabled

    return tools_enabled()



def _collect_standard_items(first_item, stream_gen):
    """同步消费上游流（在 worker 线程里跑，避免阻塞事件循环）。"""
    content_parts = []
    thinking_parts = []
    search_results = []
    final_text = ""
    final_source = ""
    for raw_item in _iter_stream_items(first_item, stream_gen):
        item = _normalize_stream_item(raw_item)
        item_type = item.get("type")
        if item_type == "final_content":
            text = str(item.get("text", "") or "").strip()
            if text:
                final_text = text
                final_source = str(item.get("source_type", "") or "")
            continue
        if item_type == "thinking":
            text = item.get("text", "")
            if text:
                thinking_parts.append(text)
            continue
        if item_type == "search":
            data = item.get("data", {})
            if data:
                search_results.append(data)
            continue
        if item_type == "content":
            chunk = item.get("text", "")
            if chunk:
                content_parts.append(chunk)
    return {
        "content": content_parts,
        "thinking": thinking_parts,
        "search": search_results,
        "final_text": final_text,
        "final_source": final_source,
    }


def _open_upstream(client: Any, transcript: List[Dict[str, Any]]) -> Any:
    """开一条上游流：启用侧车时跑在子进程里，否则退回进程内直连。

    侧车存在的唯一理由：curl_cffi 的 native 崩溃（0xC0000374）会带走整个进程。
    放进子进程后，崩的只是那一次请求。
    """
    from app.upstream_pool import get_pool

    pool = get_pool()
    if pool is not None:
        return pool.stream(client, transcript)
    return client.stream_response(transcript, thread_id=None)


def _max_attempts() -> int:
    """单请求最多换几个账号。

    之前是 12：上游大面积 502 时一次请求会连续撞 12 个账号，既慢又把
    curl_cffi 的并发压力拉满（native 堆损坏 0xC0000374 就发生在这种长重试里）。
    默认降到 6，可用 NOTION_MAX_ATTEMPTS 调整。
    """
    try:
        return max(1, int(os.getenv("NOTION_MAX_ATTEMPTS", "6")))
    except ValueError:
        return 6


def _tool_names(tools: Any) -> set:
    """客户端声明的工具名，用于裸 JSON 调用的白名单校验。"""
    names = set()
    for item in tools or []:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = str(fn.get("name") or "").strip()
        if name:
            names.add(name)
    return names


def _tool_stream_filter(req_body: ChatCompletionRequest) -> Any:
    """带 tools 且注入开关打开时，返回流式过滤器；否则 None（走原路径）。"""
    if not (getattr(req_body, "tools", None) and _tools_on()):
        return None
    from app.tool_bridge import ToolCallStreamFilter, should_offer_tools

    if not should_offer_tools(req_body.tools):
        return None
    return ToolCallStreamFilter(known_names=_tool_names(req_body.tools))


def _emit_tool_call_completion(
    response_id: str,
    model_name: str,
    calls: List[Dict[str, Any]],
    assistant_started: bool,
    *,
    prompt_tokens: int = 0,
    emit_usage: bool = False,
) -> Generator[str, None, None]:
    """按 OpenAI 流式协议补发 tool_calls，并收尾。"""
    from app.tool_bridge import tool_call_deltas

    if not assistant_started:
        yield _build_stream_chunk(response_id, model_name, role="assistant")
    deltas = tool_call_deltas(calls)
    yield _build_stream_chunk(
        response_id, model_name, tool_calls=deltas
    )
    yield _build_stream_chunk(response_id, model_name, finish_reason="tool_calls")
    if emit_usage:
        from app.usage_meter import build_usage, estimate_tokens

        completion = estimate_tokens(json.dumps(deltas, ensure_ascii=False))
        yield _build_usage_chunk(
            response_id, model_name, build_usage(prompt_tokens, completion)
        )
    yield "data: [DONE]\n\n"


def _prepare_messages(
    req_body: ChatCompletionRequest,
) -> Tuple[str, List[Tuple[str, str, str]], str]:
    system_messages = []
    dialogue_messages = []

    for msg in req_body.messages:
        if msg.role == "system":
            if msg.content.strip():
                system_messages.append(msg.content.strip())
            continue
        dialogue_messages.append((msg.role, msg.content, msg.thinking or ""))

    if not dialogue_messages:
        raise HTTPException(
            status_code=400,
            detail="The messages list must contain at least one user message.",
        )

    last_role, user_prompt, _ = dialogue_messages[-1]
    raw_user_prompt = user_prompt
    history_messages = dialogue_messages[:-1]

    if last_role != "user":
        # agent 框架常把历史停在 assistant（含 tool_calls），这里补一句续写指令，
        # 而不是直接 400，避免整轮请求失败。
        if last_role == "assistant":
            history_messages = dialogue_messages
            last_role = "user"
            user_prompt = "Continue."
            raw_user_prompt = user_prompt
        else:
            raise HTTPException(
                status_code=400, detail="The last message must be from role 'user'."
            )
    if not user_prompt.strip():
        raise HTTPException(
            status_code=400, detail="The last user message cannot be empty."
        )

    if system_messages:
        from app.conversation import reframe_system_prompt_for_notion

        merged_system_prompt = "\n".join(system_messages)
        reframed = reframe_system_prompt_for_notion(merged_system_prompt)
        if reframed:
            user_prompt = f"{reframed}\n\n{user_prompt}"

    # 客户端带了 tools：上游不认 function calling，渲染成文本说明注入
    if getattr(req_body, "tools", None) and _tools_on():
        from app.tool_bridge import build_tool_preamble, build_tool_tail, should_offer_tools

        if should_offer_tools(req_body.tools):
            preamble = build_tool_preamble(req_body.tools, req_body.tool_choice)
            if preamble:
                user_prompt = f"{preamble}\n\n{user_prompt}"
            tail = build_tool_tail(req_body.tools)
            if tail:
                user_prompt = f"{user_prompt}\n\n{tail}"

    return user_prompt, history_messages, raw_user_prompt


def _prepare_messages_lite(req_body: ChatCompletionRequest) -> str:
    """Lite 模式：只提取最后一条 user 消息，支持 system 指令合并"""
    system_messages = []
    user_prompt = ""

    for msg in req_body.messages:
        if msg.role == "system" and msg.content.strip():
            system_messages.append(msg.content.strip())
        elif msg.role == "user":
            user_prompt = msg.content

    if not user_prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="The messages list must contain at least one user message.",
        )

    if system_messages:
        from app.conversation import reframe_system_prompt_for_notion

        reframed = reframe_system_prompt_for_notion(" ".join(system_messages))
        if reframed:
            user_prompt = f"{reframed}\n\n{user_prompt}"

    if getattr(req_body, "tools", None) and _tools_on():
        from app.tool_bridge import build_tool_preamble, build_tool_tail, should_offer_tools

        if should_offer_tools(req_body.tools):
            preamble = build_tool_preamble(req_body.tools, req_body.tool_choice)
            if preamble:
                user_prompt = f"{preamble}\n\n{user_prompt}"
            tail = build_tool_tail(req_body.tools)
            if tail:
                user_prompt = f"{user_prompt}\n\n{tail}"

    return user_prompt


def _record_stream_failure(
    lease: Any, exc: BaseException, fallback_reason: str
) -> None:
    if lease is None:
        return
    if isinstance(exc, NotionUpstreamError):
        lease.record_upstream_error(exc)
    else:
        lease.mark_failed(reason=fallback_reason)


def _with_lease_release(
    stream: Iterable[str], lease: Any
) -> Generator[str, None, None]:
    """Release a streaming lease only after the response generator is closed."""

    try:
        yield from stream
    finally:
        if lease is not None:
            lease.release()


def _create_lite_stream_generator(
    response_id: str,
    model_name: str,
    first_item: Any,
    stream_gen: Iterable[Any],
    *,
    lease: Any,
    tool_filter: Any = None,
    prompt_tokens: int = 0,
    emit_usage: bool = False,
) -> Generator[str, None, None]:
    """Lite 模式流式生成器：只输出 content，忽略 thinking 和 search"""
    streamed_content_accumulator = ""
    authoritative_final_content = ""
    authoritative_final_source_type = ""
    assistant_started = False
    failure_hint = ""
    client_disconnected = False

    try:
        for raw_item in _iter_stream_items(first_item, stream_gen):
            item = _normalize_stream_item(raw_item)
            item_type = item.get("type")

            if item_type == "final_content":
                final_text = str(item.get("text", "") or "").strip()
                if final_text:
                    authoritative_final_content = final_text
                    authoritative_final_source_type = str(
                        item.get("source_type", "") or ""
                    )
                continue

            # Lite 模式忽略 thinking 和 search
            if item_type in ("thinking", "search"):
                continue

            if item_type != "content":
                continue

            chunk_text = item.get("text", "")
            if not chunk_text:
                continue

            if tool_filter is not None:
                chunk_text = tool_filter.feed(chunk_text)
                if not chunk_text:
                    continue

            streamed_content_accumulator += chunk_text
            if not assistant_started:
                assistant_started = True
                yield _build_stream_chunk(
                    response_id,
                    model_name,
                    role="assistant",
                    content=chunk_text,
                )
            else:
                yield _build_stream_chunk(response_id, model_name, content=chunk_text)
        lease.mark_success()
    except GeneratorExit:
        client_disconnected = True
        logger.info(
            "Lite streaming closed by client",
            extra={"request_info": {"event": "lite_stream_client_disconnected"}},
        )
        raise
    except asyncio.CancelledError:
        client_disconnected = True
        logger.info(
            "Lite streaming cancelled by client",
            extra={"request_info": {"event": "lite_stream_cancelled"}},
        )
        raise
    except BaseException as exc:
        if _is_client_disconnect_error(exc):
            logger.info(
                "Lite streaming connection closed by client",
                extra={"request_info": {"event": "lite_stream_client_disconnected"}},
            )
            client_disconnected = True
            return
        _record_stream_failure(lease, exc, "lite_stream_interrupted")
        logger.error(
            "Lite streaming interrupted",
            exc_info=True,
            extra={"request_info": {"event": "lite_stream_interrupted"}},
        )
        failure_hint = "\n\n[上游连接中断，请稍后重试。]"
        streamed_content_accumulator += failure_hint
    finally:
        if client_disconnected:
            return

        if failure_hint:
            if not assistant_started:
                assistant_started = True
                yield _build_stream_chunk(
                    response_id,
                    model_name,
                    role="assistant",
                    content=failure_hint,
                )
            else:
                yield _build_stream_chunk(
                    response_id, model_name, content=failure_hint
                )

        # 先把流式过滤器里扣留的尾巴放出来，并判断有没有工具调用
        tool_calls: List[Dict[str, Any]] = []
        if tool_filter is not None:
            tail_text, tool_calls = tool_filter.finish()
            if tail_text:
                streamed_content_accumulator += tail_text
                if not assistant_started:
                    assistant_started = True
                    yield _build_stream_chunk(
                        response_id, model_name, role="assistant", content=tail_text
                    )
                else:
                    yield _build_stream_chunk(
                        response_id, model_name, content=tail_text
                    )

        if tool_calls:
            yield from _emit_tool_call_completion(
                response_id,
                model_name,
                tool_calls,
                assistant_started,
                prompt_tokens=prompt_tokens,
                emit_usage=emit_usage,
            )
            return

        # 选择最佳最终回复
        final_reply, _ = _select_best_final_reply(
            streamed_content_accumulator,
            authoritative_final_content,
            authoritative_final_source_type,
        )

        # 发送缺失的后缀（如果有）
        missing_suffix = _compute_missing_suffix(
            streamed_content_accumulator, final_reply
        )
        if missing_suffix:
            if not assistant_started:
                assistant_started = True
                yield _build_stream_chunk(
                    response_id,
                    model_name,
                    role="assistant",
                    content=missing_suffix,
                )
            else:
                yield _build_stream_chunk(
                    response_id, model_name, content=missing_suffix
                )
            streamed_content_accumulator += missing_suffix
        elif final_reply != streamed_content_accumulator:
            # 处理分叉内容（使用最终内容）
            if not streamed_content_accumulator and final_reply:
                if not assistant_started:
                    assistant_started = True
                    yield _build_stream_chunk(
                        response_id,
                        model_name,
                        role="assistant",
                        content=final_reply,
                    )
                else:
                    yield _build_stream_chunk(
                        response_id, model_name, content=final_reply
                    )
                streamed_content_accumulator = final_reply

        yield _build_stream_chunk(response_id, model_name, finish_reason="stop")
        if emit_usage:
            from app.usage_meter import build_usage, estimate_tokens

            yield _build_usage_chunk(
                response_id,
                model_name,
                build_usage(
                    prompt_tokens, estimate_tokens(streamed_content_accumulator)
                ),
            )
        yield "data: [DONE]\n\n"


def _create_standard_stream_generator(
    response_id: str,
    model_name: str,
    first_item: Any,
    stream_gen: Iterable[Any],
    client_type: str = "",
    *,
    lease: Any,
    tool_filter: Any = None,
    prompt_tokens: int = 0,
    emit_usage: bool = False,
) -> Generator[str, None, None]:
    """
    Standard 模式流式生成器：使用前端定义的 SSE 事件类型

    前端协议：
    - thinking_chunk: 流式思考片段
    - thinking_replace: 完整思考替换
    - search_metadata: 搜索结果
    - choices[0].delta.content: 正文内容

    对非 web 客户端（如 opencode 等严格 OpenAI 兼容客户端），thinking 使用
    delta.reasoning_content；search 以 markdown 注入 content（对齐 Heavy），
    避免自定义 SSE 字段触发 strict validation，同时不丢搜索结果。
    """
    streamed_content_accumulator = ""
    streamed_thinking_accumulator = ""
    collected_search_sources = []
    collected_search_queries = []
    authoritative_final_content = ""
    authoritative_final_source_type = ""
    assistant_started = False
    is_web_client = client_type == "web"
    failure_hint = ""
    client_disconnected = False

    try:
        for raw_item in _iter_stream_items(first_item, stream_gen):
            item = _normalize_stream_item(raw_item)
            item_type = item.get("type")

            if item_type == "final_content":
                final_text = str(item.get("text", "") or "").strip()
                if final_text:
                    authoritative_final_content = final_text
                    authoritative_final_source_type = str(
                        item.get("source_type", "") or ""
                    )
                continue

            # Standard 模式：处理 thinking
            if item_type == "thinking":
                thinking_text = item.get("text", "")
                if thinking_text:
                    streamed_thinking_accumulator += thinking_text
                    if is_web_client:
                        # Web UI: 前端协议 thinking_chunk
                        yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': thinking_text}, ensure_ascii=False)}\n\n"
                    else:
                        # 严格 OpenAI 客户端：reasoning_content；首包带 role
                        if not assistant_started:
                            assistant_started = True
                            yield _build_stream_chunk(
                                response_id,
                                model_name,
                                role="assistant",
                                thinking=thinking_text,
                            )
                        else:
                            yield _build_stream_chunk(
                                response_id, model_name, thinking=thinking_text
                            )
                continue

            # Standard 模式：处理 search（收集起来，最后输出）
            if item_type == "search":
                search_data = item.get("data", {})
                if isinstance(search_data, dict):
                    # 提取 queries 和 sources
                    queries = search_data.get("queries", [])
                    sources = search_data.get("sources", [])

                    if queries:
                        collected_search_queries.extend(queries)
                    if sources:
                        collected_search_sources.extend(sources)
                continue

            if item_type != "content":
                continue

            chunk_text = item.get("text", "")
            if not chunk_text:
                continue

            if tool_filter is not None:
                chunk_text = tool_filter.feed(chunk_text)
                if not chunk_text:
                    continue

            streamed_content_accumulator += chunk_text

            # 输出标准 OpenAI 格式的 delta
            if not assistant_started:
                assistant_started = True
                yield _build_stream_chunk(
                    response_id,
                    model_name,
                    role="assistant",
                    content=chunk_text,
                )
            else:
                yield _build_stream_chunk(response_id, model_name, content=chunk_text)
        lease.mark_success()
    except GeneratorExit:
        client_disconnected = True
        logger.info(
            "Standard streaming closed by client",
            extra={"request_info": {"event": "standard_stream_client_disconnected"}},
        )
        raise
    except asyncio.CancelledError:
        client_disconnected = True
        logger.info(
            "Standard streaming cancelled by client",
            extra={"request_info": {"event": "standard_stream_cancelled"}},
        )
        raise
    except BaseException as exc:
        if _is_client_disconnect_error(exc):
            logger.info(
                "Standard streaming connection closed by client",
                extra={
                    "request_info": {"event": "standard_stream_client_disconnected"}
                },
            )
            client_disconnected = True
            return
        _record_stream_failure(lease, exc, "standard_stream_interrupted")
        logger.error(
            "Standard streaming interrupted",
            exc_info=True,
            extra={"request_info": {"event": "standard_stream_interrupted"}},
        )
        failure_hint = "\n\n[上游连接中断，请稍后重试。]"
        streamed_content_accumulator += failure_hint
    finally:
        if client_disconnected:
            return

        if failure_hint:
            if not assistant_started:
                assistant_started = True
                yield _build_stream_chunk(
                    response_id,
                    model_name,
                    role="assistant",
                    content=failure_hint,
                )
            else:
                yield _build_stream_chunk(
                    response_id, model_name, content=failure_hint
                )

        # 先把流式过滤器里扣留的尾巴放出来，并判断有没有工具调用
        tool_calls: List[Dict[str, Any]] = []
        if tool_filter is not None:
            tail_text, tool_calls = tool_filter.finish()
            if tail_text:
                streamed_content_accumulator += tail_text
                if not assistant_started:
                    assistant_started = True
                    yield _build_stream_chunk(
                        response_id, model_name, role="assistant", content=tail_text
                    )
                else:
                    yield _build_stream_chunk(
                        response_id, model_name, content=tail_text
                    )

        if tool_calls:
            yield from _emit_tool_call_completion(
                response_id,
                model_name,
                tool_calls,
                assistant_started,
                prompt_tokens=prompt_tokens,
                emit_usage=emit_usage,
            )
            return

        # 选择最佳最终回复
        final_reply, _ = _select_best_final_reply(
            streamed_content_accumulator,
            authoritative_final_content,
            authoritative_final_source_type,
        )

        # 发送缺失的后缀（如果有）
        missing_suffix = _compute_missing_suffix(
            streamed_content_accumulator, final_reply
        )
        if missing_suffix:
            if not assistant_started:
                assistant_started = True
                yield _build_stream_chunk(
                    response_id,
                    model_name,
                    role="assistant",
                    content=missing_suffix,
                )
            else:
                yield _build_stream_chunk(
                    response_id, model_name, content=missing_suffix
                )
            streamed_content_accumulator += missing_suffix
        elif final_reply != streamed_content_accumulator:
            # 处理分叉内容（使用最终内容）
            if not streamed_content_accumulator and final_reply:
                if not assistant_started:
                    assistant_started = True
                    yield _build_stream_chunk(
                        response_id,
                        model_name,
                        role="assistant",
                        content=final_reply,
                    )
                else:
                    yield _build_stream_chunk(
                        response_id, model_name, content=final_reply
                    )
                streamed_content_accumulator = final_reply

        # 输出搜索结果
        if collected_search_sources or collected_search_queries:
            search_payload = {
                "queries": collected_search_queries,
                "sources": collected_search_sources,
            }
            if is_web_client:
                # Web UI：扩展 search_metadata（带 OpenAI chunk 外壳）
                yield _build_local_ui_chunk(
                    response_id,
                    model_name,
                    "search_metadata",
                    searches=search_payload,
                )
            else:
                # 严格客户端：markdown 注入 content，不丢结果
                search_md = _format_search_results_md(search_payload)
                if search_md:
                    if not assistant_started:
                        assistant_started = True
                        yield _build_stream_chunk(
                            response_id,
                            model_name,
                            role="assistant",
                            content=search_md,
                        )
                    else:
                        yield _build_stream_chunk(
                            response_id, model_name, content=search_md
                        )

        yield _build_stream_chunk(response_id, model_name, finish_reason="stop")
        if emit_usage:
            from app.usage_meter import build_usage, estimate_tokens

            yield _build_usage_chunk(
                response_id,
                model_name,
                build_usage(
                    prompt_tokens,
                    estimate_tokens(
                        streamed_content_accumulator + streamed_thinking_accumulator
                    ),
                ),
            )
        yield "data: [DONE]\n\n"


def _persist_round(
    manager,
    background_tasks: BackgroundTasks,
    conversation_id: str,
    user_prompt: str,
    assistant_reply: str,
    assistant_thinking: str = "",
) -> None:
    """
    持久化一轮对话并触发异步预压缩。

    预压缩逻辑：
    - 当 round >= WINDOW_ROUNDS//2 时，提前压缩滑出窗口的轮次
    - 使用 BackgroundTasks 确保不阻塞当前对话
    """
    round_index = manager.persist_round(
        conversation_id,
        user_prompt,
        assistant_reply,
        assistant_thinking=assistant_thinking,
    )

    # 异步预压缩：当窗口快满时提前压缩
    WINDOW_ROUNDS = 8  # 与 conversation.py 保持一致
    PRECOMPRESS_THRESHOLD = WINDOW_ROUNDS // 2  # 在第 4 轮时开始预压缩

    if round_index >= PRECOMPRESS_THRESHOLD:
        # 计算需要压缩的轮次（滑出窗口的轮次）
        round_to_compress = round_index - WINDOW_ROUNDS + 1
        if round_to_compress >= 0:
            background_tasks.add_task(
                compress_sliding_window_round,
                manager=manager,
                conversation_id=conversation_id,
                round_number=round_to_compress,
            )
            logger.info(
                "Triggered async pre-compression",
                extra={
                    "request_info": {
                        "event": "async_precompress_triggered",
                        "conversation_id": conversation_id,
                        "current_round": round_index,
                        "compress_round": round_to_compress,
                    }
                },
            )

    # 保留原有的压缩逻辑作为兜底
    background_tasks.add_task(
        compress_round_if_needed,
        manager=manager,
        conversation_id=conversation_id,
    )


def _persist_history_messages(
    manager, conversation_id: str, history_messages: List[Tuple[str, str, str]]
) -> None:
    for role, content, thinking in history_messages:
        manager.add_message(conversation_id, role, content, thinking)


def _is_client_disconnect_error(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return True
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        return exc.errno in {32, 54, 104, 10053, 10054}
    return False


async def _handle_lite_request(
    request: Request,
    req_body: ChatCompletionRequest,
    response: Response,
) -> JSONResponse | StreamingResponse | ChatCompletionResponse:
    """处理 Lite 模式请求（无记忆，单轮问答）"""
    pool = request.app.state.account_pool

    # 提取用户问题
    user_prompt = _prepare_messages_lite(req_body)
    # Lite 无历史，只需要给 usage 一个估算基准
    from app.usage_meter import build_usage, estimate_tokens as _estimate_tokens

    lite_prompt_tokens = _estimate_tokens(user_prompt)
    lite_stream_usage = _wants_stream_usage(req_body)

    # 验证模型
    if not is_supported_model(req_body.model):
        available_models = list_available_models()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req_body.model}'. Available models: {', '.join(available_models)}",
        )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    # 之前等于账号总数（792），意味着一次失败调用会把所有账号都试一遍：
    # 单请求耗时几分钟，还会把整池账号全打进冷却。
    # 上游 502/超时是全局问题，换号没用，限制在少数几次即可。
    max_retries = max(3, min(len(pool.clients), _max_attempts()))
    attempted_account_keys: set[str] = set()

    for attempt in range(1, max_retries + 1):
        lease = None
        client = None
        try:
            lease = await pool.acquire_async(
                exclude_keys=attempted_account_keys, model=req_body.model
            )
            attempted_account_keys.add(lease.account_key)
            client = lease.client

            # 构建 Lite transcript（无历史记忆）
            transcript = build_lite_transcript(
        user_prompt,
        req_body.model,
        resolve_reasoning_effort(
            getattr(req_body, "reasoning_effort", None),
            getattr(req_body, "thinking", None),
        ),
    )

            # 调用 Notion API（不使用 thread_id）
            stream_gen = _open_upstream(client, transcript)
            try:
                first_item = await asyncio.to_thread(next, stream_gen, None)
            except NotionUpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001 - curl_cffi 传输层异常
                raise NotionUpstreamError(
                    f"upstream transport error: {type(exc).__name__}: {exc}"[:200],
                    retriable=True,
                ) from exc

            if first_item is None:
                raise NotionUpstreamError(
                    "Notion upstream returned empty content.", retriable=True
                )

            # 流式响应
            if req_body.stream:
                stream_headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
                streaming_response = StreamingResponse(
                    _with_lease_release(
                        _with_heartbeat(
                            _create_lite_stream_generator(
                                response_id,
                                req_body.model,
                                first_item,
                                stream_gen,
                                lease=lease,
                                tool_filter=_tool_stream_filter(req_body),
                                prompt_tokens=lite_prompt_tokens,
                                emit_usage=lite_stream_usage,
                            )
                        ),
                        lease,
                    ),
                    media_type="text/event-stream",
                    headers=stream_headers,
                )
                lease = None
                return streaming_response

            # 非流式响应：整段上游读取放到线程里，事件循环继续服务别的请求
            collected = await asyncio.to_thread(
                _collect_standard_items, first_item, stream_gen
            )
            content_parts = collected["content"]
            thinking_parts = collected["thinking"]
            search_results = collected["search"]
            authoritative_final_content = collected["final_text"]
            authoritative_final_source_type = collected["final_source"]

            full_text, _ = _select_best_final_reply(
                "".join(content_parts),
                authoritative_final_content,
                authoritative_final_source_type,
            )

            if not full_text.strip():
                raise NotionUpstreamError(
                    "Notion upstream returned empty content.", retriable=True
                )

            response_text = (
                full_text if full_text.strip() else "[assistant_no_visible_content]"
            )
            lease.mark_success()
            _msg: Dict[str, Any] = {"role": "assistant", "content": response_text}
            _finish = "stop"
            if getattr(req_body, "tools", None) and _tools_on():
                from app.tool_bridge import clean_tool_call_content, extract_tool_calls

                _cleaned, _calls = extract_tool_calls(response_text, _tool_names(getattr(req_body, "tools", None)))
                if _calls:
                    _msg["content"] = clean_tool_call_content(_cleaned) or None
                    _msg["tool_calls"] = _calls
                    _finish = "tool_calls"
            return ChatCompletionResponse(
                id=response_id,
                model=req_body.model,
                choices=[
                    ChatMessageResponseChoice(message=_msg, finish_reason=_finish)
                ],
                usage=build_usage(
                    lite_prompt_tokens, _estimate_tokens(response_text)
                ),
            )

        except NotionUpstreamError as exc:
            if lease is not None:
                lease.record_upstream_error(exc)
            logger.warning(
                "Lite mode: Notion upstream failed",
                extra={
                    "request_info": {
                        "event": "lite_notion_upstream_failed",
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "status_code": exc.status_code,
                        "retriable": exc.retriable,
                        "response_excerpt": exc.response_excerpt,
                    }
                },
            )
            if attempt == max_retries or not exc.retriable:
                return _upstream_error_response(exc)
        except RuntimeError as exc:
            logger.error(
                "Lite mode: No available client in account pool",
                extra={
                    "request_info": {
                        "event": "lite_account_pool_unavailable",
                        "detail": str(exc),
                    }
                },
            )
            return _build_error_response(
                503,
                code="POOL_COOLING",
                message=str(exc),
                error_type="account_pool_cooling",
                suggestion="所有账号暂时冷却中，请等待几秒后重试",
            )
        except HTTPException:
            raise
        except Exception:
            if lease is not None:
                lease.mark_failed(reason="lite_unhandled_exception")
            logger.error(
                "Lite mode: Unhandled error",
                exc_info=True,
                extra={
                    "request_info": {
                        "event": "lite_unhandled_exception",
                        "attempt": attempt,
                    }
                },
            )
            if attempt == max_retries:
                return _build_error_response(
                    500,
                    code="INTERNAL_ERROR",
                    message="服务内部异常",
                    error_type="internal_error",
                    suggestion="请稍后重试，如持续出现请联系管理员",
                )
        finally:
            if lease is not None:
                lease.release()

    return _build_error_response(
        503,
        code="RETRIES_EXHAUSTED",
        message="所有重试均已失败",
        error_type="upstream_error",
        suggestion="Notion 上游服务暂时不可用，请稍后重试",
    )


async def _handle_standard_request(
    request: Request,
    req_body: ChatCompletionRequest,
    response: Response,
) -> JSONResponse | StreamingResponse | ChatCompletionResponse:
    """
    处理 Standard 模式请求（完整上下文，支持 thinking 和搜索）

    类似 Lite 模式，但：
    1. 发送完整 messages 历史
    2. 保留 thinking 输出
    3. 保留搜索结果输出
    """
    from app.conversation import build_standard_transcript

    pool = request.app.state.account_pool

    # 验证模型
    if not is_supported_model(req_body.model):
        available_models = list_available_models()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req_body.model}'. Available models: {', '.join(available_models)}",
        )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    # 之前等于账号总数（792），意味着一次失败调用会把所有账号都试一遍：
    # 单请求耗时几分钟，还会把整池账号全打进冷却。
    # 上游 502/超时是全局问题，换号没用，限制在少数几次即可。
    max_retries = max(3, min(len(pool.clients), _max_attempts()))
    attempted_account_keys: set[str] = set()
    repair_used = False  # 非流式：模型整段拒绝时补一次格式修复重试

    # ── 输入预算守卫（借鉴 sticture/notion2api）──────────────────────
    # 先注入工具清单，再按「JSON 转义后的 UTF-8 字节」压一次：
    # 超长工具结果保留头尾，仍然超限就丢最旧轮次，丢不动就 413。
    # 只做一次，重试复用同一份，避免反复压缩。
    prepared_messages: List[Dict[str, Any]] = [msg.dict() for msg in req_body.messages]
    if getattr(req_body, "tools", None) and _tools_on():
        from app.tool_bridge import (
            build_tool_preamble,
            build_tool_tail,
            should_offer_tools,
        )

        if should_offer_tools(req_body.tools):
            _pre = build_tool_preamble(req_body.tools, req_body.tool_choice)
            _tail = build_tool_tail(req_body.tools)
            for _i in range(len(prepared_messages) - 1, -1, -1):
                if prepared_messages[_i].get("role") == "user":
                    _body = prepared_messages[_i].get("content") or ""
                    prepared_messages[_i]["content"] = (
                        f"{_pre}\n\n{_body}\n\n{_tail}".strip()
                    )
                    break

    try:
        prepared_messages, budget_stats = enforce_budget(prepared_messages)
    except InputTooLarge as exc:
        stats = exc.stats
        logger.warning(
            "Input exceeds upstream budget",
            extra={"request_info": {"event": "context_too_large", **stats}},
        )
        return _build_error_response(
            413,
            code="CONTEXT_TOO_LARGE",
            message=(
                f"输入 {stats['bytes_after']} 字节，超过上游上限 "
                f"{stats['max_bytes']} 字节（已压缩 "
                f"{stats['compressed_tool_results']} 条工具结果、丢弃 "
                f"{stats['dropped_messages']} 条消息）"
            ),
            error_type="context_length_exceeded",
            suggestion="减少历史轮次或分片读取文件（用更小的 offset/limit）",
            detail=json.dumps(stats, ensure_ascii=False),
        )

    if budget_stats["dropped_messages"] or budget_stats["compressed_tool_results"]:
        logger.info(
            "Input trimmed to upstream budget",
            extra={"request_info": {"event": "context_trimmed", **budget_stats}},
        )

    prompt_tokens = _prompt_tokens_of(prepared_messages)
    stream_usage = _wants_stream_usage(req_body)

    for attempt in range(1, max_retries + 1):
        lease = None
        client = None
        try:
            lease = await pool.acquire_async(
                exclude_keys=attempted_account_keys, model=req_body.model
            )
            attempted_account_keys.add(lease.account_key)
            client = lease.client

            # 构建 Standard transcript（完整上下文）
            # 从 client 提取账号信息
            account = {
                "user_id": client.user_id,
                "space_id": client.space_id,
            }
            messages = [dict(message) for message in prepared_messages]

            transcript = build_standard_transcript(
                messages,
                req_body.model,
                account,
                resolve_reasoning_effort(
                    getattr(req_body, "reasoning_effort", None),
                    getattr(req_body, "thinking", None),
                ),
            )

            # 调用 Notion API（不使用 thread_id，让 Notion ��动处理）
            stream_gen = _open_upstream(client, transcript)
            try:
                first_item = next(stream_gen, None)
            except NotionUpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001 - curl_cffi 传输层异常
                raise NotionUpstreamError(
                    f"upstream transport error: {type(exc).__name__}: {exc}"[:200],
                    retriable=True,
                ) from exc

            if first_item is None:
                raise NotionUpstreamError(
                    "Notion upstream returned empty content.", retriable=True
                )

            # 流式响应
            if req_body.stream:
                stream_headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
                client_type = request.headers.get("X-Client-Type", "").lower()
                streaming_response = StreamingResponse(
                    _with_lease_release(
                        _with_heartbeat(
                            _create_standard_stream_generator(
                                response_id,
                                req_body.model,
                                first_item,
                                stream_gen,
                                client_type=client_type,
                                lease=lease,
                                tool_filter=_tool_stream_filter(req_body),
                                prompt_tokens=prompt_tokens,
                                emit_usage=stream_usage,
                            )
                        ),
                        lease,
                    ),
                    media_type="text/event-stream",
                    headers=stream_headers,
                )
                lease = None
                return streaming_response

            # 非流式响应
            content_parts: list[str] = []
            thinking_parts: list[str] = []
            search_results: list[dict] = []
            authoritative_final_content = ""
            authoritative_final_source_type = ""

            for raw_item in _iter_stream_items(first_item, stream_gen):
                item = _normalize_stream_item(raw_item)
                item_type = item.get("type")

                if item_type == "final_content":
                    final_text = str(item.get("text", "") or "").strip()
                    if final_text:
                        authoritative_final_content = final_text
                        authoritative_final_source_type = str(
                            item.get("source_type", "") or ""
                        )
                    continue

                # Standard 模式：处理 thinking
                if item_type == "thinking":
                    thinking_text = item.get("text", "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)
                    continue

                # Standard 模式：处理 search
                if item_type == "search":
                    search_data = item.get("data", {})
                    if search_data:
                        search_results.append(search_data)
                    continue

                if item_type != "content":
                    continue

                chunk_text = item.get("text", "")
                if chunk_text:
                    content_parts.append(chunk_text)

            full_text, _ = _select_best_final_reply(
                "".join(content_parts),
                authoritative_final_content,
                authoritative_final_source_type,
            )

            if not full_text.strip():
                raise NotionUpstreamError(
                    "Notion upstream returned empty content.", retriable=True
                )

            response_text = (
                full_text if full_text.strip() else "[assistant_no_visible_content]"
            )

            # 客户端带了 tools：把模型输出的 <tool_call> 解析成标准 tool_calls
            msg_payload: Dict[str, Any] = {"role": "assistant", "content": response_text}
            finish_reason = "stop"
            if getattr(req_body, "tools", None) and _tools_on():
                from app.tool_bridge import (
                    clean_tool_call_content,
                    extract_tool_calls,
                    looks_like_refusal,
                )

                cleaned_text, parsed_calls = extract_tool_calls(response_text, _tool_names(getattr(req_body, "tools", None)))
                if parsed_calls:
                    msg_payload["content"] = clean_tool_call_content(cleaned_text) or None
                    msg_payload["tool_calls"] = parsed_calls
                    finish_reason = "tool_calls"
                elif not repair_used and looks_like_refusal(response_text):
                    # 模型整段拒绝、一个调用都没发：补一条格式修复指令重试一次。
                    # 同时去掉 system —— 实测身份断言是拒绝的主因，重试时先卸掉它。
                    # 流式路径没法撤回已发出的文字，所以只在非流式做。
                    repair_used = True
                    prepared_messages = [
                        message for message in prepared_messages
                        if message.get("role") != "system"
                    ]
                    prepared_messages.append({
                        "role": "user",
                        "content": (
                            "上一条回复没有按约定输出调用。执行由外部执行器负责，"
                            "你只需要输出一行调用表示法，不要解释，也不要回答内容：\n"
                            '<tool_call>{"name": "<函数名>", "arguments": {<参数>}}</tool_call>'
                        ),
                    })
                    logger.info(
                        "Tool-call repair retry",
                        extra={"request_info": {"event": "tool_call_repair_retry",
                                                "attempt": attempt}},
                    )
                    continue

            # 如果有 thinking，添加到扩展字段（前端会读取）
            if thinking_parts:
                msg_payload["thinking"] = "".join(thinking_parts)

            # 与流式路径统一：搜索结果同样以 Markdown 注入正文，
            # 否则同一请求在 stream=false 时看不到「已搜索 / 来源」这段内容
            if search_results:
                _q, _src = [], []
                for _r in search_results:
                    if isinstance(_r, dict):
                        _q.extend(_r.get("queries", []))
                        _src.extend(_r.get("sources", []))
                _md = _format_search_results_md({"queries": _q, "sources": _src})
                if _md:
                    msg_payload["content"] = (msg_payload.get("content") or "") + _md

            # 构建响应
            from app.usage_meter import build_usage, estimate_tokens

            response_obj = ChatCompletionResponse(
                id=response_id,
                model=req_body.model,
                choices=[ChatMessageResponseChoice(
                    message=msg_payload, finish_reason=finish_reason)],
                usage=build_usage(prompt_tokens, estimate_tokens(response_text)),
            )

            # 如果有搜索结果，添加到扩展字段（前端会读取）
            if search_results:
                # 提取 queries 和 sources
                all_queries = []
                all_sources = []
                for result in search_results:
                    if isinstance(result, dict):
                        all_queries.extend(result.get("queries", []))
                        all_sources.extend(result.get("sources", []))

                if all_queries or all_sources:
                    # 添加到自定义字段
                    response_obj.search_metadata = {
                        "queries": all_queries,
                        "sources": all_sources,
                    }

            lease.mark_success()
            return response_obj

        except NotionUpstreamError as exc:
            if lease is not None:
                lease.record_upstream_error(exc)
            logger.warning(
                "Standard mode: Notion upstream failed",
                extra={
                    "request_info": {
                        "event": "standard_notion_upstream_failed",
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "status_code": exc.status_code,
                        "retriable": exc.retriable,
                        "response_excerpt": exc.response_excerpt,
                    }
                },
            )
            if attempt == max_retries or not exc.retriable:
                return _upstream_error_response(exc)
        except RuntimeError as exc:
            logger.error(
                "Standard mode: No available client in account pool",
                extra={
                    "request_info": {
                        "event": "standard_account_pool_unavailable",
                        "detail": str(exc),
                    }
                },
            )
            return _build_error_response(
                503,
                code="POOL_COOLING",
                message=str(exc),
                error_type="account_pool_cooling",
                suggestion="所有账号暂时冷却中，请等待几秒后重试",
            )
        except HTTPException:
            raise
        except Exception:
            if lease is not None:
                lease.mark_failed(reason="standard_unhandled_exception")
            logger.error(
                "Standard mode: Unhandled error",
                exc_info=True,
                extra={
                    "request_info": {
                        "event": "standard_unhandled_exception",
                        "attempt": attempt,
                    }
                },
            )
            if attempt == max_retries:
                return _build_error_response(
                    500,
                    code="INTERNAL_ERROR",
                    message="服务内部异常",
                    error_type="internal_error",
                    suggestion="请稍后重试，如持续出现请联系管理员",
                )
        finally:
            if lease is not None:
                lease.release()

    return _build_error_response(
        503,
        code="RETRIES_EXHAUSTED",
        message="所有重试均已失败",
        error_type="upstream_error",
        suggestion="Notion 上游服务暂时不可用，请稍后重试",
    )


@router.post("/chat/completions", tags=["chat"])
async def create_chat_completion(
    request: Request,
    req_body: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    response: Response,
):
    """
    创建聊天请求，严格兼容 OpenAI API。

    速率限制：
    - Lite 模式：30/分钟（适合单轮问答）
    - Standard 模式：25/分钟（完整上下文，支持 thinking 和搜索）
    - Heavy 模式：20/分钟（包含会话管理）
    """
    from app.config import is_standard_mode

    # 把本次请求的模型放进 context，供调用明细记账时读取
    try:
        from app.call_context import set_current_model

        set_current_model(req_body.model)
    except Exception:  # noqa: BLE001
        pass

    # 诊断：记录本次请求带进来的消息结构，用于判断「串台文本」是客户端发来的
    # 还是上游返回的。只记前 60 字，避免完整内容落盘。
    try:
        messages_in = req_body.messages or []
        logger.info(
            "Chat request summary",
            extra={
                "request_info": {
                    "event": "chat_request_summary",
                    "message_count": len(messages_in),
                    "roles": [m.role for m in messages_in],
                    "reasoning_effort": resolve_reasoning_effort(
                        getattr(req_body, "reasoning_effort", None),
                        getattr(req_body, "thinking", None),
                    ),
                    "system_len": sum(
                        len(m.content) for m in messages_in if m.role == "system"
                    ),
                    "first_user": next(
                        (m.content for m in messages_in if m.role == "user"), ""
                    )[:60],
                    "last_user": (messages_in[-1].content if messages_in else "")[:60],
                }
            },
        )
    except Exception:  # noqa: BLE001
        pass

    # Lite 模式：单轮问答，无记忆
    if is_lite_mode():
        return await _handle_lite_request(request, req_body, response)

    # Standard 模式：完整上下文，支持 thinking 和搜索
    if is_standard_mode():
        return await _handle_standard_request(request, req_body, response)

    # Heavy 模式：完整会话管理
    pool = request.app.state.account_pool
    manager = request.app.state.conversation_manager

    user_prompt, history_messages, raw_user_prompt = _prepare_messages(req_body)
    # Heavy 的 payload 由会话管理器拼装，这里只按「本轮 user 文本」估个基准；
    # 精确值要等上游返回，本网关拿不到 token 数，只能估算（见 usage_meter）。
    from app.usage_meter import build_usage as _build_usage
    from app.usage_meter import estimate_tokens as _estimate_heavy_tokens

    heavy_prompt_tokens = _estimate_heavy_tokens(user_prompt)
    heavy_stream_usage = _wants_stream_usage(req_body)
    recall_query = (
        _extract_recall_query(raw_user_prompt)
        if _contains_recall_intent(raw_user_prompt)
        else None
    )

    if not is_supported_model(req_body.model):
        available_models = list_available_models()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model '{req_body.model}'. Available models: {', '.join(available_models)}",
        )

    conversation_id = (
        req_body.conversation_id.strip() if req_body.conversation_id else ""
    )
    restore_history = False
    if not conversation_id:
        conversation_id = manager.new_conversation()
        restore_history = True
    elif not manager.conversation_exists(conversation_id):
        logger.warning(
            "Conversation id not found, creating a fresh conversation",
            extra={
                "request_info": {
                    "event": "conversation_id_not_found",
                    "provided_conversation_id": conversation_id,
                }
            },
        )
        conversation_id = manager.new_conversation()
        restore_history = True

    # 关键修复：总是持久化客户端发送的历史消息，避免上下文丢失
    # 即使 conversation_id 已存在，也需要同步客户端发送的完整历史
    if history_messages:
        # 检查是否需要持久化（避免重复）
        with manager._get_conn() as conn:
            existing_count = manager._count_messages(conn, conversation_id)
            history_count = len(history_messages)

            # 只有当客户端发送的历史消息多于数据库中的消息时才持久化
            # 这样可以：
            # 1. 避免重复持久化相同的历史
            # 2. 确保客户端发送的完整历史被保存
            # 3. 解决"滑动窗口缺失 AI 回复"的 bug
            if history_count > existing_count:
                _persist_history_messages(manager, conversation_id, history_messages)
                restored_user_count = sum(
                    1 for role, *_ in history_messages if role == "user"
                )
                restored_assistant_count = sum(
                    1 for role, *_ in history_messages if role == "assistant"
                )

                logger.info(
                    "Restored history into conversation",
                    extra={
                        "request_info": {
                            "event": "conversation_history_restored",
                            "conversation_id": conversation_id,
                            "restore_history_flag": restore_history,
                            "existing_count": existing_count,
                            "history_count": history_count,
                            "restored_total": len(history_messages),
                            "restored_user_count": restored_user_count,
                            "restored_assistant_count": restored_assistant_count,
                        }
                    },
                )

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    # 之前等于账号总数（792），意味着一次失败调用会把所有账号都试一遍：
    # 单请求耗时几分钟，还会把整池账号全打进冷却。
    # 上游 502/超时是全局问题，换号没用，限制在少数几次即可。
    max_retries = max(3, min(len(pool.clients), _max_attempts()))
    attempted_account_keys: set[str] = set()

    for attempt in range(1, max_retries + 1):
        lease = None
        client = None
        try:
            # 获取或创建 thread_id 以保持对话上下文
            thread_id = manager.get_conversation_thread_id(conversation_id)
            thread_account_key = manager.get_conversation_thread_account(conversation_id)

            # 检测用户是否在对话中途切换了模型。Notion 的 thread 会把 config 中的 model
            # 粘在服务端 thread 对象上，复用同一 thread 即使 transcript 里写了新 model，
            # 上游实际仍按原始模型执行。必须丢掉旧 thread 让 Notion 按新 model 新建，
            # 对话上下文由我们自己的滑动窗口 + 压缩摘要重建，不会失忆。
            if thread_id:
                bound_model = manager.get_conversation_thread_model(conversation_id)
                # bound_model 为 None 表示升级前的遗留对话，没记录过模型绑定，
                # 无法判断 thread 当初绑的是什么模型。为避免继续踩老 bug，
                # 一律当作"可能不匹配"处理，丢弃老 thread 重新开。
                if not bound_model or bound_model != req_body.model:
                    logger.info(
                        "Recreating Notion thread: model changed or legacy binding",
                        extra={
                            "request_info": {
                                "event": "thread_model_switched",
                                "conversation_id": conversation_id,
                                "old_model": bound_model,
                                "new_model": req_body.model,
                                "reason": "model_mismatch" if bound_model else "legacy_no_binding",
                            }
                        },
                    )
                    manager.clear_conversation_thread(conversation_id)
                    thread_id = None
                    thread_account_key = None
                elif not thread_account_key:
                    # 升级前的旧数据没有账号绑定，无法证明 thread 属于当前账号。
                    logger.info(
                        "Recreating Notion thread: legacy account binding missing",
                        extra={
                            "request_info": {
                                "event": "thread_account_switched",
                                "conversation_id": conversation_id,
                                "reason": "legacy_no_account_binding",
                            }
                        },
                    )
                    manager.clear_conversation_thread(conversation_id)
                    thread_id = None

            affinity_key = thread_account_key if thread_id else None
            lease = await pool.acquire_async(
                affinity_key=affinity_key,
                exclude_keys=attempted_account_keys,
                model=req_body.model,
            )
            attempted_account_keys.add(lease.account_key)
            client = lease.client

            if thread_id and thread_account_key != lease.account_key:
                logger.info(
                    "Recreating Notion thread: routed to a different account",
                    extra={
                        "request_info": {
                            "event": "thread_account_switched",
                            "conversation_id": conversation_id,
                            "reason": "account_unavailable",
                        }
                    },
                )
                manager.clear_conversation_thread(conversation_id)
                thread_id = None
                thread_account_key = None

            transcript_payload = manager.get_transcript_payload(
                notion_client=client,
                conversation_id=conversation_id,
                new_prompt=user_prompt,
                model_name=req_body.model,
                recall_query=recall_query,
            )
            transcript = transcript_payload["transcript"]
            memory_degraded = bool(transcript_payload.get("memory_degraded"))
            memory_headers = {"X-Memory-Status": "degraded"} if memory_degraded else {}

            new_thread = thread_id is None
            if new_thread:
                thread_id = str(uuid.uuid4())

            stream_gen = client.stream_response(
                transcript,
                thread_id=thread_id,
                force_new_thread=new_thread,
            )
            try:
                first_item = next(stream_gen, None)
            except NotionUpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001 - curl_cffi 传输层异常
                raise NotionUpstreamError(
                    f"upstream transport error: {type(exc).__name__}: {exc}"[:200],
                    retriable=True,
                ) from exc

            if first_item is None:
                raise NotionUpstreamError(
                    "Notion upstream returned empty content.", retriable=True
                )

            # 只有上游真正开始返回数据后才保存绑定；调用方生成的 thread_id
            # 避免共享客户端上的 current_thread_id 在并发请求间串号。
            if new_thread:
                manager.set_conversation_thread_id(
                    conversation_id,
                    thread_id,
                    model_name=req_body.model,
                    account_key=lease.account_key,
                )

            def openai_stream_generator(stream_lease: Any) -> Generator[str, None, None]:
                streamed_content_accumulator = ""
                thinking_accumulator = ""
                authoritative_final_content = ""
                authoritative_final_source_type = ""
                assistant_started = False
                pending_search_md = ""
                client_type = request.headers.get("X-Client-Type", "").lower()
                recent_thinking_buffer: list[str] = []
                failure_hint = ""
                client_disconnected = False
                tool_filter = _tool_stream_filter(req_body)

                try:
                    for raw_item in _iter_stream_items(first_item, stream_gen):
                        item = _normalize_stream_item(raw_item)
                        item_type = item.get("type")

                        if item_type == "search":
                            search_data = item.get("data")
                            if isinstance(search_data, dict) and search_data:
                                pending_search_md += _format_search_results_md(
                                    search_data
                                )
                                if client_type == "web":
                                    yield _build_local_ui_chunk(
                                        response_id,
                                        req_body.model,
                                        "search_metadata",
                                        searches=search_data,
                                    )
                            continue

                        if item_type == "final_content":
                            final_text = str(item.get("text", "") or "").strip()
                            if final_text:
                                authoritative_final_content = final_text
                                authoritative_final_source_type = str(
                                    item.get("source_type", "") or ""
                                )
                            continue

                        if item_type == "thinking":
                            thinking_text = item.get("text", "")
                            if thinking_text:
                                thinking_accumulator += thinking_text
                                # Track recent thinking for overlap detection
                                recent_thinking_buffer.append(thinking_text)
                                # Keep buffer manageable (max 40 recent chunks)
                                if len(recent_thinking_buffer) > 40:
                                    recent_thinking_buffer.pop(0)

                                if not assistant_started:
                                    assistant_started = True
                                    yield _build_stream_chunk(
                                        response_id,
                                        req_body.model,
                                        role="assistant",
                                        thinking=thinking_text,
                                    )
                                else:
                                    yield _build_stream_chunk(
                                        response_id,
                                        req_body.model,
                                        thinking=thinking_text,
                                    )
                            continue

                        if item_type != "content":
                            continue

                        chunk_text = item.get("text", "")
                        if not chunk_text and not pending_search_md:
                            continue

                        if tool_filter is not None:
                            chunk_text = tool_filter.feed(chunk_text)
                            if not chunk_text and not pending_search_md:
                                continue

                        # Check if content overlaps with recent thinking (prevents thinking leakage)
                        if recent_thinking_buffer and chunk_text.strip():
                            combined_recent_thinking = "".join(recent_thinking_buffer)
                            chunk_normalized = chunk_text.strip()

                            # Use normalized text without spaces for robust comparison
                            combined_norm = re.sub(r"\s+", "", combined_recent_thinking)
                            chunk_norm = re.sub(r"\s+", "", chunk_normalized)

                            # Check for significant overlap - skip duplicate content
                            # We only skip if a sufficiently long chunk matches to avoid swallowing short common characters.
                            if (
                                chunk_norm
                                and len(chunk_norm) > 3
                                and (
                                    chunk_norm in combined_norm
                                    or (
                                        len(chunk_norm) > 10
                                        and chunk_norm[:10] in combined_norm
                                    )
                                )
                            ):
                                # Skip this chunk as it's likely duplicated thinking content
                                logger.debug(
                                    "Skipping duplicate content chunk that overlaps with thinking",
                                    extra={
                                        "request_info": {
                                            "event": "content_overlap_with_thinking",
                                            "chunk_length": len(chunk_text),
                                            "overlap_detected": True,
                                        }
                                    },
                                )
                                continue

                        # 在第一个正文内容发出前，把积攒的搜索信息拼上去
                        if pending_search_md and client_type != "web":
                            chunk_text = pending_search_md + chunk_text

                        if pending_search_md:
                            pending_search_md = ""

                        streamed_content_accumulator += chunk_text
                        if not assistant_started:
                            assistant_started = True
                            yield _build_stream_chunk(
                                response_id,
                                req_body.model,
                                role="assistant",
                                content=chunk_text,
                            )
                        else:
                            yield _build_stream_chunk(
                                response_id, req_body.model, content=chunk_text
                            )
                    stream_lease.mark_success()
                except GeneratorExit:
                    client_disconnected = True
                    logger.info(
                        "Streaming response closed by downstream client",
                        extra={
                            "request_info": {
                                "event": "stream_client_disconnected",
                                "conversation_id": conversation_id,
                                "attempt": attempt,
                            }
                        },
                    )
                    raise
                except asyncio.CancelledError:
                    client_disconnected = True
                    logger.info(
                        "Streaming response cancelled by downstream client",
                        extra={
                            "request_info": {
                                "event": "stream_cancelled_by_client",
                                "conversation_id": conversation_id,
                                "attempt": attempt,
                            }
                        },
                    )
                    raise
                except BaseException as exc:
                    if _is_client_disconnect_error(exc):
                        logger.info(
                            "Streaming connection closed by downstream client",
                            extra={
                                "request_info": {
                                    "event": "stream_client_disconnected",
                                    "conversation_id": conversation_id,
                                    "attempt": attempt,
                                }
                            },
                        )
                        client_disconnected = True
                        return
                    _record_stream_failure(
                        stream_lease, exc, "stream_interrupted"
                    )
                    try:
                        manager.clear_conversation_thread(conversation_id)
                    except Exception:
                        logger.warning(
                            "Failed to clear poisoned thread binding",
                            exc_info=True,
                            extra={
                                "request_info": {
                                    "event": "thread_binding_clear_failed",
                                    "conversation_id": conversation_id,
                                }
                            },
                        )
                    log_method = (
                        logger.warning
                        if isinstance(exc, NotionUpstreamError)
                        else logger.error
                    )
                    log_method(
                        "Streaming response interrupted",
                        exc_info=True,
                        extra={
                            "request_info": {
                                "event": "stream_interrupted",
                                "conversation_id": conversation_id,
                                "attempt": attempt,
                                "is_upstream_error": isinstance(
                                    exc, NotionUpstreamError
                                ),
                            }
                        },
                    )
                    failure_hint = "\n\n[上游连接中断，请稍后重试。]"
                    streamed_content_accumulator += failure_hint
                finally:
                    if client_disconnected:
                        return

                    if failure_hint:
                        if not assistant_started:
                            assistant_started = True
                            yield _build_stream_chunk(
                                response_id,
                                req_body.model,
                                role="assistant",
                                content=failure_hint,
                            )
                        else:
                            yield _build_stream_chunk(
                                response_id,
                                req_body.model,
                                content=failure_hint,
                            )

                    # 先把流式过滤器里扣留的尾巴放出来，并判断有没有工具调用
                    tool_calls: List[Dict[str, Any]] = []
                    if tool_filter is not None:
                        tail_text, tool_calls = tool_filter.finish()
                        if tail_text:
                            if pending_search_md and client_type != "web":
                                tail_text = pending_search_md + tail_text
                                pending_search_md = ""
                            streamed_content_accumulator += tail_text
                            if not assistant_started:
                                assistant_started = True
                                yield _build_stream_chunk(
                                    response_id,
                                    req_body.model,
                                    role="assistant",
                                    content=tail_text,
                                )
                            else:
                                yield _build_stream_chunk(
                                    response_id, req_body.model, content=tail_text
                                )

                    if tool_calls:
                        yield from _emit_tool_call_completion(
                            response_id, req_body.model, tool_calls, assistant_started
                        )
                        return

                    final_reply, reply_decision = _select_best_final_reply(
                        streamed_content_accumulator,
                        authoritative_final_content,
                        authoritative_final_source_type,
                    )

                    missing_suffix = _compute_missing_suffix(
                        streamed_content_accumulator, final_reply
                    )
                    if missing_suffix:
                        suffix_to_emit = missing_suffix
                        if (
                            pending_search_md
                            and client_type != "web"
                            and not streamed_content_accumulator
                        ):
                            suffix_to_emit = pending_search_md + suffix_to_emit
                            pending_search_md = ""
                        if not assistant_started:
                            assistant_started = True
                            yield _build_stream_chunk(
                                response_id,
                                req_body.model,
                                role="assistant",
                                content=suffix_to_emit,
                            )
                        else:
                            yield _build_stream_chunk(
                                response_id, req_body.model, content=suffix_to_emit
                            )
                        streamed_content_accumulator += suffix_to_emit
                    elif final_reply != streamed_content_accumulator:
                        # Diverged bodies cannot be safely "patched" in plain OpenAI deltas.
                        # Web client supports replace event to keep rendered body aligned with persisted final reply.
                        if client_type == "web":
                            yield _build_local_ui_chunk(
                                response_id,
                                req_body.model,
                                "content_replace",
                                content=final_reply,
                                source_type=authoritative_final_source_type,
                                decision=reply_decision,
                            )
                            streamed_content_accumulator = final_reply
                        elif not streamed_content_accumulator and final_reply:
                            # Non-web fallback when nothing has been shown yet.
                            emit_text = final_reply
                            if pending_search_md and client_type != "web":
                                emit_text = pending_search_md + emit_text
                                pending_search_md = ""
                            if not assistant_started:
                                assistant_started = True
                                yield _build_stream_chunk(
                                    response_id,
                                    req_body.model,
                                    role="assistant",
                                    content=emit_text,
                                )
                            else:
                                yield _build_stream_chunk(
                                    response_id, req_body.model, content=emit_text
                                )
                            streamed_content_accumulator = final_reply

                    thinking_replacement = _build_thinking_replacement(
                        streamed_content_accumulator,
                        thinking_accumulator,
                        final_reply,
                        authoritative_final_source_type,
                    )
                    if client_type == "web" and thinking_replacement is not None:
                        yield _build_local_ui_chunk(
                            response_id,
                            req_body.model,
                            "thinking_replace",
                            thinking=thinking_replacement["thinking"],
                            decision=thinking_replacement["decision"],
                            overlap_ratio=thinking_replacement["overlap_ratio"],
                            source_type=thinking_replacement["source_type"],
                            reply_decision=reply_decision,
                        )

                    persisted_thinking = (
                        str(thinking_replacement["thinking"])
                        if thinking_replacement is not None
                        else thinking_accumulator
                    )
                    if final_reply.strip() or persisted_thinking.strip():
                        try:
                            _persist_round(
                                manager,
                                background_tasks,
                                conversation_id,
                                user_prompt,
                                final_reply,
                                persisted_thinking,
                            )
                        except Exception:
                            logger.error(
                                "Failed to persist conversation round",
                                exc_info=True,
                                extra={
                                    "request_info": {
                                        "event": "conversation_persist_failed",
                                        "conversation_id": conversation_id,
                                    }
                                },
                            )
                    yield _build_stream_chunk(
                        response_id, req_body.model, finish_reason="stop"
                    )
                    if heavy_stream_usage:
                        from app.usage_meter import build_usage, estimate_tokens

                        yield _build_usage_chunk(
                            response_id,
                            req_body.model,
                            build_usage(
                                heavy_prompt_tokens,
                                estimate_tokens(
                                    streamed_content_accumulator + thinking_accumulator
                                ),
                            ),
                        )
                    yield "data: [DONE]\n\n"

            if req_body.stream:
                stream_headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "X-Conversation-Id": conversation_id,
                    **memory_headers,
                }
                stream_lease = lease
                streaming_response = StreamingResponse(
                    _with_lease_release(
                        _with_heartbeat(
                            openai_stream_generator(stream_lease)
                        ),
                        stream_lease,
                    ),
                    media_type="text/event-stream",
                    headers=stream_headers,
                )
                lease = None
                return streaming_response

            content_parts: list[str] = []
            thinking_parts: list[str] = []
            search_results: list[dict] = []
            authoritative_final_content = ""
            authoritative_final_source_type = ""
            for raw_item in _iter_stream_items(first_item, stream_gen):
                item = _normalize_stream_item(raw_item)
                item_type = item.get("type")
                if item_type == "final_content":
                    final_text = str(item.get("text", "") or "").strip()
                    if final_text:
                        authoritative_final_content = final_text
                        authoritative_final_source_type = str(
                            item.get("source_type", "") or ""
                        )
                    continue
                if item_type == "thinking":
                    thinking_text = str(item.get("text", "") or "")
                    if thinking_text:
                        thinking_parts.append(thinking_text)
                    continue
                if item_type != "content":
                    continue
                chunk_text = item.get("text", "")
                if chunk_text:
                    content_parts.append(chunk_text)

            full_text, _ = _select_best_final_reply(
                "".join(content_parts),
                authoritative_final_content,
                authoritative_final_source_type,
            )
            merged_thinking = "".join(thinking_parts).strip()
            if not full_text.strip() and not merged_thinking:
                raise NotionUpstreamError(
                    "Notion upstream returned empty content.", retriable=True
                )

            _persist_round(
                manager,
                background_tasks,
                conversation_id,
                user_prompt,
                full_text,
                merged_thinking,
            )
            response.headers["X-Conversation-Id"] = conversation_id
            if memory_degraded:
                response.headers["X-Memory-Status"] = "degraded"

            response_text = (
                full_text if full_text.strip() else "[assistant_no_visible_content]"
            )
            lease.mark_success()
            _msg: Dict[str, Any] = {"role": "assistant", "content": response_text}
            _finish = "stop"
            if getattr(req_body, "tools", None) and _tools_on():
                from app.tool_bridge import clean_tool_call_content, extract_tool_calls

                _cleaned, _calls = extract_tool_calls(response_text, _tool_names(getattr(req_body, "tools", None)))
                if _calls:
                    _msg["content"] = clean_tool_call_content(_cleaned) or None
                    _msg["tool_calls"] = _calls
                    _finish = "tool_calls"
            return ChatCompletionResponse(
                id=response_id,
                model=req_body.model,
                choices=[
                    ChatMessageResponseChoice(message=_msg, finish_reason=_finish)
                ],
                usage=_build_usage(
                    heavy_prompt_tokens, _estimate_heavy_tokens(response_text)
                ),
            )
        except NotionUpstreamError as exc:
            if lease is not None:
                lease.record_upstream_error(exc)
            try:
                manager.clear_conversation_thread(conversation_id)
            except Exception:
                logger.warning(
                    "Failed to clear thread binding after upstream failure",
                    exc_info=True,
                    extra={
                        "request_info": {
                            "event": "thread_binding_clear_failed",
                            "conversation_id": conversation_id,
                        }
                    },
                )
            logger.warning(
                "Notion upstream failed",
                extra={
                    "request_info": {
                        "event": "notion_upstream_failed",
                        "attempt": attempt,
                        "max_retries": max_retries,
                        "conversation_id": conversation_id,
                        "status_code": exc.status_code,
                        "retriable": exc.retriable,
                        "response_excerpt": exc.response_excerpt,
                    }
                },
            )
            if attempt == max_retries or not exc.retriable:
                return _upstream_error_response(exc)
        except RuntimeError as exc:
            logger.error(
                "No available client in account pool",
                extra={
                    "request_info": {
                        "event": "account_pool_unavailable",
                        "detail": str(exc),
                    }
                },
            )
            return _build_error_response(
                503,
                code="POOL_COOLING",
                message=str(exc),
                error_type="account_pool_cooling",
                suggestion="所有账号暂时冷却中，请等待几秒后重试",
            )
        except HTTPException:
            raise
        except Exception:
            if lease is not None:
                lease.mark_failed(reason="chat_completion_unhandled_exception")
            try:
                manager.clear_conversation_thread(conversation_id)
            except Exception:
                logger.warning(
                    "Failed to clear thread binding after internal error",
                    exc_info=True,
                    extra={
                        "request_info": {
                            "event": "thread_binding_clear_failed",
                            "conversation_id": conversation_id,
                        }
                    },
                )
            logger.error(
                "Unhandled chat completion error",
                exc_info=True,
                extra={
                    "request_info": {
                        "event": "chat_completion_unhandled_exception",
                        "attempt": attempt,
                        "conversation_id": conversation_id,
                    }
                },
            )
            if attempt == max_retries:
                return _build_error_response(
                    500,
                    code="INTERNAL_ERROR",
                    message="服务内部异常",
                    error_type="internal_error",
                    suggestion="请稍后重试，如持续出现请联系管理员",
                )
        finally:
            if lease is not None:
                lease.release()

    return _build_error_response(
        503,
        code="RETRIES_EXHAUSTED",
        message="所有重试均已失败",
        error_type="upstream_error",
        suggestion="Notion 上游服务暂时不可用，请稍后重试",
    )


@router.delete("/conversations/{conversation_id}", tags=["chat"])
async def delete_conversation(conversation_id: str, request: Request):
    manager = request.app.state.conversation_manager
    deleted = manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"id": conversation_id, "deleted": True}
