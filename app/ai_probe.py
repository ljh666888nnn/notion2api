# -*- coding: utf-8 -*-
"""ai_probe.py - 真实推理探活。

额度/套餐能探到不代表 AI 真的能用：账号可能额度耗尽、被限流、
或者 AI 功能被单独关掉。只有发一次真实推理才能确认。

巡检用它把 exhausted 的号找出来，避免池子把请求派给废号。
"""
from __future__ import annotations

import time
from typing import Any, Dict

from app.logger import logger

PROBE_PROMPT = "Reply with exactly: PONG"
PROBE_MODEL = "gpt-5.2"
TEXT_CHUNK_TYPES = ("content", "final_content")


def _build_transcript(prompt: str, user_id: str = "") -> list[dict[str, Any]]:
    """复用 chat 主链路那份 transcript。

    自己拼精简版会拿到 empty stream——上游要求完整 config + context 块，
    这里直接用 build_lite_transcript，保证和真实调用走同一条路。
    """
    from app.conversation import build_lite_transcript

    return build_lite_transcript(prompt, PROBE_MODEL)


def probe_inference(
    account: Dict[str, Any],
    prompt: str = PROBE_PROMPT,
    max_chars: int = 120,
    max_events: int = 400,
) -> Dict[str, Any]:
    """对一个账号发一次真实推理，返回是否拿到文本。

    判定：
        ok=True   拿到了非空文本 → AI 可用
        ok=False  空流 / 429 / 额度耗尽 / 凭据问题
    """
    started = time.monotonic()
    text_parts: list[str] = []
    events = 0
    error = ""

    try:
        from app.notion_client import NotionOpusAPI

        api = NotionOpusAPI(account)
        user_id = str(getattr(api, "user_id", "") or account.get("user_id") or "")
        # 必须显式 thread_id=None，和 chat 主链路一致；
        # 不传会走默认的建线程路径，拿到空流。
        for chunk in api.stream_response(
            _build_transcript(prompt, user_id=user_id), thread_id=None
        ):
            events += 1
            if not isinstance(chunk, dict):
                continue
            ctype = str(chunk.get("type") or "")
            value = chunk.get("text")
            if isinstance(value, str) and value and ctype in TEXT_CHUNK_TYPES:
                text_parts.append(value)
            if sum(len(p) for p in text_parts) > max_chars:
                break
            if events > max_events:
                break
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:200]

    text = "".join(text_parts).strip()
    latency = round((time.monotonic() - started) * 1000, 1)
    result = {
        "ok": bool(text),
        "events": events,
        "chars": len(text),
        "text": text[:max_chars],
        "error": error,
        "latency_ms": latency,
    }
    if not text and not error:
        result["error"] = "empty stream（可能额度耗尽或 AI 不可用）"
    return result


def probe_and_record(account: Dict[str, Any]) -> Dict[str, Any]:
    """探活并把结果写进 ai_probe 表。"""
    result = probe_inference(account)
    key = str(account.get("account_key") or account.get("profile_name") or "")
    if key:
        try:
            from app.store import get_store

            get_store().save_probe(
                key,
                ok=bool(result["ok"]),
                chars=int(result["chars"]),
                latency_ms=result["latency_ms"],
                error=result["error"],
            )
        except Exception:  # noqa: BLE001
            pass
    if not result["ok"]:
        logger.info("AI probe failed", extra={"request_info": {
            "event": "ai_probe_failed",
            "account": str(account.get("user_email") or "")[:32],
            "error": result["error"][:120],
        }})
    return result
