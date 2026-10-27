# -*- coding: utf-8 -*-
"""quota_meter.py - 实测一个账号在某个模型下还能用多少次。

Notion 没有暴露用量查询接口（试过 15 个候选端点，全 404），
space 记录里也只有 subscription_tier，没有额度。
所以只能**行为法**：一直发最小请求，直到上游拒绝。

判定要区分三种情况，否则会把限流当成额度耗尽：
    quota_exhausted  upstream 空流 / 明确额度提示 → 真的用完了
    rate_limited     429 → 退避后重试，不计入消耗
    auth/invalid     401/403 → 凭据问题，不是额度

代价：测到失败会把该账号的免费额度用光。
所以默认 max_calls 很小，抽样测即可。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.logger import logger

PROBE_MODEL = "gpt-5.2"
PROBE_PROMPT = "Reply with exactly: PONG"
TEXT_CHUNK_TYPES = ("content", "final_content")


def _build_transcript(prompt: str, model: str = PROBE_MODEL) -> list[dict[str, Any]]:
    from app.conversation import build_lite_transcript

    return build_lite_transcript(prompt, model)


def _classify(exc: Exception, text: str) -> str:
    """把一次失败归类。"""
    low = f"{type(exc).__name__} {exc} {text}".lower()
    if "429" in low or "rate" in low or "usage limit" in low:
        return "rate_limited"
    if "401" in low or "403" in low or "unauthorized" in low:
        return "auth_failed"
    if "empty stream" in low:
        return "quota_exhausted"
    if "quota" in low or "exceed" in low or "limit" in low:
        return "quota_exhausted"
    return "error"


def measure_quota(
    client: Any,
    model: str = PROBE_MODEL,
    max_calls: int = 50,
    cooldown_on_429: float = 8.0,
    max_retries_on_429: int = 3,
) -> Dict[str, Any]:
    """对一个账号连续发请求，直到失败或达到上限。

    返回：
        success_calls  成功次数（= 已消耗的额度单位）
        stop_reason    quota_exhausted / rate_limited / auth_failed / max_calls / error
        latencies      每轮耗时
        exhausted     是否真的跑到额度耗尽
    """
    from app.model_registry import get_notion_model

    transcript_builder = _build_transcript
    latencies: List[float] = []
    success = 0
    stop_reason = "max_calls"
    error_text = ""
    started = time.time()
    retries_429 = 0

    for index in range(1, max_calls + 1):
        attempt_started = time.monotonic()
        text = ""
        try:
            transcript = transcript_builder(PROBE_PROMPT, model)
            parts: List[str] = []
            for chunk in client.stream_response(transcript, thread_id=None):
                if not isinstance(chunk, dict):
                    continue
                ctype = str(chunk.get("type") or "")
                value = chunk.get("text")
                if isinstance(value, str) and value and ctype in TEXT_CHUNK_TYPES:
                    parts.append(value)
                if sum(len(p) for p in parts) > 40:
                    break
            text = "".join(parts).strip()
            if not text:
                raise RuntimeError("empty stream")
            success += 1
            latencies.append(round((time.monotonic() - attempt_started) * 1000, 1))
        except Exception as exc:  # noqa: BLE001
            kind = _classify(exc, text)
            error_text = f"{type(exc).__name__}: {exc}"[:180]
            if kind == "rate_limited" and retries_429 < max_retries_on_429:
                retries_429 += 1
                logger.info("Quota meter backoff", extra={"request_info": {
                    "event": "quota_meter_backoff", "retry": retries_429,
                    "sleep_s": cooldown_on_429}})
                time.sleep(cooldown_on_429)
                continue
            stop_reason = kind
            break

    elapsed = round(time.time() - started, 1)
    result = {
        "model": model,
        "notion_model": get_notion_model(model),
        "success_calls": success,
        "stop_reason": stop_reason,
        "exhausted": stop_reason == "quota_exhausted",
        "elapsed_s": elapsed,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "error": error_text,
    }
    logger.info("Quota meter done", extra={"request_info": {
        "event": "quota_meter_done", **{k: v for k, v in result.items() if k != "error"}}})
    return result


def measure_by_index(
    pool: Any,
    account_index: int,
    model: str = PROBE_MODEL,
    max_calls: int = 50,
) -> Dict[str, Any]:
    """用池子里已经建好的 client 测（新建 client 会拿到空流）。"""
    state = pool._state_at(account_index)  # noqa: SLF001
    if state is None:
        return {"error": f"account_index {account_index} 不存在"}
    client = state.client
    result = measure_quota(client, model=model, max_calls=max_calls)
    result["account_index"] = account_index
    result["user_email"] = getattr(state.account, "user_email", "")
    return result
