# -*- coding: utf-8 -*-
"""reasoning.py — 把各家客户端的「强度」参数归一成 Notion 的 reasoningEffort。

Notion 的 transcript config 里有 `reasoningEffort`（浏览器抓包里的默认值是
medium），但网关一直写死 medium，客户端根本无法选择。这里做一层归一：

    OpenAI      reasoning_effort: "low" | "medium" | "high"（也接受 minimal / xhigh）
    Anthropic   thinking: {type: "enabled", budget_tokens: N}  -> 按预算分档
    Responses   reasoning: {effort: "low" | "medium" | "high"}

注意：上游只认小写的 low/medium/high；实测传 "ultra" 或 "HIGH" 会得到空流，
所以这里严格收敛到这三个值，未知一律回 medium。
"""
from __future__ import annotations

from typing import Any, Optional

VALID_EFFORTS = ("low", "medium", "high")
DEFAULT_EFFORT = "medium"

_ALIASES = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
    "xhigh": "high",
    "maximum": "high",
}


def normalize_effort(value: Any) -> Optional[str]:
    """把单个字符串收敛成 low/medium/high；无法识别时返回 None。"""
    if not isinstance(value, str):
        return None
    return _ALIASES.get(value.strip().lower())


def _effort_from_thinking(thinking: Any) -> Optional[str]:
    """Anthropic 的 thinking.budget_tokens → 档位。"""
    if not isinstance(thinking, dict):
        return None
    if str(thinking.get("type") or "").lower() not in {"enabled", "auto", "adaptive"}:
        return None
    try:
        budget = int(thinking.get("budget_tokens") or 0)
    except (TypeError, ValueError):
        return None
    if budget <= 0:
        return None
    if budget < 4096:
        return "low"
    if budget < 16384:
        return "medium"
    return "high"


def resolve_reasoning_effort(
    reasoning_effort: Any = None, thinking: Any = None
) -> str:
    """请求里的强度 -> 上游 reasoningEffort；缺省 medium。"""
    effort = normalize_effort(reasoning_effort)
    if effort:
        return effort
    effort = _effort_from_thinking(thinking)
    return effort or DEFAULT_EFFORT
