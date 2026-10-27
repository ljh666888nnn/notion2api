# -*- coding: utf-8 -*-
"""usage_meter.py — 估算 token 用量，供 `/v1/chat/completions` 的 usage 字段使用。

为什么要它：Notion 上游不返回 token 数，本仓库之前把 `usage` 固定写成 0。
对聊天客户端无所谓，但 agent 框架（Cline / DSH / OpenCode 等）会拿 usage 做
上下文占用、成本与压缩决策；恒为 0 等于「永远没占用」，要么不压缩、要么费用面板
全空。

口径（与 sticture/notion2api 的「估算，不是官方账单」一致）：
    CJK / 假名 / 谚文：1 字 ≈ 1 token
    其余字符：4 字符 ≈ 1 token

只是估算，不用于计费。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

_CJK_RANGES = (
    (0x3040, 0x30FF),   # 日文假名
    (0x3400, 0x4DBF),   # 扩展 A
    (0x4E00, 0x9FFF),   # 基本汉字
    (0xAC00, 0xD7AF),   # 谚文
    (0xF900, 0xFAFF),   # 兼容汉字
    (0xFF00, 0xFFEF),   # 全角标点
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def estimate_tokens(text: Any) -> int:
    """粗略估算一段文本的 token 数。"""
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    cjk = 0
    for char in text:
        if _is_cjk(char):
            cjk += 1
    other = len(text) - cjk
    return int(cjk + math.ceil(other / 4))


def messages_text(messages: Iterable[Dict[str, Any]]) -> str:
    """把消息列表拼成用于估算的文本（只取 content，忽略结构开销）。"""
    parts: List[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif content:
            parts.append(str(content))
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict):
                    parts.append(str(fn.get("name") or ""))
                    parts.append(str(fn.get("arguments") or ""))
    return "\n".join(parts)


def build_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    """OpenAI 形状的 usage。纯整数，避免严格客户端校验失败。"""
    prompt_tokens = max(0, int(prompt_tokens))
    completion_tokens = max(0, int(completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
