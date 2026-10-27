# -*- coding: utf-8 -*-
"""context_budget.py — 上游输入预算守卫（借鉴 sticture/notion2api）。

背景：Notion 的 `runInferenceTranscript` 对单次输入有硬上限（社区实测新线程约
19.5 万字节 JSON 转义后的 UTF-8，写入上限 40.96 万字节）。本网关之前是
「客户端发多少就转多少」，一个 agent 会话读几次文件就会顶到上限，上游直接返回
空流 / 502，前端只看到 `NOTION_502`，很难定位。

这里做三件事，顺序固定：

1. **压缩工具结果**：单条 `[tool result: ...]` 超过 `NOTION_TOOL_RESULT_MAX_CHARS`
   时保留头尾，中间写明截断长度，提示模型用更小的 offset/limit 重新读取
   （sticture 的做法是「保留头尾共约 16,000 字符」）。
2. **丢最旧轮次**：仍然超预算就从最早的对话开始丢，永远保留 system 与最后一条
   user，并在保留下来的最早消息前加一行说明，避免模型误以为没有历史。
3. **413 兜底**：丢到只剩最后一条仍然超限，抛 `InputTooLarge`，由路由层返回
   413 `CONTEXT_TOO_LARGE`，而不是让上游静默失败。

环境变量：
    NOTION_INPUT_MAX_BYTES        默认 195000（JSON 转义后的 UTF-8 字节）
    NOTION_TOOL_RESULT_MAX_CHARS  默认 16000（单条工具结果上限）
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

DEFAULT_MAX_INPUT_BYTES = 195_000
DEFAULT_TOOL_RESULT_MAX_CHARS = 16_000
TOOL_RESULT_HEAD_CHARS = 12_000
TOOL_RESULT_TAIL_CHARS = 4_000

TRIMMED_NOTE = "[上下文说明] 更早的 {count} 条消息因上游长度上限被省略；如需原文请重新读取。"


class InputTooLarge(Exception):
    """压缩 + 裁剪之后仍然超过上游输入上限。"""

    def __init__(self, stats: Dict[str, Any]) -> None:
        self.stats = stats
        super().__init__(
            f"input {stats.get('bytes_after')} bytes > limit {stats.get('max_bytes')} bytes"
        )


def max_input_bytes() -> int:
    raw = os.getenv("NOTION_INPUT_MAX_BYTES", "").strip()
    try:
        return max(4096, int(raw)) if raw else DEFAULT_MAX_INPUT_BYTES
    except ValueError:
        return DEFAULT_MAX_INPUT_BYTES


def tool_result_max_chars() -> int:
    raw = os.getenv("NOTION_TOOL_RESULT_MAX_CHARS", "").strip()
    try:
        return max(512, int(raw)) if raw else DEFAULT_TOOL_RESULT_MAX_CHARS
    except ValueError:
        return DEFAULT_TOOL_RESULT_MAX_CHARS


def payload_bytes(messages: List[Dict[str, Any]]) -> int:
    """按上游的实际计费口径量：JSON 转义后的 UTF-8 字节。"""
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))


def _is_tool_result(message: Dict[str, Any]) -> bool:
    if message.get("role") == "tool":
        return True
    content = message.get("content")
    return isinstance(content, str) and content.startswith("[tool result:")


def compress_tool_results(
    messages: List[Dict[str, Any]], limit: int | None = None
) -> int:
    """把超长工具结果压成头 + 尾，返回被压缩的条数。"""
    limit = limit or tool_result_max_chars()
    # 大预算用「头 12k + 尾 4k」；预算更小时按 3:1 缩放，保证两头都有内容
    head = min(TOOL_RESULT_HEAD_CHARS, max(1, int(limit * 0.75)))
    tail = max(0, min(TOOL_RESULT_TAIL_CHARS, limit - head))
    touched = 0
    for message in messages:
        if not _is_tool_result(message):
            continue
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= limit:
            continue
        omitted = len(content) - head - tail
        marker = (
            f"\n\n[... 已截断 {omitted} 个字符（原长 {len(content)}）；"
            f"需要完整内容请用更小的 offset/limit 重新读取 ...]\n\n"
        )
        message["content"] = content[:head] + marker + (content[-tail:] if tail else "")
        touched += 1
    return touched


def _shrink_message(messages: List[Dict[str, Any]], index: int, limit: int) -> bool:
    """把单条消息压到预算内：保留头 75% + 尾 25%，中间标注截断。

    场景：客户端一次贴了 200KB 的文件内容。丢历史没用（它本身就是最后一条
    user），这时宁可截断也要让请求跑起来，而不是直接 413。
    """
    text = str(messages[index].get("content") or "")
    if not text:
        return False
    marker = "\n\n…（中间内容因超出上游输入预算被截断）…\n\n"
    low, high, best = 0, len(text), ""
    while low <= high:
        middle = (low + high) // 2
        head_len = int(middle * 0.75)
        tail_len = middle - head_len
        candidate = text[:head_len] + marker + (text[-tail_len:] if tail_len else "")
        messages[index]["content"] = candidate
        if payload_bytes(messages) <= limit:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    messages[index]["content"] = best
    return bool(best)


def _trim_to_budget(messages: List[Dict[str, Any]], limit: int) -> int:
    """从最早的非 system 消息开始丢，保留 system 与最后一条 user。"""
    system_prefix = 0
    while system_prefix < len(messages) and messages[system_prefix].get("role") == "system":
        system_prefix += 1

    last_user = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            last_user = index
            break
    if last_user < 0:
        last_user = len(messages) - 1

    dropped = 0
    while payload_bytes(messages) > limit:
        removable = [
            index
            for index in range(system_prefix, len(messages))
            if index != last_user
        ]
        if not removable:
            break
        del messages[removable[0]]
        dropped += 1
        if last_user > removable[0]:
            last_user -= 1

    if dropped:
        note = TRIMMED_NOTE.format(count=dropped)
        for message in messages[system_prefix:]:
            content = message.get("content") or ""
            message["content"] = f"{note}\n\n{content}".strip()
            break

    # 丢完历史还超限：说明最后一条 user（或 system）本身就太大，截断它
    if payload_bytes(messages) > limit:
        _shrink_message(messages, last_user, limit)
    if payload_bytes(messages) > limit:
        for index in range(system_prefix):
            if payload_bytes(messages) <= limit:
                break
            _shrink_message(messages, index, limit)
    return dropped


def enforce_budget(
    messages: List[Dict[str, Any]],
    *,
    max_bytes: int | None = None,
    reserve_bytes: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """原地压缩/裁剪 messages，返回 (messages, stats)。

    `reserve_bytes` 给「还没注入的东西」留空间（例如工具清单 preamble）。
    压缩或裁剪后仍超限时抛 `InputTooLarge`。
    """
    limit = max(4096, (max_bytes or max_input_bytes()) - max(0, reserve_bytes))
    stats: Dict[str, Any] = {
        "bytes_before": payload_bytes(messages),
        "bytes_after": 0,
        "max_bytes": limit,
        "compressed_tool_results": 0,
        "dropped_messages": 0,
    }

    stats["compressed_tool_results"] = compress_tool_results(messages)
    if stats["bytes_before"] > limit:
        stats["dropped_messages"] = _trim_to_budget(messages, limit)

    stats["bytes_after"] = payload_bytes(messages)
    if stats["bytes_after"] > limit:
        raise InputTooLarge(stats)
    return messages, stats
