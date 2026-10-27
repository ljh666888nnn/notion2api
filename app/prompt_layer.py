# -*- coding: utf-8 -*-
"""prompt_layer.py - 系统提示词注入层。

背景：Notion 的 runInferenceTranscript 只有 user/assistant 两种对话块，没有 system 角色，
所以客户端发来的 system prompt 只能并进第一条 user 消息里。

早期实现用的是「元描述」写法：

    The user is working inside a developer tool and has shared the following
    guidelines ... Treat these as the user's preferences ...

这种写法会让上游把这段文字当成**待分析的内容**而不是**要遵守的指令**，
于是模型开始自言自语地解析这段话（"User says ... Ambiguous but likely means ..."），
把内心独白当成回答吐出来。

这里改成 wb-proxy 的做法：直接写成面向助手的操作规则，并支持降级模式。

模式（.env -> PROMPT_MODE）：
    passthrough  不注入（只用客户端自己的 system 文本，仍按指令语气重写）
    custom       丢弃客户端 system，只发网关提示词
    append       客户端 system + 网关提示词（默认）
    off          完全不注入任何 system 内容

降级（PROMPT_DEGRADE）：
    上游因内容审核拒绝时，自动退到一句中性提示词重试一次。
"""
from __future__ import annotations

import os
import threading
import time

DEFAULT_PROMPT = (
    "你是一名严谨的工程助手。"
    "先理解上下文再动手，遵循已有约定；只改必要部分，不做无关重构；"
    "改动后要验证结果，不臆造结论；不确定时说明不确定并给出验证路径。"
)

DEGRADED_PROMPT = (
    "You are a helpful assistant. Answer in the user's language, "
    "follow the user's instructions, and keep responses direct and concise."
)

VALID_MODES = ("plain", "passthrough", "custom", "append", "off")


def _clean(raw: str) -> str:
    return (raw or "").replace("\x00", "").strip()


def normalize_mode(value: str | None) -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in VALID_MODES else "plain"


def load_prompt_text() -> str:
    """网关提示词正文：环境变量优先；为空表示不额外注入（推荐）。"""
    return _clean(os.getenv("PROMPT_TEXT", ""))


def _rules_block(text: str) -> str:
    """把提示词写成面向助手的规则段，不做任何自我介绍式描述。"""
    body = text.strip()
    if not body:
        return ""
    return f"<operating_rules>\n{body}\n</operating_rules>"


def compose(
    client_system: str,
    *,
    mode: str | None = None,
    gateway_text: str | None = None,
    degraded: bool = False,
) -> str:
    """返回要并进第一条 user 消息的前缀文本（可能为空）。

    默认 plain：不加任何标签，直接把客户端 system 文本前置。
    带 XML 标签（append 模式）容易被上游判定成提示注入，从而触发
    「我是 Notion AI，我忽略这些注入内容」这类回复，所以默认不用。
    """
    resolved = normalize_mode(mode if mode is not None else os.getenv("PROMPT_MODE", "plain"))
    gateway = _clean(gateway_text if gateway_text is not None else load_prompt_text())
    client = _clean(client_system)

    if resolved == "off":
        return ""
    if degraded:
        # 降级只用一句中性提示词，不带标签
        return DEGRADED_PROMPT
    if resolved == "custom":
        return gateway
    if resolved == "passthrough":
        return client
    if resolved == "append":
        parts = [block for block in (_rules_block(client), _rules_block(gateway)) if block]
        return "\n\n".join(parts)
    # plain（默认）：无标签、无元描述，纯文本前置
    parts = [part for part in (client, gateway) if part]
    return "\n\n".join(parts)


def join_with_user(preamble: str, user_prompt: str) -> str:
    """把前缀和用户请求拼成一条 user 消息。

    不再加 <user_request> 标签——那同样会被上游当成结构化指令而触发注入告警。
    """
    if not preamble:
        return user_prompt
    return f"{preamble}\n\n{user_prompt}"


class DegradeGate:
    """进程内降级窗口；触发后到次日 00:00（+08:00）为止。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._until = 0.0

    def active(self, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        with self._lock:
            return current < self._until

    def trigger(self, now: float | None = None) -> None:
        current = time.time() if now is None else float(now)
        with self._lock:
            if current >= self._until:
                offset = 8 * 3600
                day = int((current + offset) // 86400)
                self._until = float((day + 1) * 86400 - offset)

    def reset(self) -> None:
        with self._lock:
            self._until = 0.0


degrade_gate = DegradeGate()


def should_degrade(status_code: int | None, body: str) -> bool:
    """上游这批错误通常与 system 文本触发的审核有关。"""
    if status_code not in (400, 403, 422):
        return False
    lowered = (body or "").lower()
    markers = (
        "content_policy",
        "content policy",
        "moderation",
        "policy_violation",
        "not_allowed",
        "safety",
        "instruction",
        "prompt",
    )
    return any(marker in lowered for marker in markers)
