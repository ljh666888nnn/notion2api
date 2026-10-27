# -*- coding: utf-8 -*-
"""call_context.py - 把"本次请求用的模型"传到记账处。

调用明细的 model 字段由 account_pool._log_call 写，
但那个函数在 mark_success / mark_failed 内部触发，
拿不到请求体。用 contextvar 在请求入口设一次即可，
async 下能随任务上下文传播。
"""
from __future__ import annotations

import contextvars

CURRENT_MODEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_model", default=""
)


def set_current_model(model: str) -> None:
    try:
        CURRENT_MODEL.set(model or "")
    except Exception:  # noqa: BLE001
        pass


def get_current_model() -> str:
    try:
        return CURRENT_MODEL.get()
    except Exception:  # noqa: BLE001
        return ""
