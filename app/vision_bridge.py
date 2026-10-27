# -*- coding: utf-8 -*-
"""vision_bridge.py — 把图片转成文字描述，再交给只吃文本的上游。

为什么需要它：
    Notion 的 `runInferenceTranscript` 收不到图片——文件上传三步全部 200，
    但把 `{type:"file",file_id}` 用各种形态塞进 transcript，模型一律回
    「我看不到图片」（见 docs/AGENT_FEASIBILITY.md 的实测记录）。要真正支持
    图片，只能在网关本地先把图读懂。

做法：
    调本地 OpenAI 兼容的多模态网关（默认 http://127.0.0.1:8788/v1，
    模型 hy4-preview），让它把图片内容、布局、颜色、可见文字转写出来，
    再把描述作为文本块注入 prompt。

环境变量：
    NOTION_IMAGE_MODE   vision（默认）| omit | reject
    VISION_BASE         默认 http://127.0.0.1:8788/v1
    VISION_MODEL        默认 hy4-preview
    VISION_KEY          默认 any-key
    VISION_TIMEOUT      默认 60 秒
"""
from __future__ import annotations

import os
from typing import Any, Optional

import requests

DEFAULT_PROMPT = (
    "请客观描述这张图片，供另一个只能读文字的模型使用："
    "整体内容与场景、关键元素及其位置、主要颜色、以及所有可见文字（尽量逐字转写）。"
    "不要寒暄，直接给描述。"
)
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def image_mode() -> str:
    raw = os.getenv("NOTION_IMAGE_MODE", "").strip().lower()
    if raw in {"omit", "reject", "vision"}:
        return raw
    # 兼容旧变量名
    legacy = os.getenv("NOTION_IMAGE_POLICY", "").strip().lower()
    if legacy in {"omit", "reject"}:
        return legacy
    return "vision"


def _base() -> str:
    return os.getenv("VISION_BASE", "http://127.0.0.1:8788/v1").rstrip("/")


def _model() -> str:
    return os.getenv("VISION_MODEL", "hy4-preview")


def _key() -> str:
    return os.getenv("VISION_KEY", "any-key")


def _timeout() -> float:
    try:
        return max(1.0, float(os.getenv("VISION_TIMEOUT", "60")))
    except ValueError:
        return 60.0


def describe_data_url(url: str, prompt: Optional[str] = None) -> Optional[str]:
    """让本地视觉模型描述一张图；失败返回 None（调用方决定怎么降级）。"""
    if not isinstance(url, str) or not url:
        return None
    try:
        response = requests.post(
            f"{_base()}/chat/completions",
            headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
            json={
                "model": _model(),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or DEFAULT_PROMPT},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }],
            },
            timeout=_timeout(),
        )
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        body = response.json()
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except (ValueError, AttributeError):
        return None
    text = str(text).strip()
    return text or None


def describe_block_source(source: Any) -> Optional[str]:
    """Anthropic 的 image block：只接受 base64，拒绝 url/file_id。"""
    if not isinstance(source, dict) or source.get("type") != "base64":
        return None
    media_type = str(source.get("media_type") or "").strip()
    data = source.get("data")
    if not media_type or not isinstance(data, str) or not data:
        return None
    if len(data) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        return None
    return describe_data_url(f"data:{media_type};base64,{data}")


def render_description(text: str) -> str:
    """给描述加边界，避免模型把它当成用户原话。"""
    return (
        "[图片描述（网关本地视觉模型转写，非用户原文）]\n"
        f"{text}\n"
        "[图片描述结束]"
    )
