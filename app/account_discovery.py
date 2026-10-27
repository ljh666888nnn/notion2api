# -*- coding: utf-8 -*-
"""account_discovery.py — 用 token_v2 反查 space_id / user_id / space_view_id。

为什么需要：手动从 DevTools 里抄 4 个 UUID 很容易抄错，而 `token_v2`
本来就够用——Notion 的 /api/v3/getSpaces 会返回这个 token 能看到的所有空间。
所以 accounts.json 允许只写一个 token，其余字段由网关自动补全。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

NOTION_URL = os.getenv("NOTION_URL", "https://www.notion.so").rstrip("/")


class AccountDiscoveryError(RuntimeError):
    """无法从 token 解析出账号信息。"""


def _post(token_v2: str, path: str, timeout: float) -> Dict[str, Any]:
    url = f"{NOTION_URL}{path}"
    headers = {"Content-Type": "application/json", "Cookie": f"token_v2={token_v2}"}
    try:
        from curl_cffi import requests as cffi_requests  # 与主链路保持同样的 TLS 指纹

        response = cffi_requests.post(
            url, headers=headers, data="{}", impersonate="chrome", timeout=timeout
        )
    except Exception:  # noqa: BLE001 - 没装 curl_cffi 时退回 requests
        import requests

        response = requests.post(url, headers=headers, data="{}", timeout=timeout)
    if response.status_code in (401, 403):
        raise AccountDiscoveryError(
            f"token_v2 无效或已过期（HTTP {response.status_code}）：{response.text[:120]}"
        )
    if response.status_code != 200:
        raise AccountDiscoveryError(f"getSpaces 返回 HTTP {response.status_code}")
    try:
        payload = response.json()
    except ValueError as exc:
        raise AccountDiscoveryError(f"getSpaces 返回的不是 JSON: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise AccountDiscoveryError("getSpaces 返回空对象（token 可能没有可用空间）")
    return payload


def _pick_space(tables: Dict[str, Any], wanted: str = "") -> Optional[Dict[str, str]]:
    spaces = tables.get("space") or {}
    views = tables.get("space_view") or {}
    if not isinstance(spaces, dict):
        return None
    for space_id, record in spaces.items():
        if wanted and space_id != wanted:
            continue
        value = (record or {}).get("value") or {}
        view_id = ""
        for candidate, view in views.items():
            if isinstance(view, dict) and view.get("spaceId") == space_id:
                view_id = candidate
                break
        return {
            "space_id": str(space_id),
            "space_view_id": str(view_id),
            "profile_name": str(value.get("name") or "").strip(),
        }
    return None


def resolve_account(
    token_v2: str,
    *,
    space_id: str = "",
    timeout: float = 20.0,
) -> Dict[str, str]:
    """返回 {"space_id", "user_id", "space_view_id", "profile_name"}。"""

    token = (token_v2 or "").strip()
    if not token:
        raise AccountDiscoveryError("token_v2 为空")
    payload = _post(token, "/api/v3/getSpaces", timeout)
    for user_id, tables in payload.items():
        if not isinstance(tables, dict):
            continue
        picked = _pick_space(tables, wanted=space_id)
        if picked is None:
            continue
        picked["user_id"] = str(user_id)
        return picked
    raise AccountDiscoveryError(
        "token 有效，但没有解析到空间（可能该 token 不属于任何 workspace）"
    )


def describe(payload: Dict[str, Any]) -> str:
    """调试用：把 getSpaces 的顶层结构压成一行。"""
    return json.dumps({k: list(v.keys())[:6] for k, v in payload.items() if isinstance(v, dict)})[:400]
