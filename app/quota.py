# -*- coding: utf-8 -*-
"""quota.py - 账号额度/套餐探测。

Notion 没有公开「剩余 AI 次数」的接口（getAiUsage / getCredits 之类全部 404），
所以这里只统计**真实可观测**的字段，不编造百分比：

    plan_type           空间套餐类型          （getSpaces -> space.plan_type）
    subscription_tier   订阅层级              （getSpaces -> space.subscription_tier）
    ai_enabled          空间是否开启 AI        （getSpaces -> space.settings.enable_ai_feature）
    quota_state         由真实调用结果推导：
                          invalid   凭据失效/被禁用
                          limited   上游返回 429 / 限流类错误
                          ok        最近一次调用成功
                          unknown   还没有调用记录
    last_status_code    最近一次上游 HTTP 状态
    last_error          最近一次失败原因
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.logger import logger

NOTION_URL = "https://www.notion.so"
# 实测修正：free 账户**没有总额度上限**，只有速率限制。
#   - 连续快发：第 6 次被拒（empty stream 或 429）
#   - 被拒后等 1 分钟即恢复
#   - 每 12 秒发 1 次：5 分钟连续 23 次全成功，不触顶
# 所以这里的 5 表示**速率窗口内的突发上限**，不是总配额。
# 详见 docs/QUOTA_IS_RATE_LIMIT_NOT_QUOTA.md
FREE_AI_QUOTA = 5
FREE_QUOTA_WINDOW_SECONDS = 60

CACHE_TTL_SECONDS = 300.0

_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_cache_loaded = False


def _ensure_cache_loaded() -> None:
    """启动后第一次访问时，把上次落库的额度结果载回内存。"""
    global _cache_loaded
    if _cache_loaded:
        return
    with _cache_lock:
        if _cache_loaded:
            return
        try:
            from app.store import get_store

            for key, payload in get_store().load_quota_all().items():
                fetched = float(payload.get("fetched_at") or 0.0)
                data = {k: v for k, v in payload.items() if k != "fetched_at"}
                _cache[key] = {"fetched_at": fetched, "data": data}
        except Exception:  # noqa: BLE001
            pass
        _cache_loaded = True


def _persist(key: str, data: Dict[str, Any]) -> None:
    try:
        from app.store import get_store

        get_store().save_quota(key, data)
    except Exception:  # noqa: BLE001
        pass


def _session_for(account: Dict[str, Any]) -> requests.Session:
    session = requests.Session()
    cookies = dict(account.get("cookies") or {})
    token = str(account.get("token_v2") or "")
    cookies["token_v2"] = token
    if account.get("user_id"):
        cookies.setdefault("notion_user_id", str(account["user_id"]))
    session.cookies.update(cookies)
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "notion-audit-log-platform": "web",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
        ),
    })
    return session


def _fetch_space(
    session: requests.Session,
    space_id: str,
    retries: int = 2,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """取 space 记录，返回 (space, reason)。

    reason 用来区分「真失效」和「被限流」：
        ok          拿到了
        http_401    凭据无效
        http_429    限流（并发探测常见）
        no_space    200 但确实没有这个 space
        error       网络异常
    以前一律判成"凭据失效"，导致并发探测时把好账号全标成 invalid。
    """
    payload: Any = None
    last = "error"
    for attempt in range(retries + 1):
        try:
            response = session.post(
                f"{NOTION_URL}/api/v3/getSpaces", json={}, timeout=20
            )
            if response.status_code in (401, 403):
                return None, "http_401"
            if response.status_code == 429:
                last = "http_429"
                time.sleep(2.0 + attempt * 2.0)
                continue
            if response.status_code != 200:
                last = f"http_{response.status_code}"
                continue
            payload = response.json()
            break
        except Exception:  # noqa: BLE001
            last = "error"
            time.sleep(1.0 + attempt)
            continue

    if not isinstance(payload, dict):
        return None, last
    for _uid, tables in payload.items():
        if not isinstance(tables, dict):
            continue
        space_table = tables.get("space")
        if not isinstance(space_table, dict):
            continue
        raw = space_table.get(space_id)
        if raw is None:
            continue
        value = raw
        for _ in range(3):
            if isinstance(value, dict) and isinstance(value.get("value"), dict):
                value = value["value"]
            else:
                break
        return (value if isinstance(value, dict) else None), "ok"
    return None, "no_space"


def probe_quota(
    account: Dict[str, Any],
    force: bool = False,
    cache_only: bool = False,
) -> Dict[str, Any]:
    """探一个账号的套餐/额度信息（带 5 分钟缓存）。

    cache_only=True 时只读缓存，没有缓存就返回占位结果，绝不发起网络请求。
    面板默认走这个模式，避免 47 个账号串行探测把前端请求拖超时
    （之前表现为 "signal is aborted without reason"）。
    """
    _ensure_cache_loaded()
    key = str(account.get("user_id") or account.get("profile_name") or account.get("user_email") or "")
    now = time.time()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and not force and now - cached["fetched_at"] < CACHE_TTL_SECONDS:
            return dict(cached["data"], cached=True)
        if cache_only:
            if cached:
                return dict(cached["data"], cached=True, stale=True)
            return {
                "account_key": account.get("account_key") or "",
                "profile_name": account.get("profile_name") or "",
                "user_email": account.get("user_email") or "",
                "space_id": str(account.get("space_id") or ""),
                "plan_type": "",
                "subscription_tier": "",
                "ai_enabled": None,
                "reachable": False,
                "error": "",
                "pending": True,
            }

    space_id = str(account.get("space_id") or "")
    data: Dict[str, Any] = {
        "account_key": account.get("account_key") or "",
        "profile_name": account.get("profile_name") or "",
        "user_email": account.get("user_email") or "",
        "space_id": space_id,
        "plan_type": "",
        "subscription_tier": "",
        "ai_enabled": None,
        "reachable": False,
        "error": "",
        "fetched_at": now,
    }
    if not space_id:
        data["error"] = "缺少 space_id"
    else:
        session = _session_for(account)
        space, reason = _fetch_space(session, space_id)
        if space is None:
            if reason in ("http_429", "error", "http_5xx"):
                # 限流/网络问题：账号可能是好的，只是这次没探到
                data["error"] = f"探测未完成（{reason}），稍后重试"
                data["transient"] = True
            elif reason == "http_401":
                data["error"] = "凭据失效（401）"
            else:
                data["error"] = "getSpaces 未返回该空间（凭据可能失效）"
        else:
            data["reachable"] = True
            data["plan_type"] = str(space.get("plan_type") or "")
            data["subscription_tier"] = str(space.get("subscription_tier") or "")
            settings = space.get("settings") if isinstance(space.get("settings"), dict) else {}
            enabled = settings.get("enable_ai_feature")
            disabled = settings.get("disable_ai_feature")
            if isinstance(enabled, bool):
                data["ai_enabled"] = enabled and not bool(disabled)
            elif isinstance(disabled, bool):
                data["ai_enabled"] = not disabled

    with _cache_lock:
        _cache[key] = {"fetched_at": now, "data": dict(data)}
    _persist(key, dict(data))
    return dict(data, cached=False)


def derive_quota_state(
    snapshot: Dict[str, Any],
    quota: Dict[str, Any],
    probe_result: Dict[str, Any] | None = None,
) -> str:
    """结合账号池快照、套餐探测与真实 AI 探测，给出可解释的额度状态。"""
    # 有真实推理探测结果时优先采信
    if probe_result is not None:
        if probe_result.get("ok"):
            return "ok"
        err = str(probe_result.get("error") or "").lower()
        if "429" in err or "rate" in err or "quota" in err or "limit" in err:
            return "limited"
        if "401" in err or "403" in err or "auth" in err:
            return "invalid"
        if "empty stream" in err:
            # 空流是速率限制（令牌桶被打空），1 分钟就自愈，
            # 不是额度耗尽——之前判成 exhausted 是错的。
            return "limited"
    if snapshot.get("disabled") or snapshot.get("status") == "disabled":
        return "invalid"
    if not quota.get("reachable"):
        # 限流/网络导致的探测失败不等于凭据失效，标 unknown 等下次重探
        if quota.get("transient"):
            return "unknown"
        return "invalid" if quota.get("error") else "unknown"
    if quota.get("ai_enabled") is False:
        return "disabled_ai"
    # 探到了 space 且 AI 已开启，就可以认为可用，
    # 不必等到"实际调用成功过"——新注册的号永远没调用记录，
    # 按旧判据会一直显示 unknown。
    if quota.get("reachable") and quota.get("ai_enabled") is True:
        status_code = snapshot.get("last_status_code")
        last_error = str(snapshot.get("last_error") or "").lower()
        if status_code == 429 or "429" in last_error or "rate" in last_error:
            return "limited"
        if status_code in (401, 403):
            return "invalid"
        return "ok"
    status_code = snapshot.get("last_status_code")
    last_error = str(snapshot.get("last_error") or "").lower()
    if status_code == 429 or "429" in last_error or "rate" in last_error or "usage" in last_error:
        return "limited"
    if status_code in (401, 403):
        return "invalid"
    if (snapshot.get("total_successes") or 0) > 0:
        return "ok"
    return "unknown"


def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for item in items:
        state = item.get("quota_state") or "unknown"
        counts[state] = counts.get(state, 0) + 1
    plans: Dict[str, int] = {}
    for item in items:
        plan = item.get("plan_type") or "unknown"
        plans[plan] = plans.get(plan, 0) + 1
    return {
        "total": len(items),
        "by_state": counts,
        "by_plan": plans,
        "ai_enabled": sum(1 for item in items if item.get("ai_enabled") is True),
    }


def log_failure(message: str) -> None:
    try:
        logger.warning(
            "Quota probe issue",
            extra={"request_info": {"event": "quota_probe_issue", "error": message[:160]}},
        )
    except Exception:  # noqa: BLE001
        pass
