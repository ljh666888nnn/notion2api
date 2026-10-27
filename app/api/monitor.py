from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.accounts import _require_admin
from app.call_log import get_call_log
from app.store import get_store
from app.quota import FREE_AI_QUOTA, derive_quota_state, probe_quota, summarize

router = APIRouter()

# 后台额度探测状态（避免前端请求被 47 个账号的串行 HTTP 探测拖超时）
_refresh_lock = threading.Lock()
_refresh_state: Dict[str, Any] = {
    "running": False,
    "done": 0,
    "total": 0,
    "errors": 0,
    "started_at": 0.0,
    "finished_at": 0.0,
}


def _refresh_snapshot() -> Dict[str, Any]:
    with _refresh_lock:
        return dict(_refresh_state)


def _start_refresh(pool: Any, force: bool = True) -> bool:
    with _refresh_lock:
        if _refresh_state["running"]:
            return False
        accounts = pool.accounts_raw()
        _refresh_state.update(
            running=True, done=0, total=len(accounts), errors=0,
            started_at=time.time(), finished_at=0.0,
        )

    def worker() -> None:
        # 并发探测：串行跑 150 个账号要 7~8 分钟，前端轮询会超时
        # （表现为 "signal is aborted without reason"）
        import os
        from concurrent.futures import ThreadPoolExecutor

        try:
            parallelism = max(1, int(os.getenv("QUOTA_PROBE_CONCURRENCY", "12")))

            def probe_one(account: Any) -> None:
                try:
                    probe_quota(account, force=force)
                except Exception:  # noqa: BLE001
                    with _refresh_lock:
                        _refresh_state["errors"] += 1
                finally:
                    with _refresh_lock:
                        _refresh_state["done"] += 1

            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                list(executor.map(probe_one, accounts))
        finally:
            with _refresh_lock:
                _refresh_state["running"] = False
                _refresh_state["finished_at"] = time.time()

    threading.Thread(target=worker, daemon=True, name="quota-refresh").start()
    return True


@router.get("/admin/calls", tags=["admin"])
def list_calls(
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    only_failed: bool = Query(default=False),
    account_key: str = Query(default="", max_length=64),
) -> Dict[str, Any]:
    """真实调用明细（最新在前，支持分页与筛选）。"""
    _require_admin(request)
    return get_call_log().page(offset=offset, limit=limit,
                               only_failed=only_failed, account_key=account_key)


@router.get("/admin/calls/summary", tags=["admin"])
def calls_summary(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    return get_call_log().summary()


@router.get("/admin/quota", tags=["admin"])
def quota_overview(request: Request) -> Dict[str, Any]:
    """每个账号的套餐与额度状态。

    默认**只读缓存**，不发起任何网络请求——47 个账号串行探测会把前端拖到
    超时（表现为 "signal is aborted without reason"）。
    需要刷新时调用 POST /admin/quota/refresh，它在后台线程里跑。
    """
    _require_admin(request)
    pool = request.app.state.account_pool
    snapshots = pool.snapshot()
    probe_map = get_store().load_probe_all()
    # 只统计最近 1 小时的用量：实测有账号成功调用 7 次仍可用，
    # 说明 5 次是滑动窗口内的配额，累计统计会低估。
    usage_map = get_store().account_usage(since=time.time() - 3600)
    raw_accounts = {item.get("account_key", ""): item for item in pool.accounts_raw()}

    items: List[Dict[str, Any]] = []
    pending = 0
    for snapshot in snapshots:
        account_key = snapshot.get("account_key", "")
        account = raw_accounts.get(account_key, {})
        quota = probe_quota(account, cache_only=True) if account else {
            "reachable": False,
            "error": "账号配置缺失",
            "plan_type": "",
            "subscription_tier": "",
            "ai_enabled": None,
        }
        if quota.get("pending"):
            pending += 1
        # 剩余额度 = 免费额度 - 该账号成功调用次数（实测额度与模型无关）
        used_calls = int((usage_map.get(account_key) or {}).get("ok") or 0)
        _state = derive_quota_state(snapshot, quota, probe_map.get(account_key))
        # 探测已判定耗尽时，剩余直接归零，不要和 calls 统计打架
        remaining = 0 if _state == "exhausted" else max(0, FREE_AI_QUOTA - used_calls)
        item = {
            "account_index": snapshot.get("account_index"),
            "account_key": account_key,
            "profile_name": snapshot.get("profile_name", ""),
            "user_email": snapshot.get("user_email", ""),
            "status": snapshot.get("status"),
            "plan_type": quota.get("plan_type", ""),
            "subscription_tier": quota.get("subscription_tier", ""),
            "ai_enabled": quota.get("ai_enabled"),
            "reachable": quota.get("reachable", False),
            "probe_error": quota.get("error", ""),
            "pending": bool(quota.get("pending")),
            "used_calls": used_calls,
            "total_quota": FREE_AI_QUOTA,
            "remaining_quota": remaining,
            "quota_state": _state,
            "probe": probe_map.get(account_key),
            "total_requests": snapshot.get("total_requests", 0),
            "total_successes": snapshot.get("total_successes", 0),
            "total_failures": snapshot.get("total_failures", 0),
            "last_status_code": snapshot.get("last_status_code"),
            "last_error": snapshot.get("last_error", ""),
            "last_used_at": snapshot.get("last_used_at"),
            "average_latency_ms": snapshot.get("average_latency_ms"),
        }
        items.append(item)

    return {"summary": summarize(items), "items": items, "pending": pending,
            "refresh": _refresh_snapshot()}


@router.post("/admin/quota/refresh", tags=["admin"])
def quota_refresh(request: Request, force: bool = Query(default=True)) -> Dict[str, Any]:
    """后台线程重新探测全部账号额度，立即返回，不阻塞前端。"""
    _require_admin(request)
    pool = request.app.state.account_pool
    started = _start_refresh(pool, force=force)
    return {"started": started, **_refresh_snapshot()}


@router.get("/admin/quota/refresh", tags=["admin"])
def quota_refresh_status(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    return _refresh_snapshot()


@router.get("/admin/quota/{account_index}", tags=["admin"])
def quota_one(account_index: int, request: Request,
              refresh: bool = Query(default=False)) -> Dict[str, Any]:
    _require_admin(request)
    pool = request.app.state.account_pool
    snapshots = {item["account_index"]: item for item in pool.snapshot()}
    snapshot = snapshots.get(account_index)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="账号不存在，请先刷新列表。")
    raw_accounts = {item.get("account_key", ""): item for item in pool.accounts_raw()}
    account = raw_accounts.get(snapshot.get("account_key", ""), {})
    quota = probe_quota(account, force=refresh, cache_only=not refresh) if account else {
        "reachable": False, "error": "账号配置缺失"}
    return {
        "account_index": account_index,
        "quota_state": derive_quota_state(snapshot, quota),
        **{k: v for k, v in quota.items() if k != "fetched_at"},
    }


@router.get("/admin/samples", tags=["admin"])
def pool_samples(request: Request, limit: int = Query(default=48, ge=1, le=500)) -> Dict[str, Any]:
    """账号池采样历史（折线图），来自 SQLite。"""
    _require_admin(request)
    return {"items": get_store().list_samples(limit)}
