# -*- coding: utf-8 -*-
"""api/ops.py - 备份、账号体检、运维总览。

关于 AI 探活：试过直接构造 NotionOpusAPI 发最小推理，拿到的一直是 empty stream，
而走服务自己的 /v1/chat/completions 是正常的（返回 PONG）。
说明探活不能绕过池子的租约机制单独造 client。

所以这里不做"假探活"，改成**基于真实调用统计**判断哪些号该死：
    - 额度 exhausted
    - 有失败记录且从未成功过
    - 最近一次是 401/403
这些都是真实请求留下的痕迹，比一次探测更可信。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, Request

from app.backup import (
    create_backup,
    get_scheduler,
    list_backups,
    prune_backups,
    restore_backup,
)
from app.logger import logger

router = APIRouter()


def _require_admin(request: Request) -> None:
    from app.api.accounts import _require_admin as check

    check(request)


# ---------- 备份 ----------
@router.post("/admin/backup", tags=["admin"])
def make_backup(request: Request, tag: str = "") -> Dict[str, Any]:
    """打一个快照（accounts.json + SQLite）。"""
    _require_admin(request)
    result = create_backup(tag=tag)
    logger.info("Backup created", extra={"request_info": {
        "event": "backup_created", "path": result["path"], "bytes": result["bytes"]}})
    return result


@router.get("/admin/backups", tags=["admin"])
def get_backups(request: Request) -> Dict[str, Any]:
    _require_admin(request)
    items = list_backups()
    return {
        "items": items,
        "total": len(items),
        "total_bytes": sum(item["bytes"] for item in items),
        "scheduler": get_scheduler().state(),
    }


@router.post("/admin/backups/prune", tags=["admin"])
def prune(request: Request, keep: int = Query(default=24, ge=1, le=500)) -> Dict[str, Any]:
    _require_admin(request)
    removed = prune_backups(keep)
    return {"removed": removed, "keep": keep}


@router.post("/admin/backups/restore", tags=["admin"])
def restore(request: Request, path: str, dry_run: bool = False) -> Dict[str, Any]:
    """从指定 zip 恢复。真实恢复会把现有文件改名成 .before-restore-<时间>。"""
    _require_admin(request)
    result = restore_backup(path, dry_run=dry_run)
    if not dry_run and not result.get("restored"):
        raise HTTPException(status_code=400, detail=result.get("error") or "恢复失败")
    if not dry_run:
        logger.warning("Backup restored", extra={"request_info": {
            "event": "backup_restored", "path": path, "files": result.get("files")}})
    return result


# ---------- 账号体检 ----------
def _collect_unhealthy(pool: Any) -> Dict[str, List[Dict[str, Any]]]:
    """按真实调用痕迹给账号分类。"""
    from app.quota import probe_quota

    snapshot = pool.snapshot()          # list[dict]
    states = {item.get("account_key"): item for item in snapshot}
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "exhausted": [], "never_worked": [], "auth_failed": [], "disabled": [],
    }

    for account in pool.accounts_raw():
        key = str(account.get("account_key") or "")
        state = states.get(key) or {}
        if state.get("disabled"):
            buckets["disabled"].append({
                "account_index": state.get("account_index"),
                "account_key": key,
                "user_email": account.get("user_email", ""),
                "reason": state.get("disabled_reason") or "已禁用",
            })
            continue

        quota = probe_quota(account, cache_only=True)
        if quota.get("quota_state") == "exhausted":
            buckets["exhausted"].append({
                "account_index": state.get("account_index"),
                "account_key": key,
                "user_email": account.get("user_email", ""),
                "reason": "额度耗尽",
            })
            continue

        failures = int(state.get("total_failures") or 0)
        successes = int(state.get("total_successes") or 0)
        last_code = state.get("last_status_code")
        if last_code in (401, 403):
            buckets["auth_failed"].append({
                "account_index": state.get("account_index"),
                "account_key": key,
                "user_email": account.get("user_email", ""),
                "reason": f"最近返回 {last_code}",
            })
        elif failures >= 3 and successes == 0:
            buckets["never_worked"].append({
                "account_index": state.get("account_index"),
                "account_key": key,
                "user_email": account.get("user_email", ""),
                "reason": f"失败 {failures} 次且从未成功",
            })
    return buckets


@router.get("/admin/accounts/audit", tags=["admin"])
def audit_accounts(request: Request) -> Dict[str, Any]:
    """体检：列出该下线的账号，不动任何东西。"""
    _require_admin(request)
    pool = request.app.state.account_pool
    buckets = _collect_unhealthy(pool)
    total = sum(len(v) for k, v in buckets.items() if k != "disabled")
    return {
        "total_unhealthy": total,
        "buckets": {k: {"count": len(v), "items": v[:50]} for k, v in buckets.items()},
        "generated_at": time.time(),
    }


@router.post("/admin/accounts/prune", tags=["admin"])
def prune_accounts(
    request: Request,
    dry_run: bool = Query(default=True),
    include: str = Query(default="exhausted,auth_failed",
                         description="要下线的分组，逗号分隔"),
    restore: bool = Query(default=False, description="改成重新启用这些账号"),
) -> Dict[str, Any]:
    """把体检出来的坏号批量下线（或重新启用）。

    默认 dry_run=true，只报告不改动。
    """
    _require_admin(request)
    pool = request.app.state.account_pool
    buckets = _collect_unhealthy(pool)
    wanted = [name.strip() for name in include.split(",") if name.strip()]

    targets: List[Dict[str, Any]] = []
    for name in wanted:
        targets.extend(buckets.get(name, []))

    if dry_run:
        return {"dry_run": True, "would_change": len(targets),
                "items": targets[:100], "groups": wanted}

    changed = 0
    errors: List[str] = []
    for item in targets:
        index = item.get("account_index")
        if index is None:
            continue
        try:
            if restore:
                pool.enable_account(int(index))
            else:
                pool.disable_account(int(index), reason=item.get("reason") or "体检下线")
            changed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{item.get('user_email')}: {type(exc).__name__}")

    logger.info("Accounts pruned", extra={"request_info": {
        "event": "accounts_pruned", "changed": changed,
        "restore": restore, "groups": wanted}})
    return {"dry_run": False, "changed": changed, "errors": errors[:20]}


# ---------- 总览 ----------
@router.get("/admin/overview", tags=["admin"])
def overview(request: Request) -> Dict[str, Any]:
    """一屏看全：账号 / 备份 / 告警。"""
    _require_admin(request)
    pool = request.app.state.account_pool
    snapshot = pool.snapshot() if hasattr(pool, "snapshot") else []
    total = len(snapshot)
    available = sum(1 for item in snapshot if item.get("available"))
    summary = {"total": total, "available": available}

    backups = list_backups()

    alarms: List[Dict[str, str]] = []
    if not backups:
        alarms.append({"level": "warning", "text": "还没有任何备份"})
    last = get_scheduler().state().get("last") or {}
    if last and not last.get("ok"):
        alarms.append({"level": "warning",
                       "text": f"最近一次自动备份失败：{str(last.get('error'))[:60]}"})
    if summary:
        total = int(summary.get("total") or 0)
        available = int(summary.get("available") or 0)
        if total and available / total < 0.5:
            alarms.append({"level": "critical",
                           "text": f"可用账号只占 {available}/{total}，低于 50%"})

    return {
        "accounts": summary,
        "backups": {"total": len(backups), "latest": backups[0] if backups else None},
        "alarms": alarms,
        "generated_at": time.time(),
    }
