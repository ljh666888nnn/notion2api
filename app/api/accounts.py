from __future__ import annotations

import hmac

from fastapi import APIRouter, HTTPException, Query, Request

from app.account_store import load_accounts
from app.config import API_KEY
from app.logger import logger


router = APIRouter()


def _require_admin(request: Request) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=403,
            detail="账号管理接口未启用：请先在服务端设置 API_KEY。",
        )
    auth_header = request.headers.get("Authorization", "")
    scheme, separator, credential = auth_header.partition(" ")
    if (
        not separator
        or scheme.lower() != "bearer"
        or not credential
        or not hmac.compare_digest(credential, API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="账号管理接口认证失败：请提供有效的 Bearer API_KEY。",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _account_action_response(
    pool, account_index: int, action: str, account: dict
) -> dict:
    return {
        "updated": True,
        "action": action,
        "account_index": account_index,
        "account": account,
        "summary": pool.get_status_summary(),
    }


def _log_admin_action(action: str, account_index: int, extra: dict | None = None) -> None:
    logger.info(
        "Account admin action",
        extra={
            "request_info": {
                "event": "account_admin_action",
                "action": action,
                "account_index": account_index,
                **(extra or {}),
            }
        },
    )


@router.get("/admin/accounts", tags=["admin"])
def get_account_status(request: Request) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    return {
        "strategy": pool.strategy,
        "summary": pool.get_status_summary(),
        "accounts": pool.snapshot(),
    }


@router.post("/admin/accounts/reload", tags=["admin"])
def reload_accounts(request: Request) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    try:
        accounts = load_accounts()
        summary = pool.reload(accounts)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"账号重载失败：{exc}",
        ) from exc

    return {
        "reloaded": True,
        "strategy": pool.strategy,
        "summary": summary,
        "accounts": pool.snapshot(),
    }


@router.post("/admin/accounts/{account_index}/enable", tags=["admin"])
def enable_account(account_index: int, request: Request) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    try:
        account = pool.enable_account(account_index)
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"账号启用失败：{exc} 请先刷新账号列表。",
        ) from exc
    return _account_action_response(pool, account_index, "enabled", account)


@router.post("/admin/accounts/{account_index}/disable", tags=["admin"])
def disable_account(
    account_index: int,
    request: Request,
    reason: str = Query(default="manual_disable", max_length=120),
) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    normalized_reason = " ".join(reason.split()) or "manual_disable"
    try:
        account = pool.disable_account(account_index, reason=normalized_reason)
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"账号禁用失败：{exc} 请先刷新账号列表。",
        ) from exc
    return _account_action_response(pool, account_index, "disabled", account)


@router.post("/admin/accounts/{account_index}/reset", tags=["admin"])
def reset_account(account_index: int, request: Request) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    try:
        account = pool.reset_account(account_index)
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"账号重置失败：{exc} 请先刷新账号列表。",
        ) from exc
    _log_admin_action("reset", account_index)
    return _account_action_response(pool, account_index, "reset", account)


@router.post("/admin/accounts/reset-cooling", tags=["admin"])
def reset_cooling(request: Request) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    reset_count = pool.reset_cooling()
    _log_admin_action("reset_cooling", 0, {"reset_count": reset_count})
    return {
        "updated": True,
        "action": "reset_cooling",
        "reset_count": reset_count,
        "summary": pool.get_status_summary(),
    }


@router.post("/admin/accounts/{account_index}/probe", tags=["admin"])
def probe_account(account_index: int, request: Request) -> dict:
    _require_admin(request)
    pool = request.app.state.account_pool
    try:
        result = pool.probe_account(account_index)
    except IndexError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"账号探测失败：{exc} 请先刷新账号列表。",
        ) from exc
    _log_admin_action(
        "probe",
        account_index,
        {
            "ok": result.get("ok"),
            "status_code": result.get("status_code"),
            "latency_ms": result.get("latency_ms"),
        },
    )
    return result
