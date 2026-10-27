from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from dataclasses import replace

from app.account_schema import AccountConfig, AccountValidationError, validate_accounts


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ACCOUNTS_FILE = PROJECT_ROOT / "accounts.json"


def _autodiscover_enabled() -> bool:
    return os.getenv("ACCOUNT_AUTODISCOVER", "1").strip() not in {"0", "false", "False"}


def _complete_accounts(accounts: list[AccountConfig], source: str) -> list[AccountConfig]:
    """只写了 token_v2 的账号，在这里补齐 space_id / user_id / space_view_id。

    走一次 /api/v3/getSpaces（同一个 token 本来就能查到），省掉让人手抄 UUID。
    失败时报错直接点名该怎么办，而不是让服务带着半个账号跑起来。
    """
    if not _autodiscover_enabled():
        return accounts

    completed: list[AccountConfig] = []
    for account in accounts:
        if account.space_id and account.user_id:
            completed.append(account)
            continue
        from app.account_discovery import resolve_account

        try:
            info = resolve_account(account.token_v2, space_id=account.space_id)
        except Exception as exc:  # noqa: BLE001
            raise AccountValidationError(
                f"{source}: 无法用 token_v2 自动解析账号信息（{exc}）；"
                "请在 accounts.json 中补上 space_id / user_id，"
                "或确认该 token 仍然有效"
            ) from exc
        completed.append(
            replace(
                account,
                space_id=info["space_id"],
                user_id=info["user_id"],
                space_view_id=account.space_view_id or info.get("space_view_id", ""),
                profile_name=account.profile_name or info.get("profile_name", ""),
            )
        )
    return completed


def parse_accounts_payload(payload: Any, source: str = "accounts.json") -> list[AccountConfig]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AccountValidationError(f"{source} JSON 解析失败: {exc}") from exc
    return _complete_accounts(validate_accounts(payload, source=source), source)


def load_accounts_from_file(path: str | Path) -> list[AccountConfig]:
    accounts_path = Path(path)
    try:
        payload = accounts_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AccountValidationError(f"读取账号文件失败 {accounts_path}: {exc}") from exc
    return parse_accounts_payload(payload, source=str(accounts_path))


def load_accounts_from_env(env: Mapping[str, str] | None = None) -> list[AccountConfig] | None:
    env_map = env if env is not None else os.environ
    payload = str(env_map.get("NOTION_ACCOUNTS") or "").strip()
    if not payload:
        return None
    return parse_accounts_payload(payload, source="NOTION_ACCOUNTS")


def load_accounts(
    accounts_file: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[AccountConfig]:
    """Load accounts.json first, then fall back to NOTION_ACCOUNTS.

    开源场景下 accounts.json 不会进仓库。没有账号时返回空列表，让服务能启动
    （/health 显示 0 个账号、聊天返回 503），而不是 import 阶段直接抛异常——
    否则新克隆连测试都跑不起来。

    需要严格模式的部署可以设置 `NOTION_REQUIRE_ACCOUNTS=1` 恢复旧的抛错行为。
    """

    accounts_path = Path(accounts_file) if accounts_file else DEFAULT_ACCOUNTS_FILE
    if accounts_path.exists():
        return load_accounts_from_file(accounts_path)

    env_accounts = load_accounts_from_env(env)
    if env_accounts is not None:
        return env_accounts

    if os.getenv("NOTION_REQUIRE_ACCOUNTS", "").strip() in {"1", "true", "True"}:
        raise AccountValidationError(
            "未找到账号配置：请创建 accounts.json 文件或设置 NOTION_ACCOUNTS 环境变量。"
        )
    return []


def accounts_to_dicts(accounts: list[AccountConfig]) -> list[dict[str, Any]]:
    return [account.to_dict() for account in accounts]


def save_accounts_atomic(accounts: list[AccountConfig], path: str | Path) -> None:
    """Validate and atomically replace an accounts JSON file."""

    if not accounts:
        raise AccountValidationError("拒绝写入空账号列表。")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    payload = json.dumps(accounts_to_dicts(accounts), ensure_ascii=False, indent=2) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
