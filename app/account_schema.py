from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple


# 只强制 token_v2：space_id / user_id / space_view_id 可以由 account_discovery
# 在加载时自动补全（见 app/account_discovery.py）。
REQUIRED_ACCOUNT_FIELDS = ("token_v2",)
AUTO_FIELDS = ("space_id", "user_id", "space_view_id")


class AccountValidationError(ValueError):
    """Raised when an account entry is structurally invalid."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def mask_secret(value: str) -> str:
    text = _text(value)
    if not text:
        return ""
    if len(text) <= 8:
        return f"{text[:2]}***"
    return f"{text[:6]}***{text[-4:]}"


def mask_email(value: str) -> str:
    email = _text(value)
    if not email:
        return ""
    local, separator, domain = email.partition("@")
    if not separator:
        return mask_secret(email)
    return f"{local[:1] if local else ''}***@{domain}"


def _bounded_int(
    raw: Mapping[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
    index: int,
) -> int:
    value = raw.get(key, default)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise AccountValidationError(f"账号配置[{index}].{key} 必须是整数。")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise AccountValidationError(f"账号配置[{index}].{key} 必须是整数。") from exc
    if parsed < minimum or parsed > maximum:
        raise AccountValidationError(
            f"账号配置[{index}].{key} 必须在 {minimum} 到 {maximum} 之间。"
        )
    return parsed


@dataclass(frozen=True)
class AccountConfig:
    """Canonical account configuration used by the pool and import tools."""

    token_v2: str
    space_id: str
    user_id: str
    profile_name: str = ""
    space_view_id: str = ""
    user_name: str = "user"
    user_email: str = ""
    weight: int = 1
    priority: int = 0
    max_concurrency: int = 0
    cookies: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], index: int = 0) -> "AccountConfig":
        if not isinstance(raw, Mapping):
            raise AccountValidationError(f"账号配置[{index}] 必须是对象。")

        values: Dict[str, str] = {}
        missing = []
        for key in REQUIRED_ACCOUNT_FIELDS:
            value = _text(raw.get(key))
            values[key] = value
            if not value:
                missing.append(key)
        if missing:
            raise AccountValidationError(
                f"账号配置[{index}] 缺少必要字段: {', '.join(missing)}"
            )
        # 这三个允许留空，交给 account_store 在加载时用 token 反查。
        for key in ("space_id", "user_id"):
            values[key] = _text(raw.get(key))

        cookies_raw = raw.get("cookies") or {}
        if not isinstance(cookies_raw, Mapping):
            raise AccountValidationError(f"账号配置[{index}].cookies 必须是对象。")
        cookies = {
            str(key): str(value)
            for key, value in cookies_raw.items()
            if value is not None
        }

        return cls(
            token_v2=values["token_v2"],
            space_id=values["space_id"],
            user_id=values["user_id"],
            profile_name=_text(raw.get("profile_name")),
            space_view_id=_text(raw.get("space_view_id")),
            user_name=_text(raw.get("user_name")) or "user",
            user_email=_text(raw.get("user_email")),
            weight=_bounded_int(raw, "weight", 1, 1, 1000, index),
            priority=_bounded_int(raw, "priority", 0, -1000, 1000, index),
            max_concurrency=_bounded_int(
                raw, "max_concurrency", 0, 0, 1000, index
            ),
            cookies=cookies,
        )

    def identity(self) -> Tuple[str, ...]:
        if self.profile_name:
            return ("profile", self.profile_name.lower())
        if self.user_id or self.space_id:
            return ("account", self.user_id, self.space_id)
        # 只有 token 的账号也要有稳定 key，否则池子里会互相覆盖。
        return ("token", hashlib.sha256(self.token_v2.encode("utf-8")).hexdigest()[:16])

    @property
    def account_key(self) -> str:
        """Stable, non-secret routing key used for affinity and diagnostics."""

        raw = "|".join(self.identity()).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "token_v2": self.token_v2,
            "space_id": self.space_id,
            "user_id": self.user_id,
            "space_view_id": self.space_view_id,
            "user_name": self.user_name,
            "user_email": self.user_email,
            "weight": self.weight,
            "priority": self.priority,
            "max_concurrency": self.max_concurrency,
            "cookies": dict(self.cookies),
        }

    def public_dict(self, index: int | None = None) -> Dict[str, Any]:
        profile_name = self.profile_name
        if not profile_name and index is not None:
            profile_name = f"account-{index}"
        return {
            "profile_name": profile_name,
            "account_key": self.account_key,
            "user_name": self.user_name,
            "user_email": mask_email(self.user_email),
            "user_id": mask_secret(self.user_id),
            "space_id": mask_secret(self.space_id),
            "space_view_id": mask_secret(self.space_view_id),
            "weight": self.weight,
            "priority": self.priority,
            "max_concurrency": self.max_concurrency,
        }


def validate_accounts(raw_accounts: Sequence[Any], source: str = "accounts") -> list[AccountConfig]:
    if isinstance(raw_accounts, (str, bytes)) or not isinstance(raw_accounts, Sequence):
        raise AccountValidationError(f"{source} 必须是非空 JSON 数组。")
    if not raw_accounts:
        raise AccountValidationError(f"{source} 不能为空。")

    accounts: list[AccountConfig] = []
    for index, raw in enumerate(raw_accounts):
        if isinstance(raw, AccountConfig):
            accounts.append(raw)
        else:
            accounts.append(AccountConfig.from_mapping(raw, index=index))
    return accounts


def duplicate_identities(accounts: Sequence[AccountConfig]) -> list[Tuple[str, ...]]:
    seen: set[Tuple[str, ...]] = set()
    duplicates: list[Tuple[str, ...]] = []
    for account in accounts:
        identity = account.identity()
        if identity in seen and identity not in duplicates:
            duplicates.append(identity)
        seen.add(identity)
    return duplicates
