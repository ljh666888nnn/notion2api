from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence

from app.account_schema import (
    AccountConfig,
    duplicate_identities,
    mask_email,
    mask_secret,
)
from app.logger import logger


ClientFactory = Callable[[Dict[str, Any]], Any]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
RandomFunc = Callable[[], float]

# 速率限制/空流的冷却时长（秒）。实测 1 分钟自愈，留点余量。
RATE_LIMIT_COOLDOWN = 60.0

SUPPORTED_STRATEGIES = {
    "fastest",
    "round_robin",
    "weighted_round_robin",
    "least_recently_used",
    "least_inflight",
}

def _account_health_rank(state: "AccountState") -> int:
    """账号健康度：0 最近 200 且无连续失败，1 有成功记录，2 从未试过，3 5xx，4 鉴权失败。"""
    status = state.last_status_code
    successes = int(getattr(state, "total_successes", 0) or 0)
    failures = int(getattr(state, "consecutive_failures", 0) or 0)
    if status == 200 or successes > 0:
        return 0 if failures == 0 else 1
    if status is None:
        return 2
    if status in (401, 403):
        return 4
    return 3


def _default_client_factory(account: Dict[str, Any]) -> Any:
    from app.notion_client import NotionOpusAPI

    return NotionOpusAPI(account)


@dataclass
class AccountState:
    account: AccountConfig
    client: Any
    cooldown_until: float = 0.0
    disabled: bool = False
    disabled_reason: str = ""
    consecutive_failures: int = 0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    inflight: int = 0
    last_error: str = ""
    last_status_code: int | None = None
    last_used_at: float = 0.0
    last_success_at: float = 0.0
    last_failure_at: float = 0.0
    last_latency_seconds: float | None = None
    total_latency_seconds: float = 0.0
    latency_samples: int = 0
    # 成功请求的指数滑动平均延迟（秒），供 fastest 策略排序；0 表示还没有样本
    ewma_latency_seconds: float = 0.0
    smooth_weight: float = 0.0

    def max_parallel(self, default_max_concurrency: int) -> int:
        configured = self.account.max_concurrency
        if configured > 0:
            return configured
        return max(0, default_max_concurrency)

    def saturated(self, default_max_concurrency: int) -> bool:
        limit = self.max_parallel(default_max_concurrency)
        return limit > 0 and self.inflight >= limit

    def status(self, now: float, default_max_concurrency: int = 0) -> str:
        if self.disabled:
            return "disabled"
        if self.cooldown_until > now:
            return "cooling"
        if self.saturated(default_max_concurrency):
            return "saturated"
        return "active"

    def accepting(self, now: float, default_max_concurrency: int = 0) -> bool:
        return (
            not self.disabled
            and self.cooldown_until <= now
            and not self.saturated(default_max_concurrency)
        )


class AccountLease:
    """A request-scoped account reservation.

    Call ``mark_success`` or one of the failure methods before ``release`` so
    latency and outcome statistics are recorded exactly once.
    """

    def __init__(self, pool: "AccountPool", state: AccountState, model: str = "") -> None:
        self._pool = pool
        self._state = state
        # 请求用的模型显式挂在租约上：流式响应会在别的线程里结束，
        # 那里拿不到请求的 contextvar，明细就会缺 model。
        self.model = model
        self._started_at = pool._clock()
        self._released = False
        self._outcome_recorded = False

    @property
    def client(self) -> Any:
        return self._pool._ensure_client(self._state)

    @property
    def account_key(self) -> str:
        return self._state.account.account_key

    @property
    def profile_name(self) -> str:
        return self._state.account.profile_name

    @property
    def released(self) -> bool:
        return self._released

    @property
    def age_seconds(self) -> float:
        return max(0.0, self._pool._clock() - self._started_at)

    def mark_success(self) -> bool:
        if self._released or self._outcome_recorded:
            return False
        self._outcome_recorded = True
        return self._pool._mark_success_state(
            self._state, self.age_seconds, model=self.model
        )

    def record_upstream_error(self, exc: BaseException, reason: str = "") -> bool:
        if self._released or self._outcome_recorded:
            return False
        self._outcome_recorded = True
        return self._pool._record_upstream_error_state(
            self._state, exc, reason=reason, latency_seconds=self.age_seconds,
            model=self.model,
        )

    def mark_failed(
        self,
        reason: str = "",
        *,
        cooldown_seconds: float | None = None,
        permanent: bool = False,
        status_code: int | None = None,
    ) -> bool:
        if self._released or self._outcome_recorded:
            return False
        self._outcome_recorded = True
        return self._pool._mark_failed_state(
            self._state,
            reason=reason,
            cooldown_seconds=cooldown_seconds,
            permanent=permanent,
            status_code=status_code,
            latency_seconds=self.age_seconds,
            model=self.model,
        )

    def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        return self._pool._release_lease_state(self._state)

    def __enter__(self) -> "AccountLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()



def _log_call(
    state: "AccountState",
    *,
    ok: bool,
    status_code: int | None = None,
    latency_seconds: float | None = None,
    reason: str = "",
    model: str = "",
) -> None:
    """写一条真实调用明细；日志失败不能影响主流程。"""
    try:
        from app.call_context import get_current_model
        from app.call_log import get_call_log

        # 状态码为空时给个可读的代替值：上游失败记 502，其余传输层问题记 599，
        # 免得明细里一片空白、看不出是"没请求到"还是"请求被拒"。
        resolved_status = status_code
        if resolved_status is None:
            lowered = (reason or "").lower()
            resolved_status = 502 if ("upstream" in lowered or "empty" in lowered) else 599
        get_call_log().record(
            account_key=state.account.account_key,
            account_label=mask_email(state.account.user_email) or state.account.profile_name,
            ok=ok,
            status_code=resolved_status,
            latency_ms=(latency_seconds * 1000) if latency_seconds is not None else None,
            reason=reason,
            model=model or get_current_model(),
        )
    except Exception:  # noqa: BLE001
        pass


class AccountPool:
    def __init__(
        self,
        accounts: Sequence[AccountConfig | Mapping[str, Any]],
        *,
        strategy: str = "round_robin",
        client_factory: ClientFactory | None = None,
        clock: Clock = time.time,
        sleep_func: Sleeper = time.sleep,
        base_cooldown: float = 3.0,
        max_cooldown: float = 60.0,
        max_wait: float = 15.0,
        default_max_concurrency: int = 0,
        cooldown_jitter_ratio: float = 0.15,
        random_func: RandomFunc = random.random,
        condition_wait: bool = True,
        metrics_path: str | Path | None = None,
    ) -> None:
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"不支持的账号池策略: {strategy}，可选: {', '.join(sorted(SUPPORTED_STRATEGIES))}"
            )

        self.strategy = strategy
        self._client_factory = client_factory or _default_client_factory
        self._clock = clock
        self._sleep = sleep_func
        self._base_cooldown = max(0.1, float(base_cooldown))
        self._max_cooldown = max(self._base_cooldown, float(max_cooldown))
        self._max_wait = max(0.0, float(max_wait))
        self._default_max_concurrency = max(0, int(default_max_concurrency))
        self._cooldown_jitter_ratio = min(1.0, max(0.0, float(cooldown_jitter_ratio)))
        self._random = random_func
        self._condition_wait = bool(condition_wait)
        self._condition = threading.Condition()
        self._lock = self._condition
        self._metrics_path = Path(metrics_path) if metrics_path else None
        self._states = self._build_states(accounts)
        self._current_index = 0

    @staticmethod
    def _normalize_accounts(
        accounts: Sequence[AccountConfig | Mapping[str, Any]],
    ) -> list[AccountConfig]:
        if not accounts:
            raise ValueError("账号池初始化失败：未提供任何账号配置。")
        normalized: list[AccountConfig] = []
        for index, raw in enumerate(accounts):
            if isinstance(raw, AccountConfig):
                normalized.append(raw)
            else:
                normalized.append(AccountConfig.from_mapping(raw, index=index))
        duplicates = duplicate_identities(normalized)
        if duplicates:
            raise ValueError(
                "账号池初始化失败：发现重复账号标识。请确保每个 profile_name 唯一；"
                "未设置 profile_name 时，user_id 与 space_id 的组合也必须唯一。"
            )
        return normalized

    # 这些字段在 reload / 重启后要继承，否则前端统计会突然归零
    _CARRIED_METRIC_FIELDS = (
        "total_requests",
        "total_successes",
        "total_failures",
        "total_latency_seconds",
        "latency_samples",
        "smooth_weight",
        "last_success_at",
        "last_failure_at",
        "last_latency_seconds",
        "last_used_at",
        "last_status_code",
        "consecutive_failures",
    )

    def _build_states(
        self,
        accounts: Sequence[AccountConfig | Mapping[str, Any]],
        previous: Sequence["AccountState"] | None = None,
    ) -> list[AccountState]:
        normalized = self._normalize_accounts(accounts)
        carried: Dict[str, "AccountState"] = {}
        for state in previous or ():
            carried[state.account.account_key] = state
        # 只有显式配置了 metrics_path 才读写持久化存储，
        # 否则（例如单元测试）会读到真实库里的历史计数，污染调度排序。
        persisted = self._load_persisted_metrics() if self._metrics_file() else {}

        states: list[AccountState] = []
        for account in normalized:
            state = AccountState(
                account=account,
                # 客户端延迟到第一次使用时再建：992 个 cloudscraper 实例
                # 要把启动拖到 30 秒，崩溃重启的恢复时间也跟着变长。
                client=None,
            )
            source = carried.get(account.account_key)
            if source is not None:
                for field in self._CARRIED_METRIC_FIELDS:
                    setattr(state, field, getattr(source, field, None))
            else:
                saved = persisted.get(account.account_key)
                if isinstance(saved, dict):
                    for field in self._CARRIED_METRIC_FIELDS:
                        if field in saved:
                            setattr(state, field, saved[field])
            states.append(state)

        # 冷启动优化：按历史健康度做一次稳定排序。
        # 账号池 980 个里有不少注册后从未成功过的死号；纯 round_robin 会先撞它们，
        # 单请求只有 3~6 次重试，于是开局几条请求经常直接 NOTION_502。
        # 已知 200 / 有成功记录的排前面，从未试过的居中，4xx/5xx 排后面。
        states.sort(key=_account_health_rank)
        return states

    def _metrics_file(self) -> Path | None:
        path = getattr(self, "_metrics_path", None)
        return Path(path) if path else None

    def _load_persisted_metrics(self) -> Dict[str, Dict[str, Any]]:
        """从 SQLite 读取上次保存的累计计数（兼容旧的 json 文件导入一次）。"""
        try:
            from app.store import get_store

            data = get_store().load_metrics_all()
            if data:
                return data
        except Exception:  # noqa: BLE001
            pass
        # 首次迁移：把旧的 state/pool_metrics.json 导进来
        path = self._metrics_file()
        if path is None or not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        try:
            from app.store import get_store

            get_store().save_metrics_bulk(payload)
        except Exception:  # noqa: BLE001
            pass
        return payload

    def persist_metrics(self) -> None:
        """把累计计数写入 SQLite，进程重启后可以接上。"""
        payload: Dict[str, Dict[str, Any]] = {}
        for state in self._states:
            payload[state.account.account_key] = {
                field: getattr(state, field, 0) for field in self._CARRIED_METRIC_FIELDS
            }
        try:
            from app.store import get_store

            get_store().save_metrics_bulk(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Persist pool metrics failed",
                extra={"request_info": {"event": "pool_metrics_persist_failed", "error": str(exc)[:120]}},
            )

    @property
    def clients(self) -> List[Any]:
        return [state.client for state in self._states]

    def accounts_raw(self) -> List[Dict[str, Any]]:
        """返回账号原始配置（含 cookies），仅供管理接口做额度探测使用。"""
        with self._lock:
            out: List[Dict[str, Any]] = []
            for state in self._states:
                data = state.account.to_dict()
                data["account_key"] = state.account.account_key
                out.append(data)
            return out

    @property
    def account_keys(self) -> List[str]:
        return [state.account.account_key for state in self._states]

    def _state_snapshot(self, state: AccountState, now: float) -> Dict[str, Any]:
        outcome_total = state.total_successes + state.total_failures
        success_rate = (
            round(state.total_successes / outcome_total, 4) if outcome_total else None
        )
        average_latency_ms = (
            round(state.total_latency_seconds / state.latency_samples * 1000, 1)
            if state.latency_samples
            else None
        )
        last_latency_ms = (
            round(state.last_latency_seconds * 1000, 1)
            if state.last_latency_seconds is not None
            else None
        )
        effective_limit = state.max_parallel(self._default_max_concurrency)
        return {
            **state.account.public_dict(),
            "status": state.status(now, self._default_max_concurrency),
            "available": state.accepting(now, self._default_max_concurrency),
            "cooldown_remaining": round(max(0.0, state.cooldown_until - now), 1),
            "disabled_reason": state.disabled_reason,
            "consecutive_failures": state.consecutive_failures,
            "total_requests": state.total_requests,
            "total_successes": state.total_successes,
            "total_failures": state.total_failures,
            "success_rate": success_rate,
            "inflight": state.inflight,
            "max_concurrency_effective": effective_limit,
            "last_error": state.last_error,
            "last_status_code": state.last_status_code,
            "last_latency_ms": last_latency_ms,
            "average_latency_ms": average_latency_ms,
            "last_used_at": int(state.last_used_at) if state.last_used_at else None,
            "last_success_at": int(state.last_success_at)
            if state.last_success_at
            else None,
            "last_failure_at": int(state.last_failure_at)
            if state.last_failure_at
            else None,
        }

    def _find_state(self, client: Any) -> AccountState | None:
        for state in self._states:
            if state.client is client:
                return state
        return None

    def _state_at(self, account_index: int) -> AccountState:
        if account_index < 1 or account_index > len(self._states):
            raise IndexError(f"账号序号 {account_index} 不存在。")
        return self._states[account_index - 1]

    def _wake_waiters(self) -> None:
        self._condition.notify_all()

    def _select_state(
        self,
        now: float,
        *,
        exclude_keys: set[str] | None = None,
        affinity_key: str | None = None,
    ) -> AccountState | None:
        excluded = exclude_keys or set()

        if affinity_key and affinity_key not in excluded:
            for state in self._states:
                if state.account.account_key != affinity_key:
                    continue
                if state.accepting(now, self._default_max_concurrency):
                    return state
                break

        available = [
            index
            for index, state in enumerate(self._states)
            if state.account.account_key not in excluded
            and state.accepting(now, self._default_max_concurrency)
        ]
        if not available:
            return None

        highest_priority = max(self._states[index].account.priority for index in available)
        available = [
            index
            for index in available
            if self._states[index].account.priority == highest_priority
        ]

        if self.strategy == "fastest":
            # 排序优先级：账号健康度 → 并发占用 → 连续失败 → 近期延迟 → 最近使用时间。
            # 目标不是"均匀轮询"，而是"用最可能一次成功的账号"，减少 502 重试带来的等待。
            index = min(
                available,
                key=lambda item: (
                    _account_health_rank(self._states[item]),
                    self._states[item].inflight,
                    self._states[item].consecutive_failures,
                    round(
                        self._states[item].ewma_latency_seconds
                        or self._states[item].last_latency_seconds
                        or 0.0,
                        2,
                    ),
                    self._states[item].last_used_at,
                    item,
                ),
            )
            return self._states[index]

        if self.strategy == "least_inflight":
            index = min(
                available,
                key=lambda item: (
                    self._states[item].inflight,
                    self._states[item].last_used_at,
                    item,
                ),
            )
            return self._states[index]

        if self.strategy == "least_recently_used":
            index = min(
                available,
                key=lambda item: (self._states[item].last_used_at, item),
            )
            return self._states[index]

        if self.strategy == "weighted_round_robin":
            total_weight = 0
            for item in available:
                weight = max(1, self._states[item].account.weight)
                self._states[item].smooth_weight += weight
                total_weight += weight
            index = max(
                available,
                key=lambda item: (self._states[item].smooth_weight, -item),
            )
            self._states[index].smooth_weight -= total_weight
            self._current_index = (index + 1) % len(self._states)
            return self._states[index]

        for offset in range(len(self._states)):
            index = (self._current_index + offset) % len(self._states)
            if index in available:
                self._current_index = (index + 1) % len(self._states)
                return self._states[index]
        return self._states[available[0]]

    def _next_cooling_wait(self, now: float, excluded: set[str]) -> float | None:
        waits = [
            state.cooldown_until - now
            for state in self._states
            if not state.disabled
            and state.account.account_key not in excluded
            and state.cooldown_until > now
        ]
        return min(waits) if waits else None

    def _ensure_client(self, state: AccountState) -> Any:
        if state.client is None:
            state.client = self._client_factory(state.account.to_dict())
        return state.client

    def _claim_state(self, state: AccountState, now: float, model: str = "") -> AccountLease:
        self._ensure_client(state)
        state.total_requests += 1
        state.last_used_at = now
        state.inflight += 1
        return AccountLease(self, state, model=model)

    def _wait_plan_locked(
        self,
        now: float,
        *,
        excluded: set[str],
        deadline: float,
        wait_if_cooling: bool,
    ) -> tuple[float, str]:
        """Return how long to wait and why, or raise when this request cannot proceed."""

        non_excluded = [
            item
            for item in self._states
            if item.account.account_key not in excluded
        ]
        if not non_excluded:
            if all(item.disabled for item in self._states):
                raise RuntimeError(
                    "所有账号均已禁用，请在配置中启用或导入新账号。"
                )
            raise RuntimeError("没有更多可尝试的账号，请稍后重试。")

        usable = [item for item in non_excluded if not item.disabled]
        if not usable:
            raise RuntimeError("所有账号均已禁用，请在配置中启用或导入新账号。")

        cooling_wait = self._next_cooling_wait(now, excluded)
        remaining = deadline - now
        if remaining <= 0:
            if cooling_wait is not None:
                raise RuntimeError(
                    f"所有账号冷却中，请在 {max(1, int(cooling_wait))} 秒后重试。"
                )
            if any(item.saturated(self._default_max_concurrency) for item in usable):
                raise RuntimeError("所有账号并发已满，请稍后重试。")
            raise RuntimeError("账号池繁忙，请稍后重试。")

        if cooling_wait is not None:
            if not wait_if_cooling:
                raise RuntimeError(
                    f"所有账号冷却中，请在 {max(1, int(cooling_wait))} 秒后重试。"
                )
            return min(cooling_wait, remaining), "cooling"
        return remaining, "concurrency"

    def _log_pool_wait(self, wait_seconds: float, reason: str) -> None:
        logger.info(
            "Account pool waiting for availability",
            extra={
                "request_info": {
                    "event": "account_pool_wait",
                    "reason": reason,
                    "wait_seconds": round(wait_seconds, 2),
                    "inflight": sum(item.inflight for item in self._states),
                }
            },
        )

    def acquire(
        self,
        *,
        affinity_key: str | None = None,
        wait_if_cooling: bool = True,
        exclude_keys: set[str] | None = None,
        max_wait: float | None = None,
        model: str = "",
    ) -> AccountLease:
        excluded = set(exclude_keys or ())
        wait_budget = self._max_wait if max_wait is None else max(0.0, float(max_wait))
        deadline = self._clock() + wait_budget

        while True:
            wait_seconds: float | None = None
            with self._lock:
                now = self._clock()
                state = self._select_state(
                    now, exclude_keys=excluded, affinity_key=affinity_key
                )
                if state is not None:
                    return self._claim_state(state, now, model=model)

                wait_seconds, reason = self._wait_plan_locked(
                    now,
                    excluded=excluded,
                    deadline=deadline,
                    wait_if_cooling=wait_if_cooling,
                )
                self._log_pool_wait(wait_seconds, reason)
                if self._condition_wait:
                    self._condition.wait(wait_seconds)
            if not self._condition_wait:
                self._sleep(wait_seconds)

    async def acquire_async(
        self,
        *,
        affinity_key: str | None = None,
        wait_if_cooling: bool = True,
        exclude_keys: set[str] | None = None,
        max_wait: float | None = None,
        model: str = "",
    ) -> AccountLease:
        """Async-safe acquisition that never blocks the FastAPI event loop."""

        excluded = set(exclude_keys or ())
        wait_budget = self._max_wait if max_wait is None else max(0.0, float(max_wait))
        deadline = self._clock() + wait_budget
        last_logged_reason = ""

        while True:
            with self._lock:
                now = self._clock()
                state = self._select_state(
                    now, exclude_keys=excluded, affinity_key=affinity_key
                )
                if state is not None:
                    return self._claim_state(state, now, model=model)

                wait_seconds, reason = self._wait_plan_locked(
                    now,
                    excluded=excluded,
                    deadline=deadline,
                    wait_if_cooling=wait_if_cooling,
                )
                if reason != last_logged_reason:
                    self._log_pool_wait(wait_seconds, reason)
                    last_logged_reason = reason

            await asyncio.sleep(min(0.1, max(0.01, wait_seconds)))

    def get_client(self, wait_if_cooling: bool = True) -> Any:
        """Legacy compatibility helper. New request code should use acquire()."""

        lease = self.acquire(wait_if_cooling=wait_if_cooling)
        try:
            return lease.client
        finally:
            lease.release()

    def _mark_success_state(
        self, state: AccountState, latency_seconds: float | None = None, model: str = ""
    ) -> bool:
        with self._lock:
            if state not in self._states:
                return False
            state.consecutive_failures = 0
            state.last_error = ""
            state.last_status_code = 200
            state.cooldown_until = 0.0
            state.total_successes += 1
            state.last_success_at = self._clock()
            _log_call(
                state, ok=True, status_code=200,
                latency_seconds=latency_seconds, model=model,
            )
            if latency_seconds is not None:
                elapsed = max(0.0, float(latency_seconds))
                state.last_latency_seconds = elapsed
                state.total_latency_seconds += elapsed
                state.latency_samples += 1
                # EWMA：首样本直接取值，之后 0.7*旧 + 0.3*新，近期表现权重更高
                state.ewma_latency_seconds = (
                    elapsed
                    if state.ewma_latency_seconds <= 0
                    else state.ewma_latency_seconds * 0.7 + elapsed * 0.3
                )
            self._wake_waiters()
            return True

    def mark_success(self, client: Any, latency_seconds: float | None = None) -> bool:
        with self._lock:
            state = self._find_state(client)
            if state is None:
                return False
        return self._mark_success_state(state, latency_seconds)

    def _cooldown_for_failure(
        self, state: AccountState, cooldown_seconds: float | None
    ) -> float:
        if cooldown_seconds is not None:
            return min(self._max_cooldown, max(0.0, float(cooldown_seconds)))

        exponent = min(max(0, state.consecutive_failures - 1), 10)
        raw = self._base_cooldown * (2**exponent)
        if self._cooldown_jitter_ratio > 0:
            jitter = (float(self._random()) * 2.0 - 1.0) * self._cooldown_jitter_ratio
            raw *= max(0.05, 1.0 + jitter)
        return min(self._max_cooldown, max(0.1, raw))

    def _mark_failed_state(
        self,
        state: AccountState,
        *,
        cooldown_seconds: float | None = None,
        reason: str = "",
        permanent: bool = False,
        status_code: int | None = None,
        latency_seconds: float | None = None,
        model: str = "",
    ) -> bool:
        with self._lock:
            if state not in self._states:
                return False

            state.total_failures += 1
            state.consecutive_failures += 1
            state.last_error = reason or "upstream_failure"
            state.last_status_code = status_code
            state.last_failure_at = self._clock()
            _log_call(state, ok=False, status_code=status_code,
                      latency_seconds=latency_seconds, reason=reason, model=model)
            if latency_seconds is not None:
                state.last_latency_seconds = max(0.0, float(latency_seconds))

            if permanent:
                state.disabled = True
                state.disabled_reason = state.last_error
                state.cooldown_until = 0.0
                wait_seconds = 0.0
            else:
                wait_seconds = self._cooldown_for_failure(state, cooldown_seconds)
                state.cooldown_until = self._clock() + wait_seconds

            logger.warning(
                "Account marked as failed",
                extra={
                    "request_info": {
                        "event": "account_failed",
                        "account": mask_email(state.account.user_email)
                        or mask_secret(state.account.user_id),
                        "account_key": state.account.account_key,
                        "space_id": mask_secret(state.account.space_id),
                        "reason": state.last_error,
                        "status_code": status_code,
                        "cooldown_seconds": round(wait_seconds, 1),
                        "permanent": permanent,
                    }
                },
            )
            self._wake_waiters()
            return True

    def mark_failed(
        self,
        client: Any,
        cooldown_seconds: float | None = None,
        reason: str = "",
        permanent: bool = False,
        status_code: int | None = None,
        latency_seconds: float | None = None,
    ) -> bool:
        with self._lock:
            state = self._find_state(client)
        if state is None:
            logger.warning(
                "Attempted to mark unknown account as failed",
                extra={"request_info": {"event": "account_failed_unknown"}},
            )
            return False
        return self._mark_failed_state(
            state,
            cooldown_seconds=cooldown_seconds,
            reason=reason,
            permanent=permanent,
            status_code=status_code,
            latency_seconds=latency_seconds,
        )

    def _record_upstream_error_state(
        self,
        state: AccountState,
        exc: BaseException,
        *,
        reason: str = "",
        latency_seconds: float | None = None,
        model: str = "",
    ) -> bool:
        status = getattr(exc, "status_code", None)
        retriable = bool(getattr(exc, "retriable", False))
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            try:
                retry_after = max(0.0, float(retry_after))
            except (TypeError, ValueError):
                retry_after = None
        with self._lock:
            state.last_status_code = status if isinstance(status, int) else None

        if status == 401:
            return self._mark_failed_state(
                state,
                reason=reason or "upstream_auth_error",
                permanent=True,
                status_code=status,
                latency_seconds=latency_seconds, model=model,
            )

        should_cool = (
            retriable
            or status in {403, 408, 425, 429}
            or (isinstance(status, int) and status >= 500)
        )
        if not should_cool:
            # 400 这类不冷却的失败以前直接 return，明细里既没有状态码也没有耗时。
            # 账号不罚，但这一次调用要如实记下来。
            _log_call(
                state,
                ok=False,
                status_code=status,
                latency_seconds=latency_seconds,
                reason=reason or f"upstream_{status or 'error'}",
                model=model,
            )
            return False

        # 速率限制（429 / 空流）一分钟就自愈，不该走指数退避。
        # 实测：连续快发第 6 次被拒 → 等 1 分钟恢复 → 每 12 秒一次可无限跑。
        # 按指数退避会把它冷却成几小时，等于把好账号判死。
        reason_text = str(reason or "").lower()
        if (
            status == 429
            or "empty stream" in reason_text
            or "rate" in reason_text
            or "429" in reason_text
        ) and retry_after is None:
            return self._mark_failed_state(
                state,
                reason=reason or "rate_limited",
                cooldown_seconds=min(90.0, max(30.0, float(RATE_LIMIT_COOLDOWN))),
                status_code=status if isinstance(status, int) else None,
                latency_seconds=latency_seconds, model=model,
            )

        return self._mark_failed_state(
            state,
            reason=reason or f"upstream_{status or 'error'}",
            cooldown_seconds=retry_after,
            status_code=status if isinstance(status, int) else None,
            latency_seconds=latency_seconds, model=model,
        )

    def record_upstream_error(
        self,
        client: Any,
        exc: BaseException,
        reason: str = "",
        latency_seconds: float | None = None,
    ) -> bool:
        with self._lock:
            state = self._find_state(client)
        if state is None:
            return False
        return self._record_upstream_error_state(
            state,
            exc,
            reason=reason,
            latency_seconds=latency_seconds,
        )

    def disable_client(self, client: Any, reason: str = "upstream_auth_error") -> bool:
        return self.mark_failed(client, reason=reason, permanent=True)

    def _reset_runtime_state(self, state: AccountState) -> None:
        state.disabled = False
        state.disabled_reason = ""
        state.cooldown_until = 0.0
        state.consecutive_failures = 0
        state.last_error = ""
        state.last_status_code = None

    def _reset_metrics(self, state: AccountState) -> None:
        state.total_requests = 0
        state.total_successes = 0
        state.total_failures = 0
        state.last_success_at = 0.0
        state.last_failure_at = 0.0
        state.last_latency_seconds = None
        state.total_latency_seconds = 0.0
        state.latency_samples = 0
        state.smooth_weight = 0.0

    def enable_client(self, client: Any) -> bool:
        with self._lock:
            state = self._find_state(client)
            if state is None:
                return False
            self._reset_runtime_state(state)
            self._wake_waiters()
            return True

    def enable_account(self, account_index: int) -> Dict[str, Any]:
        with self._lock:
            state = self._state_at(account_index)
            self._reset_runtime_state(state)
            result = self._state_snapshot(state, self._clock())
            if not result.get("profile_name"):
                result["profile_name"] = f"account-{account_index}"
            result["account_index"] = account_index
            self._wake_waiters()
            return result

    def disable_account(
        self, account_index: int, reason: str = "manual_disable"
    ) -> Dict[str, Any]:
        with self._lock:
            state = self._state_at(account_index)
            state.disabled = True
            state.disabled_reason = reason or "manual_disable"
            state.cooldown_until = 0.0
            result = self._state_snapshot(state, self._clock())
            if not result.get("profile_name"):
                result["profile_name"] = f"account-{account_index}"
            result["account_index"] = account_index
            self._wake_waiters()
            return result

    def reset_account(self, account_index: int) -> Dict[str, Any]:
        with self._lock:
            state = self._state_at(account_index)
            self._reset_runtime_state(state)
            self._reset_metrics(state)
            result = self._state_snapshot(state, self._clock())
            if not result.get("profile_name"):
                result["profile_name"] = f"account-{account_index}"
            result["account_index"] = account_index
            self._wake_waiters()
            return result

    def reset_cooling(self) -> int:
        now = self._clock()
        reset_count = 0
        with self._lock:
            for state in self._states:
                if state.disabled or state.cooldown_until <= now:
                    continue
                state.cooldown_until = 0.0
                state.consecutive_failures = 0
                state.last_error = ""
                reset_count += 1
            if reset_count:
                self._wake_waiters()
        return reset_count

    def _release_lease_state(self, state: AccountState) -> bool:
        with self._lock:
            if state not in self._states:
                return False
            state.inflight = max(0, state.inflight - 1)
            self._wake_waiters()
            return True

    def probe_account(self, account_index: int) -> Dict[str, Any]:
        with self._lock:
            state = self._state_at(account_index)
            client = self._ensure_client(state)
            account_key = state.account.account_key

        checker = getattr(client, "check_health", None)
        if not callable(checker):
            return {
                "account_index": account_index,
                "account_key": account_key,
                "supported": False,
                "ok": False,
                "message": "当前客户端不支持健康探测。",
            }

        started = self._clock()
        try:
            raw_result = checker()
        except Exception as exc:
            raw_result = {
                "ok": False,
                "status_code": None,
                "message": f"探测异常：{type(exc).__name__}",
            }
        elapsed_ms = round(max(0.0, self._clock() - started) * 1000, 1)

        if isinstance(raw_result, Mapping):
            result = dict(raw_result)
        else:
            result = {"ok": bool(raw_result), "status_code": None, "message": ""}
        status_code = result.get("status_code")
        if not isinstance(status_code, int):
            status_code = None
        message = " ".join(str(result.get("message") or "").split())[:200]

        if status_code == 401:
            self.disable_client(client, reason="probe_auth_error")

        with self._lock:
            current = self._find_state(client)
            if current is not None:
                current.last_status_code = status_code
                if status_code is not None and status_code != 200:
                    current.last_error = f"probe_http_{status_code}"
                snapshot = self._state_snapshot(current, self._clock())
            else:
                snapshot = None

        return {
            "account_index": account_index,
            "account_key": account_key,
            "supported": True,
            "ok": bool(result.get("ok")),
            "status_code": status_code,
            "message": message or ("ok" if result.get("ok") else "probe_failed"),
            "latency_ms": elapsed_ms,
            "account": snapshot,
        }

    def reload(self, accounts: Sequence[AccountConfig | Mapping[str, Any]]) -> Dict[str, int]:
        # 继承旧 states 的计数，避免注册后自动 reload 把统计清零
        new_states = self._build_states(accounts, previous=self._states)
        with self._lock:
            self._states = new_states
            self._current_index = 0
            self._wake_waiters()
        self.persist_metrics()

        summary = self.get_status_summary()
        logger.info(
            "Account pool reloaded",
            extra={"request_info": {"event": "account_pool_reloaded", **summary}},
        )
        return summary

    def get_status_summary(self) -> Dict[str, int]:
        now = self._clock()
        with self._lock:
            active = sum(
                1
                for state in self._states
                if not state.disabled and state.cooldown_until <= now
            )
            cooling = sum(
                1
                for state in self._states
                if not state.disabled and state.cooldown_until > now
            )
            disabled = sum(1 for state in self._states if state.disabled)
            saturated = sum(
                1
                for state in self._states
                if not state.disabled
                and state.cooldown_until <= now
                and state.saturated(self._default_max_concurrency)
            )
            available = sum(
                1
                for state in self._states
                if state.accepting(now, self._default_max_concurrency)
            )
            inflight = sum(state.inflight for state in self._states)
            total_successes = sum(state.total_successes for state in self._states)
            total_failures = sum(state.total_failures for state in self._states)
            return {
                "total": len(self._states),
                "active": active,
                "available": available,
                "cooling": cooling,
                "disabled": disabled,
                "saturated": saturated,
                "inflight": inflight,
                "total_successes": total_successes,
                "total_failures": total_failures,
            }

    def snapshot(self) -> List[Dict[str, Any]]:
        now = self._clock()
        result: List[Dict[str, Any]] = []
        with self._lock:
            for index, state in enumerate(self._states, start=1):
                item = self._state_snapshot(state, now)
                item["account_index"] = index
                if not item.get("profile_name"):
                    item["profile_name"] = f"account-{index}"
                result.append(item)
        return result
