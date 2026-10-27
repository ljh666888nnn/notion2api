# -*- coding: utf-8 -*-
"""call_log.py - 调用明细（SQLite 持久化）。

原实现把明细放在内存 deque 里，进程重启即清空。现在全部落到
state/notion2api.db 的 calls 表，重启后仍可查询。
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Dict, Optional

from app.store import get_store


class CallLog:
    """兼容旧接口的薄封装，实际存储交给 Store。"""

    def __init__(self, path: Optional[str] = None) -> None:  # path 保留兼容，忽略
        self._store = get_store()
        # 记账是 fire-and-forget：account_pool 在持锁状态下调用它，
        # 同步写 SQLite（fsync 可能几秒到十几秒）会把 /health 一起拖死，
        # 进而被 watchdog 误判成挂掉。这里改成入队 + 后台线程落盘。
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=20000)
        self._dropped = 0
        threading.Thread(target=self._writer, name="call-log-writer", daemon=True).start()

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._store.record_call(**item)
            except Exception:  # noqa: BLE001
                pass
            finally:
                self._queue.task_done()

    def record(
        self,
        *,
        account_key: str = "",
        account_label: str = "",
        ok: bool,
        status_code: Optional[int] = None,
        latency_ms: Optional[float] = None,
        reason: str = "",
        model: str = "",
    ) -> Dict[str, Any]:
        ts = time.time()
        payload = {
            "account_key": account_key,
            "account_label": account_label,
            "ok": bool(ok),
            "status_code": status_code,
            "latency_ms": latency_ms,
            "reason": reason,
            "model": model,
            "ts": ts,
        }
        # 同步写：放后台线程时一旦消费者出问题，明细会静默丢失（踩过一次）。
        # 单条写入是 WAL 下的一次小事务，代价可以接受。
        try:
            self._store.record_call(**payload)
        except Exception:  # noqa: BLE001 - 记账失败不能影响主流程
            pass
        return {
            "ts": ts,
            "account_key": account_key,
            "account_label": account_label,
            "ok": bool(ok),
            "status_code": status_code,
            "latency_ms": latency_ms,
            "reason": reason,
            "model": model,
        }

    def page(
        self,
        offset: int = 0,
        limit: int = 20,
        only_failed: bool = False,
        account_key: str = "",
    ) -> Dict[str, Any]:
        return self._store.page_calls(
            offset=offset, limit=limit, only_failed=only_failed, account_key=account_key
        )

    def summary(self) -> Dict[str, Any]:
        return self._store.call_summary()


_call_log: Optional[CallLog] = None
_call_log_lock = threading.Lock()


def get_call_log(path: Optional[str] = None) -> CallLog:
    global _call_log
    with _call_log_lock:
        if _call_log is None:
            _call_log = CallLog(path)
        return _call_log
