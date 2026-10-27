# -*- coding: utf-8 -*-
"""store.py - 统一 SQLite 持久化。

之前的问题：调用记录只存在内存 deque、额度只在内存 dict、趋势采样只在前端
localStorage，进程一重启全没了。这里把四类数据都落到一个 SQLite 文件：

    calls        调用明细
    quota_cache  账号额度探测结果（带 fetched_at，按 TTL 复用）
    pool_metrics 账号累计计数（成功/失败/延迟…）
    pool_samples 账号池采样点（折线图历史）

连接按线程持有，WAL 模式，写入用事务，读写失败不影响主流程。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_PATH = Path(
    os.getenv("DB_PATH")
    or (Path(__file__).resolve().parent.parent / "state" / "notion2api.db")
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    account_key   TEXT    NOT NULL DEFAULT '',
    account_label TEXT    NOT NULL DEFAULT '',
    ok            INTEGER NOT NULL,
    status_code   INTEGER,
    latency_ms    REAL,
    reason        TEXT    NOT NULL DEFAULT '',
    model         TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls(ts DESC);
CREATE INDEX IF NOT EXISTS idx_calls_account ON calls(account_key, ts DESC);
CREATE INDEX IF NOT EXISTS idx_calls_ok ON calls(ok, ts DESC);

CREATE TABLE IF NOT EXISTS quota_cache (
    account_key TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    fetched_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pool_metrics (
    account_key TEXT PRIMARY KEY,
    payload     TEXT NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pool_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    available       INTEGER NOT NULL,
    inflight        INTEGER NOT NULL,
    success_rate    REAL,
    average_latency REAL,
    total_successes INTEGER NOT NULL,
    total_failures  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON pool_samples(ts DESC);


CREATE TABLE IF NOT EXISTS ai_probe (
    account_key  TEXT PRIMARY KEY,
    ok           INTEGER NOT NULL,
    chars        INTEGER NOT NULL DEFAULT 0,
    latency_ms   REAL,
    error        TEXT    NOT NULL DEFAULT '',
    probed_at    REAL    NOT NULL
);

"""


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        raw = path or os.getenv("STATE_DB_PATH") or _DEFAULT_PATH
        self.path = Path(raw)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # ---------- 连接 ----------
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        with self._write_lock:
            conn = self._conn()
            conn.executescript(_SCHEMA)

    def _execute(self, sql: str, args: Tuple[Any, ...] = ()) -> None:
        with self._write_lock:
            self._conn().execute(sql, args)

    def _query(self, sql: str, args: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
        return list(self._conn().execute(sql, args).fetchall())

    # ---------- calls ----------
    def record_call(
        self,
        *,
        account_key: str,
        account_label: str,
        ok: bool,
        status_code: Optional[int],
        latency_ms: Optional[float],
        reason: str,
        model: str = "",
        ts: Optional[float] = None,
    ) -> None:
        self._execute(
            "INSERT INTO calls (ts, account_key, account_label, ok, status_code,"
            " latency_ms, reason, model) VALUES (?,?,?,?,?,?,?,?)",
            (
                ts if ts is not None else time.time(),
                account_key or "",
                account_label or "",
                1 if ok else 0,
                status_code,
                latency_ms,
                (reason or "")[:200],
                model or "",
            ),
        )

    def page_calls(
        self,
        offset: int = 0,
        limit: int = 20,
        only_failed: bool = False,
        account_key: str = "",
    ) -> Dict[str, Any]:
        where: List[str] = []
        args: List[Any] = []
        if only_failed:
            where.append("ok = 0")
        if account_key:
            where.append("account_key = ?")
            args.append(account_key)
        clause = f"WHERE {' AND '.join(where)}" if where else ""

        total = self._query(f"SELECT COUNT(*) AS n FROM calls {clause}", tuple(args))[0]["n"]
        rows = self._query(
            f"SELECT * FROM calls {clause} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
            tuple(args) + (int(limit), int(offset)),
        )
        return {
            "total": int(total),
            "offset": int(offset),
            "limit": int(limit),
            "items": [
                {
                    "id": row["id"],
                    "ts": row["ts"],
                    "account_key": row["account_key"],
                    "account_label": row["account_label"],
                    "ok": bool(row["ok"]),
                    "status_code": row["status_code"],
                    "latency_ms": row["latency_ms"],
                    "reason": row["reason"],
                    "model": row["model"],
                }
                for row in rows
            ],
        }

    def call_summary(self) -> Dict[str, Any]:
        row = self._query(
            "SELECT COUNT(*) AS total, SUM(ok) AS ok, AVG(latency_ms) AS avg_latency,"
            " MAX(ts) AS last_ts FROM calls"
        )[0]
        total = int(row["total"] or 0)
        ok = int(row["ok"] or 0)
        return {
            "total": total,
            "ok": ok,
            "failed": total - ok,
            "success_rate": round(ok / total, 4) if total else None,
            "avg_latency_ms": round(row["avg_latency"], 1) if row["avg_latency"] else None,
            "last_ts": row["last_ts"],
        }

    # ---------- quota ----------
    def save_quota(self, account_key: str, payload: Dict[str, Any]) -> None:
        if not account_key:
            return
        self._execute(
            "INSERT INTO quota_cache (account_key, payload, fetched_at) VALUES (?,?,?)"
            " ON CONFLICT(account_key) DO UPDATE SET payload=excluded.payload,"
            " fetched_at=excluded.fetched_at",
            (account_key, json.dumps(payload, ensure_ascii=False), time.time()),
        )

    def load_quota_all(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in self._query("SELECT account_key, payload, fetched_at FROM quota_cache"):
            try:
                payload = json.loads(row["payload"])
            except Exception:  # noqa: BLE001
                continue
            payload["fetched_at"] = row["fetched_at"]
            out[row["account_key"]] = payload
        return out

    # ---------- pool metrics ----------
    def save_metrics(self, account_key: str, payload: Dict[str, Any]) -> None:
        if not account_key:
            return
        self._execute(
            "INSERT INTO pool_metrics (account_key, payload, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(account_key) DO UPDATE SET payload=excluded.payload,"
            " updated_at=excluded.updated_at",
            (account_key, json.dumps(payload, ensure_ascii=False), time.time()),
        )

    def save_metrics_bulk(self, payloads: Dict[str, Dict[str, Any]]) -> None:
        if not payloads:
            return
        now = time.time()
        rows = [
            (key, json.dumps(value, ensure_ascii=False), now)
            for key, value in payloads.items()
            if key
        ]
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN")
            try:
                conn.executemany(
                    "INSERT INTO pool_metrics (account_key, payload, updated_at) VALUES (?,?,?)"
                    " ON CONFLICT(account_key) DO UPDATE SET payload=excluded.payload,"
                    " updated_at=excluded.updated_at",
                    rows,
                )
                conn.execute("COMMIT")
            except Exception:  # noqa: BLE001
                conn.execute("ROLLBACK")
                raise

    def load_metrics_all(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in self._query("SELECT account_key, payload FROM pool_metrics"):
            try:
                out[row["account_key"]] = json.loads(row["payload"])
            except Exception:  # noqa: BLE001
                continue
        return out

    # ---------- pool samples（折线图） ----------
    def add_sample(
        self,
        *,
        available: int,
        inflight: int,
        success_rate: Optional[float],
        average_latency: Optional[float],
        total_successes: int,
        total_failures: int,
        ts: Optional[float] = None,
    ) -> None:
        self._execute(
            "INSERT INTO pool_samples (ts, available, inflight, success_rate,"
            " average_latency, total_successes, total_failures) VALUES (?,?,?,?,?,?,?)",
            (
                ts if ts is not None else time.time(),
                int(available),
                int(inflight),
                success_rate,
                average_latency,
                int(total_successes),
                int(total_failures),
            ),
        )

    def list_samples(self, limit: int = 48) -> List[Dict[str, Any]]:
        rows = self._query(
            "SELECT * FROM pool_samples ORDER BY ts DESC, id DESC LIMIT ?", (int(limit),)
        )
        items = [
            {
                "at": row["ts"],
                "available": row["available"],
                "inflight": row["inflight"],
                "successRate": row["success_rate"],
                "averageLatency": row["average_latency"],
                "totalSuccesses": row["total_successes"],
                "totalFailures": row["total_failures"],
            }
            for row in rows
        ]
        items.reverse()
        return items

    def prune_samples(self, keep: int = 2000) -> None:
        self._execute(
            "DELETE FROM pool_samples WHERE id NOT IN"
            " (SELECT id FROM pool_samples ORDER BY ts DESC, id DESC LIMIT ?)",
            (int(keep),),
        )

    # ---------- 账号用量（用于估算剩余免费额度）----------
    def account_usage(self, since: Optional[float] = None) -> Dict[str, Dict[str, int]]:
        """每个账号的成功/失败调用次数。

        free 计划的 AI 额度是按**次**计的（实测 24 个模型都是 5 次），
        但存在时间窗口——有账号成功调用 7 次仍然可用，
        说明 5 次是"某个窗口内的配额"，窗口长度还没确定。
        所以这里支持 since 参数，按最近一段时间统计。
        """
        if since:
            rows = self._query(
                "SELECT account_key,"
                " SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok_count,"
                " SUM(CASE WHEN ok THEN 0 ELSE 1 END) AS fail_count,"
                " MAX(ts) AS last_ts"
                " FROM calls WHERE ts >= ? GROUP BY account_key",
                (float(since),),
            )
        else:
            rows = self._query(
                "SELECT account_key,"
                " SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS ok_count,"
                " SUM(CASE WHEN ok THEN 0 ELSE 1 END) AS fail_count,"
                " MAX(ts) AS last_ts"
                " FROM calls GROUP BY account_key"
            )
        return {
            row["account_key"]: {
                "ok": int(row["ok_count"] or 0),
                "failed": int(row["fail_count"] or 0),
                "last_ts": float(row["last_ts"] or 0),
            }
            for row in rows
        }

    # ---------- ai probe ----------
    def save_probe(
        self,
        account_key: str,
        *,
        ok: bool,
        chars: int = 0,
        latency_ms: Optional[float] = None,
        error: str = "",
        probed_at: Optional[float] = None,
    ) -> None:
        if not account_key:
            return
        self._execute(
            "INSERT INTO ai_probe (account_key, ok, chars, latency_ms, error, probed_at)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(account_key) DO UPDATE SET ok=excluded.ok, chars=excluded.chars,"
            " latency_ms=excluded.latency_ms, error=excluded.error, probed_at=excluded.probed_at",
            (account_key, 1 if ok else 0, int(chars), latency_ms, (error or "")[:200],
             probed_at if probed_at is not None else time.time()),
        )

    def load_probe_all(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for row in self._query("SELECT * FROM ai_probe"):
            out[row["account_key"]] = {
                "ok": bool(row["ok"]),
                "chars": row["chars"],
                "latency_ms": row["latency_ms"],
                "error": row["error"],
                "probed_at": row["probed_at"],
            }
        return out

    # ---------- 维护 ----------
    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._local.conn = None


_store: Optional[Store] = None
_store_lock = threading.Lock()


def get_store(path: str | Path | None = None) -> Store:
    global _store
    with _store_lock:
        if _store is None:
            _store = Store(path)
        return _store
