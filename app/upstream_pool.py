# -*- coding: utf-8 -*-
"""upstream_pool.py — 管理上游 sidecar 子进程池。

每个子进程一次只跑一个上游流；父进程只做 JSON 转发与解析。子进程被
curl_cffi 的 native 崩溃带走时，父进程读到 EOF → 该请求按「可重试」失败，
同时后台把子进程补回来，网关自身不受影响。

开关：UPSTREAM_SIDECAR=1 启用；UPSTREAM_SIDECAR_WORKERS 默认 2。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from app.logger import logger

ROOT = Path(__file__).resolve().parent.parent


def sidecar_enabled() -> bool:
    return os.getenv("UPSTREAM_SIDECAR", "0").strip() in {"1", "true", "True"}


def _workers() -> int:
    try:
        return max(1, min(8, int(os.getenv("UPSTREAM_SIDECAR_WORKERS", "2"))))
    except ValueError:
        return 2


class _Worker:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.busy = False

    def start(self) -> None:
        # 全新克隆里没有 logs/ 目录，先建好再打开；失败就退回 DEVNULL。
        try:
            log_dir = ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stderr_target: Any = open(log_dir / "sidecar.err", "ab")
        except OSError:
            stderr_target = subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "app.upstream_sidecar"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_target,
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        self.proc = None


class UpstreamPool:
    def __init__(self, workers: int | None = None) -> None:
        self._workers = [_Worker() for _ in range(workers or _workers())]
        self._cond = threading.Condition()
        self._closed = False
        for worker in self._workers:
            worker.start()
        # 守护线程：子进程被 native 崩溃带走后，几秒内补回来，
        # 不依赖「下一次请求」去触发修复。
        threading.Thread(target=self._janitor, name="sidecar-janitor", daemon=True).start()

    def _janitor(self) -> None:
        while True:
            time.sleep(5)
            with self._cond:
                if self._closed:
                    return
                for worker in self._workers:
                    if not worker.busy and not worker.alive():
                        try:
                            worker.start()
                        except Exception:  # noqa: BLE001
                            pass
                self._cond.notify_all()

    # ---- 生命周期 ----
    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        for worker in self._workers:
            worker.stop()

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "workers": len(self._workers),
            "alive": sum(1 for worker in self._workers if worker.alive()),
            "busy": sum(1 for worker in self._workers if worker.busy),
        }

    # ---- 调度 ----
    def _acquire(self) -> _Worker:
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("upstream pool closed")
                for worker in self._workers:
                    if not worker.busy and worker.alive():
                        worker.busy = True
                        return worker
                # 先把死掉的 worker 补回来（不占用本次请求）
                for worker in self._workers:
                    if not worker.busy and not worker.alive():
                        try:
                            worker.start()
                        except Exception:  # noqa: BLE001
                            pass
                for worker in self._workers:
                    if not worker.busy and worker.alive():
                        worker.busy = True
                        return worker
                self._cond.wait(timeout=0.5)

    def _release(self, worker: _Worker, *, usable: bool) -> None:
        with self._cond:
            worker.busy = False
            if not usable and worker.alive():
                worker.stop()
            if not worker.alive():
                try:
                    worker.start()
                except Exception:  # noqa: BLE001
                    logger.error(
                        "upstream sidecar respawn failed",
                        extra={"request_info": {"event": "sidecar_respawn_failed"}},
                    )
            self._cond.notify()

    @staticmethod
    def _account_of(client: Any) -> Dict[str, Any]:
        return {
            "token_v2": getattr(client, "token_v2", ""),
            "space_id": getattr(client, "space_id", ""),
            "user_id": getattr(client, "user_id", ""),
            "space_view_id": getattr(client, "space_view_id", ""),
            "user_name": getattr(client, "user_name", "user"),
            "user_email": getattr(client, "user_email", ""),
            "cookies": dict(getattr(client, "cookies", {}) or {}),
        }

    def stream(
        self, client: Any, transcript: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """把一次上游流跑在子进程里，逐个 yield 解析后的条目。"""
        from app.notion_client import NotionUpstreamError

        worker = self._acquire()
        usable = True
        try:
            job = json.dumps(
                {"account": self._account_of(client), "transcript": transcript},
                ensure_ascii=False,
            ).encode("utf-8") + b"\n"
            assert worker.proc and worker.proc.stdin and worker.proc.stdout
            worker.proc.stdin.write(job)
            worker.proc.stdin.flush()

            while True:
                raw = worker.proc.stdout.readline()
                if not raw:
                    # 子进程死了（大概率是 curl_cffi 的 native 崩溃）
                    usable = False
                    logger.warning(
                        "upstream sidecar exited mid-stream",
                        extra={"request_info": {"event": "sidecar_crashed"}},
                    )
                    raise NotionUpstreamError(
                        "upstream sidecar crashed", retriable=True
                    )
                try:
                    message = json.loads(raw.decode("utf-8", "replace"))
                except ValueError:
                    continue
                kind = message.get("t")
                if kind == "item":
                    item = message.get("v")
                    if isinstance(item, dict):
                        yield item
                    continue
                if kind == "done":
                    return
                if kind == "error":
                    raise NotionUpstreamError(
                        str(message.get("v") or "sidecar error")[:300], retriable=True
                    )
        finally:
            self._release(worker, usable=usable)


_pool: Optional[UpstreamPool] = None
_lock = threading.Lock()


def get_pool() -> Optional[UpstreamPool]:
    """侧车池单例；未启用时返回 None（调用方回退到进程内直连）。"""
    global _pool
    if not sidecar_enabled():
        return None
    with _lock:
        if _pool is None:
            _pool = UpstreamPool()
            logger.info(
                "Upstream sidecar pool started",
                extra={"request_info": {"event": "sidecar_started", **_pool.stats()}},
            )
    return _pool


def shutdown() -> None:
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close()
            _pool = None
