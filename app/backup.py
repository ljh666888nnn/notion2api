# -*- coding: utf-8 -*-
"""backup.py - 账号与数据库的备份 / 恢复。

437 个账号 + SQLite 里的调用记录、额度缓存、代理池，
这些一旦丢了没法重建（账号要重新注册，还受 IP 限制）。
所以备份是必需项，不是锦上添花。

打包内容：
    accounts.json
    state/*.db（WAL 模式要一起带上 -wal / -shm，或者先 checkpoint）
    state/pool_metrics.json（如果还在用）

策略：
    - 手动：POST /v1/admin/backup
    - 自动：每 BACKUP_INTERVAL 秒一次，保留最近 BACKUP_KEEP 份
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.logger import logger

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "backups"


def _db_paths() -> List[Path]:
    state = ROOT / "state"
    if not state.exists():
        return []
    return sorted(state.glob("*.db"))


def _checkpoint(path: Path) -> None:
    """把 WAL 落盘，保证拷出来的是完整快照。"""
    try:
        conn = sqlite3.connect(str(path), timeout=10)
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        pass


def create_backup(target_dir: Optional[Path] = None, tag: str = "") -> Dict[str, Any]:
    """打一个 zip 快照，返回路径与内容清单。"""
    out_dir = Path(target_dir) if target_dir else DEFAULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"-{tag}" if tag else ""
    archive = out_dir / f"notion2api-{stamp}{suffix}.zip"

    included: List[Dict[str, Any]] = []
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        accounts = ROOT / "accounts.json"
        if accounts.exists():
            bundle.write(accounts, "accounts.json")
            included.append({"name": "accounts.json",
                             "bytes": accounts.stat().st_size})

        for db in _db_paths():
            _checkpoint(db)
            bundle.write(db, f"state/{db.name}")
            included.append({"name": f"state/{db.name}",
                             "bytes": db.stat().st_size})

        metrics = ROOT / "state" / "pool_metrics.json"
        if metrics.exists():
            bundle.write(metrics, "state/pool_metrics.json")
            included.append({"name": "state/pool_metrics.json",
                             "bytes": metrics.stat().st_size})

        # 附一份清单，方便日后核对
        manifest = {
            "created_at": time.time(),
            "created_at_text": stamp,
            "files": included,
        }
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    return {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "files": included,
        "created_at": time.time(),
    }


def list_backups(target_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    out_dir = Path(target_dir) if target_dir else DEFAULT_DIR
    if not out_dir.exists():
        return []
    items: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("notion2api-*.zip"), reverse=True):
        stat = path.stat()
        items.append({
            "path": str(path),
            "name": path.name,
            "bytes": stat.st_size,
            "created_at": stat.st_mtime,
        })
    return items


def prune_backups(keep: int = 24, target_dir: Optional[Path] = None) -> int:
    items = list_backups(target_dir)
    removed = 0
    for item in items[keep:]:
        try:
            Path(item["path"]).unlink()
            removed += 1
        except Exception:  # noqa: BLE001
            continue
    return removed


def restore_backup(archive: str, dry_run: bool = False) -> Dict[str, Any]:
    """从 zip 恢复。先把现有文件挪到 .before-restore 再覆盖。"""
    src = Path(archive)
    if not src.exists():
        return {"restored": False, "error": "备份文件不存在"}

    stamp = time.strftime("%Y%m%d-%H%M%S")
    restored: List[str] = []
    with zipfile.ZipFile(src) as bundle:
        names = bundle.namelist()
        for name in names:
            if name == "manifest.json":
                continue
            dest = ROOT / name
            if dry_run:
                restored.append(name)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                shutil.copy2(dest, dest.with_suffix(dest.suffix + f".before-restore-{stamp}"))
            with bundle.open(name) as handle, open(dest, "wb") as out:
                shutil.copyfileobj(handle, out)
            restored.append(name)

    return {"restored": not dry_run, "files": restored, "dry_run": dry_run}


class BackupScheduler:
    """后台定时备份。"""

    def __init__(self, interval: float = 21600.0, keep: int = 24) -> None:
        self.interval = max(300.0, interval)
        self.keep = max(1, keep)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self._running:
                return False
            self._running = True

        def loop() -> None:
            # 启动后先等一会儿，避免和冷启动抢 IO
            time.sleep(60)
            while True:
                with self._lock:
                    if not self._running:
                        return
                try:
                    result = create_backup(tag="auto")
                    prune_backups(self.keep)
                    with self._lock:
                        self._last = {"ok": True, **result}
                    logger.info("Auto backup done", extra={"request_info": {
                        "event": "backup_auto", "path": result["path"],
                        "bytes": result["bytes"]}})
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self._last = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:160]}
                try:
                    time.sleep(self.interval)
                except Exception:  # noqa: BLE001
                    return

        self._thread = threading.Thread(target=loop, daemon=True, name="backup-auto")
        self._thread.start()
        return True

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self._running,
                "interval": self.interval,
                "keep": self.keep,
                "last": dict(self._last),
            }


_scheduler: Optional[BackupScheduler] = None


def get_scheduler() -> BackupScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackupScheduler(
            interval=float(os.getenv("BACKUP_INTERVAL", "21600")),
            keep=int(os.getenv("BACKUP_KEEP", "24")),
        )
    return _scheduler
