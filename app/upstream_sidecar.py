# -*- coding: utf-8 -*-
"""upstream_sidecar.py — 在独立进程里跑 Notion 上游调用。

协议（stdin/stdout，UTF-8，一行一个 JSON）：
    父 -> 子：{"account": {...}, "transcript": [...]}
    子 -> 父：{"t": "item", "v": {...}} / {"t": "done"} / {"t": "error", "v": "..."}

SIDECAR_DEBUG=1 时把诊断信息写到 stderr（父进程重定向到 logs/sidecar.err）。
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


def _write(payload: Dict[str, Any]) -> None:
    # 必须走 buffer + 显式 UTF-8：Windows 的 sys.stdout 默认 cp936，
    # 中文会变乱码，上游拿到的就是坏 prompt。
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    sys.stdout.buffer.write(line.encode("utf-8"))
    sys.stdout.buffer.flush()


def _debug(message: str) -> None:
    if os.getenv("SIDECAR_DEBUG"):
        sys.stderr.write("[sidecar] " + message + "\n")
        sys.stderr.flush()


def main() -> int:
    from app.notion_client import NotionOpusAPI

    while True:
        raw = sys.stdin.buffer.readline()
        if not raw:
            break
        text = raw.decode("utf-8", "replace").strip()
        if not text:
            continue
        try:
            job = json.loads(text)
        except ValueError:
            continue
        if job.get("op") == "quit":
            break

        account = job.get("account") or {}
        transcript = job.get("transcript") or []

        # 诊断用：让子进程自己用同样的账号/提示词重建一份 transcript，
        # 用来区分「payload 问题」还是「子进程环境问题」。
        if job.get("selftest"):
            from app.conversation import build_standard_transcript

            transcript = build_standard_transcript(
                [{"role": "user", "content": "只回复：OK"}],
                "claude-opus4.6",
                {"user_id": account.get("user_id", ""), "space_id": account.get("space_id", "")},
            )

        try:
            client = NotionOpusAPI(account)
            _debug(
                "start blocks=" + str(len(transcript)) +
                " first=" + str((transcript[0].get("type") if transcript else "none"))
            )
            for item in client.stream_response(transcript, thread_id=None):
                if isinstance(item, dict):
                    _write({"t": "item", "v": item})
            _write({"t": "done"})
        except BaseException as exc:  # noqa: BLE001
            _debug("error " + type(exc).__name__ + ": " + str(exc)[:200])
            _write({"t": "error", "v": f"{type(exc).__name__}: {exc}"[:400]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
