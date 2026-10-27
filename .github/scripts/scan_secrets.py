#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Secret gate for CI: scan every tracked file with a fixed, reviewed rule set.

Runs after `actions/checkout`, so `git ls-files` is the source of truth for
what is actually published. Third-party assets under frontend/vendor are
skipped (KaTeX bundles contain long base64 blobs by design), as is
requirements.lock (pure version pins).

Exit code 1 when anything matches, so the job fails before a leak lands.
"""
from __future__ import annotations

import math
import re
import subprocess
from collections import Counter

SKIP_PREFIXES = ("frontend/vendor/",)
SKIP_FILES = {"requirements.lock"}

PATTERNS = {
    "notion_token": re.compile(r"v03%3A[A-Za-z0-9]{20,}|v03:[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-(?!your)[A-Za-z0-9_\-]{20,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "slack_token": re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "jwt": re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # 只报非示例邮箱：仓库里允许出现 you@example.com 这类占位
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
}

ALLOW_EMAIL = re.compile(r"(example\.(com|org|net)|@local\b|users\.noreply\.github\.com)")
ENTROPY_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")
ENTROPY_THRESHOLD = 4.5


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, encoding="utf-8"
    ).stdout.splitlines()
    return [f for f in out if not f.startswith(SKIP_PREFIXES) and f not in SKIP_FILES]


def entropy(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum(c / length * math.log2(c / length) for c in counts.values())


def main() -> int:
    files = tracked_files()
    if not files:
        print("no tracked files found - refusing to report a clean run")
        return 1

    findings: list[str] = []
    for rel in files:
        try:
            text = open(rel, encoding="utf-8", errors="ignore").read()
        except OSError as exc:
            print(f"skip {rel}: {exc}")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    value = match.group(0)
                    if kind == "email" and ALLOW_EMAIL.search(value):
                        continue
                    findings.append(f"{rel}:{lineno} {kind} {value[:60]}")
            for token in ENTROPY_TOKEN.findall(line):
                if len(set(token)) < 12:
                    continue
                # 配置名（ACCOUNT_MAX_CONCURRENCY 之类）不算密钥：
                # 真密钥几乎总带数字，且不是纯标识符形态。
                if not any(ch.isdigit() for ch in token):
                    continue
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                    continue
                if entropy(token) >= ENTROPY_THRESHOLD:
                    findings.append(f"{rel}:{lineno} high_entropy {token[:40]}")

    if findings:
        for item in findings:
            print(item)
        print(f"\n{len(findings)} potential secret(s) found")
        return 1

    print(f"no secrets detected across {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
