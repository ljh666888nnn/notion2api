# -*- coding: utf-8 -*-
"""mcp_server.py — 把本机能力以 MCP 协议暴露出去。

设计要点：工具由**网关本地执行**，不经过 Notion 上游。
这样 MCP 是否可用与"Notion 支不支持 function calling"彻底解耦——
上游只负责聊天，工具调用由客户端直连 `/mcp` 走 JSON-RPC。

默认安全边界：
  - 文件工具只能在 MCP_ROOT（默认仓库根）内活动，越界直接拒绝
  - 写文件要 MCP_ALLOW_WRITE=1
  - 执行命令要 MCP_ALLOW_EXEC=1（默认关）
  - 设了 API_KEY 就要求 Bearer / X-API-Key / ?key=

环境变量：
  MCP_ENABLED=1        是否挂载 /mcp（默认 1）
  MCP_ROOT=<path>      沙箱根目录（默认仓库根）
  MCP_ALLOW_WRITE=0    允许 write_file（默认 0）
  MCP_ALLOW_EXEC=0     允许 run_command（默认 0）
  MCP_ALLOW_OUTSIDE=0  允许访问 MCP_ROOT 之外的路径（默认 0）
  MCP_MAX_READ_BYTES   单次读取上限（默认 200000）
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.config import API_KEY

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "notion2api-mcp"
SERVER_VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_OUTPUT_CHARS = 20000
MAX_LIST_ENTRIES = 500
MAX_SEARCH_RESULTS = 100
MAX_SEARCH_FILES = 5000

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    ".edge-profile",
    ".build-tmp",
    "dist",
    "build",
}


class ToolError(Exception):
    """工具执行失败（返回给客户端时标 isError）。"""


# --------------------------------------------------------------------------- #
# 配置读取（每次调用读环境，方便测试与热改）
# --------------------------------------------------------------------------- #
def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _root() -> Path:
    raw = os.getenv("MCP_ROOT", "").strip()
    return Path(raw).expanduser().resolve() if raw else PROJECT_ROOT


def _max_read_bytes() -> int:
    try:
        return max(1024, int(os.getenv("MCP_MAX_READ_BYTES", "200000")))
    except ValueError:
        return 200000


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [已截断，共 {len(text)} 字符]"


# --------------------------------------------------------------------------- #
# 路径沙箱
# --------------------------------------------------------------------------- #
def _resolve(raw: Any) -> Path:
    text = str(raw or ".").strip() or "."
    root = _root()
    candidate = Path(text).expanduser()
    try:
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    except OSError as exc:
        raise ToolError(f"路径无法解析：{text}（{exc}）") from exc
    if not _flag("MCP_ALLOW_OUTSIDE"):
        try:
            inside = target.is_relative_to(root)
        except AttributeError:  # pragma: no cover - Python < 3.9
            inside = str(target).startswith(str(root))
        if not inside:
            raise ToolError(f"路径越界：{target} 不在沙箱 {root} 之内")
    return target


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_root()).as_posix()
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #
def tool_list_dir(args: Dict[str, Any]) -> str:
    target = _resolve(args.get("path", "."))
    if not target.exists():
        raise ToolError(f"目录不存在：{_rel(target)}")
    if not target.is_dir():
        raise ToolError(f"不是目录：{_rel(target)}")

    entries = sorted(
        target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
    )
    lines: List[str] = []
    for entry in entries[:MAX_LIST_ENTRIES]:
        try:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}\t{entry.stat().st_size} B")
        except OSError:
            lines.append(f"{entry.name}\t<unreadable>")
    if len(entries) > MAX_LIST_ENTRIES:
        lines.append(f"... 还有 {len(entries) - MAX_LIST_ENTRIES} 项未列出")
    if not lines:
        return f"{_rel(target)} 是空目录"
    return "\n".join(lines)


def tool_read_file(args: Dict[str, Any]) -> str:
    target = _resolve(args.get("path", ""))
    if not target.exists():
        raise ToolError(f"文件不存在：{_rel(target)}")
    if not target.is_file():
        raise ToolError(f"不是文件：{_rel(target)}")

    limit = _max_read_bytes()
    try:
        with target.open("rb") as fh:
            raw = fh.read(limit + 1)
    except OSError as exc:
        raise ToolError(f"读取失败：{exc}") from exc

    truncated = len(raw) > limit
    if truncated:
        raw = raw[:limit]
    text = raw.decode("utf-8", errors="replace")

    offset = max(1, int(args.get("offset") or 1))
    line_limit = max(1, int(args.get("limit") or 400))
    lines = text.splitlines()
    window = lines[offset - 1 : offset - 1 + line_limit]
    body = "\n".join(window)

    header = f"{_rel(target)}（第 {offset}-{offset + len(window) - 1} 行，共 {len(lines)} 行）"
    if truncated:
        header += f"\n[注意] 文件超过 {limit} 字节，只读了开头部分"
    if len(lines) > offset - 1 + len(window):
        header += f"\n[注意] 还有 {len(lines) - (offset - 1 + len(window))} 行未返回，用 offset 继续"
    return f"{header}\n\n{body}" if body else f"{header}\n\n（空）"


def tool_search_files(args: Dict[str, Any]) -> str:
    pattern = str(args.get("pattern") or "").strip()
    if not pattern:
        raise ToolError("pattern 不能为空")
    flags = 0 if args.get("case_sensitive") else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"正则不合法：{exc}") from exc

    base = _resolve(args.get("path", "."))
    if not base.exists():
        raise ToolError(f"路径不存在：{_rel(base)}")

    include = str(args.get("glob") or "*")
    max_results = int(args.get("max_results") or MAX_SEARCH_RESULTS)
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    limit_bytes = _max_read_bytes()

    hits: List[str] = []
    scanned = 0
    stopped = False

    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not fnmatch.fnmatch(filename, include):
                continue
            file_path = Path(dirpath) / filename
            scanned += 1
            if scanned > MAX_SEARCH_FILES:
                stopped = True
                break
            try:
                if file_path.stat().st_size > limit_bytes:
                    continue
                with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            hits.append(f"{_rel(file_path)}:{lineno}: {line.strip()[:200]}")
                            if len(hits) >= max_results:
                                stopped = True
                                break
            except OSError:
                continue
            if stopped:
                break
        if stopped:
            break

    if not hits:
        return f"在 {_rel(base)} 下未匹配到 /{pattern}/（扫描 {scanned} 个文件）"
    note = ""
    if stopped:
        note = f"\n[注意] 已达上限（结果 {max_results} 条 / 文件 {MAX_SEARCH_FILES} 个），可能还有更多"
    return "\n".join(hits) + note


def tool_project_map(args: Dict[str, Any]) -> str:
    """带深度限制的目录树，比 list_dir 更适合让 agent 建立全局印象。"""
    base = _resolve(args.get("path", "."))
    if not base.is_dir():
        raise ToolError(f"不是目录：{_rel(base)}")

    depth = max(1, min(int(args.get("depth") or 2), 6))
    max_entries = max(1, min(int(args.get("max_entries") or 200), 1000))

    lines: List[str] = [_rel(base).rstrip("/") + "/"]
    emitted = 0
    truncated = False

    for dirpath, dirnames, filenames in os.walk(base):
        rel = os.path.relpath(dirpath, base)
        level = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
        )
        if level >= depth:
            dirnames[:] = []
        if rel != ".":
            lines.append("  " * level + os.path.basename(dirpath) + "/")
            emitted += 1
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            lines.append("  " * (level + 1) + name)
            emitted += 1
            if emitted >= max_entries:
                truncated = True
                break
        if truncated:
            break

    if truncated:
        lines.append(f"...[已截断，上限 {max_entries} 项]")
    return "\n".join(lines)


def _git(base: Path, argv: List[str], timeout: int = 20) -> str:
    """只读 git：固定参数、shell=False，路径已过沙箱校验。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(base), *argv],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError:
        raise ToolError("未找到 git（PATH 里没有）") from None
    except subprocess.TimeoutExpired:
        raise ToolError(f"git 超时（{timeout}s）") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if "not a git repository" in detail.lower():
            raise ToolError(f"不在 git 仓库内：{base}")
        raise ToolError(f"git {argv[0]} 失败（exit={proc.returncode}）：{detail[:600]}")
    return proc.stdout


def tool_git_status(args: Dict[str, Any]) -> str:
    base = _resolve(args.get("path", "."))
    output = _git(base, ["status", "--porcelain", "-b"]).strip()
    return _truncate(output) if output else "工作区干净（无改动）"


def tool_git_diff(args: Dict[str, Any]) -> str:
    base = _resolve(args.get("path", "."))
    argv: List[str] = ["diff", "--no-color"]
    if args.get("staged"):
        argv.append("--cached")
    if args.get("stat"):
        argv.append("--stat")
    output = _git(base, argv).strip()
    return _truncate(output) if output else "无差异"


def tool_write_file(args: Dict[str, Any]) -> str:
    if not _flag("MCP_ALLOW_WRITE"):
        raise ToolError("写文件已禁用：设置 MCP_ALLOW_WRITE=1 后重启服务")
    target = _resolve(args.get("path", ""))
    content = args.get("content")
    if content is None:
        raise ToolError("content 不能为空")
    target.parent.mkdir(parents=True, exist_ok=True)
    data = str(content)
    try:
        target.write_text(data, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"写入失败：{exc}") from exc
    return f"已写入 {_rel(target)}（{len(data)} 字符）"


def tool_run_command(args: Dict[str, Any]) -> str:
    if not _flag("MCP_ALLOW_EXEC"):
        raise ToolError("执行命令已禁用：设置 MCP_ALLOW_EXEC=1 后重启服务")
    command = str(args.get("command") or "").strip()
    if not command:
        raise ToolError("command 不能为空")
    timeout = int(args.get("timeout_seconds") or 30)
    timeout = max(1, min(timeout, 300))
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(_root()),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"命令超时（{timeout}s）：{command}") from None
    except OSError as exc:
        raise ToolError(f"命令启动失败：{exc}") from exc

    parts = [f"$ {command}", f"exit={proc.returncode}"]
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
    return _truncate("\n".join(parts))


_HANDLERS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "project_map": tool_project_map,
    "list_dir": tool_list_dir,
    "read_file": tool_read_file,
    "search_files": tool_search_files,
    "git_status": tool_git_status,
    "git_diff": tool_git_diff,
    "write_file": tool_write_file,
    "run_command": tool_run_command,
}

# 只读、默认暴露
_DEFAULT_TOOLS = (
    "project_map",
    "list_dir",
    "read_file",
    "search_files",
    "git_status",
    "git_diff",
)

_OPTIONAL = {"write_file": "MCP_ALLOW_WRITE", "run_command": "MCP_ALLOW_EXEC"}


# --------------------------------------------------------------------------- #
# MCP 工具清单
# --------------------------------------------------------------------------- #
def _schema(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_TOOL_DEFS: Dict[str, Dict[str, Any]] = {
    "list_dir": {
        "name": "list_dir",
        "description": "列出目录内容。返回子目录（带 / 结尾）与文件及其字节数。",
        "inputSchema": _schema(
            {"path": {"type": "string", "description": "相对沙箱根的目录路径，默认 ."}},
            [],
        ),
    },
    "read_file": {
        "name": "read_file",
        "description": "按行读取文本文件。默认返回第 1 条起的 400 行。",
        "inputSchema": _schema(
            {
                "path": {"type": "string", "description": "相对沙箱根的文件路径"},
                "offset": {"type": "integer", "description": "起始行号，从 1 开始"},
                "limit": {"type": "integer", "description": "最多返回多少行"},
            },
            ["path"],
        ),
    },
    "search_files": {
        "name": "search_files",
        "description": "在目录下按正则搜索文件内容，返回 路径:行号: 内容。",
        "inputSchema": _schema(
            {
                "pattern": {"type": "string", "description": "正则表达式"},
                "path": {"type": "string", "description": "搜索起点，默认 ."},
                "glob": {"type": "string", "description": "文件名过滤，如 *.py"},
                "max_results": {"type": "integer", "description": "最多返回条数，上限 100"},
                "case_sensitive": {"type": "boolean", "description": "是否区分大小写"},
            },
            ["pattern"],
        ),
    },
    "project_map": {
        "name": "project_map",
        "description": "输出带缩进的目录树（自动跳过 node_modules/.git/.venv 等）。",
        "inputSchema": _schema(
            {
                "path": {"type": "string", "description": "起点目录，默认 ."},
                "depth": {"type": "integer", "description": "深度 1-6，默认 2"},
                "max_entries": {"type": "integer", "description": "条目上限，默认 200"},
            },
            [],
        ),
    },
    "git_status": {
        "name": "git_status",
        "description": "只读：git status --porcelain -b。",
        "inputSchema": _schema(
            {"path": {"type": "string", "description": "仓库内路径，默认 ."}},
            [],
        ),
    },
    "git_diff": {
        "name": "git_diff",
        "description": "只读：git diff（可选 --cached / --stat）。",
        "inputSchema": _schema(
            {
                "path": {"type": "string", "description": "仓库内路径，默认 ."},
                "staged": {"type": "boolean", "description": "看暂存区（--cached）"},
                "stat": {"type": "boolean", "description": "只看统计（--stat）"},
            },
            [],
        ),
    },
    "write_file": {
        "name": "write_file",
        "description": "写入（覆盖）文本文件。需服务端开启 MCP_ALLOW_WRITE。",
        "inputSchema": _schema(
            {
                "path": {"type": "string", "description": "相对沙箱根的文件路径"},
                "content": {"type": "string", "description": "文件内容"},
            },
            ["path", "content"],
        ),
    },
    "run_command": {
        "name": "run_command",
        "description": "在沙箱根执行 shell 命令并返回输出。需服务端开启 MCP_ALLOW_EXEC。",
        "inputSchema": _schema(
            {
                "command": {"type": "string", "description": "要执行的命令"},
                "timeout_seconds": {"type": "integer", "description": "超时秒数，上限 300"},
            },
            ["command"],
        ),
    },
}


def visible_tools() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, gate in _OPTIONAL.items():
        if not _flag(gate):
            continue
        out.append(_TOOL_DEFS[name])
    for name in _DEFAULT_TOOLS:
        out.append(_TOOL_DEFS[name])
    return out


# --------------------------------------------------------------------------- #
# JSON-RPC
# --------------------------------------------------------------------------- #
def _ok(msg_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _err(msg_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": code, "message": message},
    }
    if data is not None:
        payload["error"]["data"] = data
    return payload


def _text_result(text: str, is_error: bool = False) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def handle_message(message: Any) -> Optional[Dict[str, Any]]:
    """处理一条 JSON-RPC 消息；通知类返回 None。"""
    if not isinstance(message, dict):
        return _err(None, -32600, "Invalid Request")

    msg_id = message.get("id")
    method = message.get("method")
    if not isinstance(method, str):
        return _err(msg_id, -32600, "Invalid Request")

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        return _ok(
            msg_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            },
        )

    if method == "ping":
        return _ok(msg_id, {})

    if method == "tools/list":
        return _ok(msg_id, {"tools": visible_tools()})

    if method == "tools/call":
        params = message.get("params")
        if not isinstance(params, dict):
            return _err(msg_id, -32602, "params must be an object")
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            return _err(msg_id, -32602, "params.name 缺失")
        if not isinstance(args, dict):
            return _err(msg_id, -32602, "params.arguments 必须是对象")

        handler = _HANDLERS.get(name)
        if handler is None:
            return _err(msg_id, -32602, f"未知工具：{name}")
        gate = _OPTIONAL.get(name)
        if gate and not _flag(gate):
            return _err(msg_id, -32602, f"未知工具：{name}")

        try:
            text = handler(args)
        except ToolError as exc:
            return _ok(msg_id, _text_result(str(exc), is_error=True))
        except Exception as exc:  # noqa: BLE001
            return _ok(
                msg_id, _text_result(f"{type(exc).__name__}: {exc}", is_error=True)
            )
        return _ok(msg_id, _text_result(text))

    return _err(msg_id, -32601, f"Method not found: {method}")


# --------------------------------------------------------------------------- #
# HTTP 传输
# --------------------------------------------------------------------------- #
router = APIRouter()


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return True
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        if header.split(" ", 1)[1].strip() == API_KEY:
            return True
    if request.headers.get("x-api-key") == API_KEY:
        return True
    if request.query_params.get("key") == API_KEY:
        return True
    return False


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    if not _authorized(request):
        return JSONResponse(_err(None, -32001, "Unauthorized"), status_code=401)

    raw = await request.body()
    if not raw:
        return JSONResponse(_err(None, -32700, "Parse error: empty body"), status_code=400)
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse(_err(None, -32700, "Parse error"), status_code=400)

    if isinstance(payload, list):
        if not payload:
            return JSONResponse(_err(None, -32600, "Invalid Request: empty batch"), status_code=400)
        responses = [r for r in (handle_message(m) for m in payload) if r is not None]
        if not responses:
            return Response(status_code=202)
        return JSONResponse(responses)

    result = handle_message(payload)
    if result is None:
        # 通知（notifications/initialized 等）不需要响应体
        return Response(status_code=202)
    return JSONResponse(result)


@router.get("/mcp")
async def mcp_get_not_supported(request: Request):
    if not _authorized(request):
        return JSONResponse(_err(None, -32001, "Unauthorized"), status_code=401)
    return JSONResponse(
        _err(
            None,
            -32600,
            "This server uses Streamable HTTP (POST /mcp); GET SSE stream is not provided.",
        ),
        status_code=405,
        headers={"Allow": "POST"},
    )


def mcp_enabled() -> bool:
    return _flag("MCP_ENABLED", "1")
