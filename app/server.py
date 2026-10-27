import asyncio
import hmac
import time
import os
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
from app.config import (
    ACCOUNTS,
    ACCOUNT_COOLDOWN_SECONDS,
    ACCOUNT_COOLDOWN_JITTER_RATIO,
    ACCOUNT_MAX_CONCURRENCY,
    ACCOUNT_MAX_COOLDOWN_SECONDS,
    ACCOUNT_POOL_MAX_WAIT_SECONDS,
    ACCOUNT_POOL_STRATEGY,
    API_KEY,
    ALLOWED_ORIGINS,
    is_lite_mode,
    is_standard_mode,
)
from app.account_pool import AccountPool
from app.conversation import ConversationManager
from app.api.chat import router as chat_router
from app.api.accounts import router as accounts_router
from app.api.monitor import router as monitor_router
from app.api.ops import router as ops_router
from app.api.models import router as models_router
from app.logger import logger
from app.limiter import limiter


async def _sample_loop(app) -> None:
    """每 30 秒把账号池快照写入 SQLite，保证折线图历史重启后仍在。"""
    from app.store import get_store

    store = get_store()
    while True:
        try:
            pool = app.state.account_pool
            summary = pool.get_status_summary()
            accounts = pool.snapshot()
            latencies = [
                item["average_latency_ms"]
                for item in accounts
                if item.get("average_latency_ms") is not None
            ]
            successes = int(summary.get("total_successes", 0))
            failures = int(summary.get("total_failures", 0))
            outcomes = successes + failures
            store.add_sample(
                available=int(summary.get("available", 0)),
                inflight=int(summary.get("inflight", 0)),
                success_rate=(successes / outcomes) if outcomes else None,
                average_latency=(sum(latencies) / len(latencies)) if latencies else None,
                total_successes=successes,
                total_failures=failures,
            )
            store.prune_samples(2000)
            # 强制杀进程时 lifespan 的收尾不一定会跑，所以计数必须周期落库
            pool.persist_metrics()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化状态
    app.state.account_pool = AccountPool(
        ACCOUNTS,
        strategy=ACCOUNT_POOL_STRATEGY,
        base_cooldown=ACCOUNT_COOLDOWN_SECONDS,
        max_cooldown=ACCOUNT_MAX_COOLDOWN_SECONDS,
        max_wait=ACCOUNT_POOL_MAX_WAIT_SECONDS,
        default_max_concurrency=ACCOUNT_MAX_CONCURRENCY,
        cooldown_jitter_ratio=ACCOUNT_COOLDOWN_JITTER_RATIO,
        metrics_path=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "state",
            "pool_metrics.json",
        ),
    )

    # 批量注册引擎：注册成功后自动写 accounts.json 并重载账号池
    def _reload_pool_from_disk() -> dict:
        from app.account_store import load_accounts as _load_from_disk
        configs = _load_from_disk()
        return app.state.account_pool.reload(configs)

    from app.call_log import get_call_log

    get_call_log(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs",
            "calls.jsonl",
        )
    )

    # 自动备份：accounts.json + SQLite，这些丢了没法重建
    if os.getenv("BACKUP_AUTO", "1").strip() not in {"0", "false", "False"}:
        try:
            from app.backup import get_scheduler

            get_scheduler().start()
        except Exception:  # noqa: BLE001
            pass

    # 确定运行模式
    if is_lite_mode():
        logger.info("Service starting up in LITE mode", extra={"request_info": {"event": "startup", "accounts": len(ACCOUNTS), "mode": "lite"}})
    elif is_standard_mode():
        logger.info("Service starting up in STANDARD mode", extra={"request_info": {"event": "startup", "accounts": len(ACCOUNTS), "mode": "standard"}})
    else:
        app.state.conversation_manager = ConversationManager()
        logger.info("Service starting up in HEAVY mode", extra={"request_info": {"event": "startup", "accounts": len(ACCOUNTS), "mode": "heavy"}})

    app.state.start_time = time.time()
    sampler = asyncio.create_task(_sample_loop(app))
    yield
    sampler.cancel()
    try:
        await sampler
    except (asyncio.CancelledError, Exception):
        pass
    try:
        app.state.account_pool.persist_metrics()
    except Exception:  # noqa: BLE001
        pass
    # 关闭时清理
    logger.info("Service shutting down", extra={"request_info": {"event": "shutdown"}})

app = FastAPI(
    title="Notion Opus API",
    description="A FastAPI wrapper providing an OpenAI-compatible interface for Notion's Claude Opus backend.",
    version="1.0.0",
    lifespan=lifespan
)

# 允许跨域（配合本地前端）
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注入 Limiter
app.state.limiter = limiter

# 自定义 429 速率限制响应
def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests, please try again later"}
    )
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled application exception",
        exc_info=True,
        extra={
            "request_info": {
                "event": "unhandled_exception",
                "method": request.method,
                "path": request.url.path,
            }
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
            }
        },
    )

# 静态页面禁用缓存：前端改完刷新即生效（no-store 连启发式缓存也不留）
@app.middleware("http")
async def no_cache_html_middleware(request: Request, call_next):
    response = await call_next(request)
    ctype = response.headers.get("content-type", "")
    if "text/html" in ctype:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# 结构化日志中间件
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    
    # 跳过高频且不重要的日志打印，避免刷屏
    skip_logging = request.url.path in ["/health", "/favicon.ico"]
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise
    finally:
        process_time = time.time() - start_time
        client_ip = request.client.host if request.client else "unknown"
        

        if not skip_logging:
            log_level = logger.error if status_code >= 400 else logger.info
            log_level(
                "Request processed",
                extra={
                    "request_info": {
                        "method": request.method,
                        "path": request.url.path,
                        "ip": client_ip,
                        "status_code": status_code,
                        "duration_ms": round(process_time * 1000, 2)
                    }
                }
            )
            
    return response

# 简易 API Key 鉴权中间件
@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    # 如果环境配置中未设置 API_KEY，则全局不验证
    if API_KEY:
        # 跳过 OPTIONS 请求和非受保护的静态路由（如果以后有的话）
        if request.url.path.startswith("/v1") and request.method != "OPTIONS":
            auth_header = request.headers.get("Authorization", "")
            token = auth_header[7:] if auth_header.startswith("Bearer ") else ""
            if not token:
                token = (request.headers.get("x-api-key") or "").strip()
            if not token or not hmac.compare_digest(token, API_KEY):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Error: API KEY doesn't match.",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key"
                        }
                    }
                )
    return await call_next(request)


@app.api_route("/v1", methods=["GET", "POST", "HEAD"])
@app.api_route("/v1/", methods=["GET", "POST", "HEAD"])
async def v1_index(request: Request):
    """基址探测：告诉调用方鉴权已通过、以及该用哪些端点。"""
    return {
        "object": "list",
        "data": [],
        "auth": "ok",
        "endpoints": [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/responses",
        ],
        "note": "OpenAI 兼容网关。请把 base_url 设为 https://<域名>/v1，"
                "或直接 POST /v1/chat/completions。",
    }


# 挂载路由，前缀统一为 /v1
app.include_router(chat_router, prefix="/v1")
# Anthropic 协议兼容：POST /v1/messages（Claude SDK / Claude Code 用这套）
try:
    from app.api.anthropic_compat import router as anthropic_router
    app.include_router(anthropic_router, prefix="/v1")
except Exception as _anthropic_exc:  # noqa: BLE001
    logger.warning("Anthropic compat layer unavailable: %s", _anthropic_exc)
# OpenAI Responses API 兼容：POST /v1/responses（新版客户端与指纹工具走这套）
try:
    from app.api.responses_compat import router as responses_router
    app.include_router(responses_router, prefix="/v1")
except Exception as _responses_exc:  # noqa: BLE001
    logger.warning("Responses compat layer unavailable: %s", _responses_exc)
app.include_router(models_router, prefix="/v1")
app.include_router(accounts_router, prefix="/v1")
app.include_router(monitor_router, prefix="/v1")
app.include_router(ops_router, prefix="/v1")

# MCP：工具由网关本地执行，不经过 Notion 上游。
# 默认挂载在 /mcp（不在 /v1 下，避免和 OpenAI 兼容面混在一起）。
try:
    from app.mcp_server import mcp_enabled
    from app.mcp_server import router as mcp_router

    if mcp_enabled():
        app.include_router(mcp_router)
        logger.info(
            "MCP endpoint mounted at /mcp",
            extra={"request_info": {"event": "mcp_mounted", "path": "/mcp"}},
        )
except Exception:  # noqa: BLE001
    logger.warning("MCP endpoint not mounted", exc_info=True)

# 挂载健康检查
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon", status_code=204)

_HEALTH_CACHE: Dict[str, Any] = {"ts": 0.0, "value": None}


@app.get("/health", tags=["system"])
def health_check(request: Request):
    # 992 个账号的 snapshot 要几百毫秒到几秒；watchdog 每分钟都探，
    # 缓存 2 秒既省 CPU，也避免健康检查自己成为负载。
    cached = _HEALTH_CACHE["value"]
    if cached is not None and time.time() - _HEALTH_CACHE["ts"] < 2.0:
        return cached
    uptime = time.time() - request.app.state.start_time
    pool = request.app.state.account_pool
    status = pool.get_status_summary()

    # 上游侧车池：看得到就等于知道崩溃隔离还在不在
    sidecar_stats: Dict[str, Any] = {"enabled": False}
    try:
        from app.upstream_pool import get_pool, sidecar_enabled

        if sidecar_enabled():
            running = get_pool()
            sidecar_stats = running.stats() if running is not None else {"enabled": False}
    except Exception:  # noqa: BLE001
        sidecar_stats = {"enabled": False}

    # 能力清单：一眼看出这个实例能做什么，避免客户端盲试
    try:
        from app.mcp_server import mcp_enabled

        mcp_on = bool(mcp_enabled())
    except Exception:  # noqa: BLE001
        mcp_on = False
    try:
        from app.tool_bridge import bridge_frame, tools_enabled

        tool_bridge = "on" if tools_enabled() else "off"
        bridge_style = bridge_frame()
    except Exception:  # noqa: BLE001
        tool_bridge = "unknown"
        bridge_style = "unknown"
    try:
        from app.notion_config_template import agent_caps_enabled

        agent_caps = "on" if agent_caps_enabled() else "off"
    except Exception:  # noqa: BLE001
        agent_caps = "unknown"

    payload = {
        "upstream_sidecar": sidecar_stats,
        "status": "ok" if status["active"] > 0 else "degraded",
        "accounts": status["active"],
        "accounts_available": status["available"],
        "accounts_total": status["total"],
        "accounts_cooling": status["cooling"],
        "accounts_disabled": status.get("disabled", 0),
        "accounts_saturated": status.get("saturated", 0),
        "accounts_inflight": status.get("inflight", 0),
        "account_strategy": pool.strategy,
        "uptime": int(uptime),
        # 能力面
        "mcp": mcp_on,
        "tool_bridge": tool_bridge,
        "notion_agent_caps": agent_caps,
        "capabilities": {
            "chat": True,
            "streaming": True,
            # 默认 false（上游没有原生 tools 通道，工具走 /mcp）。
            # TOOL_BRIDGE=1 时是「文本仿真」，如实报 true，别让客户端盲猜。
            "tools_on_chat": tool_bridge == "on",
            "tool_bridge_frame": bridge_style if tool_bridge == "on" else None,
        },
    }
    _HEALTH_CACHE["value"] = payload
    _HEALTH_CACHE["ts"] = time.time()
    return payload

# 挂载管理仪表盘构建产物（如果已经执行 npm run build）
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
dashboard_dir = os.path.join(frontend_dir, "admin-dashboard")
if os.path.exists(os.path.join(dashboard_dir, "index.html")):
    app.mount(
        "/admin-dashboard",
        StaticFiles(directory=dashboard_dir, html=True),
        name="admin-dashboard",
    )

# 挂载静态前端到根目录
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
