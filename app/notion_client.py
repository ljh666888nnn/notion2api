import datetime as dt
import os
import threading
import time
import uuid
from email.utils import parsedate_to_datetime
from typing import Any, Generator, Optional

import cloudscraper
import requests
import urllib3

try:  # 浏览器级 TLS 指纹；缺失时回退到 requests（会被上游软拦截）
    from curl_cffi import requests as cffi_requests
except ImportError:  # pragma: no cover
    cffi_requests = None

from app.account_schema import mask_email, mask_secret
from app.logger import logger
from app.model_registry import get_notion_model
from app.stream_parser import parse_stream

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 可通过环境变量覆盖 Notion 客户端版本号（Notion 更新后可能需要同步）
NOTION_CLIENT_VERSION = os.getenv("NOTION_CLIENT_VERSION", "23.13.20260228.0625")

# Notion 站点根地址（区域镜像 / 反代时可覆盖）；未设置时保持官方默认
NOTION_URL = os.getenv("NOTION_URL", "https://www.notion.so").rstrip("/")

# curl_cffi 的浏览器指纹档位。Notion 对 runInferenceTranscript 做 TLS 指纹检测：
# 纯 requests(OpenSSL) 会被识别为非浏览器，返回 200 + subType=temporarily-unavailable。
# 换成 curl_cffi 模拟 Chrome/Edge 的 TLS+HTTP2 指纹后正常返回 NDJSON。
NOTION_IMPERSONATE = os.getenv("NOTION_IMPERSONATE", "chrome")

# curl_cffi 在 Windows 上存在并发/中断时的 native 堆损坏（0xC0000374，见
# lexiforest/curl_cffi#718「Fixes for heap corruption due to thread safety issue」，
# 该 PR 目前仍未合入）。现象是整个 uvicorn 进程被 ntdll 直接干掉、没有 Python
# traceback，客户端只看到连接中断。这里用信号量限制同一进程内并发的 upstream
# 流式请求数，默认 2；设为 0 表示不限制（回到旧行为）。
def _upstream_concurrency() -> int:
    raw = os.getenv("NOTION_UPSTREAM_MAX_CONCURRENCY", "2").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2


_UPSTREAM_LIMIT = _upstream_concurrency()
_UPSTREAM_SEM = threading.BoundedSemaphore(_UPSTREAM_LIMIT) if _UPSTREAM_LIMIT else None


def _parse_retry_after(value: Any) -> Optional[float]:
    """Parse either delta-seconds or an HTTP-date Retry-After header."""

    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if retry_at is None:
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
        seconds = (retry_at - dt.datetime.now(dt.timezone.utc)).total_seconds()
    return max(0.0, seconds)


class NotionUpstreamError(RuntimeError):
    """Notion 上游请求失败或返回异常内容。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retriable: bool = True,
        response_excerpt: str = "",
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retriable = retriable
        self.response_excerpt = response_excerpt
        self.retry_after = retry_after


class NotionOpusAPI:
    def __init__(self, account_config: dict):
        """
        从单组账号配置初始化 Notion 客户端。
        account_config 需要包含 token_v2, space_id, user_id, space_view_id, user_name, user_email
        """
        self.token_v2 = account_config.get("token_v2", "")
        self.space_id = account_config.get("space_id", "")
        self.user_id = account_config.get("user_id", "")
        self.space_view_id = account_config.get("space_view_id", "")
        self.user_name = account_config.get("user_name", "user")
        self.user_email = account_config.get("user_email", "")
        self.cookies = account_config.get("cookies", {})
        if not isinstance(self.cookies, dict):
            self.cookies = {}
        self.cookies["token_v2"] = self.token_v2

        self.url = f"{NOTION_URL}/api/v3/runInferenceTranscript"
        self.delete_url = f"{NOTION_URL}/api/v3/saveTransactions"
        # Keep logs useful without writing full email addresses or user IDs.
        self.account_key = (
            mask_email(self.user_email)
            or mask_secret(self.user_id)
            or "unknown-account"
        )

        # 复用 cloudscraper 实例：保留 Cloudflare challenge cookie，避免每次请求都重新过验证
        self._scraper = cloudscraper.create_scraper()
        self._scraper_lock = threading.Lock()

    def _build_cookie_header(self) -> str:
        cookie_jar = self.cookies.copy()
        cookie_jar["notion_user_id"] = self.user_id
        return "; ".join(f"{name}={value}" for name, value in cookie_jar.items() if value)

    def _to_notion_transcript(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for block in transcript:
            if block.get("type") != "config":
                converted.append(block)
                continue

            value = block.get("value")
            if not isinstance(value, dict):
                converted.append(block)
                continue

            notion_block = dict(block)
            notion_value = dict(value)
            notion_value["model"] = get_notion_model(str(value.get("model", "") or ""))
            notion_block["value"] = notion_value
            converted.append(notion_block)
        return converted

    def _resolve_thread_type(self, notion_transcript: list[dict[str, Any]]) -> str:
        for block in notion_transcript:
            if block.get("type") != "config":
                continue
            value = block.get("value")
            if isinstance(value, dict):
                thread_type = str(value.get("type", "") or "").strip()
                if thread_type:
                    return thread_type
        return "workflow"

    def _resolve_request_profile(self, thread_type: str) -> dict[str, Any]:
        is_markdown_chat = thread_type == "markdown-chat"
        return {
            "thread_type": thread_type,
            "create_thread": not is_markdown_chat,
            "is_partial_transcript": is_markdown_chat,
            "precreate_thread": is_markdown_chat,
            "include_debug_overrides": True,
        }

    def _build_thread_headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "cookie": self._build_cookie_header(),
            "x-notion-active-user-header": self.user_id,
            "x-notion-space-id": self.space_id,
        }

    def _create_thread(self, thread_id: str, thread_type: str) -> bool:
        payload = {
            "requestId": str(uuid.uuid4()),
            "transactions": [
                {
                    "id": str(uuid.uuid4()),
                    "spaceId": self.space_id,
                    "operations": [
                        {
                            "pointer": {"table": "thread", "id": thread_id, "spaceId": self.space_id},
                            "path": [],
                            "command": "set",
                            "args": {
                                "id": thread_id,
                                "version": 1,
                                "parent_id": self.space_id,
                                "parent_table": "space",
                                "space_id": self.space_id,
                                "created_time": int(time.time() * 1000),
                                "created_by_id": self.user_id,
                                "created_by_table": "notion_user",
                                "messages": [],
                                "data": {},
                                "alive": True,
                                "type": thread_type,
                            },
                        }
                    ],
                }
            ],
        }
        try:
            resp = requests.post(
                self.delete_url,
                json=payload,
                headers=self._build_thread_headers(),
                timeout=20,
            )
            if resp.status_code == 200:
                return True
            logger.warning(
                "Pre-create thread failed",
                extra={
                    "request_info": {
                        "event": "thread_precreate_failed",
                        "thread_id": thread_id,
                        "thread_type": thread_type,
                        "status": resp.status_code,
                    }
                },
            )
        except Exception:
            logger.warning(
                "Pre-create thread raised exception",
                exc_info=True,
                extra={
                    "request_info": {
                        "event": "thread_precreate_error",
                        "thread_id": thread_id,
                        "thread_type": thread_type,
                    }
                },
            )
        return False

    def delete_thread(self, thread_id: str) -> None:
        """
        通过 saveTransactions 接口将指定 thread 的 alive 状态设为 False，
        从而清理 Notion 主页面上的对话记录。
        此方法设计为在后台线程中调用，不影响主流输出。
        """
        headers = self._build_thread_headers()
        payload = {
            "requestId": str(uuid.uuid4()),
            "transactions": [
                {
                    "id": str(uuid.uuid4()),
                    "spaceId": self.space_id,
                    "operations": [
                        {
                            "pointer": {
                                "table": "thread",
                                "id": thread_id,
                                "spaceId": self.space_id,
                            },
                            "command": "update",
                            "path": [],
                            "args": {"alive": False},
                        }
                    ],
                }
            ],
        }
        try:
            resp = requests.post(self.delete_url, json=payload, headers=headers, timeout=15)
            if resp.status_code == 200:
                logger.info(
                    "Thread auto-deleted from Notion home",
                    extra={"request_info": {"event": "thread_deleted", "thread_id": thread_id}},
                )
            else:
                logger.warning(
                    f"Thread deletion failed: HTTP {resp.status_code}",
                    extra={"request_info": {"event": "thread_delete_failed", "thread_id": thread_id, "status": resp.status_code}},
                )
        except Exception as exc:
            logger.warning(
                f"Thread deletion raised an exception: {exc}",
                extra={"request_info": {"event": "thread_delete_error", "thread_id": thread_id}},
            )

    def check_health(self) -> dict[str, Any]:
        """Probe credentials with a lightweight, read-only Notion API call.

        The returned payload deliberately contains only a status code and a
        short diagnostic message; cookies and identifiers never leave here.
        """

        url = f"{NOTION_URL}/api/v3/loadUserContent"
        headers = self._build_thread_headers()
        response = None
        started = time.monotonic()
        try:
            response = requests.post(url, headers=headers, json={}, timeout=20)
            if response.status_code == 403:
                response.close()
                with self._scraper_lock:
                    scraper = self._scraper
                response = scraper.post(url, headers=headers, json={}, timeout=20)

            status_code = int(response.status_code)
            if status_code == 200:
                message = "凭据有效，Notion API 可访问。"
                ok = True
            elif status_code == 401:
                ok = False
                message = "Notion 认证失败 (HTTP 401)，token_v2 可能已过期。"
            elif status_code == 403:
                ok = False
                message = "Notion 拒绝访问 (HTTP 403)，可能被 Cloudflare 或工作区权限拦截。"
            elif status_code == 429:
                ok = False
                message = "Notion 请求频率受限 (HTTP 429)，请稍后重试。"
            elif status_code >= 500:
                ok = False
                message = f"Notion 服务暂时不可用 (HTTP {status_code})。"
            else:
                ok = False
                message = f"Notion 探测返回非预期状态 (HTTP {status_code})。"
            return {
                "ok": ok,
                "status_code": status_code,
                "message": message,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            }
        except requests.exceptions.Timeout:
            return {
                "ok": False,
                "status_code": None,
                "message": "连接 Notion 探测接口超时。",
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            }
        except requests.exceptions.RequestException as exc:
            return {
                "ok": False,
                "status_code": None,
                "message": f"连接 Notion 探测接口失败：{type(exc).__name__}。",
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
            }
        finally:
            if response is not None:
                response.close()

    def stream_response(
        self,
        transcript: list,
        thread_id: Optional[str] = None,
        *,
        force_new_thread: bool = False,
    ) -> Generator[dict[str, Any], None, None]:
        """
        发起 Notion API 请求并返回结构化流生成器。
        接收完整的 transcript 列表作为参数。

        Args:
            transcript: 对话历史记录列表
            thread_id: 可选的已有 thread_id。如果提供，将重用该线程以保持上下文
            force_new_thread: 为 True 时即使传入了 thread_id 也按新线程创建。
                调用方可自行生成 ID，从而避免依赖共享客户端上的 current_thread_id。
        """
        if not isinstance(transcript, list) or not transcript:
            raise ValueError("Invalid transcript payload: transcript must be a non-empty list.")

        notion_transcript = self._to_notion_transcript(transcript)
        thread_type = self._resolve_thread_type(notion_transcript)
        request_profile = self._resolve_request_profile(thread_type)

        # 如果没有提供 thread_id，创建新的；否则重用已有的
        should_create_thread = thread_id is None or force_new_thread
        thread_id = thread_id or str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        response = None

        # 保存 thread_id 以便外部访问
        self.current_thread_id = thread_id

        if request_profile["precreate_thread"] and should_create_thread:
            if not self._create_thread(thread_id, thread_type):
                should_create_thread = True
                request_profile["create_thread"] = True
                request_profile["is_partial_transcript"] = False
        elif not should_create_thread:
            # 如果重用已有线程，不要创建新线程
            request_profile["create_thread"] = False
            # 关键修复：设置 is_partial_transcript=True，让 Notion 接受客户端的历史消息
            request_profile["is_partial_transcript"] = True

        # 把 cookie 直接放进 header，绕过 cloudscraper 的 cookie jar
        # （cookie jar 可能被 Cloudflare challenge 写入含非 ASCII 字符的 cookie，导致编码错误）
        cookie_header = self._build_cookie_header()

        # ВАЖНО (2026-09-06): апстрим отдаёт пустой стрим без браузерного фингерпринта
        # (проверено A/B: те же payload/token/IP — с sec-fetch + свежим UA идёт NDJSON, без них пусто).
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
            "x-notion-space-id": self.space_id,
            "x-notion-active-user-header": self.user_id,
            "notion-audit-log-platform": "web",
            "notion-client-version": NOTION_CLIENT_VERSION,
            "origin": NOTION_URL,
            "referer": f"{NOTION_URL}/chat",
            "sec-ch-ua": '"Chromium";v="152", "Not?A_Brand";v="24", "Google Chrome";v="152"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Linux"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "cookie": cookie_header,
        }

        payload = {
            "traceId": trace_id,
            "spaceId": self.space_id,
            "threadId": thread_id,
            "threadType": thread_type,
            "createThread": request_profile["create_thread"],
            "generateTitle": True,
            "saveAllThreadOperations": True,
            "setUnreadState": True,
            "isPartialTranscript": request_profile["is_partial_transcript"],
            "asPatchResponse": True,
            "isUserInAnySalesAssistedSpace": False,
            "isSpaceSalesAssisted": False,
            "threadParentPointer": {
                "table": "space",
                "id": self.space_id,
                "spaceId": self.space_id,
            },
            "transcript": notion_transcript,
        }
        if request_profile["include_debug_overrides"]:
            payload["debugOverrides"] = {
                "emitAgentSearchExtractedResults": True,
                "cachedInferences": {},
                "annotationInferences": {},
                "emitInferences": False,
            }

        logger.info(
            "Dispatching request to Notion upstream",
            extra={
                "request_info": {
                    "event": "notion_upstream_request",
                    "trace_id": trace_id,
                    "thread_id": thread_id,
                    "thread_type": thread_type,
                    "create_thread": bool(request_profile["create_thread"]),
                    "is_partial_transcript": bool(request_profile["is_partial_transcript"]),
                    "account": self.account_key,
                    "space_id": self.space_id,
                }
            },
        )

        response = None
        acquired = False
        if _UPSTREAM_SEM is not None:
            acquired = _UPSTREAM_SEM.acquire(timeout=120)
            if not acquired:
                logger.warning(
                    "Upstream concurrency limit wait timed out; proceeding unserialized",
                    extra={"request_info": {"event": "upstream_semaphore_timeout",
                                            "account": self.account_key}},
                )
        try:
            # 2026-10: Notion 对 runInferenceTranscript 做 TLS 指纹检测。
            # 纯 requests(OpenSSL) 会被判为非浏览器，返回 200 + temporarily-unavailable。
            # 优先 curl_cffi 模拟浏览器指纹；不可用时回退 requests，403 再回退 cloudscraper。
            if cffi_requests is not None:
                response = cffi_requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    impersonate=NOTION_IMPERSONATE,
                    timeout=(15, 120),
                )
            else:
                response = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=(15, 120),
                )
            if response.status_code == 403:
                # Cloudflare challenge 可能过期，重建 scraper 后重试一次
                response.close()
                logger.warning(
                    "Got 403, rebuilding cloudscraper to refresh Cloudflare challenge",
                    extra={"request_info": {"event": "cloudflare_challenge_refresh", "account": self.account_key}},
                )
                new_scraper = cloudscraper.create_scraper()
                with self._scraper_lock:
                    self._scraper = new_scraper
                response = new_scraper.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=(15, 120),
                )
            if response.status_code != 200:
                excerpt = (response.text or "").strip().replace("\n", " ")[:300]
                # 认证/权限问题通过换账号重试；限流和 5xx 通过冷却后重试。
                retriable = (
                    response.status_code in {401, 403, 408, 425, 429}
                    or response.status_code >= 500
                )
                raise NotionUpstreamError(
                    f"Notion upstream returned HTTP {response.status_code}.",
                    status_code=response.status_code,
                    retriable=retriable,
                    response_excerpt=excerpt,
                    retry_after=_parse_retry_after(
                        response.headers.get("Retry-After")
                    ),
                )

            emitted = False
            for chunk in parse_stream(response):
                emitted = True
                yield chunk

            if not emitted:
                raise NotionUpstreamError(
                    "Notion upstream returned an empty stream.",
                    status_code=502,
                    retriable=True,
                )

            # 流结束后，不再自动删除 thread
            # 原因：Notion API 的 workflow 模式依赖于服务器端保存的对话历史
            # 删除 thread 会导致后续请求无法获取历史消息（AI 失忆）
            # 保持 thread 存活可以维持对话上下文
            logger.info(
                "Thread completed and preserved for conversation context",
                extra={
                    "request_info": {
                        "event": "thread_completed_preserved",
                        "thread_id": thread_id,
                        "was_created_new": should_create_thread,
                    }
                },
            )
        except requests.exceptions.Timeout as exc:
            logger.error(f"Request timeout: {exc}", exc_info=True)
            raise NotionUpstreamError("Request to Notion upstream timed out.", retriable=True) from exc
        except requests.exceptions.RequestException as exc:
            logger.error(f"Request failed: {exc}", exc_info=True)
            # 不暴露原始异常细节给用户
            raise NotionUpstreamError("Request to Notion upstream failed. Please try again later.", retriable=True) from exc
        finally:
            if response is not None:
                response.close()
            if acquired and _UPSTREAM_SEM is not None:
                _UPSTREAM_SEM.release()
