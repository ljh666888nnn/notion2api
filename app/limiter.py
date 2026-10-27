import os

from slowapi import Limiter
from slowapi.util import get_remote_address
from app.config import is_lite_mode, is_standard_mode

# DISABLE_RATE_LIMIT=True 时关闭限流（调试用）
_disable_flag = os.getenv("DISABLE_RATE_LIMIT", "").strip().lower()
_rate_limit_disabled = _disable_flag == "true"

# 根据 APP_MODE 动态设置速率限制
# Lite 模式：30/minute（单轮问答响应快，3-5秒/次）
# Standard 模式：25/minute（完整上下文，支持 thinking 和搜索）
# Heavy 模式：20/minute（需要数据库操作，相对慢一些）
if is_lite_mode():
    default_limit = "30/minute"
elif is_standard_mode():
    default_limit = "25/minute"
else:
    default_limit = "20/minute"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")


def _slowapi_config_filename() -> str | None:
    """避免 slowapi 用系统默认编码读 .env 时把服务搞崩。

    slowapi 内部构造的是 `starlette.config.Config(".env")`，而 starlette 的
    `_read_file` 用的是 `open(file_name)`——**没有指定 encoding**，在中文 Windows
    上就是 GBK。只要 .env 里出现任意一个非 ASCII 字节（中文注释、emoji），
    构造 Limiter 时就会抛 UnicodeDecodeError，整个服务起不来。

    这里的处理：.env 是纯 ASCII 就返回 None（保持 slowapi 原行为，仍能读到
    RATELIMIT_* 设置）；检测到非 ASCII 就返回一个不存在的路径，让 starlette
    直接跳过文件读取（限流参数本来就写在代码里，不依赖 .env）。
    """
    try:
        with open(_ENV_PATH, "rb") as handle:
            handle.read().decode("ascii")
        return None
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        return _ENV_PATH + ".slowapi-skip"


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[] if _rate_limit_disabled else [default_limit],
    enabled=not _rate_limit_disabled,
    config_filename=_slowapi_config_filename(),
)
