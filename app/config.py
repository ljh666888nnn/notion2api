import math
import os

from dotenv import load_dotenv

from app.account_store import load_accounts as _load_account_configs


# 加载 .env 文件中的环境变量（override=True 确保 .env 优先于系统环境变量）
load_dotenv(override=True)


def load_accounts() -> list[dict]:
    """从 accounts.json 或 NOTION_ACCOUNTS 加载并校验账号配置。"""

    return [account.to_dict() for account in _load_account_configs()]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数字。")
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数。") from exc


# 全局配置对象
ACCOUNTS = load_accounts()

# FastAPI 服务配置
API_KEY = os.getenv("API_KEY", "")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]

# APP_MODE: heavy（默认）、lite 或 standard
APP_MODE = os.getenv("APP_MODE", "heavy").lower().strip()

# 账号池策略与冷却参数
ACCOUNT_POOL_STRATEGY = os.getenv("ACCOUNT_POOL_STRATEGY", "fastest").strip().lower()
ACCOUNT_COOLDOWN_SECONDS = _env_float("ACCOUNT_COOLDOWN_SECONDS", 3.0)
ACCOUNT_MAX_COOLDOWN_SECONDS = _env_float("ACCOUNT_MAX_COOLDOWN_SECONDS", 60.0)
ACCOUNT_POOL_MAX_WAIT_SECONDS = _env_float("ACCOUNT_POOL_MAX_WAIT_SECONDS", 15.0)
ACCOUNT_MAX_CONCURRENCY = _env_int("ACCOUNT_MAX_CONCURRENCY", 0)
ACCOUNT_COOLDOWN_JITTER_RATIO = _env_float("ACCOUNT_COOLDOWN_JITTER_RATIO", 0.15)

if ACCOUNT_POOL_STRATEGY not in {
    "fastest",
    "round_robin",
    "weighted_round_robin",
    "least_recently_used",
    "least_inflight",
}:
    raise ValueError(
        "ACCOUNT_POOL_STRATEGY 只能是 fastest、round_robin、weighted_round_robin、least_recently_used 或 least_inflight。"
    )
if ACCOUNT_COOLDOWN_SECONDS <= 0:
    raise ValueError("ACCOUNT_COOLDOWN_SECONDS 必须大于 0。")
if ACCOUNT_MAX_COOLDOWN_SECONDS < ACCOUNT_COOLDOWN_SECONDS:
    raise ValueError("ACCOUNT_MAX_COOLDOWN_SECONDS 不能小于 ACCOUNT_COOLDOWN_SECONDS。")
if ACCOUNT_POOL_MAX_WAIT_SECONDS < 0:
    raise ValueError("ACCOUNT_POOL_MAX_WAIT_SECONDS 不能小于 0。")
if ACCOUNT_MAX_CONCURRENCY < 0:
    raise ValueError("ACCOUNT_MAX_CONCURRENCY 不能小于 0。")
if not 0 <= ACCOUNT_COOLDOWN_JITTER_RATIO <= 1:
    raise ValueError("ACCOUNT_COOLDOWN_JITTER_RATIO 必须在 0 到 1 之间。")
def is_lite_mode() -> bool:
    return APP_MODE == "lite"


def is_standard_mode() -> bool:
    """Standard 模式：发送完整上下文，支持 thinking 和搜索输出"""
    return APP_MODE == "standard"


def get_default_account():
    """获取默认账号（列表中的第一个账号）"""
    return ACCOUNTS[0]
