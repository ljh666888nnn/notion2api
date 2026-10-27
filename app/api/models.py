import time
from typing import Any, Dict

from fastapi import APIRouter

from app.model_registry import get_display_name, get_model_icon, list_available_models

router = APIRouter()

# 能力声明：让客户端不要盲试工具调用。
#
# 上游 Notion 的 runInferenceTranscript 没有 tools 通道。默认 tools=False：
# 需要工具请走 /mcp（网关本地执行，与上游无关）。
# TOOL_BRIDGE=1 时改报 true —— 那是「文本仿真」，不是原生 function calling，
# 但仍然会返回标准形状的 tool_calls，客户端可以按正常协议处理。
MODEL_CAPABILITIES: Dict[str, Any] = {
    "tools": False,
    "streaming": True,
    "reasoning": True,   # thinking 面板（standard/heavy）
    "search": True,      # 上游搜索（NOTION_WEB_SEARCH=1 时生效）
}


def _capabilities() -> Dict[str, Any]:
    caps = dict(MODEL_CAPABILITIES)
    try:
        from app.tool_bridge import bridge_frame, tools_enabled
    except Exception:  # noqa: BLE001
        return caps
    if tools_enabled():
        caps["tools"] = True
        caps["tools_bridged"] = True    # 经文本协议桥还原，非上游原生 function calling
        caps["tool_frame"] = bridge_frame()
    return caps


@router.get("/models", tags=["models"])
async def list_models() -> Dict[str, Any]:
    """
    列出可用的模型，兼容 OpenAI 格式。

    额外字段 `capabilities` 是本网关的扩展：`tools=false` 表示
    /v1/chat/completions 不会返回 tool_calls（上游不支持）；
    `tools=true` + `tools_bridged=true` 表示工具调用经文本协议桥实现。
    """
    created_at = int(time.time())
    caps = _capabilities()
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": created_at,
                "owned_by": "notion",
                "permission": [],
                "root": model_id,
                "parent": None,
                "display_name": get_display_name(model_id),
                "icon": get_model_icon(model_id),
                "capabilities": dict(caps),
            }
            for model_id in list_available_models()
        ],
    }
