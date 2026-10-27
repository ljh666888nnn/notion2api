import json
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# ================================
# 请求相关 Schema (Chat Completion)
# ================================

class ChatMessage(BaseModel):
    """单条对话消息。

    兼容 OpenAI 的完整角色集合：user / assistant / system / tool / function / developer。
    上游 Notion 只认 user/assistant/system，所以 tool 一类在解析时就被压平成文本。
    """

    role: str
    content: str = ""
    thinking: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Any]] = None

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        raw = str(v or "").strip().lower()
        if raw == "developer":
            return "system"
        if raw == "function":
            return "tool"
        if raw in {"user", "assistant", "system", "tool"}:
            return raw
        raise ValueError(
            "role must be one of: user, assistant, system, tool, function, developer"
        )

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content_parts(cls, v):
        # OpenAI-style content parts -> plain text；非文本部分（image_url 等）直接丢弃。
        if v is None:
            return ""
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            texts = []
            for part in v:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(part.get("text") or "")
                    elif isinstance(part.get("text"), str):
                        texts.append(part["text"])
                elif isinstance(part, str):
                    texts.append(part)
            return "\n".join(t for t in texts if t)
        if isinstance(v, dict):
            if isinstance(v.get("text"), str):
                return v["text"]
            return ""
        return str(v)

    @staticmethod
    def _describe_tool_calls(tool_calls: Any) -> str:
        if not isinstance(tool_calls, list):
            return ""
        lines = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name") or call.get("name") or "tool"
            args = fn.get("arguments") or call.get("arguments") or ""
            if isinstance(args, (dict, list)):
                args = json.dumps(args, ensure_ascii=False)
            args = str(args)
            if len(args) > 800:
                args = args[:800] + "..."
            lines.append(f"{name}({args})" if args else f"{name}()")
        return "\n".join(lines)

    @model_validator(mode="after")
    def _flatten_tool_turns(self):
        role = self.role

        if role == "tool":
            label = self.name or self.tool_call_id or "tool"
            body = self.content.strip()
            self.content = f"[tool result: {label}]\n{body}" if body else f"[tool result: {label}]"
            self.role = "user"
            return self

        if role == "assistant" and self.tool_calls:
            summary = self._describe_tool_calls(self.tool_calls)
            if summary:
                if self.content.strip():
                    self.content = f"{self.content.strip()}\n\n[tool calls]\n{summary}"
                else:
                    self.content = f"[tool calls]\n{summary}"
            return self

        return self


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-Compatible 发起完成请求的 Payload。
    保留 `conversation_id` 作为特定的扩展字段，若缺失则视为独立请求。

    `tools` / `tool_choice` 是 function calling 支持所必需的：
    上游 Notion 不认这两个字段，由 app/tool_bridge.py 渲染成文本注入。
    """
    model: str = Field(default="claude-opus4.6", description="Requested model.")
    messages: List[ChatMessage]
    stream: bool = Field(default=False, description="Whether to stream the response as SSE.")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature.")
    conversation_id: Optional[str] = Field(default=None, description="Extension for stateful conversation tracking.")
    tools: Optional[List[Any]] = Field(default=None, description="OpenAI function calling tools.")
    tool_choice: Optional[Any] = Field(default=None, description="'auto' | 'none' | 'required' | {type,function}.")
    parallel_tool_calls: Optional[bool] = Field(default=None)
    response_format: Optional[Any] = Field(default=None)
    # OpenAI 的流式附加项；目前只认 include_usage（末帧回 usage 估算）。
    stream_options: Optional[Dict[str, Any]] = Field(default=None)
    # 推理强度：low | medium | high（也接受 minimal / xhigh）。映射到 Notion 的
    # config.reasoningEffort；缺省 medium。
    reasoning_effort: Optional[str] = Field(default=None, description="low | medium | high")
    # Anthropic 风格：{type:"enabled", budget_tokens:N} 也能换算成强度
    thinking: Optional[Any] = Field(default=None)

# ================================
# 非流式返回 Schema
# ================================

class ChatMessageResponseChoice(BaseModel):
    """非流式响应的选项。

    message 用 dict 而不是 ChatMessage：响应里的 assistant 需要原样带出
    `tool_calls`，而 ChatMessage 的 validator 是为**入参**设计的，会把
    tool_calls 压平成文本。
    """
    index: int = 0
    message: Dict[str, Any]
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    """
    OpenAI-Compatible 完整返回 Payload。
    """
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatMessageResponseChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    # Standard 模式扩展字段
    search_metadata: Optional[Dict[str, Any]] = Field(default=None)

# ================================
# 流式返回 Schema (供内部组织)
# ================================

class ChatCompletionChunkDelta(BaseModel):
    """SSE Delta Block"""
    content: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[List[Any]] = None

class ChatCompletionChunkChoice(BaseModel):
    """SSE Choice Block"""
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    """
    OpenAI-Compatible 流式 Chunk
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
