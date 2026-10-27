"""Notion AI 推理请求的 config 块模板。

来源：从真实浏览器会话抓取的 `runInferenceTranscript` 请求体（app.notion.com）。
Notion 会校验 config 字段的完整度；只给 `type` + `model` 两个字段时，
接口返回 200 但流为空（单个 `[`），不报错、不扣额度，非常难排查。

因此这里保留完整字段集，运行时只覆盖 `type` 和 `model`。
Notion 增加新字段时，用 `scripts/edge_watch.py` 抓一次再补进来。
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

# 完整 config 模板（与浏览器抓包一致）
CONFIG_TEMPLATE: Dict[str, Any] = {
    "type": "workflow",
    "model": "amla-jam",
    "isHipaa": False,
    "isMobile": False,
    "writerMode": False,
    "searchScopes": [{"type": "everything"}],
    "useWebSearch": True,
    "isCustomAgent": False,
    "manageWorkers": False,
    "modelFromUser": True,
    "enableComputer": True,
    "internetAccess": False,
    "enableQueryMail": False,
    "reasoningEffort": "medium",
    "useReadOnlyMode": False,
    "enableAgentDiffs": True,
    "enableScriptAgent": True,
    "enableWebResearch": False,
    "isOnboardingAgent": False,
    "enableCustomAgents": True,
    "agentMemorySettings": {"useMemories": True, "excludeChatFromMemories": False},
    "enableMarkdownVNext": True,
    "enableQueryCalendar": False,
    "isCustomAgentCreate": False,
    "useCustomAgentDraft": False,
    "enableAgentAskSurvey": True,
    "enableCrdtOperations": False,
    "enableScriptAgentGtm": False,
    "isCustomAgentBuilder": False,
    "useRulePrioritization": True,
    "enableAgentAutomations": True,
    "enableAgentThreadTools": False,
    "enableScriptAgentSlack": True,
    "isAgentResearchRequest": False,
    "databaseAgentConfigMode": False,
    "enableAgentIntegrations": True,
    "enableAgentGenerateImage": False,
    "enableSystemPromptAsPage": False,
    "enableUserSessionContext": False,
    "enableScriptAgentAdvanced": False,
    "enableSoftwareFactoryPage": False,
    "enableSuggestedEditsTools": True,
    "enableCsvAttachmentSupport": True,
    "enableNotionMailDeprecated": True,
    "enableSkillsInCustomAgents": False,
    "enableScriptAgentMcpServers": True,
    "enableAgentCardCustomization": True,
    "enableUpdatePageOrderUpdates": True,
    "useContextualCoreDocsAutoLoad": False,
    "useDocPreviewsForCoreAutoLoad": True,
    "enableExperimentalIntegrations": False,
    "updatePageStaleViewGuardEnabled": True,
    "enableAgentSupportPropertyReorder": True,
    "enableMailNotificationPreferences": False,
    "showDatabaseAgentsDiscoverability": False,
    "enableMailAgentMultiProviderSupport": True,
    "enableLargeToolResultComputerOffload": False,
    "enableScriptAgentGoogleDriveInCustomAgent": False,
    "enableScriptAgentGoogleDriveOAuthInCustomAgent": False,
    "enableScriptAgentSearchConnectorsInCustomAgent": False,
}


# 这份模板是从**真实浏览器里的 agent 会话**抓的，所以自带了 Notion 的
# computer / script-agent 能力开关。对我们这种"纯文本网关"来说，这些开关会把
# 模型变成"在 Notion 沙箱里干活的 agent"——它于是回答
#   「沙盒文件系统 /data 是空的，无法访问你本地 Windows 路径」
# 而不会理我们注入的工具清单。默认全部关掉，见 _apply_agent_policy()。

# agent 相关能力（关掉后模型退回普通对话助手）
AGENT_CAPABILITY_FLAGS = (
    "enableComputer",
    "enableScriptAgent",
    "enableScriptAgentSlack",
    "enableScriptAgentMcpServers",
    "enableScriptAgentGtm",
    "enableScriptAgentAdvanced",
    "enableSuggestedEditsTools",
    "enableAgentDiffs",
    "enableAgentThreadTools",
    "enableAgentIntegrations",
    "enableCustomAgents",
    "enableAgentAutomations",
    "enableAgentAskSurvey",
    "enableAgentCardCustomization",
)

# 会话开始时自动加载 Notion 文档/上下文（同样是"工作区幻觉"的来源）
CONTEXT_AUTOLOAD_FLAGS = (
    "useContextualCoreDocsAutoLoad",
    "useDocPreviewsForCoreAutoLoad",
)


def agent_caps_enabled() -> bool:
    """上游 agent 能力是否保留。默认关闭；排查/对照时用 NOTION_AGENT_CAPS=1 还原。"""
    return os.getenv("NOTION_AGENT_CAPS", "0").strip() in {"1", "true", "True"}


def _apply_agent_policy(config: Dict[str, Any]) -> None:
    if agent_caps_enabled():
        return
    for key in AGENT_CAPABILITY_FLAGS:
        if key in config:
            config[key] = False
    for key in CONTEXT_AUTOLOAD_FLAGS:
        config[key] = False
    memory = config.get("agentMemorySettings")
    if isinstance(memory, dict):
        memory["useMemories"] = False


def build_config(
    thread_type: str, notion_model: str, reasoning_effort: str | None = None
) -> Dict[str, Any]:
    """按线程类型和模型生成完整 config 块。"""
    config = copy.deepcopy(CONFIG_TEMPLATE)

    # 允许临时改 type 做对照实验（workflow / markdown-chat）
    override_type = os.getenv("NOTION_THREAD_TYPE", "").strip()
    config["type"] = override_type or (thread_type or "workflow")
    config["model"] = notion_model

    # 强度：客户端没传就沿用模板里的 medium（与浏览器抓包一致）
    if reasoning_effort:
        config["reasoningEffort"] = reasoning_effort

    # 上游默认会做工作区/网页搜索，并在回答末尾追加
    #   > 🌐 **来源:**
    #   > 1. [user://...]
    # 这种引用块会把 agent 对话污染成"搜索摘要"。默认关闭，需要时用
    # NOTION_WEB_SEARCH=1 打开。
    if os.getenv("NOTION_WEB_SEARCH", "0").strip() not in {"1", "true", "True"}:
        config["useWebSearch"] = False
        config["enableWebResearch"] = False
        config["searchScopes"] = []

    _apply_agent_policy(config)
    return config
