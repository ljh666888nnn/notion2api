# notion2api

把 **Notion AI** 包装成 **OpenAI 兼容 API** 的本地网关。一个进程同时提供三套协议（OpenAI Chat Completions / Anthropic Messages / OpenAI Responses），自带账号池、流式输出、工具调用桥、额度探测与运维面板。

> **使用范围**：本项目面向本地自用与学习研究。上游是第三方服务，请自行确认你的使用方式符合其条款；作者不对账号封禁或数据丢失负责。

---

## 目录

- [特性](#特性)
- [架构](#架构)
- [快速开始](#快速开始)
- [账号导入](#账号导入)
- [接口](#接口)
- [模型](#模型)
- [工具调用](#工具调用)
- [推理强度](#推理强度)
- [多轮上下文](#多轮上下文)
- [图片](#图片)
- [主要配置](#主要配置)
- [运维](#运维)
- [已知限制](#已知限制)
- [目录结构](#目录结构)
- [许可](#许可)

## 特性

| 能力 | 说明 |
| --- | --- |
| OpenAI 兼容 | `POST /v1/chat/completions`，流式 / 非流式，`tools`、`reasoning_effort` |
| Anthropic 兼容 | `POST /v1/messages`，Claude Code / Anthropic SDK 可直接接入，支持 `tools` 与 `thinking` |
| Responses 兼容 | `POST /v1/responses`，Codex 等新客户端，支持函数调用与 `type:"custom"` 工具 |
| 多模型 | 31 个模型标识映射到 Notion 内部代号，见 `app/model_registry.py` |
| 账号池 | `fastest` 调度、按账号并发上限、429 冷却、EWMA 延迟、故障隔离、额度探测 |
| 流式输出 | SSE，含 thinking 与搜索结果事件；结束前与上游终稿比对并补齐缺失后缀 |
| 工具调用桥 | 上游只接受纯文本，网关把工具清单渲染进提示词，再把 `<tool_call>` 解析回标准 `tool_calls` |
| 多轮记忆 | 上游是无状态的，网关把完整历史重放进最后一条消息 |
| 推理强度 | `reasoning_effort` / `thinking.budget_tokens` 统一映射到上游 `config.reasoningEffort` |
| 图片 | 上游不收图片，网关用本地视觉模型转写成文字描述后注入（`NOTION_IMAGE_MODE=vision`） |
| 可观测 | SQLite 保存调用明细、吞吐采样、额度探测；`/v1/admin/*` 查询 |
| MCP 服务 | `POST /mcp`，只读工具：项目结构、读文件、搜索、git 状态 |

## 架构

```
客户端（OpenAI SDK / Claude Code / Codex / curl）
   │  OpenAI        Anthropic        Responses
   ▼
FastAPI 网关  app/server.py
   ├── /v1/chat/completions   协议适配、流式改写、重试、工具桥
   ├── /v1/messages           Anthropic 兼容（转发到上面的内部链路）
   ├── /v1/responses          Responses 兼容（同上）
   ├── /v1/models             模型清单 + capabilities
   ├── /v1/admin/*            账号池、额度、调用明细、备份
   ├── /mcp                   MCP JSON-RPC（MCP_ENABLED=1）
   └── /                      聊天界面（frontend/index.html）
   │
   ▼
账号池 app/account_pool.py ──fastest/冷却/并发──► Notion AI 上游
   │
   └── SQLite app/store.py     采样 / 调用明细 / 额度
```

## 快速开始

```bash
# 1. 依赖（Python 3.10+）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置：至少填 API_KEY 和账号
cp .env.example .env
#   API_KEY=sk-你的本地密钥
#   其余保持默认即可

# 3. 放入账号（见下一节）
cp accounts.example.json accounts.json

# 4. 启动
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/> 是聊天界面，`/docs` 是 OpenAPI 文档。

## 账号导入

**最省事：只填 token_v2。** 网关会用这个 token 调一次 `/api/v3/getSpaces`，
自动补全 `space_id` / `user_id` / `space_view_id`。

1. 登录 <https://www.notion.so>，按 F12；
2. Application → Cookies → `https://www.notion.so` → 复制 `token_v2` 的值；
3. 写进 `accounts.json`：

```json
[
  { "token_v2": "v03:xxxxxxxxxxxxxxxx" }
]
```

4. 启动服务，`GET /health` 里 `accounts` ≥ 1 即成功（解析结果会并入内存里的账号）。

> 自动补全可以用 `ACCOUNT_AUTODISCOVER=0` 关掉；关掉后必须自己写全 4 个字段。
> 解析失败会直接报错并提示补哪个字段，不会让服务带着半个账号跑起来。

### 想一次拿到全部字段（可选）

在已登录的 Notion 页面按 F12 → Console，粘贴：

```js
(async () => {
  const cookieToken = decodeURIComponent((document.cookie.match(/(?:^|;\s*)token_v2=([^;]+)/) || [])[1] || '');

  const res = await fetch('/api/v3/getSpaces', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
    credentials: 'include',
  });
  const raw = await res.text();
  console.log('[诊断] status =', res.status, '| 返回字节 =', raw.length, '| document.cookie 里能读到 token =', !!cookieToken);

  let data;
  try { data = JSON.parse(raw); } catch (e) {
    return console.warn('[诊断] 返回不是 JSON，前 200 字：', raw.slice(0, 200));
  }
  if (!data || !Object.keys(data).length) {
    return console.warn('[诊断] getSpaces 返回空对象：这个 token 可能不属于任何 workspace');
  }

  const accounts = [];
  for (const [userId, tables] of Object.entries(data)) {
    if (!tables || !tables.space) continue;
    const views = tables.space_view || {};
    for (const [spaceId, record] of Object.entries(tables.space)) {
      const value = (record && record.value) || {};
      const viewId = Object.keys(views).find(id => views[id] && views[id].spaceId === spaceId) || '';
      accounts.push({
        profile_name: value.name || 'main',
        token_v2: cookieToken || '<把 token_v2 粘到这里>',
        space_id: spaceId,
        user_id: userId,
        space_view_id: viewId,
        user_name: '',
        user_email: '',
      });
    }
  }

  console.log(JSON.stringify(accounts, null, 2));
  if (typeof copy === 'function') copy(JSON.stringify(accounts, null, 2));
  if (!cookieToken) {
    console.warn('[注意] document.cookie 读不到 token_v2（可能被设成 HttpOnly）。');
    console.warn('       接口仍然能拿到 id，但 token 请从 Application → Cookies 里复制后替换占位符。');
  }
})();
```

两点说明：

- **接口部分不依赖读 cookie**：`fetch` 用 `credentials: 'include'` 自动带 cookie，所以即使 `document.cookie` 读不到 `token_v2`，`space_id` / `user_id` / `space_view_id` 照样能拿到；只有 `token_v2` 本身可能需要手抄。
- **`token_v2` 是 HttpOnly 时控制台读不到**：这是浏览器的限制，不是脚本 bug。此时按上面「只填 token_v2」的方式做即可，网关照常自动补全。

### 手动兜底

| 字段 | 从哪里拿 |
| --- | --- |
| `token_v2` | DevTools → Application → Cookies → `https://www.notion.so` |
| `space_id` | Network 面板筛 `getSpaces` → Response → `space` 对象的**键** |
| `user_id` | 同一份 Response 的**顶层键** |
| `space_view_id` | `space_view` 里 `spaceId` 等于上面 `space_id` 的那一项的键 |

字段说明：

| 字段 | 是否必填 | 说明 |
| --- | --- | --- |
| `token_v2` | **必填** | Notion 会话 cookie，**泄露即等于账号被接管** |
| `space_id` / `user_id` | 可省 | 留空时由 `ACCOUNT_AUTODISCOVER` 自动反查 |
| `space_view_id` | 可省 | 同上 |
| `user_name` / `user_email` | 可省 | 只用于展示与日志脱敏 |

改完 `accounts.json` 不用重启，`POST /v1/admin/accounts/reload` 热加载即可；
多个账号就在数组里继续追加。

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/chat/completions` | OpenAI 协议，支持 `stream`、`tools`、`reasoning_effort` |
| POST | `/v1/messages` | Anthropic Messages 兼容，`x-api-key` 或 `Authorization` 均可 |
| POST | `/v1/responses` | OpenAI Responses 兼容 |
| GET | `/v1/models` | 模型清单（含 `display_name` 与 `capabilities`） |
| DELETE | `/v1/conversations/{id}` | 删除上游会话 |
| GET | `/health` | 账号池与能力快照（无需鉴权） |
| GET | `/v1/admin/accounts` | 账号明细 |
| POST | `/v1/admin/accounts/reload` | 重新从磁盘加载账号 |
| POST | `/v1/admin/accounts/{i}/disable` \| `/enable` \| `/reset` \| `/probe` | 单账号操作 |
| GET | `/v1/admin/quota` | 额度状态 |
| POST | `/v1/admin/quota/refresh` | 后台探测额度 |
| GET | `/v1/admin/calls` \| `/calls/summary` | 调用明细与汇总 |
| GET | `/v1/admin/samples` | 吞吐采样（供折线图） |
| GET | `/v1/admin/overview` | 一屏总览 + 告警 |
| POST | `/v1/admin/backup` | 立即备份；`/backups`、`/backups/restore`、`/backups/prune` |
| POST | `/mcp` | MCP JSON-RPC（`tools/list`、`tools/call`） |

除 `/health` 外都需要鉴权：`Authorization: Bearer $API_KEY` 或 `x-api-key: $API_KEY`。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"claude-opus4.6","messages":[{"role":"user","content":"你好"}]}'
```

## 模型

`GET /v1/models` 返回 31 个标识，命名如 `claude-opus4.6` / `gpt-5.6-sol` / `gemini-3.1pro` / `kimi-3` / `glm-5.2` / `grok-4.6` / `deepseek-v4pro`。每项带 `capabilities`：

```json
{"tools": true, "streaming": true, "reasoning": true, "search": true,
 "tools_bridged": true, "tool_frame": "planner"}
```

`tools_bridged: true` 表示工具调用经**文本协议桥**还原：客户端拿到的是真实可执行的 `tool_calls`，只是上游没有原生 tools 通道，调用是靠文本约定运回来的。

新增模型改 `app/model_registry.py` 的映射表即可，无需改动路由。

## 工具调用

上游没有 tools 参数，网关用**文本协议桥**把调用运回来——这不是「假调用」：客户端收到的 `tool_calls` 是网关解析出的结构化调用，可以直接执行；「桥」指的是传输方式（文本约定），不是结果。

不稳定点在模型侧：偶尔漏写 `<tool_call>` 标签，或把参数写成散文。网关对常见走样做了解析容错（见下），但格式完全跑偏时这一轮会退化成普通回答。


1. 把客户端声明的工具渲染成紧凑清单（名称 + 参数名/类型，描述按需截断），注入最后一条用户消息；
2. 模型按约定输出 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`，或漏掉标签直接输出裸 JSON；
3. 网关解析成标准 `tool_calls` 返回，流式下转成 `delta.tool_calls`。

默认开启（`.env.example` 里 `TOOL_BRIDGE=1`、`TOOL_BRIDGE_FRAME=planner`、`NOTION_CONTEXT_BLOCK=0`）。这三者的关系是实测出来的：

| 配置 | 结果 |
| --- | --- |
| 有 `context` 块 + agent 风格系统提示 | 模型以「我是 Notion 助手，无法扮演外部执行器」拒绝 |
| 去掉 `context` 块 | 稳定按 `<tool_call>` 输出，多步 agent 循环可跑 |
| `TOOL_BRIDGE=0` | 工具清单不下发，`tool_calls` 永远为空 |

容错做在三处：跨 chunk 拆分的裸 JSON（`{"na` + `me": ...`）、行内前缀（`调用：{"name" ...`）、名字白名单校验（避免把正文里的 JSON 示例误判成调用）。

## 推理强度

| 客户端 | 传法 | 映射 |
| --- | --- | --- |
| OpenAI | `"reasoning_effort": "low" \| "medium" \| "high"` | `minimal`→low、`xhigh`→high |
| Anthropic | `"thinking": {"type":"enabled","budget_tokens":N}` | <4096→low、<16384→medium、否则 high |
| Responses | `"reasoning": {"effort":"low"}` | 同上 |

最终写进上游 `config.reasoningEffort`。上游接受这个字段但不做校验（传 `ultra` 也不报错），实际效果取决于模型自身是否按档位调整；网关这层保证「如实传递 + 归一」，日志里每条请求都会打印实际档位。

## 多轮上下文

上游每次请求都是无状态的，且只认**最后一条** user 消息。网关把完整历史重放进最后一条消息：

```
以下是本次会话的历史记录，仅用于保持上下文连贯，不要复述：
用户：记住一个代号：蓝鲸-7391
助手：已记住

现在回答用户的新请求：
我的代号是什么？
```

## 图片

上游不收图片。`NOTION_IMAGE_MODE` 三个取值：

| 值 | 行为 |
| --- | --- |
| `vision`（默认） | 用本地视觉模型把图片转写成文字描述，再注入提示词 |
| `omit` | 丢弃图片，只保留文字 |
| `reject` | 遇到图片直接报错 |

视觉模型走 OpenAI 兼容接口，配置项见 `.env.example` 的 `VISION_*`。

## 主要配置

完整清单见 `.env.example`，常用项：

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `API_KEY` | 无 | 网关密钥，必填 |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | 监听地址 |
| `APP_MODE` | `standard` | `lite`（单轮）/ `standard`（完整上下文）/ `heavy`（会话管理 + SQLite） |
| `NOTION_ACCOUNTS` | 无 | 也可用 JSON 直接注入账号，优先于 `accounts.json` |
| `ACCOUNT_POOL_STRATEGY` | `fastest` | `fastest` / `round_robin` / `least_inflight` |
| `NOTION_UPSTREAM_MAX_CONCURRENCY` | `4` | 上游并发上限 |
| `NOTION_MAX_ATTEMPTS` | `6` | 单请求最多换几个账号 |
| `TOOL_BRIDGE` | `1` | 工具调用桥开关 |
| `NOTION_CONTEXT_BLOCK` | `0` | 是否发送 Notion `context` 块（工具场景必须 0） |
| `NOTION_SYSTEM_MODE` | `neutral` | 客户端系统提示的处理方式 |
| `NOTION_IMAGE_MODE` | `vision` | 图片策略 |
| `MCP_ENABLED` | `0` | 是否挂载 `/mcp` |
| `STATE_DB_PATH` | `state/notion2api.db` | SQLite 位置 |
| `BACKUP_AUTO` | `1` | 自动备份开关 |
| `UPSTREAM_SIDECAR` | `0` | 上游调用放到子进程（崩溃隔离） |
| `UPSTREAM_SIDECAR_WORKERS` | `4` | 子进程数 |

`.env` 需保持纯 ASCII（限流组件按 GBK 解码读取）。

启动后 `GET /health` 可确认账号池与能力状态；`/docs` 是 OpenAPI 文档。

## 运维

```bash
GET  /health                       # 账号池、工具桥、sidecar 状态
GET  /v1/admin/overview            # 账号 + 备份 + 告警
GET  /v1/admin/calls?limit=50      # 最近调用明细（模型、耗时、状态）
POST /v1/admin/backup              # 立即备份 accounts.json + SQLite
```

`UPSTREAM_SIDECAR=1` 会把上游调用放进独立子进程：curl_cffi 在 Windows 上有 native 崩溃（`0xC0000374`）风险，隔离后崩的只是那一次请求。

## 已知限制

1. **上游不稳定**：并发高时 Notion 会返回 502，网关如实转述为 `NOTION_502`，账号池会自动换号重试。
2. **工具调用依赖模型遵守文本约定**：模型偶尔漏写 `<tool_call>` 标签或把参数写成散文；网关对裸 JSON、跨 chunk 拆分、行内前缀都做了容错，但不保证 100%——格式完全跑偏时这一轮会退化成普通文本，重发即可。
3. **图片是转写而非原图**：视觉模型只保留文字描述，细节会丢失。
4. **多轮依赖历史重放**：超长对话会接近上游输入上限，网关会截断较早消息（`app/context_budget.py`）。
5. **额度不可精确观测**：Notion 没有公开剩余额度接口，`/v1/admin/quota` 只反映由真实调用结果推导出的状态。

## 目录结构

```
app/
  server.py              路由装配、鉴权、健康检查
  api/                   chat / anthropic_compat / responses_compat / accounts / ops / monitor / models
  account_pool.py        账号池调度、冷却、EWMA、额度
  notion_client.py       上游 HTTP 客户端（curl_cffi）
  conversation.py        transcript 构建、系统提示处理、多轮重放
  tool_bridge.py         工具清单渲染、调用解析、流式过滤器
  stream_parser.py       上游 SSE 解析、Markdown 修补
  upstream_pool.py       上游 sidecar 进程池
  upstream_sidecar.py    sidecar 子进程入口
  store.py               SQLite 持久化
  model_registry.py      模型映射
frontend/index.html      聊天界面
```

## 许可

见 [LICENSE](LICENSE)。
