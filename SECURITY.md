# 安全策略

## 最重要的：账号数据即凭据

`accounts.json` 里存的是真实 Notion 账号的**会话令牌与 Cookie**，等价于账号密码。

- **不要**把它提交到任何仓库、Issue、PR 或聊天记录里
- **不要**把它贴进截图（仓库里已忽略 `image.png` 等截图文件，但请自行留意）
- 如果已经泄露：立刻在对应 Notion 账号里退出所有设备 / 重置密码，并从 `accounts.json` 删除该条

本项目通过 `.gitignore` 默认排除以下敏感文件：

```
.env / .env.*          # 含 API_KEY、邮箱与代理凭据
accounts*.json         # 账号池（含会话 Cookie）
account.edge.json      # 浏览器登录态
mailtm_cred.json       # 临时邮箱凭据
state/ backups/ output/ logs/   # 运行期数据库、日志与导出
```

## 提供者密钥

- 网关自身的 `API_KEY` 通过 `.env` 配置；请改成你自己的强随机值，不要沿用示例值
- 反向代理到公网时，务必在最外层再加一层鉴权（Basic Auth / VPN / 内网白名单）

## 已知的风险面

| 面 | 说明 |
| --- | --- |
| `/mcp`（可选） | 暴露本地文件读写与命令执行工具。`MCP_ALLOW_WRITE` / `MCP_ALLOW_EXEC` 默认关闭，开启后等同把 shell 交给调用方 |
| `/v1/admin/*` | 仅由 `API_KEY` 保护，可读取账号邮箱、额度与调用明细 |
| 代理池 | 若把别人的免费代理配进来，你的请求内容会经过第三方 |

## 报告漏洞

请通过仓库的私密渠道（Security Advisory）报告，不要开公开 Issue。请附上：

- 受影响版本 / 提交号
- 复现步骤与最小 PoC
- 影响范围（是否需要先拿到 `API_KEY`）

我们会尽快确认并在修复后公开致谢。

---

## 本机暴露面（2026-10-25 加固记录）

审计发现本机多个数据服务监听在 `0.0.0.0 / ::`，且 MySQL、Redis 的安装器还自带了
「允许任何来源」的入站规则。已做如下处理（**未改任何服务配置，随时可回滚**）：

| 端口 | 服务 | 处理 |
| --- | --- | --- |
| 6379 | Redis | 新增 Block（RemoteAddress=Internet）；原有 Allow 收窄为 Private + LocalSubnet |
| 3306 / 33060 | MySQL | 新增 Block（Internet） |
| 1433 | SQL Server | 新增 Block（Internet） |
| 5432 | PostgreSQL | 新增 Block（Internet） |
| 8848 | node（本地开发服务） | 新增 Block（Internet） |

效果：**本机与局域网照常访问，公网来源一律被防火墙丢弃**。
即使路由器做了端口映射，外部也连不进来。

Windows 上 Block 优先于 Allow，且 `RemoteAddress=Internet` 只匹配非本地子网地址，
所以不会影响 `192.168.x.x` 内的设备连接。

### 回滚

```powershell
Get-NetFirewallRule -DisplayName 'notion2api-block-*' | Remove-NetFirewallRule
Set-NetFirewallRule -DisplayName 'Redis' -Profile Any -RemoteAddress Any
```

### 未处理的系统服务

`135 / 139 / 445`（RPC / SMB）与 `5357 / 5040 / 7680` 是 Windows 自身服务，
无法改绑、也不建议随意封锁（会影响文件共享与系统功能）。若确认不需要局域网共享，
可在「网络和共享中心 → 高级共享设置」里关闭网络发现与文件共享。

### 若要真正改绑（更彻底）

防火墙是等价防护；如需服务本身只听回环：

```ini
# redis.windows.conf
bind 127.0.0.1 ::1
requirepass <强密码>

# my.ini / my.cnf
bind-address = 127.0.0.1

# postgresql.conf
listen_addresses = 'localhost'
```

SQL Server 改绑要在「SQL Server 配置管理器 → 协议 → TCP/IP → IPAll」里
清空 TCP Dynamic Ports 并把 Listen All 设为 No，改动较重，一般用防火墙即可。
