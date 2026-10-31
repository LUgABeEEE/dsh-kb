# 接入 agent 宿主

dsh-kb 对外有两种形态，选一个即可：

| 形态 | 何时用 | 代价 |
|---|---|---|
| **CLI + `--json`** | 任何能跑命令的 agent | 一次子进程 + 一次工具往返（约 30 ms） |
| **MCP 服务器（stdio）** | 宿主支持 MCP（推荐） | 常驻 schema ≈ 385 tokens，但每次调用少一层包装 |

两条路走的是**同一份实现**（`dsh_kb.service`），所以输出结构不会漂移。

---

## MCP 服务器

```bash
dsh-kb mcp                     # 或用 python -m dsh_kb mcp
```

它在 **stdin/stdout** 上说换行分隔的 JSON-RPC 2.0，暴露四个工具：

| 原始名 | 模型看到的名字（`serverName: kb`） | 作用 |
|---|---|---|
| `search` | `mcp__kb__search` | 排序检索，返回 `path` + 行号 + 片段 |
| `read` | `mcp__kb__read` | 读行窗口（`path` 来自 `search`） |
| `status` | `mcp__kb__status` | 镜像/索引新鲜度、源、条目数 |
| `skills` | `mcp__kb__skills` | 检索 `z3-skills` 资产区的 skill 总库（独立入口，进不了主子索引） |

**不暴露写操作**：写入经 `push` 落到 `_incoming/`，由人或定时任务转正。

---

## 配置：优先用命令行参数，而不是环境变量

有些宿主在拉起 MCP 子进程时会**清洗环境变量**（凭据形状的名字、过期的前缀）。
命令行参数不会被清洗，所以把配置放进 `args` 最稳：

```yaml
- id: mcp-kb
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: kb
    transport: stdio
    command: python3
    args:
      - -m
      - dsh_kb
      - mcp
      - --home
      - /srv/dsh-kb                 # 状态目录：镜像/索引/清单/日志/配置
      - --source
      - auto                        # auto | local | mount | smb | webdav
      - --mount
      - /mnt/nas/kb                 # source=mount 时用
    env:
      PYTHONPATH: /opt/dsh-kb/src   # 未 pip install 时
```

如果更愿意用环境变量（宿主不清洗时同样有效）：

```yaml
    env:
      DSH_KB_HOME: /srv/dsh-kb
      DSH_KB_SOURCE: mount
      DSH_KB_MOUNT: /mnt/nas/kb
```

**WebDAV 的密钥只从环境/凭据文件读**，永远不要写进配置：

```yaml
    env:
      DSH_KB_WEBDAV_ROOT: https://nas.example:5000/dav/kb
      DSH_KB_WEBDAV_USER: me
      DSH_KB_WEBDAV_PASSWORD: '***'
      # mTLS 场景
      DSH_KB_WEBDAV_CERT: /etc/dsh-kb/cert.pem
      DSH_KB_WEBDAV_KEY: /etc/dsh-kb/key.pem
      DSH_KB_WEBDAV_CA: /etc/dsh-kb/ca.pem
      DSH_KB_WEBDAV_CONNECT: 192.0.2.10   # 证书 CN 与地址不同时：连 IP、TLS 用 CN
```

---

## 其它宿主

任何 MCP 客户端都行，本质就是"起一个进程、说 JSON-RPC"：

```json
{
  "mcpServers": {
    "kb": {
      "command": "python3",
      "args": ["-m", "dsh_kb", "mcp", "--home", "/srv/dsh-kb",
               "--source", "mount", "--mount", "/mnt/nas/kb"],
      "env": { "PYTHONPATH": "/opt/dsh-kb/src" }
    }
  }
}
```

---

## 行为（接进去之后会发生什么）

| 场景 | 行为 |
|---|---|
| 索引过期 | `search` 前**静默增量刷新**（本地挂载 0.0–0.1 s；WebDAV 约 0.3 s） |
| 共享盘掉线 | 刷新失败**不影响检索**，结果里标 `stale: true` 并在 `note` 说明原因 |
| 敏感分区 | 默认不搜；`zone: "z2"` 才进，且该分区**从未镜像到本地** |
| 工具失败 | 作为**工具结果**返回（`isError: true` + 人话），不是协议错误 |

---

## 定位：**补充**，按需调用

这个知识库**不是**每轮都要读的前置上下文。三条硬约束写进了代码与配置：

1. **不自动注入**：`z1`/`z2` 的 `inject` 默认 `false`（`z0-public` 默认 `true`，仅作标记供注入方参考）；本仓库自己的部署还把它显式设成 `false`。没有任何机制会把
   知识库内容塞进系统提示或每轮上下文 —— 只能由模型显式调用 `search`/`read`/`status`。
2. **指令口径是"按需"**：`initialize.instructions` 明确写着 *only when you need … not
   routinely*。这条很重要 —— 早先的措辞是"search **before reading files**"，
   那等于每轮都催模型查一次，与"补充"的定位相反。**措辞即行为**，测试里有一条断言盯着它。
3. **空转不烧资源**：内容没变时刷新**不会重建索引**（此前无条件重写整库，本库 3.4 MB）。
   定时刷新 4 次/天、以及 `search` 前的自动刷新，绝大多数时候只是一次目录走查。

## 常驻成本（实测）

| 项 | 大小 |
|---|---|
| `tools/list` 全文（4 个工具及其参数 + annotations） | **2099 字符 ≈ 525 tokens**（实测；其中 415 字符是四个 readOnly/destructive/idempotent/openWorld 标注） |
| `initialize.instructions` | 125 字符 ≈ 31 tokens |
| **合计常驻** | **≈ 556 tokens / 每次模型调用** |
| 单次 `search` 返回 | 约 2.4 KB ≈ 1.2 K tokens，**与库大小无关** |

> 这 450 tokens 是**唯一**的固定开销（相对平均 23 万 tokens 的上下文约 0.19%）。
> 测试里有一条断言把它压在 1800 字符以内（`test_tool_schemas_stay_small`）——
> 加工具、加参数都是真成本，不是免费功能。
>
> 2026-10-14 从 3 个工具（1289 字符）增到 4 个：新增 `skills`，用于检索放在
> `z3-skills` 资产区的 skill 总库（724 条，**刻意不进主子索引**，所以需要自己的入口）。

---

## 验证

```bash
# 1) 进程能不能起来、协议对不对（不需要宿主）
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
| dsh-kb mcp

# 2) 真正的检索能不能命中（打真库）
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"search","arguments":{"query":"挂载"}}}' \
| dsh-kb mcp --home /srv/dsh-kb --source mount --mount /mnt/nas/kb

# 3) 与环境无关的自检
dsh-kb doctor --home /srv/dsh-kb
```

接进宿主之后，再确认两件事：

1. 模型工具列表里出现了 `mcp__kb__search` / `__read` / `__status`；
2. 实际调一次 `search`，返回体量在 ~2.4 KB 量级（若大得多，说明 `limit` 或 `snippet` 没生效）。

---

## 排障

| 症状 | 原因与处理 |
|---|---|
| 工具没出现在列表里 | 宿主可能**静默跳过**了这条 entry（配置解析失败）；查宿主日志与 `did not activate`，并手工跑上面的验证 ① |
| `search` 报 `index missing` | 还没建索引：`dsh-kb refresh --home <home>` |
| 结果一直是 `stale: true` | 源不可达：看 `note`（它会说明是掉线、只读、还是路径不对）；`dsh-kb doctor` 逐项给出结论 |
| stdout 里混进非 JSON | 宿主或包装脚本打印了东西——stdout 是协议通道，**只能有协议帧**；日志一律走 `--quiet` / 日志文件 |
| 中文搜不到 | 用 ≥2 个字；单个拉丁标识符是**精确匹配**（不做松弛），短语与中文才会松弛 |
