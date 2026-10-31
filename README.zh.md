# dsh-kb

**一个给 AI agent 用的个人知识库：真源放在共享盘，本机留镜像与索引，检索又快又能离线。**

[English](README.md) | 中文

---

## 给谁用、跑起来要花什么、两分钟怎么试

**给谁用**：知识本来就在一个 Markdown 目录里（共享盘、NAS 或普通文件夹）、
而 agent 每次会话都要重新翻一遍的人或小队。不是一个托管服务、不是多租户、
也不是笔记软件——一台机器、一个 bundle、一个主人。

**成本**：Python ≥ 3.10，**一个依赖**（[`handrolled`](https://github.com/LUgABeEEE/handrolled)，
它本身纯标准库、无编译扩展）——没有守护进程、没有服务端、不碰数据库服务、
不需要 root，一条命令一个进程。状态（镜像 + 索引）在 `~/.dsh-kb` 下，约是 bundle
体积的 1.9 倍；索引在本地，所以共享盘慢或离线只影响新鲜度，不影响可用性。

**两分钟试起来**（不需要共享盘、不需要写配置文件）：

```bash
git clone https://github.com/LUgABeEEE/dsh-kb && cd dsh-kb
bash examples/demo.sh          # 造一个临时 bundle、建索引、搜一次
```

---

## 它解决什么问题

个人知识（笔记、教训、决策、项目档案）通常散落在好几处：会话记录、记忆插件、
若干份 Markdown 报告、NAS 上的文档。于是每个 agent 会话都要：

1. **重新翻一遍**几个来源 → 慢；
2. 把大量原文**塞进上下文** → 贵（实测某部署平均每请求重发 **23 万 tokens** 上下文）；
3. 找不到就说"我没有相关信息"。

dsh-kb 把这件事收敛成一条路径：

```
共享盘上的一个 Markdown 目录  ←（唯一真源）
        │  增量拉取
        ▼
本机镜像 + 本机 FTS5 索引     ←（毫秒级、离线可用）
        │  两个窄工具
        ▼
agent 调 search / read 拿"路径 + 行号 + 240 字符片段"
```

**一次检索的返回体量与库的大小无关**——200 页和 10 000 页都是约 2.4 KB。
这是整个设计里最重要的一条性质：库长大了，单次检索不会变贵。

---

## 核心想法

| # | 想法 | 为什么 |
|---|---|---|
| 1 | **真源唯一，镜像派生** | 镜像永远不是权威，所以**不存在合并冲突**；刷新是 `(远端清单, manifest)` 的纯函数，可中断、可频繁跑 |
| 2 | **索引在本机，绝不放网络盘** | SQLite 在网络文件系统上的锁语义不可靠；索引是可丢弃的派生物，随时重建 |
| 3 | **隐私分区编译进代码** | 敏感分区**连拉都不拉**（不落盘）、不建索引、默认查询不返回——三层拦截，不靠约定 |
| 4 | **提案制写入** | agent 只写 `_incoming/`，转正由人或定时任务做；避免"同一主题十几条渐进状态 + 互相矛盾" |
| 5 | **说清有多确定，并抢救弱查询** | 每条结果都带一个 cross-encoder 给出的置信度；首轮太弱时再走一轮扩展（同义词 → 本地模型改写），而不是直接返回空。全部可选、有缓存，且只**增加**命中——永远不会挤掉词法已经找到的结果 |
| 6 | **先精度后召回** | 查询先用 `AND` 严格匹配；无命中才松弛成 `OR`，且**先按文档频率丢掉泛化词**，否则一个 `UNIQ` 就能把无关文档拖进来 |

细节与实测依据见 [docs/design.md](docs/design.md)。

---

## 安装

纯 Python，无编译扩展，一个依赖而它自己没有依赖
（[`handrolled`](https://github.com/LUgABeEEE/handrolled) —— 本项目的文本/WebDAV/挂载/导入层抽出去的那个纯标准库工具箱）：

```bash
pip install -e .          # 或直接 PYTHONPATH=src python3 -m dsh_kb
dsh-kb --version
```

---

## 快速开始

### 1. 准备一个 bundle（共享盘上的目录）

```
<share>/kb/
  index.md                 # 可选：渐进披露索引入口
  log.md                   # 可选：变更日志
  z0-public/               # 公开区
  z1-restricted/           # 受限区
  z2-sensitive/            # 敏感区（默认不镜像、不索引）
  _incoming/               # 提案区（agent 写这里）
  _meta/                   # 约定、清单、回执
```

### 2. 配置

配置放在 `~/.dsh-kb/config.json`（**刻意放在代码仓库之外**，公开 clone 不会带上你的路径）：

```json
{
  "source": "auto",
  "mount": "/mnt/nas/kb",
  "local_root": "/srv/kb-share",
  "webdav_root": "https://nas.example.com:5000/webdav/kb",
  "stale_minutes": 15
}
```

`source=auto` 会按 **挂载 → 本地目录 → Samba（smbclient）→ WebDAV** 的顺序挑第一个可用的：
本地文件系统遍历整树是毫秒级，WebDAV 每个目录一次 PROPFIND（约 55 ms），所以能用挂载就别用 WebDAV。
配置里的路径写 POSIX、Windows 盘符或 UNC 都行（`Z:\kb`、`\\nas\share\kb`、`/mnt/nas/kb` 都会归一化）。

> **Samba/CIFS 挂载要多加小心**（我们在真实环境里踩全了）：挂载点可能 `rw` 但实际只读、
> 会话掉线会让整棵树报 `ESTALE`、时间戳只有 **1 秒**粒度。这些都被显式处理了，
> 见 [docs/deployment.md](docs/deployment.md#samba--cifs-挂载-linux-与-windows)。

### 3. 拉取、建索引、检索

```bash
dsh-kb doctor                      # 先自检：源可达？镜像干净？索引在？
dsh-kb refresh                     # sync + index，定时任务调这个
dsh-kb search "挂载点失效" -n 5     # 默认只搜 z0/z1 + _meta/_incoming
dsh-kb search "UNIQ-000123" --json  # 机器可读
dsh-kb read z0-public/mount.md --start 1 --lines 40
dsh-kb push                        # 把本地 _incoming/ 的提案推上去
```

**只有 `sync` / `push` / `doctor` 会写共享盘**（其中 `doctor` 只写两个探针文件，
见下面的 CLI 表）。`diff` 也会**读**源清单（要报告远端变了什么），但它从不写共享盘、
镜像或索引——放定时任务里是安全的。共享盘关机、证书过期、笔记本不在线，检索照常可用。
（这一条是硬要求：否则"离线可用"就是假的。）

---

## 隐私分区

| 分区 | 镜像到本地 | 建索引 | 默认查询返回 |
|---|---|---|---|
| `z0-public` | ✅ | ✅ | ✅ |
| `z1-restricted` | ✅ | ✅ | ✅（`inject` 标记为 false，供注入方参考） |
| `z2-sensitive` | ❌ **连拉都不拉** | ❌ | ❌（需显式 `--zone z2`） |
| `z3-skills` | ✅ | ❌ **永不进索引**（需先注册） | ❌（另有入口 `dsh-kb skills`；2.3k 个 `SKILL.md` 会把检索淹没） |
| `_meta/`、`_incoming/` | ✅ | ✅ | ✅（是布局，不是隐私内容） |
| 命中 `never` 列表的路径 | ❌ | ❌ | ❌ |

`never` 默认含 `credentials`、`.credentials`、`secrets`、`id_rsa`、`密钥`、`密钥库`，
可在配置里增补。三层拦截（镜像 / 索引 / 查询）都有测试覆盖。

> **`z3-skills` 不是内置分区。** 要先把 `"index": false` 的这条注册进 `config.json`，
> 再把文件搬进去——未注册的顶层目录会被**照常镜像、照常索引**。
> 参考 [`examples/config.json`](examples/config.json)。

---

## CLI

| 命令 | 会写吗 | 说明 |
|---|---|---|
| `status` | — | 配置、源、镜像与索引状态、新鲜度 |
| `doctor` | ✔ 写探针 | 逐项自检并给可执行结论（含"镜像里有没有本该排除的路径"）；会写两个探针文件，见下 |
| `diff [--incoming-days DAYS] [--json]` | —（读源清单） | 只读体检：远端变化、本地镜像缺口、索引漂移、提案积压、冲突副本、越界 |
| `rag-check [--prune] [--force]` | 仅 `--prune` | 向量库 ↔ 索引一致性，页级 **+ 块级**；默认只报告（dry-run） |
| `sync [--prune] [--full] [--dry-run]` | ✔ | 增量拉取；`--prune` 才删除远端已消失的本地文件；`--full` 忽略清单全量重拉 |
| `index` | ✔（仅本地） | 重建索引 |
| `refresh [--prune] [--full]` | ✔ | sync + index（定时任务的调用单位） |
| `search <q> [-n] [--zone] [--snippet] [--refresh] [--no-auto-refresh] [--json]` | — | 排序检索 |
| `read <path> [--start] [--lines] [--json]` | — | 行窗口 |
| `skills search\|show\|stats` | — | 查 `z3-skills` 资产区里的 skill 总库 |
| `supersede-check [--json] [--proposal FILE]` | — | 取代关系体检（A1–A5）与 R1 提案闸门 |
| `import --from PATH [--format NAME] --prefix P [--zone Z] [--write]` | 仅 `--write` | 把别的工具的导出变成页面 —— 见 [docs/import.md](docs/import.md) |
| `push [files…] [--force] [--expect SHA256] [--log [N]] [--allow-undeclared] [--json]` | ✔ | 上传 `_incoming/` 提案（可能被取代闸门整批拦下） |
| `mcp` | — | 在 stdio 上跑 MCP 服务器（给 agent 宿主用） |

全局 flag 写在子命令前后都可以（`dsh-kb --json status` ≡ `dsh-kb status --json`）。

**不是所有"诊断"命令都是严格只读的。** 唯一的例外是 `doctor`：它会在 `home` 里
创建再删掉一个探针文件；源支持探写时，还会在共享盘的 `_incoming/` 下写一个探针
再删掉（用来回答"`push` 到底能不能用"）。探针文件会清理掉，除此之外不写任何东西。
**别把它塞进"必须严格只读"的位置。**

**取代关系是一等公民。** 页面可以在 front matter 里写 `supersedes: [...]` 或
`superseded_by: <path>`，`dsh-kb supersede-check` 报告这种关系坏掉的四种样子：单向
（对方不认账）、成环、指向不存在的页、一页被两个后继认领。同一条关系还是**写入口的
闸门**：提案标题与现行页撞车、又没写 `supersedes:` 或 `not_superseding: "<理由>"` 时，
`push` 会把整批拒绝（`--allow-undeclared` 是显式绕过）。在写入口问一句，比半年后
从库里清两份都像现行的近重复页便宜得多。

**退出码**：`diff` 与 `rag-check` 共用一套约定——**0** 干净；**10** 有需要人处理的
项；**11** 本地状态读不出来 / 库坏了。`rag-check --force` **不带** `--prune` 属于
**用法错误**，退出 **1**：它什么保护都越不过，所以直接拒绝，而不是假装强制成功。

### 两份报告到底比什么

`dsh-kb diff` 比**两对东西、两个方向**，两边都按文件类型拆成 `markdown` / `other`，
免得图片/JSON 的日常变动把 `.md` 的差异淹掉：

* *远端清单 vs 快照* → `content`：共享盘上变了什么。**只报告、永不告警**——下一次
  `sync` 就会收敛。
* *快照 vs 本地镜像* → `mirror_gaps`：**这台机器的副本**哪里不对。比对域是全部
  文件，不只 Markdown。快照记录过、磁盘上却没了的本地文件会进 `mirror-gap`
  （退出码 10）——以前它是隐形的：远端和快照都没变，而 `sync` 会立刻把它取回来。
  本地多出来的文件只计数、不告警：`sync --prune` 删不掉它们，得人来决定。

`dsh-kb rag-check` 同时按**页**（`path`）和**块**（`chunk_id`）比对向量库与索引。
块级口径抓的是"一页 5 块只嵌了 2 块"——这种形状下页级计数全是 0。块身份是
**内容寻址**的 `chunk_id`（见下节），所以"同一块被编辑过"和"同一块挪了位置"能
分开报；两侧身份不同（例如索引已迁、向量库还没迁）时退化成页级，并在 `note` 里
写清是哪一侧、该怎么补。

---

## 索引 schema（内容寻址）

`meta` 一行一块，主键是**内容寻址**的块身份：

```
chunk_id = sha256(path + 0x1f + text)[:16]      -- 由内容决定，重建后不变
rid      = (int(chunk_id, 16) >> 4) + 1         -- FTS5 只吃整数 rowid，故派生 60 bit
```

`chunks` 是 `contentless_delete=1` 的 FTS5 表（只存倒排索引），`vocab` 是
`fts5vocab(chunks,'row')`。检索走 `chunks join meta m on m.rid = chunks.rowid`。

这样设计换来两件事：向量库与嵌入缓存可以**按 `chunk_id`** 对齐（块没变就不必重嵌），
以及跨重建的稳定身份。代价是索引变大（实测 47.95 MB → 88.55 MB，+85%）。

**两套 schema 都能读。** 打开库时按 `meta` 有没有 `chunk_id` 列判断：旧库（`id` 列）
继续可用，检索照常；`meta` 两列都没有时报明确错误而不是猜一个 join。升级顺序
因此可以是"先代码、后数据"。

几个必须知道的 SQLite/FTS5 事实（都实测过）：

- **删块顺序**：先 `delete from chunks where rowid=?`，再 `delete from meta where rid=?`。
  反过来会留下"幽灵行"——FTS 仍能 MATCH、join 之后看不见、`integrity-check` 抓不到。
  按列名删（`delete from chunks where chunk_id=?`）是**硬报错**：FTS 表只有 `toks` 一列。
- **`optimize` 不回收空间**（删一半后文件一字未减），只有 `VACUUM` 回收（实测 -63%）。
- **只读打开也会生成 `-wal`/`-shm`** ⇒ 备份别只拷主文件。

## 配置项

| 键 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `source` | `DSH_KB_SOURCE` | `auto` | `auto` / `local` / `mount` / `smb` / `webdav` |
| `mount` | `DSH_KB_MOUNT` | — | 挂载点（Samba/CIFS、NFS 或普通目录） |
| `local_root` | `DSH_KB_LOCAL_ROOT` | — | 本地目录（测试与单机都靠它） |
| `smb_share` | `DSH_KB_SMB_SHARE` | — | 不挂载直连共享，如 `//host/share`（需 `smbclient`） |
| `webdav_root` | `DSH_KB_WEBDAV_ROOT` | — | WebDAV 集合 URL |
| `stale_minutes` | `DSH_KB_STALE_MINUTES` | `15` | 超过则 `search` 前静默刷新 |
| `mtime_tolerance` | `DSH_KB_MTIME_TOLERANCE` | `1.0` | 多大的 mtime 差仍算"未变" |
| `snippet_chars` | `DSH_KB_SNIPPET_CHARS` | `240` | 片段宽度 |
| `default_limit` | `DSH_KB_DEFAULT_LIMIT` | `8` | 命中条数 |
| `zones` / `never` | — | 见上文 | 分区策略与排除清单 |
| `rag.*` | `DSH_KB_RAG_*` | 全部关闭 | 可选语义臂 —— 见 [docs/semantic-arm.md](docs/semantic-arm.md) |

WebDAV 的密钥**只从环境或凭据文件读**，不写进 `config.json`：

```bash
export DSH_KB_WEBDAV_USER=me
export DSH_KB_WEBDAV_PASSWORD=...            # 或 DSH_KB_WEBDAV_CREDENTIALS=<key=value 文件>
export DSH_KB_WEBDAV_CERT=... DSH_KB_WEBDAV_KEY=... DSH_KB_WEBDAV_CA=...   # 需要 mTLS 时
export DSH_KB_WEBDAV_CONNECT=192.0.2.10      # 证书 CN 与地址不同时：连 IP、TLS 用 CN
```

---

## 检索行为

- **中文**：索引时保留原文并追加 CJK 相邻二元组，所以「挂载」能命中「重新挂载」。
- **英文与标识符**：原文照常入索引，`z2-sensitive`、`page-000123.md`、`fts5` 都能搜到。
- **两段式**：先 `AND`（精确），空结果才松弛为**只含有区分度词**的 `OR`，
  结果里带 `relaxed: true` 与 `dropped: [...]`，调用方知道"这次精度换成了召回"。
- **片段定位**优先整条查询字面量，其次词边界匹配——所以查 `mount -a` 报的是那一行，
  而不是 `/proc/mounts` 所在的行。

### 查询太弱的时候

搜不到东西时，它不会只回一个空。每条结果都带**置信度档位**——用 cross-encoder
给前 3 页打分，给出 `high` / `medium` / `low` 和一句 `hint`，告诉调用方该保留多少余地。
这是**软信号**：它从不筛结果。硬阈值实测要在「拒掉 82% 域外」与「丢掉 24% 可答」之间换，
对一个「按需查阅」的工具来说不划算。

置信度低或完全没命中时，按顺序跑两个可选阶段：

| 阶段 | 代价 | 做什么 |
|---|---|---|
| 同义词扩展（`DSH_KB_DICT=1` + 词表） | 毫秒级，零模型调用 | 把用户输入的说法换成语料里的说法 |
| 查询改写 | 一次本地模型调用，按查询缓存 | 让模型给出几种别的问法，再合并 |

两者都**只补空位**（`DSH_KB_MERGE=fill`，默认）——变体永远不会挤掉原查询找到的命中。
也测过纯 RRF 融合：50 题 gold 上把 page@5 从 0.580 拉到 0.440，所以默认是 `fill`，
`top2` 需要显式开。

**标识符永远不改写。** `UNIQ-…`、`page-000123.md`、`errno2` 要么精确命中要么不命中：
改写会把它变成泛化中文，而泛化中文会命中无关页面——本该返回「没有」的查找开始返回噪音。
单 token 且含数字/分隔符的查询直接跳过扩展阶段。

---

## 实测性能

单机、Linux、内置 SQLite（FTS5），合成语料（每页约 3.4 KB 中英混排）：

| 页数 | 语料 | 建索引 | 索引大小 | 膨胀 | 引擎检索（最坏） | 精确标识符 | CLI 端到端 |
|---|---|---|---|---|---|---|---|
| 1 000 | 3.3 MB | 0.38 s | 7.7 MB | 2.3× | — | — | 46 ms |
| 5 000 | 16.7 MB | 1.6 s | 33.1 MB | 2.0× | — | — | 59 ms |
| 10 000 | 33.4 MB | 3.2 s | 62.2 MB | **1.9×** | 29.6 ms | **0.2 ms** | **72 ms** |

- 端到端含 Python 解释器启动（约 30 ms）。
- 用法差异很大：精确标识符 0.2 ms，泛化关键词（命中近全库）29.6 ms。
- 一次检索返回约 **2.4 KB**，与库大小无关。

索引为什么能压到 1.9×：FTS5 表**不存正文**（`content=''`），正文单独一张 `meta` 表，
检索时按 rowid join 回来。早期把正文内联进 FTS 表是 **8.1×**。

---

## 与 agent 集成

**定位：补充，不是前置上下文。** 不自动注入、不做每轮检索；只在真需要往事、环境事实或历史时调用一次。

两种形态，同一份实现（`dsh_kb.service`），所以输出结构不会漂移：

| 形态 | 何时用 | 代价 |
|---|---|---|
| **CLI + `--json`** | 任何能跑命令的 agent | 一次子进程 + 一次工具往返（约 30 ms） |
| **MCP 服务器（stdio）** | 宿主支持 MCP（推荐） | 常驻 **≈ 556 tokens/次**（`tools/list` 2099 字符含 annotations + 指令 125 字符，实测），每次调用少一层包装 |

```bash
dsh-kb mcp            # 在 stdin/stdout 上讲 JSON-RPC，暴露 search / read / status / skills
```

接进 DSH 是一条 `insert`（其它宿主见 [docs/agent-integration.md](docs/agent-integration.md)）：

```yaml
- id: mcp-kb
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: kb
    transport: stdio
    command: python3
    args: [-m, dsh_kb, mcp, --home, /srv/dsh-kb, --source, auto, --mount, /mnt/nas/kb]
    env: { PYTHONPATH: /opt/dsh-kb/src }
```

> **为什么配置放 `args` 而不是 `env`**：有些宿主拉起 MCP 子进程时会清洗环境变量
> （凭据形状的名字、过期的前缀），命令行参数不会被清洗。
> 实测：把 `DSH_KB_*` 全部清空、只靠 `args`，对真实共享盘依然工作。

设计取舍（只留一个可选的 `deep` 开关、不给 `mode` 之类参数、不暴露写操作、失败以工具结果返回）
见 [docs/design.md](docs/design.md#与-agent-集成)。

---

## 可选语义臂（默认关闭）

上面讲的全是词法检索。在这之上还有一层**可选**能力：向量臂、cross-encoder、
远端模型 —— 它刻意**不在**「零依赖」的承诺范围内：

| 层 | 需要 | 默认 |
|---|---|---|
| 向量臂 | `numpy` + 一个向量库 + 嵌入端点 | 关 |
| cross-encoder 重排 | `onnxruntime`/`tokenizers` + 约 280 MB 模型 | 关 |
| ONNX 嵌入后端 | 约 610 MB 模型（A/B 实验） | 关 |
| 深池重排（`deep=true`） | 一个 OpenAI-compatible 端点 + key | 关 |

保持默认时，`import dsh_kb` 不碰任何第三方代码，检索结果与 0.1.0 逐字节一致。
开启前请先读 [docs/semantic-arm.md](docs/semantic-arm.md)：那里写了每层要什么、
模型从哪来，以及**哪一层会把页面正文发到机器外**（是 `deep`；如果你的库里
有"不愿粘进聊天框"的内容，就别开它）。

## 从别的工具导入

知识已经在别处时，`import` 把那份导出变成页面，而不是让你手工拷文件：

```bash
dsh-kb import --from ~/notes/obsidian --format obsidian --prefix notes        # 只看会改什么
dsh-kb import --from ~/notes/obsidian --format obsidian --prefix notes --write
dsh-kb refresh
```

适配 `mem0`、Obsidian、Notion 的 Markdown 导出、basic-memory、通用 JSON/JSONL，
以及 dsh-kb 自己的 `records/v1`。只增不删、确定性（再跑一次是空操作）、
没 `--allow-public` 就拒绝写 `z0-public`。详见 [docs/import.md](docs/import.md)。

## 状态

Alpha（0.5.0），已有 321 个测试（单元 + 端到端 + 传输层 + CLI 冒烟 + MCP，全部离线可跑）：

```bash
python3 -m unittest discover -s tests -t . -v
```

装了官方 MCP SDK 时，还会额外跑一条**互操作测试**（用真实客户端连这个手写服务器）；
没装则自动跳过。

## Roadmap

- [x] MCP 服务器（`mcp__kb__search` / `read` / `status`）
- [x] 语义回退：关键词零命中时才走本地嵌入检索 —— **可选、默认关闭**（[docs/semantic-arm.md](docs/semantic-arm.md)）
- [ ] 定时刷新（systemd timer / cron 示例）
- [ ] 中文嵌入质量评估，决定默认模型
- [ ] 只读 HTTP 浏览界面（可选）

## License

MIT，见 [LICENSE](LICENSE)。
