# 部署与自动刷新

目标：**你不需要手动更新**。本页给出把刷新交给系统调度的具体做法。

---

## 1. 先决定"谁写、谁读"

| 角色 | 建议 |
|---|---|
| 真源 | 共享盘上的一个目录（多机可见、可人工编辑） |
| 各机镜像 + 索引 | 每台机器各自的 `~/.dsh-kb/`（**不要放到共享盘上**） |
| 写入 | 只在共享盘上发生；agent 只写 `_incoming/` |
| 刷新 | 每台机器定期跑 `dsh-kb refresh` |

一台机器刷新失败不影响另一台：镜像和索引都是**本机独立**的派生物。

---

## 2. 最小可用：一条命令

```bash
dsh-kb refresh
```

它等价于 `sync` + `index`，返回值可用于告警（非零表示失败）。**调度器只需要这一条。**

---

## 3. systemd timer（Linux）

`~/.config/systemd/user/dsh-kb-refresh.service`：

```ini
[Unit]
Description=dsh-kb: refresh mirror and index

[Service]
Type=oneshot
Environment=DSH_KB_HOME=%h/.dsh-kb
ExecStart=%h/.local/bin/dsh-kb refresh --quiet
# 共享盘没挂上时快速失败，不要卡住
TimeoutStartSec=300
```

`~/.config/systemd/user/dsh-kb-refresh.timer`：

```ini
[Unit]
Description=dsh-kb: refresh twice a day

[Timer]
OnCalendar=*-*-* 06:00:00
OnCalendar=*-*-* 18:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now dsh-kb-refresh.timer
systemctl --user list-timers dsh-kb-refresh.timer
```

> 注意：`OnCalendar` **一行只能写一个时刻**。写成
> `OnCalendar=06:00,18:00` 是无效语法——这是踩过的坑，每个时刻单独一行。

---

## 4. cron

```cron
# 每天 06:00 与 18:00 刷新（PATH 要写全，cron 环境很干净）
0 6,18 * * * DSH_KB_HOME=$HOME/.dsh-kb /home/you/.local/bin/dsh-kb refresh --quiet
```

---

## 5. Windows 计划任务（笔记本侧）

```powershell
$action  = New-ScheduledTaskAction -Execute "python.exe" `
           -Argument "-m dsh_kb refresh --quiet" -WorkingDirectory "F:\Projects\dsh-kb"
$trigger = New-ScheduledTaskTrigger -Daily -At 09:00
Register-ScheduledTask -TaskName "dsh-kb refresh" -Action $action -Trigger $trigger
```

如果不想配计划任务，也可以让 agent 在会话开始时自己跑一次 `refresh`
（把这条写进它的指令文件即可）——同样不需要你手动操作。

---

## 6. 让检索总是新鲜：自动保鲜

`search` 之前若镜像超过 `stale_minutes`（默认 15）未刷新，会**静默增量刷新**：

- 本地挂载源：0.0–0.1 s，基本无感；
- WebDAV 源：约 0.3 s；
- **刷新失败不影响检索**：照用现有镜像，并在结果里标 `"stale": true`。

所以即使调度器没跑起来，检索结果也不会离谱。

---

## 7. 自检

```bash
dsh-kb doctor          # 人类可读
dsh-kb doctor --json   # 可接监控
```

它会检查：源是否可达、镜像里**有没有本该排除的路径**、索引是否存在、home 是否可写。
把 `doctor` 挂进你的巡检（例如每天 06:00），比事后追查便宜得多。

---

## 8. WebDAV 作为兜底源

挂载不可用（网络盘掉线、容器里没有 mount 权限）时，可以直接用 WebDAV：

```bash
export DSH_KB_SOURCE=webdav
export DSH_KB_WEBDAV_ROOT=https://nas.example.com:5000/webdav/kb
export DSH_KB_WEBDAV_USER=me
export DSH_KB_WEBDAV_PASSWORD=...          # 或用 DSH_KB_WEBDAV_CREDENTIALS=<key=value 文件>
# 需要客户端证书（mTLS）时：
export DSH_KB_WEBDAV_CERT=/path/cert.pem DSH_KB_WEBDAV_KEY=/path/key.pem DSH_KB_WEBDAV_CA=/path/ca.pem
# 设备的证书 CN 与地址不同（很常见：连 IP、证书签的是域名）：
export DSH_KB_WEBDAV_CONNECT=192.0.2.10
```

`source=auto` 会在挂载不可用时自动退到 WebDAV，所以上面这些通常只要配一次。

---

<a id="samba--cifs-挂载-linux-与-windows"></a><a id="samba--cifs-mounts-linux-and-windows"></a>
## 9. Samba / CIFS 挂载

这是最常见的部署方式（个人 NAS + 一台常开主机 + 一台笔记本），也是**坑最多**的一种。
下面每条都来自真实环境。

### 9.1 Linux：fstab 示例

```fstab
# 整个共享（可读）
//192.0.2.10/kb        /mnt/nas/kb      cifs  credentials=/etc/samba/cred/kb.cred,uid=1000,gid=1000,file_mode=0644,dir_mode=0755,iocharset=utf8,vers=3.0,_netdev,nofail,x-systemd.automount  0 0
# 只给 agent 写的子目录（可写）
//192.0.2.10/kb/outbox /mnt/nas/kb-out  cifs  credentials=/etc/samba/cred/kb.cred,uid=1000,gid=1000,file_mode=0664,dir_mode=0775,iocharset=utf8,vers=3.0,_netdev,nofail,x-systemd.automount  0 0
```

值得注意的选项：

| 选项 | 为什么 |
|---|---|
| `_netdev,nofail,x-systemd.automount` | 开机时 NAS 没起来也不会卡住引导；访问时才挂 |
| `x-systemd.automount` | 产生一个 `autofs` 条目与真实挂载**同挂载点**——`dsh-kb` 会正确选择真实文件系统而不是触发器 |
| `credentials=<600 的文件>` | 不要把口令写进 fstab |
| `uid/gid/file_mode` | 让容器/服务账号能读；**显示权限不等于服务端权限**（见 9.2） |

`/etc/samba/cred/kb.cred` 格式（权限 `600`）：

```ini
username=youruser
password=yourpassword
```

### 9.2 只读视图：`rw` 挂载也可能写不进去

一个常见的安全做法是：**整份共享只读挂载，只把一个子目录可写挂载**。
于是会出现这样的组合：

```
//NAS/kb          → /mnt/nas/kb        （ro 视图）
//NAS/kb/outbox   → /mnt/nas/kb-out    （rw）
```

`dsh-kb` 会读 `/proc/mounts` 判断当前源是不是网络挂载、是不是 `ro`，
并在 `doctor` 里报出来；写操作会**提前失败并给出一句人话**，而不是抛 `PermissionError` 的栈。

> 还有一种更隐蔽的情况：挂载选项写着 `rw`，但**服务端账号**在该目录没有写权限。
> 这时只能实测——`doctor` 的 `source writable (_incoming)` 会真的写一个探针文件再删掉。
> 另外，**沙箱化的 agent 进程**即使挂载可写也写不进去（它被限制在自己的工作目录内），
> 所以这条检查是 `warn` 而不是 `fail`：读才是主业，写是可选项。

### 9.3 会话掉线：`Stale file handle`

网络存储会话丢失后，几乎所有系统调用都会返回 `ESTALE`，连 `rglob` 都会直接抛异常。
症状与处理：

```bash
ls /mnt/nas/kb          # → "Stale file handle"
sudo mount -a           # 重新挂载（需要 root）
ls /mnt/nas/kb          # 恢复正常
```

`dsh-kb` 把它翻译成可执行的句子：

```
source reachable   the share looks stale (ESTALE): the mount lost its session.
                   Remount it (`sudo mount -a`) or fall back with DSH_KB_SOURCE=webdav
```

把 `dsh-kb doctor` 挂进每日巡检，就能在"检索悄悄查不到东西"之前发现问题。

### 9.4 时间戳只有 1 秒：`--full`

实测消费级 NAS 的 `Last-Modified` 粒度是 **1 秒**。
增量同步用 `size + mtime` 判断变化，因此这种改动是**看不见**的：

> 同一秒内改完，且字节数不变。

Markdown 里这种情况不多（改一个字通常字节数也变），但确实存在（替换同长度词、
`2026` → `2027`）。需要时强制全量：

```bash
dsh-kb sync --full        # 忽略清单，全部重拉
```

容差本身可调（默认 `1.0` 秒，略小于一个刻度）：

```json
{ "mtime_tolerance": 1.0 }
```

### 9.5 Windows：盘符或 UNC

配置里三种写法都接受，`dsh-kb` 会归一化：

```json
{
  "source": "mount",
  "mount": "Z:\\kb"
}
```

```json
{
  "source": "mount",
  "mount": "\\\\192.0.2.10\\kb-1234567890\\kb"
}
```

（JSON 里反斜杠要转义；嫌麻烦就写正斜杠 `Z:/kb`，一样可用。）

注意 Windows 上的盘符映射**按用户会话**生效。计划任务或服务账号下可能看不到该盘符——
此时用 UNC 路径，或改用 `source=smb`。

### 9.6 不挂载，直连 Samba（`source=smb`）

挂载需要 root（或管理员），而 SMB 客户端不需要。当挂载不可用、或你不想为了读一个目录去动
`/etc/fstab` 时，用这条：

```bash
export DSH_KB_SOURCE=smb
export DSH_KB_SMB_SHARE="//192.0.2.10/kb-1234567890"
export DSH_KB_SMB_SUBDIR="kb"                          # 可选：共享内的子目录
export DSH_KB_SMB_CREDENTIALS=/etc/samba/cred/kb.cred  # smbclient 的 -A 文件（600）
dsh-kb sync
```

需要系统上有 `smbclient`（Debian/Ubuntu：`apt install smbclient`）。
它用**一次** `smbclient` 调用加 `recurse ON; ls` 列出整棵树，读取按文件 `get`，
写入按文件 `put`（并逐级 `mkdir`，因为 `smbclient` 的 `mkdir` 不建父目录）。

代价：每个文件一次进程 + 一次往返，比挂载慢得多。
实测一个小 bundle（4 个文件）：SMB 直连 **0.83 s**，本地挂载 **0.06 s**。
所以它能当兜底，不适合当主力。

### 9.7 `doctor` 的三档输出

```
  [ok] source usable                mount (cifs) //192.0.2.10/kb-1234567890
  [ok] network share detected       cifs at /mnt/nas/kb
  [ok] source reachable             listing succeeded
  [~~] source writable (_incoming)  permission denied ... — needed by `push`
  [ok] no excluded paths in mirror
```

| 标记 | 含义 | 影响退出码 |
|---|---|---|
| `[ok]` | 正常 | 否 |
| `[~~]` | 警告（例如写不可用） | **否**——读还能用，不该让整条巡检红掉 |
| `[!!]` | 失败（源不可达、镜像里有本该排除的路径） | 是（返回 1） |

`--json` 会把每项标成 `{"name", "status", "detail"}`，方便接监控。
