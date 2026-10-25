# dsh-kb

**A personal knowledge base for AI agents: the source of truth lives on shared
storage, each machine keeps a mirror and an index, retrieval is fast and works
offline.**

English | [中文](README.zh.md)

---

## Who it is for, what it costs, and how to try it

**For** one person (or a small team) whose knowledge already lives in a Markdown
folder — on a network share, a NAS, or just a directory — and whose agent keeps
re-reading it. Not a hosted product, not a multi-tenant service, not a
note-taking UI: one machine, one bundle, one owner.

**What it costs**: Python ≥ 3.10 and **no dependencies** — no daemon, no server,
no database service, no root. One process per command. State (mirror + index)
lives under `~/.dsh-kb` and runs about 1.9× the size of the bundle; the index is
local, so a share that is offline or slow costs you freshness, not availability.

**Try it in two minutes** — no share and no config file needed:

```bash
git clone https://github.com/LUgABeEEE/dsh-kb && cd dsh-kb
bash examples/demo.sh          # builds a throwaway bundle, indexes it, searches it
```

---

## The problem

Personal knowledge — notes, lessons learned, decisions, project records —
usually ends up scattered across session transcripts, a memory plugin, a pile
of Markdown reports and a network share. So every agent session does the same
thing: it **re-reads several sources**, **stuffs raw text into the context
window** (one measured deployment re-sent ~230 000 tokens of context per
request), and when it still cannot find the answer it says so.

dsh-kb collapses that into one path:

```
one Markdown directory on shared storage     ← single source of truth
        │  incremental pull
        ▼
local mirror + local FTS5 index              ← milliseconds, works offline
        │  two narrow tools
        ▼
the agent asks for "path + line + 240-char snippet"
```

**The size of a result is independent of the size of the corpus** — 200 pages or
10 000 pages, a search returns roughly 2.4 KB. That property is the whole point:
growing the knowledge base does not make each lookup more expensive.

---

## Core ideas

| # | Idea | Why |
|---|---|---|
| 1 | **One source of truth, a derived mirror** | The mirror is never an authority, so **merge conflicts cannot happen**; refreshing is a pure function of `(remote listing, manifest)` — interruptible and cheap to run often |
| 2 | **The index is local and never on the share** | SQLite's locking is not reliable on network filesystems; the index is a disposable derivative that can always be rebuilt |
| 3 | **Privacy zones compiled into code** | A sensitive zone is **not even copied to disk**, is not indexed, and is not returned by a default query — three layers, no conventions required |
| 4 | **Proposal-based writes** | An agent only writes to `_incoming/`; promotion is done by a human or a scheduled job — this prevents "a dozen contradictory near-duplicates about one topic" |
| 5 | **Precision first, recall second** | A query starts as a strict `AND`; only when that is empty does it relax to `OR`, and then **generic terms are dropped first** — otherwise a single shared word such as `UNIQ` drags in unrelated documents |

Rationale and measurements: [docs/design.md](docs/design.md).

---

## Install

Standard library only, no third-party dependencies (Python ≥ 3.10):

```bash
pip install -e .          # or simply PYTHONPATH=src python3 -m dsh_kb
dsh-kb --version
```

---

## Quick start

### 1. Create a bundle (a directory on shared storage)

```
<share>/kb/
  index.md                 # optional: progressive-disclosure front door
  log.md                   # optional: change log
  z0-public/               # public zone
  z1-restricted/           # restricted zone
  z2-sensitive/            # sensitive zone (not mirrored, not indexed)
  _incoming/               # proposals (the only place an agent writes)
  _meta/                   # conventions, registry, receipts
```

### 2. Configure

Configuration lives in `~/.dsh-kb/config.json` — deliberately **outside any
checkout**, so a public clone never carries your paths or private data:

```json
{
  "source": "auto",
  "mount": "/mnt/nas/kb",
  "local_root": "/srv/kb-share",
  "webdav_root": "https://nas.example.com:5000/webdav/kb",
  "stale_minutes": 15
}
```

`source=auto` picks the first usable option in the order **mount → local
directory → Samba (smbclient) → WebDAV**. Walking a local filesystem tree takes
milliseconds; WebDAV costs one PROPFIND per directory (~55 ms on a consumer
NAS), so prefer a mount when you have one. Paths may be written POSIX-style, as
a Windows drive, or as a UNC path (`Z:\kb`, `\\nas\share\kb`, `/mnt/nas/kb`) —
all are normalised.

> **Samba/CIFS mounts need extra care** (we hit all of these in production): a
> mount can be `rw` and still read-only for the account; a dropped session makes
> the whole tree raise `ESTALE`; and timestamps have **one-second** granularity.
> All three are handled explicitly — see
> [docs/deployment.md](docs/deployment.md#samba--cifs-mounts-linux-and-windows).

### 3. Pull, index, search

```bash
dsh-kb doctor                       # check source reachability and mirror hygiene
dsh-kb refresh                      # sync + index — what a scheduler should call
dsh-kb search "mount stale" -n 5
dsh-kb search "UNIQ-000123" --json  # machine-readable
dsh-kb read z0-public/mount.md --start 1 --lines 40
dsh-kb push                         # upload local _incoming/ proposals
```

**Only `sync`, `push` and `doctor` write to the share** (and `doctor` writes just
two probe files — see the CLI table). `diff` also *reads* the source listing (to
report what changed remotely), but it never writes to the share, the mirror or
the index — it is safe to call from a timer. The share can be offline, a
certificate can expire, a laptop can be away — search still works. This is a
hard requirement: otherwise "offline capable" is a lie.

---

## Privacy model

| Zone | Mirrored locally | Indexed | Returned by default |
|---|---|---|---|
| `z0-public` | ✅ | ✅ | ✅ |
| `z1-restricted` | ✅ | ✅ | ✅ (`inject: false`, for injectors to respect) |
| `z2-sensitive` | ❌ **never copied** | ❌ | ❌ (requires explicit `--zone z2`) |
| `z3-skills` | ✅ | ❌ **never indexed** (once registered) | ❌ (own entry point: `dsh-kb skills`; 2.3k `SKILL.md` would drown retrieval) |
| `_meta/`, `_incoming/` | ✅ | ✅ | ✅ (layout, not private content) |
| paths matching `never` | ❌ | ❌ | ❌ |

`never` defaults to `credentials`, `.credentials`, `secrets`, `id_rsa`, `密钥`
("keys"), `密钥库` and is extendable. All three enforcement points are covered
by tests.

> **`z3-skills` is not built in.** Register it in `config.json` with
> `"index": false` *before* moving files into it — an unregistered top-level
> directory is mirrored **and** indexed. See
> [`examples/config.json`](examples/config.json).

---

## CLI

| Command | Writes? | Purpose |
|---|---|---|
| `status` | — | configuration, source, mirror and index state, freshness |
| `doctor` | ✔ probes | per-check diagnostics, including "does the mirror contain anything it should not" (writes two probe files, see below) |
| `diff [--incoming-days DAYS] [--json]` | — (reads the source) | read-only health report: remote changes, local mirror gaps, index drift, backlog, conflicts, escapes |
| `rag-check [--prune] [--force]` | only with `--prune` | vector store ↔ index consistency, page **and chunk** level; report-only (dry-run) by default |
| `sync [--prune] [--full] [--dry-run]` | ✔ | incremental pull; `--prune` is opt-in; `--full` ignores the manifest |
| `index` | ✔ (local) | rebuild the index |
| `refresh [--prune] [--full]` | ✔ | sync + index |
| `search <q> [-n] [--zone] [--snippet] [--refresh] [--no-auto-refresh] [--json]` | — | ranked search |
| `read <path> [--start] [--lines] [--json]` | — | line window |
| `skills search\|show\|stats` | — | query the skill library in the `z3-skills` asset zone |
| `push [files…] [--force] [--expect SHA256] [--log [N]] [--json]` | ✔ | upload proposals |
| `mcp` | — | run the MCP server on stdio (for agent hosts) |

Shared flags are accepted before or after the subcommand.

**Not everything labelled "diagnostic" is read-only.** `doctor` is the one
exception: it creates and deletes a probe file in `home`, and — when the source
exposes it — writes and deletes a probe under `_incoming/` on the share, to
answer "can `push` work at all?". The probes are removed again; nothing else is
written. Don't wire `doctor` into anything that must be strictly read-only.

**Exit codes.** `diff` and `rag-check` share one contract: **0** clean, **10**
something needs a human, **11** local state is unreadable / the store is broken.
`rag-check --force` without `--prune` is a *usage* error and exits **1** — it
would override nothing, so it refuses rather than pretending it forced
something.

### What the two reports actually compare

`dsh-kb diff` compares **two pairs, in two directions**, and splits both by file
kind (`markdown` / `other`) so image or JSON churn can't drown the `.md` diff:

* *remote listing vs manifest* → `content`: what changed on the share. **Report
  only, never an alarm** — the next `sync` converges.
* *manifest vs local mirror* → `mirror_gaps`: what is wrong with **this
  machine's copy**. Covers every file, not just Markdown. A local file that the
  snapshot recorded and the disk lost is `mirror-gap` (exit 10) — it used to be
  invisible, because remote and manifest were both unchanged while `sync` would
  re-fetch it. Extra local files are counted but not alarmed: `sync --prune`
  cannot delete them, so a human has to decide.

`dsh-kb rag-check` compares the vector store against the index at **page**
(`path`) and **chunk** (`path` + `startline`) granularity. The chunk view is
what catches "one page, five chunks, only two embedded" — page-level counters
stay at 0 for that shape. On an older store whose tables have no `startline`
column the comparison degrades to page-level and says so in `note`.

---

## Configuration reference

| Key | Env | Default | Meaning |
|---|---|---|---|
| `source` | `DSH_KB_SOURCE` | `auto` | `auto` / `local` / `mount` / `smb` / `webdav` |
| `mount` | `DSH_KB_MOUNT` | — | mount point (Samba/CIFS, NFS, or a plain directory) |
| `local_root` | `DSH_KB_LOCAL_ROOT` | — | plain directory |
| `smb_share` | `DSH_KB_SMB_SHARE` | — | unmounted Samba share, e.g. `//host/share` (needs `smbclient`) |
| `webdav_root` | `DSH_KB_WEBDAV_ROOT` | — | WebDAV collection URL |
| `stale_minutes` | `DSH_KB_STALE_MINUTES` | `15` | auto-refresh threshold |
| `mtime_tolerance` | `DSH_KB_MTIME_TOLERANCE` | `1.0` | mtime delta still counted as "unchanged" |
| `snippet_chars` | `DSH_KB_SNIPPET_CHARS` | `240` | snippet width |
| `default_limit` | `DSH_KB_DEFAULT_LIMIT` | `8` | hits per query |
| `zones` / `never` | — | see above | policy |

WebDAV secrets are read **only** from the environment or a credentials file,
never from `config.json`:

```bash
export DSH_KB_WEBDAV_USER=me
export DSH_KB_WEBDAV_PASSWORD=...            # or DSH_KB_WEBDAV_CREDENTIALS=<key=value file>
export DSH_KB_WEBDAV_CERT=... DSH_KB_WEBDAV_KEY=... DSH_KB_WEBDAV_CA=...   # mutual TLS
export DSH_KB_WEBDAV_CONNECT=192.0.2.10      # connect to an IP, verify TLS against the CN
```

---

## Search behaviour

- **CJK**: indexing keeps the original text *and* appends adjacent character
  bigrams, so `挂载` matches `重新挂载` (a 2-character query, which a plain
  `unicode61` tokenizer cannot do).
- **Latin and identifiers**: the original text is indexed verbatim, so
  `z2-sensitive`, `page-000123.md` and `fts5` all match.
- **Two passes**: strict `AND` first, then a relaxed `OR` restricted to
  *selective* tokens. Results carry `relaxed: true` and `dropped: [...]` so the
  caller knows precision was traded for recall.
- **Snippet location** prefers the whole query as a literal, then word-boundary
  token hits — so searching `mount -a` reports that line, not the line holding
  `/proc/mounts`.

---

## Measured performance

Single Linux host, bundled SQLite with FTS5, synthetic corpus (~3.4 KB per page,
mixed Chinese and English):

| Pages | Corpus | Index build | Index size | Blow-up | Engine (worst case) | Exact identifier | CLI end-to-end |
|---|---|---|---|---|---|---|---|
| 1 000 | 3.3 MB | 0.38 s | 7.7 MB | 2.3× | — | — | 46 ms |
| 5 000 | 16.7 MB | 1.6 s | 33.1 MB | 2.0× | — | — | 59 ms |
| 10 000 | 33.4 MB | 3.2 s | 62.2 MB | **1.9×** | 29.6 ms | **0.2 ms** | **72 ms** |

- End-to-end includes ~30 ms of Python interpreter start-up.
- The spread is large: an exact identifier takes 0.2 ms, a generic keyword that
  matches nearly everything takes 29.6 ms.
- A search returns about **2.4 KB**, independent of corpus size.

The index stays near 1.9× because the FTS5 table stores **no content**
(`content=''`) and the text lives once in a separate `meta` table, joined back
by rowid. Inlining the text into FTS5 measured **8.1×**.

---

## Agent integration

**Positioning: a supplement, not standing context.** Nothing is auto-injected and nothing is searched per turn — it is consulted once, when past decisions, environment facts or history are actually needed.

Two shapes, one implementation (`dsh_kb.service`), so their output cannot drift:

| Shape | When | Cost |
|---|---|---|
| **CLI + `--json`** | any agent that can run a command | one subprocess and one tool round-trip (~30 ms) |
| **MCP server (stdio)** | the host speaks MCP (recommended) | **~420 tokens** standing (1559-char `tools/list` + 125-char instructions, measured), one less layer per call |

```bash
dsh-kb mcp            # JSON-RPC on stdin/stdout, exposing search / read / status / skills
```

Wiring it into DSH is one `insert` (other hosts:
[docs/agent-integration.md](docs/agent-integration.md)):

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

> **Why the configuration goes in `args`, not `env`**: some hosts scrub the
> environment of the MCP child process (credential-shaped names, stale
> prefixes). Command-line arguments are never scrubbed. Measured: with every
> `DSH_KB_*` variable removed and only `args` set, it still works against a real
> share.

Design choices — no `mode`/`rerank` knobs, no write tools, failures returned as
tool results — are in [docs/design.md](docs/design.md#agent-integration).

---

## Status

Alpha (0.1.0) with 308 tests (unit, end-to-end, transport, CLI smoke and MCP — all offline):

```bash
python3 -m unittest discover -s tests -t . -v
```

With the official MCP SDK installed, one extra **interop test** runs (a real
client against this hand-rolled server); otherwise it skips.

## Roadmap

- [x] MCP server (`mcp__kb__search` / `read` / `status` / `skills`)
- [ ] Semantic fallback, used only when keyword search returns nothing (**no vector service**)
- [ ] Scheduled refresh (systemd timer / cron examples)
- [ ] Chinese embedding evaluation to pick a default model
- [ ] Optional read-only HTTP viewer

## License

MIT — see [LICENSE](LICENSE).
