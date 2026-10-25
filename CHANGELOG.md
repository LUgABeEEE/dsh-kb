# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-10-25

### Added

- **Optional semantic arm**, four opt-in layers, every one of them off by
  default: the vector arm (`ragvec.py`), a cross-encoder reranker (`rerank.py`),
  an ONNX embedding backend kept for A/B work (`embed_onnx.py`), and a deep
  rerank that lets a remote model pick pages (`deeprank.py`).  Documented in
  [docs/semantic-arm.md](docs/semantic-arm.md), including the table of what each
  layer sends off the machine.
- `deep` on the MCP `search` tool (and `service.search`).  It only does anything
  when `rag.deep_enabled` is on; otherwise it is a no-op, not an error.
- Every `rag.*` key can be set from the environment as `DSH_KB_RAG_<FIELD>`.
  Before this, only five of them were mapped, so `rerank_*` and `deep_*`
  silently required a `config.json` — two machines could behave differently
  with no visible difference in configuration.
- Three test modules for the new code, and 327 tests total (was 308).

### Changed

- `rag.rerank_enabled` now defaults to **false**.  It used to be `true` while
  being gated by `rerank_max_shift=0`, which reads as "on" in any config dump
  even though nothing ran — the kind of default that lies.
- `rag.deep_model` defaults to empty instead of a vendor model name: the
  endpoint, the model and the key are all deployment decisions.
- The vector store defaults to `<home>/index/vectors.sqlite` rather than an
  absolute path on the author's machine.
- Optional paths have no built-in defaults any more: `DSH_KB_RERANK_ROOT`,
  `DSH_KB_EMBED_ONNX_ROOT`, `DSH_KB_ONNX_LIB` and `DSH_KB_NUMPY_PATH` are unset
  by default, and unset means the layer is off.  A clone that has never heard of
  these variables cannot accidentally load a model from someone else's disk.
- Deep-rerank configuration is environment-only: `DSH_KB_DEEP_API_KEY`,
  `DSH_KB_DEEP_CREDENTIALS`, `DSH_KB_DEEP_ENDPOINT`, `DSH_KB_DEEP_LOG`.  No
  vendor URL, key file or log path is baked into the code.

### Fixed

- HyDE no longer reads a fixed credentials file or picks a vendor endpoint from
  the shape of the key; it takes `DSH_KB_HYDE_ENDPOINT` / `_API_KEY` / `_MODEL`
  and does nothing when they are absent.
- `ragvec._numpy()` no longer falls back to a machine-specific directory.

## [0.1.0] — 2026-10-25

First public version. Written against a real deployment: a Markdown bundle on
a network share, mirrored to a Linux host and searched by an AI agent.

### Added

- **Bundle layout** with privacy zones (`z0-public`, `z1-restricted`,
  `z2-sensitive`), a layout area (`_meta`, `_incoming`) and a root index.
- **Local mirror** with a manifest-driven incremental sync: a file whose size
  and mtime are unchanged is never transferred. Opt-in `--prune`.
- **Local full-text index** on SQLite FTS5: CJK bigram expansion plus the
  original text, a contentless FTS table with a separate `meta` table, and a
  `fts5vocab` table used for selectivity.
- **Two-stage search**: strict `AND` for precision, relaxing to a selective
  `OR` only when the strict pass is empty.
- **Zone policy compiled into code**, enforced at three points: mirroring,
  indexing and querying. Sensitive zones are not even copied to disk.
- **Read sources**: a filesystem view (local directory, Samba/CIFS or NFS
  mount, Windows drive or UNC path) and a dependency-free generic WebDAV client
  (optional Basic auth and mutual TLS, with connect-host/SNI separation for
  appliances whose certificate CN differs from their address).
- **Unmounted SMB** (`source=smb`) through `smbclient`, for hosts where mounting
  needs root or a mount has gone stale. Parses the real `recurse ON; ls` output
  (verified against a live server, including names with spaces and the trailing
  `blocks of size …` line).
- **Network-mount awareness**: `/proc/mounts` inspection picks the real
  filesystem rather than its `autofs` trigger, reports `rw`/`ro`, and turns
  `ESTALE` / `EACCES` / `ENOENT` into actionable sentences. `sync --full` covers
  the one-second timestamp granularity of network shares.
- **CLI** (`dsh-kb status|doctor|sync|index|refresh|search|read|push|mcp`) with
  `--json` output, offline operation for everything except `sync`/`push`/
  `doctor`, best-effort auto-refresh before a search, and a three-level
  `doctor` report (`ok` / `~~` warning / `!!` failure).
- **MCP server** (`dsh-kb mcp`): newline-delimited JSON-RPC 2.0 on stdio,
  exposing `search` / `read` / `status`. Hand-rolled on the standard library to
  keep the zero-dependency promise; interoperability is asserted against the
  official Python SDK in the test-suite rather than assumed. Tool schemas
  measure ~1543 characters (~385 tokens) resident, with a test guarding that
  budget.
- **Use-case layer** (`dsh_kb.service`) shared by the CLI and the MCP server, so
  `--json` output and tool results cannot drift apart.
- **86→122 tests**: unit, end-to-end over a fixture bundle, transport tests with
  a stub `smbclient`, CLI smoke tests, MCP protocol/process tests, and an
  SDK interop test; no network needed.

### Fixed

- **`--home` did not relocate `config.json`.** The file was loaded from the
  environment-derived home *before* overrides were applied, so
  `dsh-kb --home X` silently ran on the built-in defaults — even though the
  CLI's own help calls `--home` the "state directory (mirror, index, manifest,
  logs, config)". Measured on a real deployment: three `never` path exclusions
  and every zone's `inject` flag were ignored, and the exclusions are a privacy
  guard. `home` is now resolved before the file is read; covered by regression
  tests that fail without the fix.

- **`WebdavSource.write` did not create parent collections.** A bare `PUT` into
  a missing collection is just `HTTP 500 errno 2 No such file or directory`, and
  because `MKCOL` does not recurse, every level must be created in turn. Found
  the honest way: a 217-file import into a freshly laid-out bundle failed 217
  times. Collections already handled are remembered per instance, so a bulk
  import does not repeat them.

- **Token and resource thrift** (the base is meant to be a supplement consulted on
  demand, not standing context): tool schemas trimmed 1543 → 1289 characters
  (~385 → ~322 tokens), the `instructions` line rewritten from "search before
  reading files" (which invited a lookup every turn) to an explicit on-demand
  stance, all zones set to `inject: false`, and an empty refresh **no longer
  rebuilds the index** — the scheduler's four daily runs and the pre-search
  auto-refresh are now a directory walk when nothing changed.
- **Unknown `config.json` keys are ignored and reported** instead of crashing the
  CLI with a `TypeError`; `doctor` lists them as a warning. Found the honest way:
  a stray note key took the whole command down.

### Known limitations

- Search is lexical only. A paraphrase with no shared wording returns nothing;
  a semantic fallback is planned but not implemented.
- Relaxation to a broader `OR` match is deliberately *not* applied to a single
  latin word or identifier — `UNIQ-…` is matched exactly or not at all.
- No write path for the model: writes go through `push` into `_incoming/` and
  are promoted to a zone by a human or a scheduled job.
- The WebDAV client speaks the subset of RFC 4918 this project needs
  (`PROPFIND`, `GET`, `PUT`, `MKCOL`, `DELETE`); no locking or `COPY`/`MOVE`.
- `source=smb` costs one process and one round trip per file; it is a fallback,
  not a replacement for a mount.
- The MCP server implements the three methods an agent needs; prompts,
  resources, sampling and subscriptions are not advertised.

### Added before the first public release

- **`skills` subcommand and MCP tool** for a curated skill library kept in a
  `z3-skills` asset zone.  Such a zone has to be registered in `config.json`
  (`index: false`) *before* anything is moved into it: `index_allows()` lets
  **unregistered top-level directories through**, so moving files alone does not
  keep them out of the index — and 2.3k `SKILL.md` files would otherwise drown
  knowledge-page retrieval.  Both entry points forward to the library's own
  `bin/find.py`, keeping tokenisation and scoring in a single implementation, and
  the library's location is resolved from configuration (`DSH_KB_SKILLS_LIB` →
  `<source root>/z3-skills/skills-library` → `~/kb/z3-skills/skills-library`) —
  never hardcoded.
- **`--prune` on the scheduled refresh**: the mirror is a derived cache, so files
  deleted upstream must not linger locally and get re-indexed.

### Changed before the first public release

- MCP resident cost 1289 → 1674 chars (~353 → ~450 tokens) with the fourth tool;
  the schema budget in tests moved 1500 → 1800 and the docs' cost table follows.
