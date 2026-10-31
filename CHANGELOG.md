# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-10-31

### Added

- **Query expansion, confidence scoring and result merging** (`dsh_kb.expansion`),
  ported from a long-running deployment where they had never been committed.
  Search now reports *how much to trust* a result and, when a query is weak,
  spends a little more work instead of returning nothing:
  - **confidence** — a cross-encoder score on the top three pages, banded into
    `high` / `medium` / `low`, plus a `hint`.  A soft signal only: it never
    filters hits, because a hard threshold had to trade "reject 82% of
    out-of-domain" against "drop 24% of answerable".
  - **synonym expansion** — an optional domain dictionary maps the word the user
    typed onto the word the corpus uses.  Zero model calls, milliseconds.  Off
    unless `DSH_KB_DICT=1` and a table exists.
  - **query rewriting** — a local model rewrites a weak query into other
    phrasings, cached by query so a repeat costs nothing.  Any failure is an
    empty list; a search is never blocked by an unreachable model.
  - **page rerank** — score a candidate page by its best passage, which recovers
    pages whose relevant part is not the snippet the index returned.  Off by
    default; it costs one model call per passage.
- Search results now carry `confidence`, `hint`, `rerank_top1`, and — when the
  extra passes ran — `expand` / `rewrite` / `page_rerank` blocks.

### Fixed

- **An identifier lookup is never rewritten.** Porting the rewrite stage as-is
  reintroduced a real defect: `UNIQ-CLI-CANARY-88888` (zero hits) was rewritten
  into generic prose, which then matched unrelated pages, so a lookup that must
  return nothing started returning noise.  The lexical layer guarantees
  identifiers match exactly or not at all; a one-token query carrying a digit or
  a separator now skips the expansion stage.  Found by an existing test that
  failed the moment the stage was enabled.

### Changed

- The rewrite cache and synonym table live under `<home>/state/` and
  `<home>/dict/` and can be redirected with `DSH_KB_REWRITE_CACHE_DB` /
  `DSH_KB_DICT_PATH`.  No absolute path is built in.
- The test-suite is hermetic: query rewriting and its endpoint are pinned off in
  `tests/__init__.py`, so a laptop with a local model running and a bare CI
  runner produce the same results.
- Requires `handrolled>=0.1.1` (adds `DSH_KB_DEEP_KEY_NAME`, so a deployment can
  keep its deep-rerank key in an existing shared credentials file instead of
  duplicating the secret).

## [0.4.0] — 2026-10-31

### Changed

- **The reusable half moved out into [`handrolled`](https://github.com/LUgABeEEE/handrolled).**
  `text`, `webdav`, `mounts`, the record importers and the optional ranking
  layers had nothing to do with knowledge bases in particular: they are
  hand-rolled, standard-library-only replacements for things you would normally
  `pip install`. They now live in a package of their own, which dsh-kb depends on.
- **The dependency story changed, and the docs say so.** "No dependencies" became
  "one dependency, which has none": `pip install dsh-kb` now pulls
  `handrolled`, which is pure Python with no compiled extensions and an empty
  `Requires-Dist` of its own.  The tree still contains no compiled code, and the
  optional model-backed layers still import lazily and return `None` when absent.
- `bash examples/demo.sh` now says what is missing instead of failing with an
  ImportError when run from a bare checkout without the dependency installed.

### Removed

- `dsh_kb.text`, `dsh_kb.webdav`, `dsh_kb.mounts`, `dsh_kb.importer`,
  `dsh_kb.importers`, `dsh_kb.rerank`, `dsh_kb.deeprank`, `dsh_kb.embed_onnx`
  (now `handrolled.*`).  Their tests moved with them; 86 tests now live in the
  new package (287 remain here).

## [0.3.4] — 2026-10-30

### Added

- **All four MCP tool annotations** (`readOnlyHint`, `destructiveHint`,
  `idempotentHint`, `openWorldHint`) on `search`, `read`, `status` and `skills`.
  OpenAI's directory rejects a tool that omits any of them, and hosts cannot warn
  before a call they know nothing about.  `search` is declared **not** read-only
  on purpose: a stale index is refreshed first, which writes to the local mirror
  and index (never to the source).  Claiming otherwise would be exactly the
  claim-vs-code gap this project treats as a red flag elsewhere.

### Changed

- Standing MCP cost rises from ~421 to ~556 tokens (the annotations are 415 of
  the 2099 `tools/list` characters). Paid knowingly: a test asserts the schema
  budget so the number cannot drift silently, and the docs now carry the new one.
- The `skills` tool description no longer claims a specific skill count — the
  hard-coded "724 third-party skills" had gone stale.

## [0.3.3] — 2026-10-30

### Fixed

- **Two groups whose names slugify to the same file no longer overwrite each
  other.** Found in a real import of 2 972 memory records: `【常守规则` and
  `常守规则` both produced `常守规则.md`, so the second page silently replaced the
  first — two records never reached the knowledge base — and every re-import
  reported four changed pages because the two groups kept rewriting each other.
  A colliding group now gets a short hash of its full name appended
  (`常守规则-a1b2c3.md`), which is stable across runs.

## [0.3.2] — 2026-10-29

### Fixed

- **Front matter tags are a summary again.** The renderer accumulated every tag
  of every record on the page, including internal `src-*:`, `workspace:` and
  `qc:` markers — on a 57-record page that was a single 2.5 KB line. Tags are now
  ranked by frequency, internal markers are dropped (they stay on the record for
  traceability) and the list is capped at 10 plus the page's types.

## [0.3.1] — 2026-10-29

### Fixed

- **Imported pages are byte-deterministic now.** `import` stamped
  `generated.at` with the wall clock, so two runs a few seconds apart produced
  different bytes for every page — which meant re-importing an unchanged export
  rewrote all of them (`changed=20`, `written=20`) instead of reporting
  `same=20`. Pages now carry `generated: { by: <generator> }` and no timestamp;
  pass `--as-of <ISO8601>` when a page really should record a date.
- A group whose name already ends in `.md` no longer produces `name.md.md`.

Found by the review gate of a real migration: the content was verified lossless
(2960 records, all present, old tags preserved), and the only blocker was that
the renderer was not reproducible.

## [0.3.0] — 2026-10-29

### Added

- **`dsh-kb import`**: turn another tool's export into pages inside the bundle.
  Dry-run by default, deterministic, additive (it never deletes), and guarded —
  a zone must be named, `z0-public` needs `--allow-public`, and a target that the
  `never` list excludes aborts the whole run instead of silently skipping files.
  Documented in [docs/import.md](docs/import.md).
- **Six adapters**, each one module under `dsh_kb/importers/`: `records` (the
  neutral `records/v1` JSON), `jsonl`, `mem0`, `obsidian`, `notion` (Markdown &
  CSV export, page ids stripped from file names) and `basic-memory` (observations
  become records; a page without observations imports whole).  Adding another is
  one module plus one table entry.
- `--group-by first-tag|none|field:<attribute>`, `--title-prefix`,
  `--wikilinks strip|keep`, `--max-page-bytes`.  Re-importing the same input is a
  no-op; pages are compared byte for byte and only differences are written.
- 31 new tests (370 total).

### Notes

- Slugs are case-folded because the bundle is served to Windows over UNC, where
  `Docker.md` and `docker.md` are the same file.
- The adapters read with the standard library only — `import dsh_kb` still pulls
  in no third-party code, and the optional-arm test keeps it that way.

## [0.2.2] — 2026-10-29

### Added

- **Documentation for the supersession model.** `dsh-kb supersede-check` existed
  but appeared nowhere in the README, which made a first-class relation look like
  a private feature. The README now covers it: what `supersedes:` /
  `superseded_by:` mean, the four ways the relation goes wrong (one-way link,
  cycle, dangling target, two successors), and the fact that the same relation is
  a **gate on writes** — `push` refuses a whole batch when a proposal collides
  with a live page's title and declares neither `supersedes:` nor
  `not_superseding: "<why>"`, with `--allow-undeclared` as the explicit bypass.

### Changed

- `supersede-check --help` is in English, like every other command.

### Fixed

- `docs/deployment.md` listed four of the fourteen `doctor` checks; it now lists
  them all, including the optional local embedding probe (`DSH_KB_RAG_OLLAMA=-`
  disables it).

## [0.2.1] — 2026-10-25

### Fixed

- **The reranker gives its memory back when idle.** Resident cost after the
  first search is ~780 MB (session 395 MB, the 17 MB `tokenizer.json` expanded
  to 214 MB, plus a 170 MB inference arena) while a cold load takes 0.92 s, so
  keeping it forever is a bad trade. `DSH_KB_RERANK_IDLE_S` (default 600 s, `0`
  disables) releases the session and tokenizer after that much quiet time,
  taking the process back to ~90 MB; the next call rebuilds in ~0.92 s.
  Eviction runs on a daemon thread and shares one lock with inference, so a
  model is never dropped mid-call, and scores are bit-identical before and after
  a reload.
- Eviction has to call `malloc_trim(0)` explicitly: `gc.collect()` alone leaves
  the process at 436 MB because glibc keeps the freed arena.  Found the honest
  way — the first version "worked" and freed nothing.
- `DSH_KB_RERANK_LOG` is opt-in: the module no longer picks a log path itself.

> 0.2.0 shipped without this fix: it existed in a deployment but never reached
> the repository, which is exactly the drift this release closes.

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

- No deployment-specific identifiers are left in the tree: the WebDAV credential
  file no longer accepts one vendor's variable names, the remote HyDE writer is
  `DSH_KB_HYDE_PROVIDER=remote` (not a product name), the CLI no longer points at
  the author's private vector builder, and the parity test for that external tool
  is gone.
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
