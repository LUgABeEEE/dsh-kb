"""Use-case layer shared by the CLI and the MCP server.

Why this module exists: the MCP server must be able to refresh, search and read
without importing argparse plumbing and — critically — **without printing
anything**.  Putting the shared logic here also means the CLI's ``--json`` shape
and the tool results cannot drift apart.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from .config import INCOMING_DIR, Config, source_available
from .index import IndexStats, build, index_age, read, search, stats
from .mirror import (
    diff_entries,
    diff_mirror,
    load_manifest,
    manifest_age,
    manifest_problem,
    mirror_files,
    mirror_size,
    sync,
)
from .sources import SourceError, SourceUnavailable, open_source


def open_configured_source(cfg: Config):
    """Open the source named by the configuration, or raise ``SourceError``."""
    source, note = open_source(cfg)
    if source is None:
        raise SourceUnavailable(note)
    return source


def file_logger(cfg: Config):
    """An ``on_event`` callback that writes to the log file and nothing else."""

    def emit(message: str) -> None:
        try:
            cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}\n"
                )
        except OSError:
            pass

    return emit


def needs_rebuild(cfg: Config, result) -> bool:
    """只有**内容真的变了**才重建索引。

    此前 `refresh`/`ensure_fresh` 无条件调用 `build()` —— 哪怕同步到 0 个变更，
    也会把整个索引删掉重建（本库 3.4 MB / 1300+ 块）。`dsh-kb` 是"按需调用的补充"，
    没变化时的刷新应当只是一次目录走查（实测 0.07 s），而不是一次全量重写。
    """
    if not cfg.index_db.exists():
        return True
    return bool(getattr(result, "fetched", 0) or getattr(result, "removed", None))


def refresh(cfg: Config, *, prune: bool = False, full: bool = False, on_event=None):
    """Sync then rebuild the index.  The unit a scheduler should call."""
    emit = on_event or (lambda _m: None)
    source = open_configured_source(cfg)
    result = sync(cfg, source, prune=prune, full=full, on_event=emit)
    if needs_rebuild(cfg, result):
        built = build(cfg, on_event=emit)
    else:
        emit("no content change — index rebuild skipped")
        built = stats(cfg)
    return result, built


def ensure_fresh(cfg: Config, *, force: bool = False, on_event=None) -> str:
    """Refresh mirror + index when they are too old.  **Never raises.**

    Best effort by design: being unable to reach the share must not stop a
    search, which is the whole point of keeping a local mirror.  The returned
    string is short and human-readable, and is surfaced in results as ``note``.
    """
    emit = on_event or (lambda _m: None)
    age = manifest_age(cfg)
    if not (force or age is None or age > cfg.stale_minutes * 60):
        return "fresh"
    try:
        source = open_configured_source(cfg)
    except SourceError as exc:
        emit(f"auto-sync skipped: {exc}")
        return f"skipped ({exc})"
    try:
        result = sync(cfg, source, on_event=emit)
        if needs_rebuild(cfg, result):
            build(cfg, on_event=emit)
            return f"refreshed ({result.fetched} fetched)"
        return f"checked (no change, {result.fetched} fetched)"
    except Exception as exc:  # noqa: BLE001 - degrade to the existing mirror
        emit(f"auto-sync failed: {type(exc).__name__}: {exc}")
        return f"refresh failed ({type(exc).__name__})"


# --------------------------------------------------------------------------- #
# payload builders — one definition, used by `--json` and by the MCP tools
# --------------------------------------------------------------------------- #
def hit_dict(hit) -> dict:
    return {"path": hit.path, "line": hit.line, "score": hit.score,
            "snippet": hit.snippet, "zone": hit.zone}


def search_payload(
    cfg: Config,
    query: str,
    *,
    zone: str = "z01",
    limit: int | None = None,
    snippet_chars: int | None = None,
    note: str = "",
    deep: bool = False,
) -> dict:
    result = search(
        cfg, query, zone=zone, limit=limit, snippet_chars=snippet_chars,
        fresh_seconds=cfg.stale_minutes * 60, deep=deep,
    )
    payload = {
        "ok": bool(result.hits),
        "query": result.query,
        "tokens": result.tokens,
        "zones": list(result.zones),
        "count": len(result.hits),
        "relaxed": result.relaxed,
        "dropped": result.dropped,
        "stale": result.stale,
        "note": " | ".join(x for x in (result.note, note) if x),
        "hits": [hit_dict(h) for h in result.hits],
    }

    # The passes below only run when the first one was weak, and each one is
    # allowed to fail into "do nothing".  See dsh_kb.expansion for why.
    from . import expansion

    fresh_seconds = cfg.stale_minutes * 60
    confidence = expansion.confidence_fields(cfg, query, result.hits)
    target = limit or (len(result.hits) or 10)
    current_hits = list(result.hits)

    if expansion.needs_rewrite(query, result.hits, confidence):
        started = time.monotonic()

        # 1) synonym substitution — milliseconds, no model call
        dict_expanded = expansion.dict_variants(cfg, query)
        extra = expansion.search_many(
            cfg, dict_expanded, search=search, zone=zone, limit=limit,
            snippet_chars=snippet_chars, fresh_seconds=fresh_seconds, deep=deep)
        if extra:
            merged = expansion.merge_variants(list(result.hits), extra, target)
            payload["hits"] = [hit_dict(h) for h in merged]
            payload["count"] = len(merged)
            payload["expand"] = {
                "source": "dict", "variants": len(dict_expanded),
                "ms": round((time.monotonic() - started) * 1000),
            }
            current_hits = merged
            confidence = expansion.confidence_fields(cfg, query, merged)

        # 2) model rewriting — only when a model is configured to answer
        variants = expansion.rewrite_variants(cfg, query) if expansion.REWRITE_MODE != "off" else []
        extra = expansion.search_many(
            cfg, variants, search=search, zone=zone, limit=limit,
            snippet_chars=snippet_chars, fresh_seconds=fresh_seconds, deep=deep)
        if extra:
            merged = expansion.merge_variants(current_hits, extra, target)
            current_hits = merged
            payload["hits"] = [hit_dict(h) for h in merged]
            payload["count"] = len(merged)
            payload["rewrite"] = {
                "used": True, "model": expansion.REWRITE_MODEL,
                "variants": len(variants),
                "ms": round((time.monotonic() - started) * 1000),
            }
            confidence = expansion.confidence_fields(cfg, query, merged)

    if expansion.PAGE_RERANK == "always" or (
            expansion.PAGE_RERANK == "auto"
            and confidence.get("confidence") in ("low", "medium", "none")):
        started = time.monotonic()
        reranked, meta = expansion.page_rerank(cfg, query, current_hits, target)
        if meta:
            payload["hits"] = [hit_dict(h) for h in reranked]
            payload["count"] = len(reranked)
            payload["page_rerank"] = {**meta, "ms": round((time.monotonic() - started) * 1000)}
            confidence = expansion.confidence_fields(cfg, query, reranked)

    payload.update(confidence)
    return payload


def search_with_refresh(
    cfg: Config,
    query: str,
    *,
    zone: str = "z01",
    limit: int | None = None,
    snippet_chars: int | None = None,
    allow_refresh: bool = True,
    on_event=None,
    deep: bool = False,
) -> dict:
    """Search, refreshing first when the index looks stale.

    A failed refresh is reported in ``note`` but never prevents a result — that
    is what makes the mirror worth having.
    """
    payload = search_payload(cfg, query, zone=zone, limit=limit, snippet_chars=snippet_chars, deep=deep)
    if payload["stale"] and allow_refresh:
        note = ensure_fresh(cfg, on_event=on_event)
        payload = search_payload(cfg, query, zone=zone, limit=limit,
                                 snippet_chars=snippet_chars, note=note, deep=deep)
    return payload


def read_payload(cfg: Config, rel: str, *, start: int = 1, lines: int = 80) -> dict:
    result = read(cfg, rel, start=start, lines=lines)
    return {
        "ok": True,
        "path": result.path,
        "start": result.start,
        "end": result.end,
        "total": result.total,
        "lines": [{"n": result.start + i, "text": text} for i, text in enumerate(result.lines)],
    }


def status_payload(cfg: Config) -> dict:
    kind, location = source_available(cfg)
    files = mirror_files(cfg)
    idx: IndexStats = stats(cfg)
    return {
        "version": _version(),
        "home": str(cfg.home),
        "source": {"kind": kind, "location": location, "note": "" if kind else location},
        "mirror": {"path": str(cfg.mirror_dir), "files": len(files),
                   "bytes": mirror_size(cfg), "age_seconds": manifest_age(cfg)},
        "index": {"path": str(cfg.index_db), "pages": idx.pages, "chunks": idx.chunks,
                  "bytes": idx.bytes, "age_seconds": index_age(cfg)},
        "zones": [{"name": z.name, "index": z.index, "inject": z.inject, "mirror": z.mirror}
                  for z in cfg.zones],
        "never": list(cfg.never),
        "stale_minutes": cfg.stale_minutes,
        "mtime_tolerance": cfg.mtime_tolerance,
    }


def _version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- #
# diff: read-only health report
# --------------------------------------------------------------------------- #
#: 待转正提案躺多久算"积压"。7 天：提案本来就是隔几天批量处理，更短的阈值只会
#: 天天报警（那会训练人忽略这个检查）；超过一周没人动，基本就是被忘了。
INCOMING_STALE_SECONDS = 7 * 86400
#: 同步/编辑器留下的冲突副本命名约定（``x.conflict-20261018.md``）。
CONFLICT_MARK = ".conflict-"
#: 每组差异最多列几条样例。计数始终精确；样例只是为了让人一眼能动手 ——
#: 一次整镜像丢失会产出几千条，不能让它把 --json 撑爆或把人读版冲掉。
DIFF_SAMPLE_LIMIT = 10


class DiffUnavailable(RuntimeError):
    """本地状态读不出来（home/mirror 形状不对、manifest 坏了）—— diff 的硬失败。

    与"源不可达"刻意分开：源不可达还能退化成纯本地报告（退出码 10），而本地
    状态读不出来时连退化都没得退（退出码 11）。
    """


def _mirror_relative(cfg: Config) -> list[tuple[Path, str]]:
    """镜像里每个文件及其 bundle 相对路径（POSIX）。"""
    out = []
    for path in mirror_files(cfg):
        try:
            out.append((path, path.relative_to(cfg.mirror_dir).as_posix()))
        except ValueError:      # pragma: no cover - mirror_files() 只返回子树内的路径
            continue
    return out


def _is_markdown(rel: str) -> bool:
    """与 ``Config.index_allows`` 同一个判据：只有 ``.md`` 会进索引。"""
    return rel.endswith(".md")


def _split_by_kind(added, removed, changed) -> dict:
    """把一组变更按 Markdown / 其他文件分开 —— 各自计数 + 少量样例。

    为什么分开：远端一次图片/JSON 更新会产出几十条非 .md 差异，混在一起时
    "Markdown 到底变了没有"就看不见了（t7 审查的原话是"别让日常噪音淹没
    Markdown 的差异"）。计数是精确的；样例按 :data:`DIFF_SAMPLE_LIMIT` 截断，
    免得整镜像丢失这种几千条的场景把 ``--json`` 撑爆或把人读版冲掉。
    """
    out = {kind: {"counts": {}, "added": [], "removed": [], "changed": []}
           for kind in ("markdown", "other")}
    for name, items in (("added", added), ("removed", removed), ("changed", changed)):
        markdown = [item for item in items if _is_markdown(item["path"])]
        other = [item for item in items if not _is_markdown(item["path"])]
        for kind, subset in (("markdown", markdown), ("other", other)):
            out[kind]["counts"][name] = len(subset)
            out[kind][name] = subset[:DIFF_SAMPLE_LIMIT]
    return out


def _mirror_group(cfg: Config, known: dict, mirror_pairs, entries) -> dict:
    """本地镜像 vs 上次快照（源可用时再加上源清单）—— **全部文件**，不只 .md。

    这是 t7 盲点 B 的修正。原来的"内容差异"只比"远端 vs 快照"：删掉一个已经
    同步过的本地文件时，manifest 与远端都没变，于是各项计数全 0、退出码 0，
    而 ``sync`` 下一次立刻 ``fetched=1``。镜像这一侧必须单独看。

    对照面：快照（manifest）是"同步进来过哪些文件"的唯一记录，所以"本地缺"
    以它为准；"本地多出"则需要源清单才判得了（否则会把上次 sync 写到一半的
    文件误判成本地多出，见 ``mirror.diff_mirror``）。源不可用时仍然能查出
    "本地缺文件"和"本地大小不符"——这是本地状态，本来就不依赖远端。
    """
    expected = {rel: record for rel, record in known.items() if cfg.mirror_allows(rel)}
    local: dict[str, int | None] = {}
    for path, rel in mirror_pairs:
        if not cfg.mirror_allows(rel):
            continue
        try:
            local[rel] = path.stat().st_size
        except OSError:         # pragma: no cover - 只在文件刚被删/权限异常时命中
            local[rel] = None
    remote = None
    if entries is not None:
        remote = {e.rel for e in entries if not e.is_dir and cfg.mirror_allows(e.rel)}

    gaps = diff_mirror(expected, local, remote=remote)
    changes = gaps.as_dict()
    group = {
        "counts": gaps.counts,
        "remote_checked": gaps.remote_checked,
        "snapshot_files": len(expected),
        "mirror_files": len(local),
        "note": "" if remote is not None else (
            "源不可用：只能按上次快照判断「本地缺文件 / 本地文件大小不符」；"
            "「本地多出」这次算不了（分不清是本地多出来的，还是上次 sync "
            "写到一半就中断了 —— manifest 是最后才写的）"),
    }
    group.update(_split_by_kind(changes["added"], changes["removed"], changes["changed"]))
    return group


def _incoming_from_source(entries, prefix: str) -> dict:
    """按**源**上的文件列表统计待审提案（mtime 是远端写入时间，最可信）。"""
    files = [e for e in entries
             if not e.is_dir and e.rel.split("/", 1)[0] == prefix and "/" in e.rel]
    oldest = min(files, key=lambda e: e.mtime or 0.0) if files else None
    return {
        "basis": "source",
        "dir": prefix,
        "files": len(files),
        "oldest": oldest.rel if oldest else None,
        "oldest_seconds": (max(0.0, time.time() - oldest.mtime)
                           if oldest is not None and oldest.mtime else None),
    }


def _incoming_from_mirror(cfg: Config) -> dict:
    """源不可用时的退化统计：数本地镜像 ``_incoming/``。

    镜像文件的 mtime 是**同步时间**而不是提案的写入时间，所以这里的年龄只会
    偏小 —— 宁可少报积压，也不要把刚同步来的旧提案误报成"刚写的"。
    """
    folder = cfg.mirror_dir / INCOMING_DIR
    files: list[tuple[Path, str]] = []
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if path.is_file():
                files.append((path, f"{INCOMING_DIR}/{path.name}"))
    oldest: tuple[Path, str] | None = None
    oldest_seconds: float | None = None
    if files:
        try:
            oldest = min(files, key=lambda item: item[0].stat().st_mtime)
            oldest_seconds = max(0.0, time.time() - oldest[0].stat().st_mtime)
        except OSError:
            oldest, oldest_seconds = None, None
    return {
        "basis": "mirror",
        "dir": str(folder),
        "files": len(files),
        "oldest": oldest[1] if oldest else None,
        "oldest_seconds": oldest_seconds,
    }


def diff_payload(cfg: Config, *, incoming_threshold: float = INCOMING_STALE_SECONDS) -> dict:
    """只读体检：相对上次快照变了什么、有没有积压/漂移/冲突/越界。

    硬约束是**只读**：不写源、不写镜像、不触发 ``sync``、不重建索引。这个命令
    会被巡检脚本按天调用，任何写动作都会把一个"看一眼"变成风险源。

    五组，两个方向：

    * ``content`` —— 远端 vs 上次快照（源上变了什么）。**只报告，不算异常**：
      下一次 ``sync`` 就会收敛。里面 ``markdown`` / ``other`` 把 Markdown 与
      非 Markdown 的增/删/改分开计数（各自带少量样例），否则图片/JSON 的日常
      变动会把 .md 的差异淹掉。
    * ``mirror_gaps`` —— 本地镜像 vs 快照（本地少了/多了/大小不符）。``removed``
      与 ``changed`` 会进 ``attention``（索引里指向的页面可能已经读不到了），
      并且它覆盖 manifest/源清单里的**全部文件**——删掉一个非 .md 的已同步文件
      不再是"一切正常"。
    * ``index`` / ``incoming`` / ``conflicts`` / ``escapes`` —— 与原来一致。

    源不可用时降级：内容差异与冲突副本算不了（写进各自的 ``note``），镜像缺口
    里"本地缺文件/大小不符"照常算（本地状态不依赖远端），索引/积压/越界照常
    从本地算出来，并在 ``attention`` 里记一条 —— 不抛异常。

    返回的 dict 就是 ``--json`` 的输出，键名固定；``attention`` 为空即退出码 0。
    """
    attention: list[str] = []

    # ---- 本地状态先做形状检查：坏了就没法降级，直接 DiffUnavailable ----
    if cfg.home.exists() and not cfg.home.is_dir():
        raise DiffUnavailable(
            f"home is not a directory: {cfg.home} — fix DSH_KB_HOME / --home")
    if cfg.mirror_dir.exists() and not cfg.mirror_dir.is_dir():
        raise DiffUnavailable(
            f"mirror path is not a directory: {cfg.mirror_dir} — move it aside and "
            f"re-run `dsh-kb sync`")
    problem = manifest_problem(cfg)
    if problem:
        raise DiffUnavailable(f"snapshot manifest unusable: {problem}")

    # ---- 源：可用性单独表达，绝不在这里抛 ----
    kind, location = source_available(cfg)
    source_info = {"kind": kind, "location": location, "available": False, "note": ""}
    entries = None
    if not kind:
        source_info["note"] = f"no usable source configured ({location})"
    else:
        try:
            source, why = open_source(cfg)
        except Exception as exc:  # noqa: BLE001 - 构建源本身也可能失败（webdav 依赖等）
            source, why = None, f"{type(exc).__name__}: {exc}"
        if source is None:
            source_info["note"] = why
        else:
            try:
                entries = source.walk()
            except Exception as exc:  # noqa: BLE001 - 源不可达/挂载失效都归到这里
                source_info["note"] = str(exc) or f"{type(exc).__name__}"
            else:
                source_info["available"] = True

    manifest = load_manifest(cfg)
    known = manifest.get("files") or {}
    mirror_pairs = _mirror_relative(cfg)
    mirror_rel = [rel for _path, rel in mirror_pairs]

    # ---- 1. 内容差异（远端 vs 上次快照）----
    if entries is not None:
        remote_files = [e for e in entries if not e.is_dir]
        content = diff_entries(
            known, [e for e in remote_files if cfg.mirror_allows(e.rel)],
            mtime_tolerance=cfg.mtime_tolerance,
        ).as_dict()
        content["remote_files"] = len(remote_files)
        content["note"] = ""
    else:
        # 本地快照还在，但没有远端清单可对比 —— 空列表 + note，而不是假装"全没了"。
        content = {"added": [], "removed": [], "changed": [],
                   "counts": {"added": 0, "removed": 0, "changed": 0},
                   "remote_files": None,
                   "note": f"源不可用（{source_info['note']}），内容差异这次算不了；"
                           f"本地快照没动，所以不把任何文件报成「消失」"}
        # 源不可用算不算"需要注意"取决于本地有没有既有状态：已经同步过的机器
        # 看不见远端是真问题（10）；全新 home 只是还没配好/还没同步过，那是
        # doctor 的活（它返回 1），这里只在 note 里说明，别让 diff 一上手就红。
        if kind or known or mirror_pairs:
            attention.append("source-unavailable")
    # Markdown 与非 Markdown 分开计数：日常噪音（图片/JSON）不该淹没 .md 的差异。
    content.update(_split_by_kind(content["added"], content["removed"], content["changed"]))
    content.update(manifest=str(cfg.manifest_path),
                   manifest_age_seconds=manifest_age(cfg),
                   last_sync=manifest.get("lastSync"))

    # ---- 1b. 镜像副本（本地 vs 上次快照/源清单）----
    # t7 盲点 B：删掉一个已同步的本地文件时，上面的"内容差异"什么都看不见
    # （远端与快照都没变），而 sync 会立刻把它取回来。覆盖全部文件、不只 .md。
    mirror_gaps = _mirror_group(cfg, known, mirror_pairs, entries)
    if mirror_gaps["counts"]["removed"] or mirror_gaps["counts"]["changed"]:
        attention.append("mirror-gap")

    # ---- 2. 索引漂移 ----
    idx = stats(cfg)
    index_present = cfg.index_db.exists()
    mirror_pages = [rel for rel in mirror_rel if rel.endswith(".md") and cfg.index_allows(rel)]
    drift = idx.pages != len(mirror_pages) or (not index_present and bool(mirror_pages))
    index_group = {
        "present": index_present,
        "path": str(cfg.index_db),
        "age_seconds": index_age(cfg),
        "meta_paths": idx.pages,
        "mirror_pages": len(mirror_pages),
        "drift": drift,
        "note": "" if index_present else "索引还没建过",
    }
    if drift:
        attention.append("index-drift")

    # ---- 3. _incoming 积压 ----
    incoming = (_incoming_from_source(entries, INCOMING_DIR) if entries is not None
                else _incoming_from_mirror(cfg))
    incoming["threshold_seconds"] = incoming_threshold
    oldest_seconds = incoming.get("oldest_seconds")
    incoming["stale"] = oldest_seconds is not None and oldest_seconds > incoming_threshold
    incoming["note"] = "" if entries is not None else (
        "源不可用，积压按本地镜像估算（镜像 mtime 是同步时间，年龄只会偏小）")
    if incoming["stale"]:
        attention.append("incoming-backlog")

    # ---- 4. 源上的冲突副本 ----
    if entries is not None:
        conflict_files = sorted(
            e.rel for e in entries
            if not e.is_dir and CONFLICT_MARK in e.rel.rsplit("/", 1)[-1].lower())
        conflict_note = ""
    else:
        conflict_files, conflict_note = [], "源不可用，共享盘上的冲突副本这次数不了"
    conflicts = {"pattern": f"*{CONFLICT_MARK}*", "files": conflict_files,
                 "count": len(conflict_files), "note": conflict_note}
    if conflict_files:
        attention.append("conflict-copy")

    # ---- 5. 镜像越界（当前策略排除、却仍躺在镜像里的路径）----
    escapes = []
    for path, rel in mirror_pairs:
        if cfg.mirror_allows(rel):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        # in_manifest=True 说明它是"策略收紧前同步进来的"（manifest 只记允许过的
        # 路径），False 则更像手工拷贝进来的 —— 两种处置方式不一样。
        escapes.append({"path": rel, "bytes": size, "in_manifest": rel in known})
    escapes_group = {"files": escapes, "count": len(escapes), "note": ""}
    if escapes:
        attention.append("mirror-escape")

    return {
        "ok": not attention,
        "home": str(cfg.home),
        "source": source_info,
        "content": content,
        "mirror_gaps": mirror_gaps,
        "index": index_group,
        "incoming": incoming,
        "conflicts": conflicts,
        "escapes": escapes_group,
        "attention": attention,
    }


__all__ = [
    "CONFLICT_MARK",
    "DiffUnavailable",
    "INCOMING_STALE_SECONDS",
    "diff_payload",
    "ensure_fresh",
    "file_logger",
    "hit_dict",
    "open_configured_source",
    "read_payload",
    "refresh",
    "search_payload",
    "search_with_refresh",
    "status_payload",
]
