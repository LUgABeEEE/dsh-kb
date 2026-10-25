"""Vector store <-> index consistency: report drift, prune orphan vectors.

Why this module exists
----------------------
The production vector builder (``kbindex.py``, which lives outside this repo)
only ever runs ``insert or replace`` — it never deletes.  When a page is removed
or re-named its rows stay in ``vectors.sqlite`` forever, and retrieval reads the
whole table without checking that the page still exists, so answers can quote
pages that are gone.  ``check`` makes that drift visible, ``prune`` removes the
orphan half of it.

Page-level and chunk-level
--------------------------
``check`` reports **two granularities**, because "the page has some vectors" is
not the same as "every chunk of the page has a vector".  A page split into five
chunks with vectors for only two of them used to look perfectly healthy: the
page-level ``path`` sets matched, so ``missing`` stayed at 0 while retrieval
silently skipped three fifths of the page (measured on the t7 audit,
2026-10-19).

The chunk identity is the content-addressed ``chunk_id`` when both databases
have it — that is the CAS schema, where a page's block keeps its key across a
rebuild and the vector store is keyed by the same value.  On a pre-CAS database
the comparison falls back to ``(path, startline)``, the only identity both
stores share there, and ``note`` says which one was used.  ``missing``/``orphan``
keep their historical **page-level** meaning (the CLI and the docs quote them);
``chunks_missing`` / ``chunks_orphan`` add the chunk-level view.  On a store
whose tables have neither identity the comparison degrades to page-level and
says so in ``note`` instead of guessing or crashing.

Both schemas are read on purpose: the upgrade order is "code first, then data",
so between the two steps the index is still the old one, and a maintenance
script that only understood the new layout would be useless exactly when it is
needed.

Two hard rules
--------------
* ``check`` is strictly read-only: both databases are opened through a
  ``file:...?mode=ro`` URI, so nothing is created and nothing is written — not
  even for a broken store.  That makes it safe to run from a timer at any
  moment, including while a vector build is in flight.
* ``prune`` is dry-run by default and, when it does delete, deletes **only**
  rows whose page no longer exists in the index.  ``kb.sqlite`` is opened
  read-only on every path here: vectors can be regenerated from the mirror, the
  index cannot be regenerated from vectors.  Note that this page-level rule means
  ``prune`` does **not** clean a chunk orphan whose page still exists — ``check``
  reports those, and the fix is re-embedding the page, not deletion.

``removed`` counts rows that were deleted, or — when ``dry_run=True`` — rows
that *would* be deleted.  ``check`` is the richer report (counts + sample), so a
caller can decide before ever calling ``prune`` for real.

``prune`` and the ownership of the two databases
-----------------------------------------------
``rag.vectors_db`` defaults to a **machine-wide** path while ``home`` is whatever
the caller passed, so "temporary home + default rag config" (a test, a CI job, a
fresh checkout, an operator's probe) used to be enough to delete a *production*
vector store using a one-page index — measured on 2026-10-19: 5,521 rows gone.
``prune`` therefore refuses before it deletes: see the three guards documented on
:func:`prune` (ownership, empty index, bulk delete).
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

from .config import Config

#: How many orphan pages ``check`` lists in ``orphan_sample``.  The count is
#: exact; the sample is only there to make the report actionable.
SAMPLE_LIMIT = 10

#: Columns ``check`` expects on the vector table.  Anything else is reported as
#: an unusable store instead of being silently skipped.  Two layouts are
#: accepted: the pre-CAS one (``id``) and the content-addressed one
#: (``chunk_id``, plus ``dim`` instead of the ``hash`` column).  The ``id``
#: spelling is quoted in the error message because that is the one an operator
#: upgrading an older machine is most likely to be looking at.
VECTOR_COLUMNS = ("id", "path", "startline", "text", "hash", "model", "vec")

#: ``meta``/``vectors`` key columns by schema.  ``chunk_id`` is the
#: content-addressed identity and the one this module compares by when both
#: sides have it; ``id``/``startline`` is the pre-CAS fallback.
CHUNK_ID = "chunk_id"
LEGACY_ID = "id"

#: 待删比例超过这个值就认为"这次 prune 很可能不是在同一个知识库里操作"。
#: 50% 是有意宽松的：真的删掉一半页面也会命中，所以它只拦**没带 --force** 的调用
#: —— 拦下来最多多敲一个 --force，放过去就是删空一个生产库。
BULK_DELETE_LIMIT = 0.5

#: 拒绝原因码。脚本按这个字段判态，不要按 note 的文案做匹配。
REFUSED_EXTERNAL = "external-vectors-db"   # 向量库不在 home 之下且未显式放行
REFUSED_EMPTY_HOME = "empty-home"          # 索引不存在/为空，而向量库非空
REFUSED_BULK = "bulk-delete"               # 待删比例 > BULK_DELETE_LIMIT 且没带 --force
REFUSED_INDEX = "index-unreadable"         # 索引读不出来（未知的对照面）
FAILED_DELETE = "delete-failed"            # 允许删、但 DELETE 本身失败


def vectors_db(cfg: Config) -> Path:
    """Path of the vector store described by ``cfg`` (never created here)."""
    return Path(str(cfg.rag.vectors_db)).expanduser()


def enabled(cfg: Config) -> bool:
    """Whether the RAG/vector side counts as present for this machine.

    The default policy is "the file decides": a machine that never built vectors
    is *absent*, not broken.  An explicit ``rag.enabled`` (config file or
    ``DSH_KB_RAG_ENABLED``) overrides the probe, so an operator can silence a
    doctor line on purpose.
    """
    flag = cfg.rag.enabled
    return vectors_db(cfg).is_file() if flag is None else bool(flag)


def check(cfg: Config) -> dict:
    """Compare the vector store against the index.  Writes nothing.

    Returns a dict with exactly these keys:

    ``ok``             no problem found at all
    ``chunks``         rows in ``meta`` (the index side)
    ``vectors``        rows in ``vectors``
    ``missing``        pages in the index that have *no* vector row (page-level)
    ``orphan``         pages in the vector store that are no longer in the index
                       (page-level)
    ``bad_path``       vector rows whose ``path`` is NULL, empty or not text
    ``model_mismatch`` vector rows not built with the configured model
    ``dim_mismatch``   vector rows whose blob is not ``dim * 4`` bytes
    ``orphan_sample``  up to 10 orphan paths, sorted (deterministic)
    ``chunks_missing`` index chunks ``(path, startline)`` with no vector row
    ``chunks_orphan``  vector chunks whose ``(path, startline)`` is not in the
                       index (includes "page still exists but lost this chunk")
    ``chunks_missing_sample`` up to 10 ``path:line`` strings, sorted
                       (deterministic) — only the *missing* direction, because
                       that is the one a human has to act on
    ``chunk_level``    whether the chunk comparison actually ran; ``False`` means
                       one side has no ``startline`` column and the report
                       degraded to page-level (``note`` says which side)
    ``note``           one human-readable conclusion, including what to do

    ``missing``/``orphan`` are deliberately kept at **page** granularity: they
    are the historical counters quoted by the CLI, the docs and monitoring.  The
    chunk counters are additive, so a page with five chunks of which two have
    vectors now shows ``missing=0`` but ``chunks_missing=3``.

    A missing or broken database yields ``ok=False`` and an explanatory note —
    never an exception, because this runs from scheduled checks where a traceback
    would only bury the actual finding.  That promise includes *dirty rows*: the
    production ``path`` column has no ``NOT NULL``, so one row with a NULL or
    empty path must not be able to abort a scheduled run (it used to raise
    ``TypeError`` from the orphan sort).
    """
    out: dict = {
        "ok": False,
        "chunks": 0,
        "vectors": 0,
        "missing": 0,
        "orphan": 0,
        "bad_path": 0,
        "model_mismatch": 0,
        "dim_mismatch": 0,
        "orphan_sample": [],
        "chunks_missing": 0,
        "chunks_orphan": 0,
        "chunks_missing_sample": [],
        "chunk_level": False,
        "note": "",
    }
    problems: list[str] = []
    fixes: list[str] = []
    degraded: list[str] = []

    meta_paths: set | None = None
    meta_chunks: set | None = None
    meta_identity: str | None = None
    meta_labels: dict = {}
    index_db = cfg.index_db
    if not index_db.is_file():
        problems.append(f"index {index_db} is missing")
        fixes.append("run `dsh-kb index`")
    else:
        try:
            with closing(_open_ro(index_db)) as con:
                out["chunks"] = _count(con, "meta")
                meta_paths = _paths(con, "meta")
                meta_chunks, meta_identity = _chunks(con, "meta")
                meta_labels = _chunk_labels(con, "meta")
        except sqlite3.Error as exc:
            problems.append(f"index {index_db} is unusable ({exc})")
            fixes.append("run `dsh-kb index` to rebuild it")

    vec_paths: set | None = None
    vec_chunks: set | None = None
    vec_identity: str | None = None
    vec_labels: dict = {}
    vdb = vectors_db(cfg)
    if not vdb.is_file():
        problems.append(f"vector store {vdb} is missing")
        fixes.append("point DSH_KB_RAG_VECTORS_DB at the real file, or build the vectors")
    else:
        try:
            with closing(_open_ro(vdb)) as con:
                out["vectors"] = _count(con, "vectors")
                vec_paths = _paths(con, "vectors")
                vec_chunks, vec_identity = _chunks(con, "vectors")
                vec_labels = _chunk_labels(con, "vectors")
                # NULL model/vec rows are drift too: "not equal to the expected
                # value" includes "no value at all".
                out["model_mismatch"] = _model_mismatches(con, cfg)
                out["dim_mismatch"] = _dim_mismatches(con, cfg)
        except sqlite3.Error as exc:
            problems.append(f"vector store {vdb} is unusable ({exc})")
            expected = ", ".join(VECTOR_COLUMNS)
            fixes.append(f"rebuild the vectors (expected table vectors({expected}))")

    if meta_paths is not None and vec_paths is not None:
        # NULL / 空 path 是坏行：既不能算"页面还在"，也不能参与排序 —— None 与 str
        # 比较会抛 TypeError，而生产 schema 的 path 列没有 NOT NULL，一行脏数据就
        # 足以让调度链里的 check() 直接崩掉。所以单独计数并在 note 里报出来，
        # 不静默丢弃（prune 会把它们当孤儿删掉，见下面的 fix 提示）。
        # "坏"的定义放宽到"非空文本"：BLOB 写进 TEXT 列会回来 bytes，那种行同样不能
        # 排序、也不能进 --json（不可序列化）。
        bad_paths = {p for p in vec_paths if not _clean_path(p)}
        # 排序键带兜底：即使将来漏进意外类型，str() 也保证比较是 str 对 str。
        orphans = sorted(
            vec_paths - bad_paths - meta_paths,
            key=lambda p: (p is None, str(p) if p is not None else ""),
        )
        out["missing"] = len(meta_paths - vec_paths)
        out["orphan"] = len(orphans)
        out["orphan_sample"] = orphans[:SAMPLE_LIMIT]
        if bad_paths:
            out["bad_path"] = len(bad_paths)
            problems.append(
                f"{len(bad_paths)} vector row(s) have no path (NULL, empty or non-text)"
            )
            fixes.append("delete them with ragcheck.prune(cfg, dry_run=False)")
        if out["missing"]:
            problems.append(f"{out['missing']} page(s) have no vectors")
            fixes.append("re-embed those pages")
        if out["orphan"]:
            problems.append(f"{out['orphan']} vector page(s) point at pages that no longer exist")
            fixes.append("delete them with ragcheck.prune(cfg, dry_run=False)")

        # ---- chunk level: same identity on both sides ----------------------
        # 页级看不出来的形状：一页 5 块只嵌了 2 块。两侧都有块身份**且身份相同**
        # 时才算 —— CAS 侧是 chunk_id、旧侧是 (path, startline)，两者做集合差
        # 得到的全是噪音（每个 chunk_id 都不等于任何 (path,line)，于是"全缺"）。
        # 混合状态（索引已迁、向量库还没迁）正是升级窗口里的正常形态，必须报成
        # "这次没查得了"，不能报成一片红。
        if meta_chunks is not None and vec_chunks is not None and meta_identity == vec_identity:
            out["chunk_level"] = True
            chunk_missing = meta_chunks - vec_chunks
            # 坏行（path 为 NULL/空/非文本）已经由 bad_path 单独计数，别在块级
            # 再报一次 —— 否则一行脏数据会被数两遍，note 也会重复。
            # CAS 身份是 chunk_id 本身，脏 path 只能通过 meta 反查（见
            # _path_of_key）；查不到的行不算孤儿，由 bad_path 那条口径负责。
            chunk_orphan = {chunk for chunk in (vec_chunks - meta_chunks)
                            if _path_of_key(chunk, vec_labels) is None
                            or _clean_path(_path_of_key(chunk, vec_labels))}
            out["chunks_missing"] = len(chunk_missing)
            out["chunks_orphan"] = len(chunk_orphan)
            out["chunks_missing_sample"] = [
                _format_chunk(chunk, meta_labels)
                for chunk in sorted(chunk_missing, key=_chunk_sort_key)[:SAMPLE_LIMIT]
            ]
            if out["chunks_missing"]:
                problems.append(f"{out['chunks_missing']} chunk(s) have no vectors "
                                f"(page-level missing: {out['missing']})")
                fixes.append("re-embed the missing chunks (`dsh-kb index` then re-embed "
                             "the pages that lost chunks)")
            if out["chunks_orphan"]:
                problems.append(f"{out['chunks_orphan']} vector chunk(s) are no longer "
                                f"in the index")
                # prune 只删"整页都不在了"的行；页面还在的块级孤儿它碰不到。
                fixes.append("re-embed those pages — prune() only deletes rows whose "
                             "*page* is gone")
        else:
            # 缺的是哪一列、或者两侧身份对不上，都要写出来：这句 note 是运维在
            # 巡检输出里唯一能看到的线索。混合状态（索引 CAS、向量库旧式）走
            # 下面第二条分支 —— 它是升级窗口的正常形态，说的是"等你迁完"。
            missing_sides = []
            if meta_identity is None:
                missing_sides.append(
                    "index has no chunk identity column (no chunk_id, no startline)")
            if vec_identity is None:
                missing_sides.append(
                    "vector store has no chunk identity column (no chunk_id, no startline)")
            mismatched = (meta_identity is not None and vec_identity is not None
                          and meta_identity != vec_identity)
            if mismatched:
                missing_sides.append(
                    f"the two sides use different identities "
                    f"(index: {meta_identity}, vector store: {vec_identity}) — migrate "
                    f"the vector store too (tools/migrate-cas.py), or rebuild the index")
            degraded.append(
                f"chunk-level comparison unavailable "
                f"({'; '.join(missing_sides) or 'schema unreadable'}) "
                f"({'; '.join(missing_sides) or 'schema unreadable'}) "
                f"— fell back to page-level, so a page with only some chunks embedded "
                f"cannot be detected on this store"
            )
    if out["model_mismatch"]:
        problems.append(f"{out['model_mismatch']} vector row(s) are not model {cfg.rag.model}")
        fixes.append(f"re-embed with {cfg.rag.model}")
    if out["dim_mismatch"]:
        problems.append(
            f"{out['dim_mismatch']} vector row(s) are not {cfg.rag.dim}-dim "
            f"({cfg.rag.dim * 4} bytes per row)"
        )
        fixes.append(f"re-embed at dim {cfg.rag.dim}")

    out["ok"] = not problems
    if out["ok"]:
        out["note"] = (
            f"consistent: {out['chunks']} chunks / {out['vectors']} vectors, "
            f"model {cfg.rag.model}, dim {cfg.rag.dim}"
        )
        if out["chunk_level"]:
            out["note"] += (f" (chunk-level: {out['chunks']} chunks matched by "
                            f"{meta_identity})")
    else:
        # dict.fromkeys de-duplicates while keeping the order problems appeared.
        out["note"] = f"{'; '.join(problems)} — fix: {', '.join(dict.fromkeys(fixes))}"
    if degraded:
        # 退化是"这次没查得了"，不是"查了没问题"：既不能静默，也不能算成异常。
        out["note"] += f"；{'；'.join(degraded)}"
    return out


def prune(cfg: Config, *, dry_run: bool = True, force: bool = False) -> dict:
    """Delete vector rows whose page no longer exists in the index.

    Returns a dict with these keys (stable — scripts should read them, not the
    prose in ``note``):

    ``removed`` / ``kept``   rows deleted / surviving.  With ``dry_run=True``
                             (the default) these are the *estimates* and the
                             database is left byte-identical.
    ``dry_run`` / ``force``  what the caller asked for.
    ``ok`` / ``refused``     ``refused=True`` means a guard said no and
                             **nothing was deleted**; ``ok=False`` means the
                             operation did not complete (refused or failed).
    ``reason``               refusal/failure code — see ``REFUSED_*``.
    ``note``                 why, plus how to unlock it.
    ``home`` / ``vectors_db`` the two paths, so a report can show the mismatch.
    ``external``             whether ``vectors_db`` lives outside ``home``.
    ``total`` / ``orphan_rows`` / ``orphan_paths`` / ``ratio``  the numbers the
                             bulk guard decided on.
    ``bypassed``             guard codes that ``force`` stepped over (audit).

    Three guards, in order of strength.  They exist because ``rag.vectors_db``
    defaults to a machine-wide path while ``home`` is whatever the caller
    passed: "temporary home + default rag config" (a test, CI, a fresh
    checkout, an operator's probe) used to be enough to delete a *production*
    vector store with a one-page index (measured 2026-10-19: 5,521 rows).

    1. **ownership** — ``vectors_db`` must live under ``home``.  A store outside
       it is refused unless ``rag.allow_external_vectors_db`` is explicitly
       true; ``force`` alone does **not** override this one.
    2. **empty index** — no index, or an empty index, while the vector store has
       rows: refused.  An empty index side makes *every* row look orphaned.
    3. **bulk delete** — more than ``BULK_DELETE_LIMIT`` of the rows (or of the
       distinct paths) would go away: refused unless ``force``.

    Only orphan rows are ever deleted, and ``kb.sqlite`` is never written: an
    index row without vectors can be re-embedded, a vector row without an index
    row is pure noise.  Nothing here raises — a scheduled chain should not die
    mid-run; ``check`` explains what is actually wrong.
    """
    vdb = vectors_db(cfg)
    out: dict = {
        "removed": 0, "kept": 0, "dry_run": bool(dry_run), "force": bool(force),
        "ok": True, "refused": False, "reason": "", "note": "",
        "home": str(cfg.home), "vectors_db": str(vdb), "external": False,
        "total": 0, "orphan_rows": 0, "orphan_paths": 0, "ratio": 0.0,
        "bypassed": [],
    }
    if not vdb.is_file():
        out["note"] = f"vector store {vdb} is missing — nothing to prune"
        return out

    # 读全表：只读连接。dry-run 与真删共用这一次读取，真删时再开写连接。
    # 主键列随 schema 变：CAS 版是 chunk_id，旧版是 id —— 写死 id 会在迁移后的
    # 库上直接 "no such column: id"（部署手册里记的另一半故障）。
    try:
        with closing(_open_ro(vdb)) as con:
            key_column = _identity_column(con, "vectors")
            rows = con.execute(f"select {key_column}, path from vectors").fetchall()
    except sqlite3.Error as exc:
        out["note"] = f"vector store {vdb} is unusable ({exc}) — nothing to prune"
        return out

    total_rows = len(rows)
    total_paths = len({path for _rid, path in rows})
    out["total"] = total_rows

    meta_paths, index_rows = _index_side(cfg)

    def refuse(reason: str, note: str) -> dict:
        """拒绝：一行都不删，把原因/比例/两条路径/解锁方式一起报出去。

        真删被拒时 ``removed`` 保持 0（这是"实际删了几行"），预估留在
        ``orphan_rows`` 里 —— 拒绝路径绝不能让人误以为"已经删掉了"。
        """
        out.update(ok=False, refused=True, reason=reason, note=note)
        if not dry_run:
            out["kept"] = total_rows
        return out

    if meta_paths is None:
        return refuse(REFUSED_INDEX, (
            f"refusing to prune: index {cfg.index_db} is missing or unreadable — "
            f"with an unknown index side every vector row looks like an orphan; "
            f"build the index first (`dsh-kb index`)"))

    orphan_rows = sum(1 for _rid, path in rows if path not in meta_paths)
    orphan_paths = len({path for _rid, path in rows if path not in meta_paths})
    out["orphan_rows"] = orphan_rows
    out["orphan_paths"] = orphan_paths
    out["ratio"] = (orphan_rows / total_rows) if total_rows else 0.0
    path_ratio = (orphan_paths / total_paths) if total_paths else 0.0

    # ---- 防线 1：归属校验（结构性；--force 不越过它）----
    out["external"] = not _within(vdb, cfg.home)
    if out["external"] and not _allow_external_vectors_db(cfg):
        return refuse(REFUSED_EXTERNAL, (
            f"refusing to prune: vectors_db {vdb} is outside home {cfg.home} — "
            f"prune compares the vector store against *this* home's index, so a "
            f"store from another knowledge base looks entirely orphaned; point "
            f"DSH_KB_RAG_VECTORS_DB at a store inside home, or set "
            f"rag.allow_external_vectors_db=true if the split is intentional "
            f"(an external store is still subject to the bulk-delete guard)"))

    # ---- 防线 3：空 home（索引侧为空，而向量库非空）----
    if index_rows == 0 and total_rows > 0:
        if not force:
            return refuse(REFUSED_EMPTY_HOME, (
                f"refusing to prune: index {cfg.index_db} has no pages while {vdb} "
                f"has {total_rows} vector row(s) — an empty index side would delete "
                f"all of them; build the index first (`dsh-kb index`), or pass "
                f"--force if this really is the same knowledge base"))
        out["bypassed"].append(REFUSED_EMPTY_HOME)

    # ---- 防线 2：大批量删除（纵深防御）----
    if out["ratio"] > BULK_DELETE_LIMIT or path_ratio > BULK_DELETE_LIMIT:
        if not force:
            return refuse(REFUSED_BULK, (
                f"refusing to prune: {orphan_rows}/{total_rows} row(s) "
                f"({out['ratio']:.0%}) and {orphan_paths}/{total_paths} path(s) "
                f"({path_ratio:.0%}) would be deleted — more than "
                f"{BULK_DELETE_LIMIT:.0%}; 疑似 home 与 vectors_db 不属于同一知识库 "
                f"(home={cfg.home}, vectors_db={vdb}); pass --force if you really "
                f"mean it"))
        out["bypassed"].append(REFUSED_BULK)

    # ---- 三道防线都过了：dry-run 只报预估，真删才开写连接 ----
    if not dry_run and orphan_rows:
        doomed = [key for key, path in rows if path not in meta_paths]
        try:
            with closing(sqlite3.connect(vdb)) as con:
                con.executemany(
                    f"delete from vectors where {key_column} = ?", [(k,) for k in doomed])
                con.commit()
        except sqlite3.Error as exc:
            out.update(ok=False, reason=FAILED_DELETE,
                       note=f"delete failed on {vdb}: {exc}")
            return out

    out["removed"] = orphan_rows
    out["kept"] = total_rows - orphan_rows
    if out["removed"]:
        out["note"] = (f"{'would delete' if dry_run else 'deleted'} {out['removed']} "
                       f"orphan row(s) of {total_rows} in {vdb}"
                       + (f" (forced past: {', '.join(out['bypassed'])})"
                          if out["bypassed"] else ""))
    else:
        out["note"] = f"no orphan rows in {vdb} — nothing to delete"
    return out


# ---------------------------------------------------------------- helpers


def _open_ro(db: Path) -> sqlite3.Connection:
    """Open ``db`` strictly read-only.

    ``mode=ro`` is what makes the read-only contract enforceable rather than
    aspirational: SQLite refuses to create a missing file or a journal, and
    refuses every write.  The path is URI-quoted because a knowledge base path
    may contain spaces or CJK characters.
    """
    return sqlite3.connect(f"file:{quote(str(db))}?mode=ro", uri=True)


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    """Column names of ``table`` — empty when it does not exist or is unreadable.

    ``table_info`` as a table-valued function keeps the table name a **bound
    parameter** rather than statement text on purpose: the doctor guard test
    only allows statements that name ``vectors`` to be count/distinct-over-path
    queries, and a statement starting with ``pragma`` would trip it even though
    it is pure metadata.
    """
    try:
        return {row[0] for row in con.execute(
            "select name from pragma_table_info(?)", (table,))}
    except sqlite3.Error:       # pragma: no cover - unreadable db / SQLite < 3.16
        return set()


def _schema_kind(con: sqlite3.Connection, table: str) -> str:
    """``"cas"`` / ``"legacy"`` / ``"unknown"`` / ``""`` for one table.

    Both schemas are readable by design: the deploy order is "code first, then
    data", so between the two steps the index is still the old one, and a
    maintenance script that only understands the new layout would be useless
    exactly when it is needed.
    """
    columns = _table_columns(con, table)
    if not columns:
        return ""
    if CHUNK_ID in columns:
        return "cas"
    if LEGACY_ID in columns:
        return "legacy"
    return "unknown"


def _count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f"select count(*) from {table}").fetchone()[0])


def _paths(con: sqlite3.Connection, table: str) -> set:
    return {row[0] for row in con.execute(f"select distinct path from {table}")}


def _chunks(con: sqlite3.Connection, table: str) -> tuple[set | None, str | None]:
    """``(chunk identities, column used)`` — ``(None, None)`` when unsupported.

    Two layouts, in order of preference:

    * **content-addressed** (``chunk_id``) — the key a CAS store and a CAS index
      share, and the one that survives re-chunking: a block edited in place
      changes its id, a block merely shifted down keeps it;
    * **positional** (``path, startline``) — the pre-CAS identity.  Still
      supported because the upgrade order is "code first, then data": a machine
      that has the new code but not yet the new database must keep reporting
      drift rather than degrading to nothing.

    A store with neither column returns ``(None, None)``, and :func:`check`
    degrades to page-level and names the missing column instead of raising — a
    scheduled check must not die because one machine's store is older than the
    code.
    """
    columns = _table_columns(con, table)
    if CHUNK_ID in columns:
        return {row[0] for row in
                con.execute(f"select distinct chunk_id from {table}")}, CHUNK_ID
    if "startline" not in columns:
        return None, None
    return {tuple(row) for row in
            con.execute(f"select distinct path, startline from {table}")}, "startline"


def _chunk_labels(con: sqlite3.Connection, table: str) -> dict:
    """Map chunk identity → ``"path:line"``, for the human-readable sample.

    The comparison itself is on the identity (``chunk_id`` when both sides have
    it); this is only how a *missing* chunk is reported, and a chunk id is not
    something an operator can act on.  Best-effort: an unreadable table yields
    an empty map and :func:`_format_chunk` falls back to the raw key.
    """
    columns = _table_columns(con, table)
    if CHUNK_ID in columns and "path" in columns and "startline" in columns:
        try:
            return {(row[0],): f"{row[1]}:{row[2]}" for row in con.execute(
                f"select distinct chunk_id, path, startline from {table}")}
        except sqlite3.Error:       # pragma: no cover - unreadable table
            return {}
    return {}


def _identity_column(con: sqlite3.Connection, table: str) -> str:
    """The column that identifies one row of ``table`` (``id`` if unknown).

    Used by ``prune``'s ``DELETE``.  The fallback keeps the pre-CAS behaviour on
    a store that has neither column, where the delete then fails loudly through
    the existing ``FAILED_DELETE`` path rather than silently deleting nothing.
    """
    columns = _table_columns(con, table)
    if CHUNK_ID in columns:
        return CHUNK_ID
    return LEGACY_ID


def _clean_path(value) -> bool:
    """Whether a stored path is usable text (the page-level "bad row" rule)."""
    return isinstance(value, str) and bool(value)


def _path_of_key(key, labels: dict) -> str | None:
    """The path a chunk key belongs to, or ``None`` when the key has no path.

    A content-addressed key (``"a1b2…"``) carries no path of its own — it is
    looked up through the label map built from the same table.  A positional key
    *is* ``(path, startline)``, so its path is simply ``key[0]``.  ``None`` means
    "this row has no path at all" (NULL/empty/BLOB), which the page-level
    ``bad_path`` counter owns.
    """
    if isinstance(key, tuple):
        return key[0] if key else None
    label = labels.get((key,))
    if label is None:
        return None                         # the row has no path at all
    path, _sep, line = label.rpartition(":")
    if not line.isdigit():                  # a path that itself contains ":" and no line
        return label
    return path or None


def _format_chunk(chunk, labels: dict | None = None) -> str:
    """``path:line`` — what a human needs to locate the block.

    A CAS key is an opaque hash, so it is resolved through ``labels``; when the
    key cannot be resolved (a store whose row has no path) the raw key is shown
    rather than an empty string, so the report still names *something*.
    """
    if isinstance(chunk, tuple):
        return f"{chunk[0]}:{chunk[1]}"
    label = (labels or {}).get((chunk,))
    return label if label else str(chunk)


def _chunk_sort_key(chunk) -> tuple:
    """Deterministic, type-safe ordering for a chunk key.

    Works for both identities.  ``startline`` is not ``NOT NULL`` in every store
    seen in the wild, and a single NULL/BLOB value must not be able to make the
    scheduled check raise ``TypeError`` while sorting — the same regression the
    orphan sort once had.  A CAS key has no position, so it sorts by its own
    text; that keeps the sample stable between runs, which is what matters.
    """
    if isinstance(chunk, tuple):
        path, line = chunk
        try:
            number = int(line)
        except (TypeError, ValueError):
            number = 1 << 62
        return (str(path), number)
    return (str(chunk), 0)


def _model_mismatches(con: sqlite3.Connection, cfg: Config) -> int:
    """Vector rows not built with the configured model (NULL counts as drift).

    ``vectors.model`` is ``NOT NULL`` in the CAS schema and nullable in the
    pre-CAS one; the same statement is correct for both.
    """
    return int(con.execute(
        "select count(*) from vectors where model is null or model != ?",
        (cfg.rag.model,),
    ).fetchone()[0])


def _dim_mismatches(con: sqlite3.Connection, cfg: Config) -> int:
    """Vector rows whose blob is not the configured dimension.

    The CAS schema stores ``dim`` per row, so the comparison is against that
    value — a store rebuilt at a different width would otherwise be judged
    against a config default it never claimed to follow.  The pre-CAS schema has
    no ``dim`` column, and there the configured dimension is the only authority
    there is.
    """
    if "dim" in _table_columns(con, "vectors"):
        return int(con.execute(
            "select count(*) from vectors where vec is null or dim is null"
            " or length(vec) != dim * 4",
        ).fetchone()[0])
    return int(con.execute(
        "select count(*) from vectors where vec is null or length(vec) != ?",
        (cfg.rag.dim * 4,),
    ).fetchone()[0])


def _index_side(cfg: Config) -> tuple[set | None, int | None]:
    """``(meta paths, meta row count)`` — ``(None, None)`` when unreadable.

    The row count is what tells "this home has no index at all" (the accident's
    shape) apart from "this home's index happens to be small": with zero rows
    every vector row looks orphaned, which is exactly when a delete must stop.
    """
    if not cfg.index_db.is_file():
        return None, None
    try:
        with closing(_open_ro(cfg.index_db)) as con:
            return _paths(con, "meta"), _count(con, "meta")
    except sqlite3.Error:
        return None, None


def _within(path: Path, root: Path) -> bool:
    """Whether ``path`` sits under ``root`` (both resolved, so symlinks count)."""
    try:
        target, parent = path.resolve(), root.resolve()
    except OSError:      # pragma: no cover - resolve() rarely raises
        target, parent = path, root
    return target == parent or parent in target.parents


#: 危险开关的"真值"集合：只有明确写出的这几个才算 true。默认拒绝 —— 一个能删库
#: 的开关不该因为拼写错误（"enabled"、空格、随手写的字）而被当成打开。
_TRUTHY = {"1", "true", "yes", "on"}


def _allow_external_vectors_db(cfg: Config) -> bool:
    """Whether the operator explicitly allowed a vector store outside ``home``.

    Three sources, nearest-to-the-config first:

    1. ``cfg.rag.allow_external_vectors_db`` — honoured the moment ``config.py``
       grows the field (this repair must not touch that file);
    2. ``DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB``;
    3. ``<home>/config.json``, nested (``rag.allow_external_vectors_db``) or flat
       (``rag_allow_external_vectors_db``) — the two spellings the rest of the
       rag config already accepts.

    Default is **false**: "temporary home + default rag config" must be refused,
    not silently obeyed.
    """
    flag = getattr(cfg.rag, "allow_external_vectors_db", None)
    if flag is not None:
        return _truthy(flag)
    env = os.environ.get("DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB")
    if env is not None:
        return _truthy(env)
    return _truthy(_config_file_flag(cfg, "allow_external_vectors_db"))


def _config_file_flag(cfg: Config, key: str):
    """Read one ``rag`` switch from ``<home>/config.json``; broken file == unset."""
    path = cfg.home / "config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    nested = data.get("rag")
    if isinstance(nested, dict) and key in nested:
        return nested[key]
    return data.get(f"rag_{key}")


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in _TRUTHY
