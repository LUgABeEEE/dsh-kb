"""Local full-text index over the mirror (SQLite FTS5).

Layout
------
``meta``   — one row per chunk: ``(rid, chunk_id, path, startline, text)``
``chunks`` — a **contentless** FTS5 table holding only the inverted index
``vocab``  — ``fts5vocab(chunks, 'row')``: term → (documents, occurrences)

Storing the text *once* (in ``meta``) instead of inline in FTS5 cut the index
from 8.1× the corpus to about 1.9× in measurement, with no latency change.
Search joins ``chunks`` back to ``meta`` on ``rid``.

Chunk identity is **content-addressed**
---------------------------------------
``chunk_id = sha256(path + 0x1f + text)[:16]`` names a chunk by what it
contains, so the same block keeps its identity across a rebuild, a re-chunk of
a neighbouring page, or a re-embed run — which is what lets ``vectors.sqlite``
and ``embed_cache`` be keyed by it (see ``tools/reindex-vectors.py``).  FTS5
only takes an integer rowid, so ``rid = (int(chunk_id, 16) >> 4) + 1`` derives a
60-bit join key from it.  Both formulas are pinned verbatim against
``tools/migrate-cas.py``, which produces the same schema from an older index:
the two must stay interchangeable, or a migrated index and a rebuilt one stop
being the same thing.

Two schemas are readable
------------------------
:func:`schema_kind` tells the two apart by looking for the ``chunk_id`` column,
and :func:`search` follows it rather than assuming one.  A pre-CAS database
still answers queries (deploy order is "code first, then data", so the window
between the two steps must not break retrieval), and a database whose ``meta``
table has neither key is **refused with an actionable note** instead of being
queried with the wrong join.

The index is always a **local, disposable artifact**: it lives next to the
mirror, never on the share (SQLite is not safe on network filesystems), and can
be rebuilt from the mirror at any time.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .supersede import page_meta
from .text import (allows_relaxation, expand_for_index, make_snippet,
                   match_expression, query_tokens, split_frontmatter)

SEARCH_OVERFETCH = 4

#: Byte between path and text in the ``chunk_id`` pre-image, and the two columns
#: the CAS schema keys a chunk by.  ``\x1f`` cannot occur in a text file we
#: index, so ``(path, text)`` maps to exactly one id.
CHUNK_ID_SEP = b"\x1f"
CHUNK_ID_HEX = 16

#: Fraction of the indexed **chunk** corpus at (or above) which a token counts as
#: generic and is dropped from the relaxed OR pass.  ``0.5`` = "half the corpus".
#: The threshold is ``max(1, int(total * ratio))`` — integer-truncated, so the
#: effective line sits just under the ratio, exactly as the old ``total // 2`` did.
#: Raise it to keep more generic words (broader recall), lower it to be stricter.
#: At the default this yields the same integer threshold as the historical
#: ``total // 2``; only the unit of ``total`` was ever wrong (see below).
#: Deliberately a module constant rather than a ``Config`` field: this fix must
#: stay a single variable, and an ablation can monkeypatch it (see the tests).
SELECTIVE_CUT_RATIO = 0.5

#: Corpus size in the **same unit** as ``vocab.doc``.  ``vocab`` is
#: ``fts5vocab(chunks, 'row')``, so ``doc`` counts *chunks*; ``meta`` holds
#: exactly one row per chunk, so this counts that identical universe.  Pages are
#: *not* interchangeable here — a page is several chunks (measured when the fix
#: landed: 1,321 pages vs 6,133 chunks, i.e. 4.6 chunks/page; the ratio, not the
#: absolute counts, is what matters and it is what the old code ignored).
#: One-line rollback to the pre-fix (page-level, ~4.6x more aggressive) behaviour
#: -- do not, it silently discards mid-frequency terms such as ``目录``/``kb``:
#:     _CORPUS_SIZE_SQL = "select count(distinct path) from meta"
_CORPUS_SIZE_SQL = "select count(*) from meta"


@dataclass
class IndexStats:
    pages: int = 0
    chunks: int = 0
    bytes: int = 0
    duration: float = 0.0
    #: Pages whose frontmatter says they are replaced (``superseded_by``), and
    #: the number of ``supersedes`` edges found.  Reported so a corpus that
    #: starts using the fields says so in one line instead of silently.
    superseded_pages: int = 0
    supersede_edges: int = 0

    @property
    def summary(self) -> str:
        extra = ""
        if self.superseded_pages or self.supersede_edges:
            extra = f" · {self.superseded_pages} 已取代 / {self.supersede_edges} 取代边"
        return (f"{self.pages} pages / {self.chunks} chunks / "
                f"{self.bytes / 1048576:.1f} MB / {self.duration:.2f}s{extra}")


@dataclass
class Hit:
    path: str
    line: int
    score: float
    snippet: str
    zone: str = ""
    #: Supersede annotation.  ``superseded`` is the machine-readable bit;
    #: ``superseded_by`` / ``valid_until`` are what the snippet prefix shows.
    superseded: bool = False
    superseded_by: str = ""
    valid_until: str = ""


@dataclass
class SearchResult:
    query: str
    tokens: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    zones: tuple[str, ...] = ()
    stale: bool = False
    index_age: float | None = None
    note: str = ""
    relaxed: bool = False
    dropped: list[str] = field(default_factory=list)
    #: How many of ``hits`` are marked superseded (0 when the index predates the
    #: pages table).  Callers can surface "N of these are outdated" in one line.
    superseded_shown: int = 0


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    """Whether ``name`` exists — used to keep a query working on an older index."""
    try:
        return con.execute(
            "select count(*) from sqlite_master where type in ('table','view') and name=?",
            (name,)).fetchone()[0] > 0
    except sqlite3.Error:                              # pragma: no cover - unreadable db
        return False


def _document_frequency(con, tokens: list[str]) -> dict[str, int]:
    """How many indexed documents contain each token (0 when absent)."""
    lowered = [t.lower() for t in tokens]
    if not lowered:
        return {}
    placeholders = ",".join("?" * len(lowered))
    try:
        rows = con.execute(
            f"select term, doc from vocab where term in ({placeholders})", lowered
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {term: int(doc) for term, doc in rows}


def _selective_tokens(con, tokens: list[str]) -> list[str]:
    """Keep only the tokens that actually discriminate.

    The unit of "corpus" is a **chunk**, not a page.  ``vocab`` is
    ``fts5vocab(chunks, 'row')``, so ``vocab.doc`` counts the *chunks* that
    contain a term, and the threshold is computed over that same chunk universe
    (``_CORPUS_SIZE_SQL``).  The two must stay in one unit: mixing them means the
    cut silently describes something other than what this docstring says.

    A token that appears in **at least ``SELECTIVE_CUT_RATIO`` of all indexed
    chunks** (default 50% — "half the corpus") carries almost no signal; if it
    survives into the relaxed pass it drags in unrelated documents.  When every
    token is generic the original list is returned, so the caller still has a
    usable (if broad) query.

    Regression this guards against: the threshold used to be
    ``count(distinct path) // 2`` — half the *pages* — while ``doc`` counted
    *chunks*.  With ~4.6 chunks per page the effective line was ~10.8% of chunks
    instead of 50%: about 4.6x more aggressive than advertised, and it discarded
    useful mid-frequency terms (measured when the fix landed: ``目录`` doc=771,
    ``kb`` doc=1450, ``dsh`` doc=2944, all above the bogus cut of 660).
    See ``tests/test_index_zone.py`` for the fixture that pins the unit.
    """
    freq = _document_frequency(con, tokens)
    try:
        total = con.execute(_CORPUS_SIZE_SQL).fetchone()[0]
    except sqlite3.DatabaseError:
        return tokens
    if not freq or total <= 1:
        return tokens
    cut = max(1, int(total * SELECTIVE_CUT_RATIO))
    kept = [t for t in tokens if freq.get(t.lower(), 0) < cut]
    return kept or tokens


def _index_files(cfg: Config):
    """Yield ``(rel_path, absolute_path)`` for every indexable markdown file."""
    root = cfg.mirror_dir
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if cfg.index_allows(rel):
            yield rel, path


#: Schema ids returned by :func:`schema_kind`.
SCHEMA_CAS = "cas"              # (rid, chunk_id, …) — what :func:`build` writes
SCHEMA_LEGACY = "legacy"        # (id, …) — pre-2026-10 indexes, still readable
SCHEMA_UNKNOWN = "unknown"      # a meta table with neither key
SCHEMA_ABSENT = ""              # no index (or no meta table) at all

#: Note for a database whose key column is not recognised.  Written as the
#: operator's next command: this state is reachable (a half-finished migration,
#: a hand-edited file) and guessing a join would return wrong rows silently.
_UNUSABLE_NOTE = (
    "index schema not recognised (meta table has neither `chunk_id` nor `id`) — "
    "run `dsh-kb index` to rebuild it, or migrate an older index with "
    "tools/migrate-cas.py"
)


def chunk_id_of(path: str, text: str) -> str:
    """Content-addressed chunk key: ``sha256(path + 0x1f + text)[:16]``.

    Must stay byte-for-byte identical to ``tools/migrate-cas.py``'s
    ``chunk_id_of``; the two producers of the CAS schema have to agree, because
    ``vectors.sqlite`` and the embed cache are keyed by the result.
    """
    digest = hashlib.sha256(
        path.encode("utf-8") + CHUNK_ID_SEP + text.encode("utf-8")
    ).hexdigest()
    return digest[:CHUNK_ID_HEX]


def rid_of(chunk_id: str) -> int:
    """FTS5 join key derived from a ``chunk_id`` (60 bits, never 0).

    ``>> 4`` keeps the top 60 bits and ``+ 1`` moves the result off rowid 0;
    collisions are astronomically unlikely (64-bit prefix of a SHA-256) and are
    handled explicitly by :class:`_ChunkIds` rather than silently overwritten —
    a unique-index violation here would abort the whole rebuild.
    """
    return (int(chunk_id, 16) >> 4) + 1


class _ChunkIds:
    """Allocate ``chunk_id`` / ``rid`` for one build without ever colliding.

    Same policy as ``migrate-cas.py``'s ``IdAllocator`` (truncate-and-retry for
    a colliding id, linear probe for a colliding rid) but scoped to a single
    build: the FTS join key only has to be unique *within* one database, and a
    capture-the-whole-corpus allocator would double peak memory on a corpus
    this size for a case that cannot happen.

    Recorded collisions are logged by :func:`build` — a *silent* adjustment
    would make the rebuild irreproducible for no visible reason.
    """

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._rids: set[int] = set()
        self.id_collisions: list[tuple[str, str]] = []
        self.rid_collisions: list[tuple[int, int]] = []

    def allocate(self, path: str, text: str) -> tuple[int, str]:
        """Return ``(rid, chunk_id)`` for one chunk, resolving any collision."""
        cid = chunk_id_of(path, text)
        if cid in self._ids:
            base = cid
            for attempt in range(1, 0x100):
                candidate = base[:CHUNK_ID_HEX - 2] + f"{attempt:02x}"
                if candidate not in self._ids:
                    cid = candidate
                    self.id_collisions.append((base, candidate))
                    break
            else:                                   # pragma: no cover - needs 255 colliding ids
                raise RuntimeError(f"chunk_id collision storm around {base}; refusing to guess")
        self._ids.add(cid)

        rid = rid_of(cid)
        if rid in self._rids:
            base_rid = rid
            while rid in self._rids:                # pragma: no cover - needs a 64-bit collision
                rid += 1
                if rid >= (1 << 62):                # pragma: no cover - defensive
                    raise RuntimeError(f"rid probe exhausted for {cid}")
            self.rid_collisions.append((base_rid, rid))
        self._rids.add(rid)
        return rid, cid


def schema_kind(cfg: Config) -> str:
    """Which schema ``cfg.index_db`` uses — see ``SCHEMA_*``.

    Opened **read-only** (``mode=ro``): probing must never create a database,
    and the answer has to come from the file even while another process is
    rebuilding it.  An unreadable or absent file is ``SCHEMA_ABSENT`` — the
    "no index" note already covers that, and this function's job is to describe
    a database that exists.
    """
    db = cfg.index_db
    if not db.exists():
        return SCHEMA_ABSENT
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:                           # pragma: no cover - connect rarely fails
        return SCHEMA_ABSENT
    try:
        return _schema_kind(con)
    except sqlite3.Error:                           # pragma: no cover - unreadable file
        return SCHEMA_ABSENT
    finally:
        con.close()


def _schema_kind(con: sqlite3.Connection) -> str:
    """:func:`schema_kind` on an open connection (shared with ``search``)."""
    try:
        columns = {row[1] for row in con.execute("pragma table_info(meta)")}
    except sqlite3.Error:                           # pragma: no cover - unreadable file
        return SCHEMA_ABSENT
    if not columns:
        return SCHEMA_ABSENT                        # no meta table at all
    if "chunk_id" in columns:
        return SCHEMA_CAS
    if "id" in columns:
        return SCHEMA_LEGACY
    return SCHEMA_UNKNOWN


#: The CAS DDL, verbatim from ``tools/migrate-cas.py`` (``KB_DDL_CAS``).  The
#: `meta_path` index is part of it: `zone` filters and the doctor both ask
#: ``count(distinct path)`` / ``like`` questions that would otherwise scan.
_CAS_DDL = """
create table meta(
    rid       integer primary key,
    chunk_id  text    not null unique,
    path      text    not null,
    startline integer not null,
    text      text    not null
);
create index meta_path on meta(path);
create virtual table chunks using fts5(
    toks, content='', contentless_delete=1, tokenize='unicode61'
);
-- term -> (documents containing it, occurrences). Lets search drop
-- generic words when relaxing, instead of matching on "UNIQ" alone.
create virtual table vocab using fts5vocab(chunks, 'row');
-- one row per page: the frontmatter fields a *machine* needs (``supersede.py``
-- explains why ``status`` and ``superseded_by`` are separate questions).
create table pages(
    path          text primary key,
    type          text,
    title         text,
    description   text,
    tags          text,                -- space separated; small, never queried by term
    status        text,                -- draft | stable | deprecated
    superseded_by text,                -- null/'' = still in force
    valid_until   text,                -- ISO8601 | null
    fm_at         text,                -- generated.at: "who is newer"
    indexed_at    text
);
create index pages_live on pages(status, valid_until);
create index pages_succ on pages(superseded_by);
create index pages_type on pages(type, status);
-- A page may supersede many pages, so this direction is its own table.  Edge
-- direction: ``child`` declares ``supersedes: [parent]``, i.e. child is the
-- newer page; the parent's ``superseded_by`` is the inverse, and
-- ``supersede.check()`` (A1) asserts the two agree.
create table supersedes(
    child  text not null,
    parent text not null,
    primary key(child, parent)
) without rowid;
create index supersedes_parent on supersedes(parent);
-- (path, startline) is what search tie-breaks on; let SQLite walk the index.
create index meta_path_line on meta(path, startline);
"""


def build(cfg: Config, *, on_event=None) -> IndexStats:
    """Rebuild the index from scratch, in the CAS schema.

    The rebuild is written to ``<db>.tmp`` and swapped in with :func:`os.replace`,
    then the previous database is kept as ``<db>.prev``.  Two reasons:

    * **crash safety** — the old implementation deleted the live database first,
      so a crash mid-build left an empty index and search returned nothing
      (2026-10-24 post-mortem);
    * **size** — ``DROP TABLE`` does not return pages to the filesystem, so a
      rebuild after a shrink would otherwise keep the file at its historical
      maximum (measured: 271 MB that would not come down); a fresh temp file
      is inherently compact.

    Chunks are keyed by :func:`chunk_id_of`; the FTS rowid is :func:`rid_of`
    applied to it, so the same corpus always yields the same keys.  Collisions
    are resolved by :class:`_ChunkIds` and reported through ``on_event`` —
    they are a "this should be impossible" diagnostic, not a normal path.
    """
    emit = on_event or (lambda _m: None)
    started = time.perf_counter()
    db = cfg.index_db
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(db) + ".tmp")
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(tmp) + suffix)
        if stale.exists():
            stale.unlink()

    stats = IndexStats()
    ids = _ChunkIds()
    con = sqlite3.connect(str(tmp))
    try:
        con.executescript(_CAS_DDL)
        for rel, path in _index_files(cfg):
            stats.pages += 1
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # One frontmatter parse per page: the chunks below still index the
            # whole file (header included) — changing that would change every
            # chunk_id and force a full re-embed, so it is its own decision.
            page = page_meta(rel, raw)
            con.execute(
                "insert or replace into pages(path, type, title, description, tags,"
                " status, superseded_by, valid_until, fm_at, indexed_at)"
                " values (?,?,?,?,?,?,?,?,?,?)",
                (rel, page.type, page.title, page.description, page.tags_text,
                 page.status, page.superseded_by, page.valid_until, page.fm_at,
                 time.strftime("%Y-%m-%dT%H:%M:%S%z")),
            )
            for parent in page.supersedes:
                con.execute("insert or ignore into supersedes(child, parent) values (?,?)",
                            (rel, parent))
                stats.supersede_edges += 1
            if page.superseded_by:
                stats.superseded_pages += 1
            lines = raw.splitlines()
            for start in range(0, len(lines), cfg.chunk_lines):
                block = "\n".join(lines[start:start + cfg.chunk_lines])
                if not block.strip():
                    continue
                rid, cid = ids.allocate(rel, block)
                con.execute(
                    "insert into meta(rid, chunk_id, path, startline, text) values (?,?,?,?,?)",
                    (rid, cid, rel, start + 1, block),
                )
                con.execute(
                    "insert into chunks(rowid, toks) values (?,?)",
                    (rid, expand_for_index(block)),
                )
                stats.chunks += 1
        con.execute("insert into chunks(chunks) values ('optimize')")
        con.commit()
    finally:
        con.close()

    if stats.pages == 0 or stats.chunks == 0:
        for suffix in ("", "-wal", "-shm"):
            junk = Path(str(tmp) + suffix)
            if junk.exists():
                junk.unlink()
        raise RuntimeError("rebuild produced an empty index; previous index kept")

    if db.exists():
        os.replace(db, Path(str(db) + ".prev"))
    for suffix in ("-wal", "-shm"):
        junk = Path(str(db) + suffix)
        if junk.exists():
            junk.unlink()
    os.replace(tmp, db)

    if ids.id_collisions:
        emit(f"!! {len(ids.id_collisions)} chunk_id collision(s) resolved by suffix: "
             f"{ids.id_collisions[:3]}")
    if ids.rid_collisions:                          # pragma: no cover - needs a 64-bit collision
        emit(f"!! {len(ids.rid_collisions)} rid collision(s) resolved by probing: "
             f"{ids.rid_collisions[:3]}")

    stats.bytes = db.stat().st_size if db.exists() else 0
    stats.duration = time.perf_counter() - started
    emit(f"indexed {stats.summary}")
    return stats


def index_age(cfg: Config) -> float | None:
    """Seconds since the index was written, or ``None`` if it does not exist."""
    db = cfg.index_db
    if not db.exists():
        return None
    return max(0.0, time.time() - db.stat().st_mtime)


def _zone_clause(cfg: Config, prefixes: tuple[str, ...]) -> tuple[str, list[str]]:
    """SQL fragment restricting results to the selected zones.

    Root-level documents (``index.md``, ``log.md``) are always allowed: they are
    the bundle's front door and carry no private content by construction.
    """
    if not prefixes:
        return "1=1", []
    placeholders = " or ".join("m.path like ?" for _ in prefixes)
    return f"(m.path not like '%/%' or {placeholders})", [f"{p}%" for p in prefixes]


def search(
    cfg: Config,
    query: str,
    *,
    zone: str = "z01",
    limit: int | None = None,
    snippet_chars: int | None = None,
    fresh_seconds: float | None = None,
    include_superseded: bool = True,
    deep: bool = False,
) -> SearchResult:
    """Run a ranked full-text query against the local index.

    ``fresh_seconds`` marks the result ``stale`` when the mirror is older than
    that many seconds, so a caller can decide to refresh and retry.
    """
    wanted = limit or cfg.default_limit
    _shift = int(getattr(cfg.rag, "rerank_max_shift", 0) or 0)
    _deep = bool(deep) and bool(getattr(cfg.rag, "deep_enabled", False))
    # 限幅重排要能看到「更多候选」才有意义：内部多取 2σ 条，末尾再截回 wanted。
    _pool_extra = int(getattr(cfg.rag, "rerank_shift_pool", 5) or 0)
    limit = wanted + (_pool_extra if _shift > 0 else 0)
    if _deep:
        limit = max(limit, int(getattr(cfg.rag, "deep_candidates", 40) or 40))
    width = snippet_chars or cfg.snippet_chars
    zones = cfg.zones_for_query(zone)
    prefixes = cfg.query_prefixes(zone)
    result = SearchResult(query=query, zones=zones, index_age=index_age(cfg))

    if not cfg.index_db.exists():
        result.note = "index missing — run `dsh-kb index`"
        return result

    tokens = query_tokens(query)
    result.tokens = tokens
    clause, params = _zone_clause(cfg, prefixes)
    ceiling = max(limit * SEARCH_OVERFETCH, 16)

    con = sqlite3.connect(f"file:{cfg.index_db}?mode=ro", uri=True)
    try:
        # The join key follows the schema on disk.  A CAS database keys chunks
        # by `rid` (the integer derived from `chunk_id`, which is what the FTS
        # rowid holds) and has no `id` at all, so `m.id = chunks.rowid` raises
        # "no such column: m.id"; an older database has `id` and no `rid`.
        # `chunk_id` is text and can never equal a rowid, so joining on it would
        # silently return nothing — the failure mode this whole change is about.
        kind = _schema_kind(con)
        if kind == SCHEMA_UNKNOWN:
            result.note = _UNUSABLE_NOTE
            return result
        key = "rid" if kind == SCHEMA_CAS else "id"
        # Supersede data lives in `pages`/`supersedes`.  A database built before
        # those tables existed must keep answering queries — the whole point of
        # the "code first, data later" deploy order — so the join is conditional
        # instead of assumed.  `superseded` is computed in SQL because it has to
        # drive the ORDER BY, not just decorate the row.
        has_pages = _has_table(con, "pages")
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        select_ = "m.path, m.startline, m.text, bm25(chunks) as score"
        join_ = f"from chunks join meta m on m.{key} = chunks.rowid"
        lead_params: tuple = ()
        if has_pages:
            select_ += (
                ", coalesce(p.superseded_by,'') as succ"
                ", coalesce(p.valid_until,'') as vu"
                ", case when coalesce(p.superseded_by,'') <> ''"
                "     or (coalesce(p.valid_until,'') <> '' and p.valid_until <= ?)"
                "   then 1 else 0 end as superseded"
            )
            join_ += " left join pages p on p.path = m.path"
            lead_params = (now,)
        sql = (
            f"select {select_} {join_} "
            f"where chunks match ? and {clause} "
            # 全序：bm25 完全相同的行（同内容的两份拷贝、同模板页）必须有稳定的
            # 先后。只按 score 排时，并列行的顺序由 FTS5 的 rowid 顺序决定 ——
            # 那既随重建而变（插入顺序变了），又在两种 schema 之间不同（CAS 的
            # rowid 由 chunk_id 派生），于是"同一批题、重建一次、答案换个位置"。
            # (path, startline) 是内容里的天然次序，两个 schema 都能给出，
            # 所以它同时买到「跨重建稳定」与「迁移前后逐条一致」。
            # 现行页永远排在已被取代的页之前（spec：标注 + 降权，不隐藏）。
            + ("order by superseded asc, score, m.path, m.startline limit ?"
               if has_pages else
               "order by score, m.path, m.startline limit ?")
        )

        def run(expression: str, ceiling_: int | None = None):
            return con.execute(
                sql, (*lead_params, expression, *params, ceiling_ or ceiling)).fetchall()

        # Pass 1 — precision: every expanded token must be present, so an exact
        # identifier cannot be answered by a document that merely shares "UNIQ".
        expression_used = match_expression(tokens, "AND")
        try:
            rows = run(expression_used)
        except sqlite3.OperationalError as exc:
            result.note = f"query failed: {exc}"
            return result

        # Pass 2 — recall: relax to OR, but only for vague queries, and only
        # over *selective* tokens. Words appearing in at least half the chunks
        # (SELECTIVE_CUT_RATIO of the chunk corpus) are dropped first; otherwise
        # a single generic word would drag in unrelated documents (measured). A
        # lone latin identifier is never relaxed.
        if not rows and len(tokens) > 1 and allows_relaxation(query):
            selective = _selective_tokens(con, tokens)
            if selective != tokens:
                result.dropped = [t for t in tokens if t not in selective]
            expression_used = match_expression(selective or tokens, "OR")
            try:
                rows = run(expression_used)
            except sqlite3.OperationalError:
                rows = []
                expression_used = ""
            result.relaxed = bool(rows)
    finally:
        con.close()

    seen: set[str] = set()
    first_cov = None
    for row in rows:
        path, start_line, text, score = row[:4]
        succ = row[4] if len(row) > 4 else ""
        valid_until = row[5] if len(row) > 5 else ""
        superseded = bool(row[6]) if len(row) > 6 else False
        if include_superseded is False and superseded:
            continue
        if path in seen:
            continue
        seen.add(path)
        # The first chunk of a page carries its YAML header, and indexing the
        # header is a deliberate open question (it holds title/description, which
        # *are* useful terms) — but *showing* it is not: a snippet that starts
        # with `generated: { by: ... }` is noise.  So strip it for the snippet
        # and add back the lines it occupied, which keeps reported line numbers
        # pointing at real file lines.
        body, header_lines = split_frontmatter(text)
        fragment, relative_line = make_snippet(body, tokens, width, phrase=query)
        relative_line += header_lines
        if superseded:
            # Annotate, do not hide: the model — not the retriever — decides
            # which of two conflicting conclusions still holds.  Hiding a
            # historically-true page recreates "I wrote that down and search
            # cannot find it", which is the complaint this field must not cause.
            mark = f"[已被 {succ or '新页'} 取代" + (f"，有效至 {valid_until}" if valid_until else "") + "] "
            fragment = mark + fragment
            result.superseded_shown += 1
        own_zone = cfg.zone_for(path)
        if first_cov is None:
            _m = len(set(tokens) & set(query_tokens(text)))
            first_cov = (_m / len(tokens)) if tokens else 1.0
        result.hits.append(
            Hit(
                path=path,
                line=start_line + relative_line - 1,
                score=round(float(score), 2),
                snippet=fragment,
                zone=own_zone.name if own_zone else "",
                superseded=superseded,
                superseded_by=succ,
                valid_until=valid_until,
            )
        )
        if len(result.hits) >= limit:
            break

    # Hidden mode must still return `limit` live hits: the over-fetched window
    # can be entirely superseded, in which case widening it is the only honest
    # fix (filtering after ranking always has this shape — see the spec §3).
    if include_superseded is False and len(result.hits) < limit:
        for factor in (4, 16, 64):
            with sqlite3.connect(f"file:{cfg.index_db}?mode=ro", uri=True) as con2:
                if _schema_kind(con2) != kind or not has_pages:
                    break
                wider = con2.execute(
                    sql, (*lead_params, expression_used or match_expression(tokens, "AND"),
                          *params, ceiling * factor)).fetchall()
            for row in wider:
                if len(result.hits) >= limit:
                    break
                path, start_line, _, score = row[:4]
                if row[6] or path in seen:
                    continue
                seen.add(path)
                body2, header2 = split_frontmatter(row[2])
                fragment, relative_line = make_snippet(body2, tokens, width, phrase=query)
                relative_line += header2
                own_zone = cfg.zone_for(path)
                result.hits.append(Hit(path=path, line=start_line + relative_line - 1,
                                      score=round(float(score), 2), snippet=fragment,
                                      zone=own_zone.name if own_zone else ""))
            if len(result.hits) >= limit:
                break
    if result.superseded_shown:
        result.note = (f"{result.superseded_shown} 条命中已被取代（片段带 [已被 … 取代] 前缀，"
                       f"排序已降到现行页之后）")
    if fresh_seconds is not None:
        age = index_age(cfg)
        result.stale = age is None or age > fresh_seconds
    # 可选向量臂：只在关键词命中不足 limit 时补一路（默认关闭）。
    # 位置刻意放在所有关键词逻辑之后 —— 关闭时行为与旧版逐字节一致；
    # 任何异常都只降级成 note，绝不影响关键词结果。
    trigger = str(getattr(cfg.rag, "vector_trigger", "auto") or "auto")
    weak_ratio = float(getattr(cfg.rag, "vector_weak_ratio", 0.35) or 0.0)
    weak = (first_cov is not None and first_cov < weak_ratio)
    reserve = int(getattr(cfg.rag, "vector_weak_reserve", 2) or 0) if (weak and trigger == "auto") else 0
    keyword_slots = max(1, wanted - reserve)
    need_vector = cfg.rag.vector_enabled and (
        bool(getattr(cfg.rag, 'vector_always', False))
        or len(result.hits) < wanted or reserve > 0)
    if need_vector:
        try:
            from .ragvec import augment, hyde_answer
            seed = result.hits[:keyword_slots] if reserve else result.hits
            hyde = None
            if weak and getattr(cfg.rag, "hyde_enabled", False):
                import os as _os
                if _os.environ.get("DSH_KB_HYDE_PROVIDER") in ("remote", "http"):
                    from .ragvec import hyde_remote
                    hyde = hyde_remote(query)
                else:
                    hyde = hyde_answer(query, getattr(cfg.rag, "hyde_model", "qwen2.5:7b"), int(getattr(cfg.rag, "hyde_prompt_tokens", 100) or 100))
            merged, note = augment(cfg, query, limit, seed, width, zone=zone,
                                   mode=getattr(cfg.rag, "vector_mode", "fill"), hyde=hyde, rerank=weak or bool(getattr(cfg.rag, 'rerank_always', False)))
            if merged:
                result.hits = merged
                tag = "weak-match " if weak else ""
                result.note = (result.note + " | " if result.note else "") + tag + note
        except Exception as exc:  # noqa: BLE001 - 语义路永不影响关键词路径
            extra = f"vector arm skipped: {type(exc).__name__}: {exc}"
            result.note = (result.note + " | " if result.note else "") + extra
    # 限幅重排（maxshift）：最终顺序上再跑一次 cross-encoder，但每条位移不超过 σ。
    # 默认 0 = 关闭，行为与旧版逐字节一致；异常只降级成 note。
    shift = _shift
    if shift > 0 and getattr(cfg.rag, "rerank_enabled", False) and len(result.hits) > 1:
        try:
            from . import rerank as _rr
            from .ragvec import _snippet as _snp
            _chars = int(getattr(cfg.rag, "rerank_passage_chars", 480) or 480)
            _passages = [_snp(cfg.mirror_dir, h.path, h.line, _chars) for h in result.hits]
            _scores = _rr.score_pairs(query, _passages)
            if _scores is not None and len(_scores) == len(result.hits):
                before = [(h.path, h.line) for h in result.hits]
                result.hits = _clamp_shift(result.hits, _scores, shift)
                if before != [(h.path, h.line) for h in result.hits]:
                    result.note = (result.note + " | " if result.note else "") + f"maxshift ±{shift}"
        except Exception:
            pass
    if _deep and len(result.hits) > 1:
        try:
            from . import deeprank as _dr
            from .ragvec import _snippet as _snp
            _seen: set = set()
            _uniq: list = []
            for h in result.hits:
                if h.path in _seen:
                    continue
                _seen.add(h.path)
                _uniq.append(h)
            _chars = int(getattr(cfg.rag, "deep_passage_chars", 340) or 340)
            _cap = int(getattr(cfg.rag, "deep_candidates", 40) or 40)
            _pages = []
            for h in _uniq[:_cap]:
                b = (getattr(h, "snippet", "") or "").strip()
                if len(b) < min(_chars, 200):
                    b = _snp(cfg.mirror_dir, h.path, h.line, _chars)
                _pages.append((h.path, b[:_chars]))
            _model = str(getattr(cfg.rag, "deep_model", "") or "")
            _order = _dr.pick_pages(
                query, _pages, model=_model,
                pick=int(getattr(cfg.rag, "deep_pick", 3) or 3),
                timeout=float(getattr(cfg.rag, "deep_timeout_s", 25) or 25),
            )
            if _order:
                _picked = [_pages[i - 1][0] for i in _order if 0 < i <= len(_pages)]
                _by = {}
                for h in _uniq:
                    _by.setdefault(h.path, h)
                _head = [_by[x] for x in _picked if x in _by]
                _pset = set(_picked)
                result.hits = _head + [h for h in _uniq if h.path not in _pset]
                result.note = (result.note + " | " if result.note else "") + "deep=true：LLM 挑 %d 页（%s）" % (len(_head), _model)
            else:
                result.note = (result.note + " | " if result.note else "") + "deep=true：LLM 不可用，回退普通排序"
        except Exception as _exc:
            result.note = (result.note + " | " if result.note else "") + "deep=true 跳过：%s" % type(_exc).__name__
    if len(result.hits) > wanted:
        result.hits = result.hits[:wanted]
    return result


def _clamp_shift(hits: list, scores: list, sigma: int) -> list:
    """按 cross-encoder 分排，但每条落位相对原名次位移不超过 sigma（贪心指派）。

    为什么不是直接采用重排顺序：池内多数题的正确答案只与前后几名差 0.1~0.2，
    全量重排会把 p-06 这类「并列页」挤出前 5；限幅只允许「近邻上移」，
    2026-10-24 实测 +2 题 / −0（33 题口径）。
    """
    n = len(hits)
    if n < 2 or sigma <= 0:
        return hits
    desired = [h for _s, h in sorted(zip(scores, hits), key=lambda t: -t[0])]
    orig = {id(h): i for i, h in enumerate(hits)}
    slots: list = [None] * n
    placed: set = set()
    for h in desired:
        o = orig[id(h)]
        lo, hi = max(0, o - sigma), min(n - 1, o + sigma)
        for pos in range(lo, hi + 1):
            if slots[pos] is None:
                slots[pos] = h
                placed.add(id(h))
                break
    for h in hits:
        if id(h) not in placed:
            for pos in range(n):
                if slots[pos] is None:
                    slots[pos] = h
                    break
    return [h for h in slots if h is not None]

@dataclass
class ReadResult:
    path: str
    start: int
    end: int
    total: int
    lines: list[str] = field(default_factory=list)


def read(cfg: Config, rel: str, *, start: int = 1, lines: int = 80) -> ReadResult:
    """Read a line window from the mirror, refusing paths outside it."""
    root = cfg.mirror_dir.resolve()
    target = (root / rel).resolve()
    if root not in target.parents or not target.is_file():
        raise FileNotFoundError(f"not in mirror: {rel}")

    body = target.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    window = max(1, min(lines, cfg.max_read_lines))
    end = min(len(body), start + window - 1)
    return ReadResult(path=rel, start=start, end=end, total=len(body), lines=body[start - 1:end])


def stats(cfg: Config) -> IndexStats:
    """Describe the current index without rebuilding it."""
    out = IndexStats()
    if not cfg.index_db.exists():
        return out
    out.bytes = cfg.index_db.stat().st_size
    con = sqlite3.connect(f"file:{cfg.index_db}?mode=ro", uri=True)
    try:
        out.chunks = con.execute("select count(*) from chunks").fetchone()[0]
        out.pages = con.execute("select count(distinct path) from meta").fetchone()[0]
    except sqlite3.DatabaseError:
        pass
    finally:
        con.close()
    return out
