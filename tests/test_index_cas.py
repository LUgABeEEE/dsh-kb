"""Content-addressed (CAS) index schema: build, detect, search, delete.

Why this file exists
--------------------
The operational index used to key a chunk by an implicitly assigned ``meta.id``
and join FTS back with ``m.id = chunks.rowid``.  The CAS schema replaces that
identity with ``chunk_id = sha256(path + 0x1f + text)[:16]`` plus a derived
64-bit FTS join key ``rid = (int(chunk_id, 16) >> 4) + 1``.  Two consequences
are pinned here, because both are deployment-blocking:

* a legacy index that meets CAS-era code (or the other way round) must **not**
  silently return wrong results — the schema is detected before the query runs;
* ``join meta m on m.id`` raises ``no such column: m.id`` on a CAS database, so
  the join has to follow the schema (and the result set must not move).

The equivalence test is the load-bearing one: the same corpus is indexed twice
— once with the verbatim production *legacy* DDL, once through the real
``index.build()`` — and ten queries must come back with the **same ordered
sequence** of hits and the same per-hit scores.  Order is part of the contract:
bm25 ties used to be resolved by whatever row order SQLite happened to walk,
which differs between the two rowid spaces (and, in CAS, moves as the corpus
grows).  ``search`` now ends with ``order by score, m.path, m.startline``, so a
tie is broken by where the chunk actually lives in the bundle.

Everything is built in a temporary directory: no network, no production paths,
no writes outside the temp dir.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb import index  # noqa: E402
from dsh_kb.config import Config  # noqa: E402
from handrolled.text import expand_for_index  # noqa: E402

#: Corpus used by every test here.  Each page is longer than ``chunk_lines``
#: (15), so a page holds several chunks and every chunk id can still be
#: computed by hand.  The two canary terms in ``alpha.md`` are 21 lines apart,
#: i.e. in **different** chunks — a page that is a single chunk would hide the
#: leftover FTS row behind the same rowid.
PAGES = {
    "z0-public/alpha.md": (
        "---\ntype: lesson\ntitle: alpha page\n---\n"
        "alpha body about 挂载点失效 and CIFS recovery.\n"
        "canary ALPHA-ONLY-11111 lives in this first chunk.\n"
        + "\n".join(f"alpha filler line {i}" for i in range(1, 13))
        + "\n"
        "second chunk opens here with the SECOND-CANARY-33333 marker.\n"
        + "\n".join(f"alpha tail line {i}" for i in range(1, 17))
        + "\n"
    ),
    "z1-restricted/bravo.md": (
        "---\ntype: preference\ntitle: bravo page\n---\n"
        "bravo body about 交付偏好 and reports.\n"
        "canary BRAVO-ONLY-22222 lives in this chunk.\n"
        + "\n".join(f"bravo filler line {i}" for i in range(1, 14))
        + "\n"
    ),
}

#: A trunk of queries hitting every path the search code can take: exact latin
#: identifier (AND pass), CJK bigram, mixed script, a term that only exists in
#: one page, and a two-token phrase that needs the relaxation pass.
QUERIES = (
    "挂载点失效",
    "ALPHA-ONLY-11111",
    "BRAVO-ONLY-22222",
    "交付偏好 reports",
    "alpha body",
    "CIFS recovery",
    "filler",
    "挂载",
    "z0-public",
    "reports 报告",
)

#: Verbatim production **legacy** DDL (``index.py`` before CAS).  Kept as a copy
#: rather than imported so that deleting the legacy branch from the shipped code
#: cannot quietly delete the counter-example this file compares against.
LEGACY_DDL = """
create table meta(
    id        integer primary key,
    path      text    not null,
    startline integer not null,
    text      text    not null
);
create virtual table chunks using fts5(
    toks, content='', tokenize='unicode61'
);
create virtual table vocab using fts5vocab(chunks, 'row');
"""


def chunk_id_of(path: str, text: str) -> str:
    """Spec formula — recomputed here so the test fails if the code drifts."""
    import hashlib

    return hashlib.sha256(path.encode("utf-8") + b"\x1f" + text.encode("utf-8")).hexdigest()[:16]


def rid_of(chunk_id: str) -> int:
    """Spec formula for the FTS join key."""
    return (int(chunk_id, 16) >> 4) + 1


def _page_chunks(text: str, chunk_lines: int = 15) -> list[str]:
    """Split a page exactly the way ``index.build`` does."""
    lines = text.splitlines()
    out = []
    for start in range(0, len(lines), chunk_lines):
        block = "\n".join(lines[start:start + chunk_lines])
        if block.strip():
            out.append(block)
    return out


def write_mirror(mirror: Path, pages: dict[str, str] = PAGES) -> None:
    for rel, text in pages.items():
        target = mirror / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def write_legacy_index(db: Path, pages: dict[str, str] = PAGES) -> None:
    """Build the pre-CAS index by hand, with the same chunking as ``build``."""
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    try:
        con.executescript(LEGACY_DDL)
        rowid = 0
        for rel, text in sorted(pages.items()):
            for start, block in enumerate(_page_chunks(text), start=1):
                rowid += 1
                con.execute(
                    "insert into meta(id, path, startline, text) values (?,?,?,?)",
                    (rowid, rel, (start - 1) * 15 + 1, block),
                )
                con.execute(
                    "insert into chunks(rowid, toks) values (?,?)",
                    (rowid, expand_for_index(block)),
                )
        con.execute("insert into chunks(chunks) values ('optimize')")
        con.commit()
    finally:
        con.close()


class CasFixture(unittest.TestCase):
    """A CAS index + mirror in a temp home, rebuilt by the real ``build()``."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "kb"
        self.cfg = Config(home=self.home)
        write_mirror(self.cfg.mirror_dir)

    def build(self) -> index.IndexStats:
        return index.build(self.cfg)

    def con(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.cfg.index_db}?mode=ro", uri=True)


class TestCasSchema(CasFixture):
    """① the new build produces the CAS schema and can answer queries."""

    def test_build_creates_the_cas_schema(self):
        self.build()
        con = self.con()
        try:
            cols = [row[1] for row in con.execute("pragma table_info(meta)")]
            self.assertEqual(cols, ["rid", "chunk_id", "path", "startline", "text"])
            fts_sql = con.execute(
                "select sql from sqlite_master where name='chunks'").fetchone()[0]
            self.assertIn("contentless_delete=1", fts_sql)
            self.assertIn("content=''", fts_sql)
            self.assertEqual(
                con.execute("select count(*) from vocab").fetchone()[0] > 0, True)
        finally:
            con.close()

    def test_schema_kind_detects_cas(self):
        self.build()
        self.assertEqual(index.schema_kind(self.cfg), "cas")

    def test_every_chunk_has_the_spec_chunk_id_and_derived_rid(self):
        stats = self.build()
        con = self.con()
        try:
            rows = con.execute("select rid, chunk_id, path, text from meta").fetchall()
        finally:
            con.close()
        self.assertEqual(len(rows), stats.chunks)
        for rid, cid, path, text in rows:
            self.assertEqual(cid, chunk_id_of(path, text))
            self.assertEqual(rid, rid_of(cid))
        self.assertEqual(len({cid for _r, cid, _p, _t in rows}), len(rows), "chunk_id unique")
        self.assertEqual(len({rid for rid, _c, _p, _t in rows}), len(rows), "rid unique")

    def test_rid_is_the_fts_rowid_for_every_chunk(self):
        """The join key must actually be the FTS row — that is its whole job."""
        self.build()
        con = self.con()
        try:
            meta = {rid for (rid,) in con.execute("select rid from meta")}
            fts = {rowid for (rowid,) in con.execute("select rowid from chunks")}
        finally:
            con.close()
        self.assertEqual(meta, fts)

    def test_search_works_on_the_new_schema(self):
        self.build()
        result = index.search(self.cfg, "挂载点失效")
        self.assertEqual(result.note, "")
        self.assertTrue(result.hits, "CAS index must answer queries")
        self.assertEqual(result.hits[0].path, "z0-public/alpha.md")

    def test_search_uses_the_rid_join_not_the_legacy_id(self):
        """The regression this whole change is about.

        Run the CAS SQL with the legacy join by hand: it must raise
        ``no such column: m.id``.  That is the measured failure mode on a
        migrated database, and it is why the join follows the schema.
        """
        self.build()
        con = self.con()
        try:
            with self.assertRaises(sqlite3.OperationalError) as ctx:
                con.execute(
                    "select m.path from chunks join meta m on m.id = chunks.rowid "
                    "where chunks match ?", ("alpha",),
                ).fetchall()
            self.assertIn("m.id", str(ctx.exception))
            # ... while the CAS join answers the same query, one row per matching
            # chunk of alpha.md (every chunk carries an "alpha" filler line).
            rows = con.execute(
                "select m.path from chunks join meta m on m.rid = chunks.rowid "
                "where chunks match ?", ("alpha",),
            ).fetchall()
            self.assertEqual([r[0] for r in rows], ["z0-public/alpha.md"] * 3)
            # Joining on `chunk_id` (text) against a rowid is the *silent* wrong
            # answer: no error, no rows.  Pin it, because "join the column that
            # names the chunk" reads correct.
            self.assertEqual(con.execute(
                "select count(*) from chunks join meta m on m.chunk_id = chunks.rowid "
                "where chunks match ?", ("alpha",)).fetchone()[0], 0)
        finally:
            con.close()


class TestCasLegacyEquivalence(unittest.TestCase):
    """② same corpus, both schemas → the same hits with the same scores."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        home = Path(cls._tmp.name)
        cls.cas_cfg = Config(home=home / "cas")
        cls.legacy_cfg = Config(home=home / "legacy")
        for cfg in (cls.cas_cfg, cls.legacy_cfg):
            write_mirror(cfg.mirror_dir)
        cls.stats = index.build(cls.cas_cfg)
        write_legacy_index(cls.legacy_cfg.index_db)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _rows(result) -> list[tuple]:
        """Order-independent identity of a result: (path, line, score, snippet)."""
        return sorted((h.path, h.line, h.score, h.snippet) for h in result.hits)

    def test_both_indexes_hold_the_same_corpus(self):
        self.assertEqual(index.schema_kind(self.cas_cfg), "cas")
        self.assertEqual(index.schema_kind(self.legacy_cfg), "legacy")
        cas = sqlite3.connect(f"file:{self.cas_cfg.index_db}?mode=ro", uri=True)
        old = sqlite3.connect(f"file:{self.legacy_cfg.index_db}?mode=ro", uri=True)
        try:
            projection = "select path, startline, text from meta order by path, startline"
            self.assertEqual(
                cas.execute(projection).fetchall(), old.execute(projection).fetchall(),
                "the two fixtures must differ only in the chunk key",
            )
            self.assertEqual(
                cas.execute("select count(*) from chunks").fetchone()[0],
                old.execute("select count(*) from chunks").fetchone()[0],
            )
        finally:
            cas.close()
            old.close()

    def test_query_results_are_equivalent(self):
        for query in QUERIES:
            with self.subTest(query=query):
                cas = index.search(self.cas_cfg, query)
                old = index.search(self.legacy_cfg, query)
                self.assertEqual(self._rows(cas), self._rows(old),
                                 f"hit sets differ for {query!r}")
                # 顺序也是契约的一部分：迁移后逐条位置都不能动，否则 recall@5
                # 之类的截断指标会在这里悄悄改变（2026-10-19 副本演练实测到 4 条
                # 查询因并列次序不同而错位 —— 靠 (path, startline) 全序修掉）。
                self.assertEqual([(h.path, h.line, h.score) for h in cas.hits],
                                 [(h.path, h.line, h.score) for h in old.hits],
                                 f"hit order differs for {query!r}")
                self.assertEqual(cas.note, old.note)
                self.assertEqual(cas.relaxed, old.relaxed)
                self.assertEqual(cas.dropped, old.dropped)
                self.assertEqual(cas.tokens, old.tokens)

    def test_the_multiset_of_scores_is_identical(self):
        """Scores are rounded by ``search``; the rounded values must not move."""
        for query in QUERIES:
            with self.subTest(query=query):
                cas = sorted(h.score for h in index.search(self.cas_cfg, query).hits)
                old = sorted(h.score for h in index.search(self.legacy_cfg, query).hits)
                self.assertEqual(cas, old)


class TestTieOrderIsDeterministic(unittest.TestCase):
    """并列 bm25 必须有内容里的次序，不能随 rowid 走。

    实测到的两个后果（副本演练 34 条查询命中 4 条）：重建一次插入顺序变了，
    答案就换位置；迁移成 CAS（rowid 由 chunk_id 派生）后与旧库逐条不一致。
    前者让 10-26 的趋势对比变成"比排序噪音"，后者让"迁移等价"这句话失真。
    """

    @staticmethod
    def _fixture(pad: int):
        """两页**逐字节相同**的正文 ⇒ 对同一查询 bm25 必然并列。

        ``pad`` 只加在正文里（两页完全一样），所以并列保持、hash 变、rowid 序变
        —— 用它来挑一组"rid 序 ≠ 路径序"的数据，否则这个用例证明不了什么。
        """
        tmp = tempfile.TemporaryDirectory()
        cfg = Config(home=Path(tmp.name) / "kb")
        body = ("alpha beta" + " pad" * pad + "\n"
                + "".join(f"filler line {i}\n" for i in range(20)))
        write_mirror(cfg.mirror_dir, {"a.md": body, "b.md": body})
        index.build(cfg)
        return tmp, cfg

    def test_tied_rows_come_out_in_path_order(self):
        for pad in range(24):
            tmp, cfg = self._fixture(pad)
            try:
                with sqlite3.connect(f"file:{cfg.index_db}?mode=ro", uri=True) as con:
                    rid_order = [p for (p,) in con.execute(
                        "select path from meta where startline = 1 order by rid")]
                if rid_order == sorted(rid_order):
                    continue          # 这组数据的 rid 序恰好也是路径序，换一组
                hits = index.search(cfg, "alpha", limit=5)
                self.assertEqual([h.path for h in hits.hits], sorted(rid_order),
                                 "并列分数必须按 (path, startline) 定序")
                return
            finally:
                tmp.cleanup()
        self.fail("24 组数据都没能得到 rid 序 != 路径序；fixture 需要换")

    def test_a_new_page_does_not_reshuffle_the_ties_in_front_of_it(self):
        """语料长大一轮后，原来那些并列行的先后不得改变。"""
        tmp, cfg = self._fixture(0)
        try:
            before = [h.path for h in index.search(cfg, "alpha", limit=5).hits]
            (cfg.mirror_dir / "0-first.md").write_text(
                "alpha beta\n" + "".join(f"filler line {i}\n" for i in range(20)),
                encoding="utf-8")
            index.build(cfg)
            after = [h.path for h in index.search(cfg, "alpha", limit=5).hits]
            self.assertEqual([p for p in after if p != "0-first.md"], before,
                             "插入新页不得改变既有并列行的相对次序")
        finally:
            tmp.cleanup()


class TestCasDeleteOrder(CasFixture):
    """③ deleting a chunk: FTS first, then ``meta`` — with ``changes()==1``.

    Measured (``migrate-cas.py --self-test``): ``DELETE FROM chunks WHERE
    <column>=?`` is a hard error, because the FTS table's only column is
    ``toks``.  The safe sequence is ``delete from chunks where rowid=rid``
    followed by ``delete from meta where rid=rid``; deleting ``meta`` first
    leaves a ghost row that still MATCHes but is invisible through the join,
    and ``integrity-check`` does not catch it.
    """

    def setUp(self):
        super().setUp()
        self.build()

    def _row_for(self, needle: str) -> tuple[int, str]:
        con = self.con()
        try:
            rid, cid = con.execute(
                "select rid, chunk_id from meta where text like ?", (f"%{needle}%",),
            ).fetchone()
        finally:
            con.close()
        return rid, cid

    def _matches(self, con, term: str) -> list[int]:
        return [rowid for (rowid,) in con.execute(
            "select rowid from chunks where chunks match ?", (f'"{term}"',))]

    def test_deleting_by_column_name_is_rejected(self):
        """Guard: the FTS table has one column, so column predicates must fail.

        Measured on ``migrate-cas.py --self-test``: the *only* column is
        ``toks``, so ``delete from chunks where <any other name>=?`` is a hard
        ``no such column`` — which is why the two-step order below exists.
        A predicate on ``toks`` itself does not raise; it matches 0 rows, so it
        deletes nothing.  Both halves are pinned so neither can be mistaken for
        a working delete.
        """
        con = sqlite3.connect(self.cfg.index_db)
        try:
            with self.assertRaises(sqlite3.OperationalError) as ctx:
                con.execute("delete from chunks where chunk_id=?", ("0" * 16,))
            self.assertIn("chunk_id", str(ctx.exception))
            con.execute("begin immediate")
            con.execute("delete from chunks where toks=?", ("alpha",))
            self.assertEqual(con.execute("select changes()").fetchone()[0], 0,
                             "a toks predicate silently matches nothing")
            con.execute("commit")
        finally:
            con.close()

    def test_delete_order_fts_then_meta_is_the_only_correct_one(self):
        rid, cid = self._row_for("ALPHA-ONLY-11111")
        second_rid, _second_cid = self._row_for("SECOND-CANARY-33333")
        self.assertEqual(self._matches(sqlite3.connect(self.cfg.index_db), "ALPHA-ONLY-11111"),
                         [rid])
        self.assertNotEqual(rid, second_rid, "the page must hold more than one chunk")

        con = sqlite3.connect(self.cfg.index_db)
        try:
            con.execute("begin immediate")
            con.execute("delete from chunks where rowid=?", (rid,))
            fts_changes = con.execute("select changes()").fetchone()[0]
            con.execute("delete from meta where rid=?", (rid,))
            meta_changes = con.execute("select changes()").fetchone()[0]
            con.execute("commit")

            # The measured contract: one row touched on each side.
            self.assertEqual((fts_changes, meta_changes), (1, 1))
            self.assertEqual(self._matches(con, "ALPHA-ONLY-11111"), [], "still MATCHable")
            self.assertIsNone(con.execute(
                "select 1 from meta where chunk_id=?", (cid,)).fetchone())
            self.assertEqual(con.execute(
                "insert into chunks(chunks) values('integrity-check')").fetchall(), [])
            # the *other* chunks are untouched
            self.assertEqual(self._matches(con, "SECOND-CANARY-33333"), [second_rid])
        finally:
            con.close()

    def test_meta_first_leaves_a_row_the_join_can_no_longer_see(self):
        """The wrong order is silent — pin it so nobody 'simplifies' to it.

        A contentless FTS table cannot be asked "do you still hold this row?" by
        value, so the leftover is evidenced the way ``migrate-cas.py
        --self-test`` does it: the FTS row is still MATCHable, the join hides it,
        ``integrity-check`` is clean, and a later ``delete ... where rowid``
        reports ``changes()==1`` — proof the row really was still there.
        """
        rid, _cid = self._row_for("ALPHA-ONLY-11111")
        second_rid, _second = self._row_for("SECOND-CANARY-33333")
        con = sqlite3.connect(self.cfg.index_db)
        try:
            con.execute("begin immediate")
            con.execute("delete from meta where rid=?", (rid,))
            con.execute("commit")

            matches = self._matches(con, "ALPHA-ONLY-11111")
            self.assertEqual(len(matches), 1, "FTS still holds a row for the removed chunk")
            self.assertNotIn(second_rid, matches, "the other chunk is unaffected")
            joined = con.execute(
                "select count(*) from chunks join meta m on m.rid = chunks.rowid "
                "where chunks match ?", ('"ALPHA-ONLY-11111"',)).fetchone()[0]
            self.assertEqual(joined, 0, "the join hides the leftover")
            self.assertEqual(con.execute(
                "insert into chunks(chunks) values('integrity-check')").fetchall(), [],
                "integrity-check cannot see it either")
            # It is still deletable by rowid, which is how a repair cleans up.
            con.execute("delete from chunks where rowid=?", (matches[0],))
            self.assertEqual(con.execute("select changes()").fetchone()[0], 1,
                             "the leftover row was physically present")
            con.commit()
            self.assertEqual(self._matches(con, "ALPHA-ONLY-11111"), [])
        finally:
            con.close()

    def test_changes_is_zero_for_an_unknown_key(self):
        """``changes()==1`` is only meaningful next to the 0 case."""
        con = sqlite3.connect(self.cfg.index_db)
        try:
            con.execute("begin immediate")
            con.execute("delete from chunks where rowid=?", (int("0" * 16, 16),))
            self.assertEqual(con.execute("select changes()").fetchone()[0], 0)
            con.execute("delete from meta where chunk_id=?", ("0" * 16,))
            self.assertEqual(con.execute("select changes()").fetchone()[0], 0)
            con.execute("commit")
        finally:
            con.close()


class TestLegacySchemaIsStillReadable(CasFixture):
    """④ dual schema: a pre-CAS index keeps working, an unusable one is refused.

    The deploy runbook requires "code first, then data" precisely because the
    reverse order breaks search.  Keeping the legacy read path means the window
    between the two steps has a working index instead of a hard failure, and
    the refusal test covers the case where neither key is recognised.
    """

    def test_legacy_index_is_detected_and_searchable(self):
        write_legacy_index(self.cfg.index_db)
        self.assertEqual(index.schema_kind(self.cfg), "legacy")
        result = index.search(self.cfg, "ALPHA-ONLY-11111")
        self.assertEqual(result.note, "")
        self.assertTrue(result.hits, "a legacy index must still answer queries")
        self.assertEqual(result.hits[0].path, "z0-public/alpha.md")
        # 3 chunks in alpha.md + 2 in bravo.md, counted from the fixture itself.
        expected = sum(len(_page_chunks(text)) for text in PAGES.values())
        self.assertEqual(index.stats(self.cfg).chunks, expected)
        self.assertEqual(index.stats(self.cfg).pages, len(PAGES))

    def test_legacy_and_cas_agree_on_one_query(self):
        """The cheapest cross-check: rebuild the same corpus both ways."""
        write_legacy_index(self.cfg.index_db)
        legacy_hits = sorted((h.path, h.line, h.score)
                             for h in index.search(self.cfg, "交付偏好").hits)
        self.build()
        cas_hits = sorted((h.path, h.line, h.score)
                          for h in index.search(self.cfg, "交付偏好").hits)
        self.assertEqual(legacy_hits, cas_hits)

    def test_unrecognised_meta_schema_is_refused_not_guessed(self):
        """A ``meta`` table with neither key must produce a note, never a guess."""
        self.cfg.index_db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.cfg.index_db)
        try:
            con.execute("create table meta(note text)")
            con.commit()
        finally:
            con.close()
        result = index.search(self.cfg, "anything")
        self.assertEqual(result.hits, [])
        self.assertIn("schema", result.note.lower())
        self.assertIn("dsh-kb index", result.note)

    def test_missing_index_still_reports_missing(self):
        result = index.search(self.cfg, "anything")
        self.assertIn("index missing", result.note)

    def test_no_such_column_m_id_on_a_cas_db(self):
        """Named for the failure the runbook measured; the guard is the fix."""
        self.build()
        con = sqlite3.connect(f"file:{self.cfg.index_db}?mode=ro", uri=True)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                con.execute("select m.id from meta m").fetchall()
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
