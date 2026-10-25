"""Tests for the vector store ↔ index consistency checker (``dsh_kb.ragcheck``).

What matters operationally, and is therefore asserted here:

* the counts report exactly the drift the fixture was built with;
* ``check`` is strictly read-only — both databases keep their bytes, mtime and
  no journal/wal file appears;
* a missing or broken vector store is *reported*, never created and never a
  traceback;
* ``prune`` is a dry-run no-op by default and, when run for real, removes only
  rows whose page no longer exists — and is idempotent.

Everything is built in a temporary directory: no network, no shared fixtures,
no writes outside the temp dir.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb import ragcheck  # noqa: E402
from dsh_kb.config import (  # noqa: E402
    DEFAULT_RAG_DIM,
    DEFAULT_RAG_MODEL,
    DEFAULT_RAG_VECTORS_DB,
    Config,
    RagConfig,
    load_config,
)

#: The fixture is deliberately small but covers every counter:
#: 5 meta rows over 3 pages, 7 vector rows of which 2 are orphans,
#: 1 was built with the wrong model and 1 has a wrong blob length.
INDEX_ROWS = [
    (1, "z0-public/alpha.md", 1, "alpha one"),
    (2, "z0-public/alpha.md", 16, "alpha two"),
    (3, "z1-restricted/bravo.md", 1, "bravo one"),
    (4, "z1-restricted/bravo.md", 16, "bravo two"),
    (5, "z0-public/charlie.md", 1, "charlie one"),
]
ORPHAN_PATHS = ["z0-public/gone-one.md", "z0-public/gone-two.md"]

CHECK_KEYS = {
    "ok", "chunks", "vectors", "missing", "orphan", "bad_path", "model_mismatch",
    "dim_mismatch", "orphan_sample", "note",
    # 块级口径（t7 盲点 A）：一页 5 块坏 3 块时页级计数是 0，页级看不见。
    "chunks_missing", "chunks_orphan", "chunks_missing_sample", "chunk_level",
}
PRUNE_KEYS = {
    "removed", "kept", "dry_run", "force", "ok", "refused", "reason", "note",
    "home", "vectors_db", "external", "total", "orphan_rows", "orphan_paths",
    "ratio", "bypassed",
}


def _blob(dim: int = DEFAULT_RAG_DIM, fill: float = 0.5) -> bytes:
    """A float32 little-endian embedding row of exactly ``dim * 4`` bytes."""
    return struct.pack(f"<{dim}f", *([fill] * dim))


def _snapshot(path: Path) -> tuple[str, int, int]:
    """(sha256, size, mtime_ns) — enough to prove a file was not touched."""
    data = path.read_bytes()
    st = path.stat()
    return hashlib.sha256(data).hexdigest(), st.st_size, st.st_mtime_ns


class RagFixture(unittest.TestCase):
    """Builds a fresh index DB + vector store for every test."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "home"
        (self.home / "index").mkdir(parents=True)
        self.index_db = self.home / "index" / "kb.sqlite"
        self.vdb = self.home / "index" / "vectors.sqlite"
        self._write_index(INDEX_ROWS)
        self._write_vectors()
        self.cfg = self._load(str(self.vdb))

    # ---- fixture builders -------------------------------------------
    def _write_index(self, rows):
        con = sqlite3.connect(self.index_db)
        with con:
            con.execute("drop table if exists meta")
            con.execute(
                "create table meta(id integer primary key, path text, "
                "startline integer, text text)"
            )
            con.executemany(
                "insert into meta(id, path, startline, text) values (?,?,?,?)", rows
            )
        con.close()

    def _write_vectors(self, rows=None):
        good = _blob()
        rows = rows if rows is not None else [
            (1, "z0-public/alpha.md", 1, "alpha one", "h1", DEFAULT_RAG_MODEL, good),
            (2, "z0-public/alpha.md", 16, "alpha two", "h2", DEFAULT_RAG_MODEL, good),
            (3, "z1-restricted/bravo.md", 1, "bravo one", "h3", DEFAULT_RAG_MODEL, good),
            # built with an older model name → model_mismatch
            (4, "z1-restricted/bravo.md", 16, "bravo two", "h4", "stale-model", good),
            # blob is not dim*4 bytes → dim_mismatch
            (5, "z0-public/charlie.md", 1, "charlie one", "h5", DEFAULT_RAG_MODEL, b"\x01" * 100),
            (6, "z0-public/gone-one.md", 1, "gone", "h6", DEFAULT_RAG_MODEL, good),
            (7, "z0-public/gone-two.md", 1, "gone", "h7", DEFAULT_RAG_MODEL, good),
        ]
        self._write_vectors_at(self.vdb, rows)

    def _write_vectors_at(self, path: Path, rows) -> None:
        """把向量行写进任意路径 —— "外部向量库"的场景要用它。"""
        con = sqlite3.connect(path)
        with con:
            con.execute("drop table if exists vectors")
            con.execute(
                "create table vectors(id integer primary key, path text, "
                "startline integer, text text, hash text, model text, vec blob)"
            )
            con.executemany(
                "insert into vectors(id, path, startline, text, hash, model, vec) "
                "values (?,?,?,?,?,?,?)",
                rows,
            )
        con.close()

    def _load(self, vectors_db: str | None = None, **env: str) -> Config:
        """load_config with a clean ``DSH_KB_*`` slate so ambient env cannot leak in."""
        with mock.patch.dict(os.environ):
            for key in [k for k in os.environ if k.startswith("DSH_KB_")]:
                os.environ.pop(key, None)
            if vectors_db is not None:
                os.environ["DSH_KB_RAG_VECTORS_DB"] = vectors_db
            os.environ.update(env)
            return load_config(home=self.home)

    # ---- convenience -----------------------------------------------
    def _rows(self) -> list[tuple]:
        con = sqlite3.connect(self.vdb)
        try:
            return con.execute("select id, path, model, length(vec) from vectors order by id").fetchall()
        finally:
            con.close()


class TestCheck(RagFixture):
    def test_reports_the_exact_drift_in_the_fixture(self):
        report = ragcheck.check(self.cfg)
        self.assertEqual(set(report), CHECK_KEYS)
        self.assertEqual(report["chunks"], 5)
        self.assertEqual(report["vectors"], 7)
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["orphan"], 2)
        self.assertEqual(report["model_mismatch"], 1)
        self.assertEqual(report["dim_mismatch"], 1)
        self.assertEqual(report["orphan_sample"], ORPHAN_PATHS)
        self.assertFalse(report["ok"])
        # The note is for a human: it must say what to do next.
        self.assertTrue(report["note"].strip())
        self.assertIn("prune", report["note"])

    def test_missing_pages_are_counted(self):
        """A page that lost every vector row is drift in the other direction."""
        con = sqlite3.connect(self.index_db)
        with con:
            con.execute(
                "insert into meta(id, path, startline, text) values (?,?,?,?)",
                (6, "z0-public/brand-new.md", 1, "no vectors yet"),
            )
        con.close()
        report = ragcheck.check(self.cfg)
        self.assertEqual(report["chunks"], 6)
        self.assertEqual(report["missing"], 1)
        self.assertEqual(report["orphan"], 2)

    def test_orphan_sample_is_sorted_and_capped(self):
        rows = [
            (i, f"z0-public/gone-{i:02d}.md", 1, "gone", f"h{i}", DEFAULT_RAG_MODEL, _blob())
            for i in range(2, 15)
        ]
        self._write_vectors(rows)
        report = ragcheck.check(self.cfg)
        self.assertEqual(report["orphan"], 13)
        self.assertEqual(len(report["orphan_sample"]), 10)
        self.assertEqual(report["orphan_sample"], sorted(p for _i, p, *_rest in rows)[:10])

    def test_null_and_empty_paths_are_reported_not_raised(self):
        """回归（t4 发现）：一行 NULL path 就能让排序抛 TypeError。

        生产 schema 的 ``vectors.path`` 没有 NOT NULL，而调度链会直接调
        ``check()`` —— 脏行必须变成"坏行计数 + note"，不能变成异常（那会中断整条
        巡检，也违反"坏库一律 ok=False + note"的承诺）。
        """
        good = _blob()
        self._write_vectors([
            (1, "z0-public/alpha.md", 1, "alpha one", "h1", DEFAULT_RAG_MODEL, good),
            (2, None, 1, "no path", "h2", DEFAULT_RAG_MODEL, good),
            (3, "", 1, "empty path", "h3", DEFAULT_RAG_MODEL, good),
            (4, "z0-public/gone.md", 1, "gone", "h4", DEFAULT_RAG_MODEL, good),
        ])

        report = ragcheck.check(self.cfg)          # 不许抛

        self.assertEqual(set(report), CHECK_KEYS)
        self.assertFalse(report["ok"], "有坏行就必须 ok=False")
        self.assertEqual(report["bad_path"], 2)
        self.assertEqual(report["orphan"], 1, "坏行单独计数，不混进孤儿")
        self.assertEqual(report["orphan_sample"], ["z0-public/gone.md"])
        self.assertIn("2 vector row(s) have no path", report["note"])
        self.assertIn("prune", report["note"], "note 要给出怎么清掉它们")
        # 坏行也是"页面不在 meta 里"的行，所以 prune 会把它们一起删掉
        # （3/4 = 75% > 50%，所以要显式 --force —— 这正是大批量保护该做的事）
        self.assertEqual(ragcheck.prune(self.cfg, force=True)["removed"], 3)

    def test_a_non_text_path_is_a_bad_row_not_a_crash(self):
        """兜底键的另一半：BLOB 写进 TEXT 列会回来 bytes，排序与 --json 都不许崩。"""
        self._write_vectors([
            (1, "z0-public/alpha.md", 1, "alpha one", "h1", DEFAULT_RAG_MODEL, _blob()),
            (2, b"z0-public/blob-path.md", 1, "blob", "h2", DEFAULT_RAG_MODEL, _blob()),
        ])

        report = ragcheck.check(self.cfg)          # 不许抛

        self.assertEqual(report["bad_path"], 1)
        self.assertEqual(report["orphan"], 0)
        json.dumps(report)                         # 坏行不进 orphan_sample，报告可序列化

    def test_check_is_strictly_read_only(self):
        """Bytes, mtime and directory layout must be identical afterwards —
        a checker that 'fixes' the store while looking at it is worse than none."""
        before = {p.name: _snapshot(p) for p in (self.index_db, self.vdb)}
        layout_before = sorted(p.name for p in self.index_db.parent.iterdir())
        ragcheck.check(self.cfg)
        after = {p.name: _snapshot(p) for p in (self.index_db, self.vdb)}
        layout_after = sorted(p.name for p in self.index_db.parent.iterdir())
        self.assertEqual(before, after)
        self.assertEqual(layout_before, layout_after)

    def test_healthy_fixture_can_report_ok(self):
        """Zero drift across every counter → ok, with a conclusion in the note."""
        good = _blob()
        self._write_vectors([
            (1, "z0-public/alpha.md", 1, "a", "h1", DEFAULT_RAG_MODEL, good),
            (2, "z0-public/alpha.md", 16, "b", "h2", DEFAULT_RAG_MODEL, good),
            (3, "z1-restricted/bravo.md", 1, "c", "h3", DEFAULT_RAG_MODEL, good),
            (4, "z1-restricted/bravo.md", 16, "d", "h4", DEFAULT_RAG_MODEL, good),
            (5, "z0-public/charlie.md", 1, "e", "h5", DEFAULT_RAG_MODEL, good),
        ])
        report = ragcheck.check(self.cfg)
        self.assertTrue(report["ok"])
        self.assertEqual(report["orphan_sample"], [])
        self.assertTrue(report["note"].strip())


class TestChunkLevel(RagFixture):
    """块级口径（t7 盲点 A）：页级"有向量"不等于"每一块都有向量"。

    真实形状：一页 5 个 chunk，向量表只给了其中 2 个。页级 path 集合看起来
    "这页有向量"（missing=0），而检索会静默漏掉 3 块内容。
    """

    PAGE = "z0-public/alpha.md"
    LINES = (1, 16, 31, 46, 61)

    def _five_chunks(self):
        self._write_index(
            [(i + 1, self.PAGE, line, f"chunk at {line}") for i, line in enumerate(self.LINES)]
        )
        return self.LINES

    def test_partial_page_coverage_is_invisible_at_page_level(self):
        lines = self._five_chunks()
        good = _blob()
        self._write_vectors([
            (i + 1, self.PAGE, line, f"chunk at {line}", f"h{i}", DEFAULT_RAG_MODEL, good)
            for i, line in enumerate(lines[:2])          # 只给了前两块
        ])

        report = ragcheck.check(self.cfg)

        # 页级：这一页"有向量"，所以两个老计数都是 0 —— 这正是盲点。
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["orphan"], 0)
        # 块级：5 块里 3 块没有向量。
        self.assertIs(report["chunk_level"], True)
        self.assertEqual(report["chunks"], 5)
        self.assertEqual(report["chunks_missing"], 3)
        self.assertEqual(report["chunks_orphan"], 0)
        self.assertEqual(
            report["chunks_missing_sample"],
            [f"{self.PAGE}:{line}" for line in lines[2:]],
            "样例要能直接指出是哪一块（path:line）",
        )
        self.assertFalse(report["ok"], "块级缺口必须让 ok=False，否则 doctor 不会响")
        self.assertIn("3 chunk(s) have no vectors", report["note"])

    def test_a_chunk_that_no_longer_exists_is_an_orphan(self):
        """页面还在、但索引里那一块没了：页级 orphan=0，块级必须看得见。"""
        self._write_index([(1, self.PAGE, 1, "a"), (2, self.PAGE, 31, "b")])
        good = _blob()
        self._write_vectors([
            (1, self.PAGE, 1, "a", "h1", DEFAULT_RAG_MODEL, good),
            (2, self.PAGE, 16, "old", "h2", DEFAULT_RAG_MODEL, good),
        ])

        report = ragcheck.check(self.cfg)

        self.assertEqual(report["missing"], 0, "页面还在，页级不是 missing")
        self.assertEqual(report["orphan"], 0, "页面还在，页级不是 orphan")
        self.assertEqual(report["chunks_missing"], 1, "第 31 行那块没有向量")
        self.assertEqual(report["chunks_orphan"], 1, "第 16 行那块已不在索引里")
        self.assertEqual(report["chunks_missing_sample"], [f"{self.PAGE}:31"])
        self.assertFalse(report["ok"])
        self.assertIn("vector chunk", report["note"])

    def test_a_store_without_startline_degrades_to_page_level(self):
        """老库没有 ``startline`` 列：不许炸，退化成页级并在 note 里说明。"""
        good = _blob()
        con = sqlite3.connect(self.vdb)
        with con:
            con.execute("drop table vectors")
            con.execute(
                "create table vectors(id integer primary key, path text, text text, "
                "hash text, model text, vec blob)"
            )
            con.executemany(
                "insert into vectors values (?,?,?,?,?,?)",
                [
                    (i + 1, path, "text", f"h{i}", DEFAULT_RAG_MODEL, good)
                    for i, path in enumerate(
                        ("z0-public/alpha.md", "z1-restricted/bravo.md", "z0-public/charlie.md")
                    )
                ],
            )
        con.close()

        report = ragcheck.check(self.cfg)               # 不许抛

        self.assertIs(report["chunk_level"], False)
        self.assertEqual(report["chunks_missing"], 0)
        self.assertEqual(report["chunks_orphan"], 0)
        self.assertEqual(report["chunks_missing_sample"], [])
        # 页级口径仍然完整，所以老库上依然是一份可用的报告
        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["orphan"], 0)
        self.assertTrue(report["ok"])
        self.assertIn("startline", report["note"], "退化必须在 note 里说出来，不能假装比对过")

    def test_an_index_without_startline_degrades_too(self):
        con = sqlite3.connect(self.index_db)
        with con:
            con.execute("drop table meta")
            con.execute("create table meta(id integer primary key, path text, text text)")
            con.executemany(
                "insert into meta values (?,?,?)",
                [(i + 1, path, "text")
                 for i, path in enumerate(
                     ("z0-public/alpha.md", "z1-restricted/bravo.md", "z0-public/charlie.md"))],
            )
        con.close()

        report = ragcheck.check(self.cfg)

        self.assertIs(report["chunk_level"], False)
        self.assertEqual(report["chunks_missing"], 0)
        self.assertEqual(report["chunks_orphan"], 0)
        self.assertIn("startline", report["note"])


class TestCheckFailures(RagFixture):
    """Missing / broken databases must produce a note, never a traceback."""

    def test_absent_vector_store_is_reported_not_created(self):
        absent = self.home / "index" / "absent.sqlite"
        cfg = self._load(str(absent))
        report = ragcheck.check(cfg)
        self.assertFalse(report["ok"])
        self.assertTrue(report["note"].strip())
        self.assertFalse(absent.exists(), "check() must never create the database it inspects")
        self.assertEqual(report["chunks"], 5)   # the index side is still readable
        self.assertEqual(report["vectors"], 0)

    def test_absent_index_is_reported(self):
        self.index_db.unlink()
        report = ragcheck.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertTrue(report["note"].strip())

    def test_corrupt_vector_store_returns_note(self):
        self.vdb.write_bytes(b"this is not a sqlite database at all")
        report = ragcheck.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertTrue(report["note"].strip())

    def test_missing_table_returns_note(self):
        con = sqlite3.connect(self.vdb)
        with con:
            con.execute("drop table vectors")
            con.execute("create table other(x)")
        con.close()
        report = ragcheck.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertTrue(report["note"].strip())

    def test_missing_columns_return_note(self):
        for absent_column in ("model", "vec"):
            with self.subTest(column=absent_column):
                con = sqlite3.connect(self.vdb)
                with con:
                    con.execute("drop table vectors")
                    cols = [c for c in ("id", "path", "startline", "text", "hash", "model", "vec")
                            if c != absent_column]
                    con.execute(
                        "create table vectors("
                        + ", ".join(f"{c} blob" for c in cols)
                        + ")"
                    )
                    con.execute(f"insert into vectors({', '.join(cols)}) "
                                f"values ({', '.join('?' * len(cols))})",
                                [1] * len(cols))
                con.close()
                report = ragcheck.check(self.cfg)
                self.assertFalse(report["ok"])
                self.assertTrue(report["note"].strip())

    def test_corrupt_index_returns_note(self):
        self.index_db.write_bytes(b"nope")
        report = ragcheck.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertTrue(report["note"].strip())


class TestPrune(RagFixture):
    def test_dry_run_is_the_default_and_deletes_nothing(self):
        before_db = _snapshot(self.vdb)
        before_rows = self._rows()
        result = ragcheck.prune(self.cfg)
        self.assertEqual(set(result), PRUNE_KEYS)
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["refused"])
        self.assertTrue(result["ok"])
        # `removed` counts rows that would go away; the store itself must not move.
        self.assertEqual((result["removed"], result["kept"]), (2, 5))
        self.assertEqual(result["total"], 7)
        self.assertEqual(result["orphan_rows"], 2)
        self.assertFalse(result["external"], "fixture 的向量库就在 home 里")
        self.assertEqual(_snapshot(self.vdb), before_db)
        self.assertEqual(self._rows(), before_rows)

    def test_real_run_removes_only_orphans_and_is_idempotent(self):
        index_before = _snapshot(self.index_db)
        result = ragcheck.prune(self.cfg, dry_run=False)
        self.assertEqual((result["removed"], result["kept"]), (2, 5))
        self.assertFalse(result["dry_run"])
        self.assertFalse(result["refused"])
        self.assertEqual(result["reason"], "")
        survivors = self._rows()
        self.assertEqual([row[1] for row in survivors if row[1] in ORPHAN_PATHS], [])
        self.assertEqual(len(survivors), 5)
        # model/dim drift is reported by check(), not silently deleted here.
        self.assertEqual(sorted(row[2] for row in survivors).count("stale-model"), 1)
        self.assertEqual(sorted(row[3] for row in survivors).count(100), 1)
        # the index is never written from this path
        self.assertEqual(_snapshot(self.index_db), index_before)

        again = ragcheck.prune(self.cfg, dry_run=False)
        self.assertEqual((again["removed"], again["kept"]), (0, 5))
        self.assertFalse(again["refused"])
        self.assertEqual(self._rows(), survivors)

    def test_prune_on_absent_store_is_a_no_op(self):
        absent = self.home / "index" / "absent.sqlite"
        cfg = self._load(str(absent))
        result = ragcheck.prune(cfg)
        self.assertEqual((result["removed"], result["kept"]), (0, 0))
        self.assertFalse(result["refused"], "没有库不是「拒绝」，是「没得删」")
        self.assertIn("missing", result["note"])
        real = ragcheck.prune(cfg, dry_run=False)
        self.assertEqual((real["removed"], real["kept"]), (0, 0))
        self.assertFalse(real["refused"])
        self.assertFalse(absent.exists())

    def test_prune_never_deletes_when_the_index_is_unreadable(self):
        """With the index side unknown, every vector row would look like an
        orphan — so the safe answer is "delete nothing", reported as a refusal
        (a silent no-op would look like "nothing to do")."""
        self.index_db.write_bytes(b"nope")
        before = _snapshot(self.vdb)
        result = ragcheck.prune(self.cfg, dry_run=False)
        self.assertEqual((result["removed"], result["kept"]), (0, 7))
        self.assertTrue(result["refused"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], ragcheck.REFUSED_INDEX)
        self.assertIn("index", result["note"])
        self.assertEqual(_snapshot(self.vdb), before)

    # ---- the accident this repair exists for -----------------------------
    def test_prune_refuses_a_foreign_vector_store_the_2026_10_19_incident(self):
        """复现 2026-10-19 00:39 的事故组合。

        当时：``DSH_KB_HOME=<sandbox>/home`` 而 **漏设** ``DSH_KB_RAG_VECTORS_DB``
        ⇒ 向量库落到默认的生产路径。prune 拿 sandbox 那份只有 1 页的索引当对照面，
        判定生产库 5,521 行全是孤儿并全部删除（removed=5521, kept=0）。

        这里把同一个组合钉成回归：sandbox home（1 页）+ tmpdir 里 5,521 行的外部库。
        ①默认拒绝且外部库 sha256/mtime 完全未变；②显式放行外部库但没 --force 仍拒绝
        （5,520/5,521 > 50%）；③放行 + --force 才允许，且只删孤儿。
        """
        self._write_index([(1, "z0-public/only.md", 1, "only page")])
        external_dir = tempfile.TemporaryDirectory(prefix="dsh-kb-external-")
        self.addCleanup(external_dir.cleanup)
        external = Path(external_dir.name) / "vectors.sqlite"
        rows = [(1, "z0-public/only.md", 1, "only", "h1", DEFAULT_RAG_MODEL, _blob())]
        rows += [(i, f"z9-gone/page-{i:04d}.md", 1, "gone", f"h{i}", DEFAULT_RAG_MODEL, _blob())
                 for i in range(2, 5522)]
        self._write_vectors_at(external, rows)
        cfg = self._load(str(external))

        # ① 默认（没放行外部库）→ 拒绝，外部库一个字节都没动
        before = _snapshot(external)
        result = ragcheck.prune(cfg, dry_run=False)
        self.assertTrue(result["refused"])
        self.assertEqual(result["reason"], ragcheck.REFUSED_EXTERNAL)
        self.assertEqual(result["removed"], 0, "拒绝路径不许执行任何 DELETE")
        self.assertEqual(result["kept"], 5521)
        self.assertTrue(result["external"])
        self.assertEqual(result["total"], 5521)
        self.assertIn(str(external), result["note"])
        self.assertIn(str(self.home), result["note"])
        self.assertIn("allow_external_vectors_db", result["note"])
        self.assertEqual(_snapshot(external), before, "外部库 sha256/大小/mtime 必须完全未变")

        # ② 显式放行外部库、但没有 --force → 仍拒绝（99.98% > 50%）
        with mock.patch.dict(os.environ, {"DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB": "1"}):
            result = ragcheck.prune(cfg, dry_run=False)
        self.assertTrue(result["refused"])
        self.assertEqual(result["reason"], ragcheck.REFUSED_BULK)
        self.assertGreater(result["ratio"], 0.5)
        self.assertEqual(result["removed"], 0)
        self.assertIn("不属于同一知识库", result["note"])
        self.assertEqual(_snapshot(external), before)

        # ③ 放行 + --force → 才真的删，而且只删孤儿（非孤儿那一行留着）
        with mock.patch.dict(os.environ, {"DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB": "1"}):
            result = ragcheck.prune(cfg, dry_run=False, force=True)
        self.assertFalse(result["refused"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["removed"], 5520)
        self.assertEqual(result["kept"], 1)
        self.assertEqual(result["bypassed"], [ragcheck.REFUSED_BULK])
        con = sqlite3.connect(external)
        try:
            left = con.execute("select path from vectors order by id").fetchall()
        finally:
            con.close()
        self.assertEqual(left, [("z0-public/only.md",)])

    def test_prune_refuses_an_empty_index_against_a_non_empty_store(self):
        """防线 3：home 的索引为空（0 行）而向量库非空 —— 这正是"临时 home +
        默认生产向量库"的另一半形状。"""
        self._write_index([])                      # 索引存在但一行都没有
        before = _snapshot(self.vdb)

        result = ragcheck.prune(self.cfg, dry_run=False)

        self.assertTrue(result["refused"])
        self.assertEqual(result["reason"], ragcheck.REFUSED_EMPTY_HOME)
        self.assertEqual(result["removed"], 0)
        self.assertEqual(_snapshot(self.vdb), before)
        # --force 是明确的解锁方式；此时 7/7 = 100% 也命中大批量保护，两条一起记进
        # bypassed（纵深防御：同一个危险组合会被两条防线同时看见）
        forced = ragcheck.prune(self.cfg, dry_run=False, force=True)
        self.assertFalse(forced["refused"])
        self.assertEqual(forced["removed"], 7)
        self.assertEqual(forced["bypassed"],
                         [ragcheck.REFUSED_EMPTY_HOME, ragcheck.REFUSED_BULK])

    def test_allow_external_can_come_from_the_config_file(self):
        """放行开关也接受 config.json 的两种写法（嵌套 / 扁平）。"""
        external_dir = tempfile.TemporaryDirectory(prefix="dsh-kb-external-")
        self.addCleanup(external_dir.cleanup)
        external = Path(external_dir.name) / "vectors.sqlite"
        self._write_vectors_at(external, [
            (1, "z0-public/alpha.md", 1, "a", "h1", DEFAULT_RAG_MODEL, _blob()),
            (2, "z9-gone/x.md", 1, "x", "h2", DEFAULT_RAG_MODEL, _blob()),
        ])
        (self.home / "config.json").write_text(
            json.dumps({"rag": {"allow_external_vectors_db": True}}), encoding="utf-8")
        cfg = self._load(str(external))
        before = _snapshot(external)

        result = ragcheck.prune(cfg, dry_run=False)

        self.assertFalse(result["refused"], "配置里显式放行了")
        self.assertTrue(result["external"])
        self.assertEqual(result["removed"], 1, "2 行里 1 行孤儿（50%，未超阈值）")
        self.assertNotEqual(_snapshot(external), before)

        # 同一件事现在也走"真字段"这条路：``RagConfig`` 读了、``prune`` 认它，
        # 不再依赖 ragcheck 自己去翻 config.json（翻文件降级成兜底）。
        (self.home / "config.json").write_text(
            json.dumps({"rag": {"allow_external_vectors_db": True}}), encoding="utf-8")
        typed = self._load(str(external))
        self.assertIs(typed.rag.allow_external_vectors_db, True)
        self.assertTrue(ragcheck._allow_external_vectors_db(typed))
        # 扁平写法同样认；写成假值则继续拒绝
        (self.home / "config.json").write_text(
            json.dumps({"rag_allow_external_vectors_db": "no"}), encoding="utf-8")
        self.assertTrue(ragcheck.prune(self._load(str(external)), dry_run=False)["refused"])


class TestCasSchema(RagFixture):
    """The content-addressed schema: ``chunk_id`` identity, ``chunk_id`` deletes.

    A CAS store keys its rows by ``chunk_id`` and has **no** ``id`` column at
    all.  Two things therefore have to change and are pinned here: the chunk
    comparison uses ``chunk_id`` (which is what makes it survive a re-chunk),
    and ``prune`` must delete by that key — ``select id`` / ``delete ... where
    id=?`` is the second half of the "migrated database breaks the tooling"
    failure the deploy runbook measured.
    """

    #: (chunk_id, path, startline, text) — 5 chunks over 3 pages, mirroring
    #: ``INDEX_ROWS`` so the same expectations can be reused.
    CAS_INDEX = [
        ("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
        ("b" * 16, "z0-public/alpha.md", 16, "alpha two"),
        ("c" * 16, "z1-restricted/bravo.md", 1, "bravo one"),
        ("d" * 16, "z1-restricted/bravo.md", 16, "bravo two"),
        ("e" * 16, "z0-public/charlie.md", 1, "charlie one"),
    ]

    def _write_cas_index(self, rows=None):
        """Verbatim CAS DDL (``migrate-cas.py``) — no ``id`` column on purpose."""
        con = sqlite3.connect(self.index_db)
        try:
            with con:
                con.execute("drop table if exists meta")
                con.execute(
                    "create table meta(rid integer primary key, chunk_id text not null unique,"
                    " path text not null, startline integer not null, text text not null)"
                )
                for rid, (cid, path, line, text) in enumerate(rows or self.CAS_INDEX, start=1):
                    con.execute(
                        "insert into meta(rid, chunk_id, path, startline, text)"
                        " values (?,?,?,?,?)", (rid, cid, path, line, text))
        finally:
            con.close()

    def _write_cas_vectors(self, rows):
        """CAS vectors: ``chunk_id`` primary key, ``dim`` column, no ``id``/``hash``.

        Columns are named explicitly — a positional insert here once wrote the
        model into ``dim`` and the test still "passed" its own fixture check.
        """
        con = sqlite3.connect(self.vdb)
        try:
            with con:
                con.execute("drop table if exists vectors")
                con.execute(
                    "create table vectors(chunk_id text not null primary key, model text not null,"
                    " dim integer not null, path text not null, startline integer not null,"
                    " text text not null, vec blob not null, updated integer not null) without rowid"
                )
                con.executemany(
                    "insert into vectors(chunk_id, model, dim, path, startline, text, vec, updated)"
                    " values (?,?,?,?,?,?,?,?)", rows)
        finally:
            con.close()

    def _cas_row(self, cid, path, line, text, model=DEFAULT_RAG_MODEL, dim=DEFAULT_RAG_DIM,
                 blob=None):
        return (cid, model, dim, path, line, text,
                _blob(dim) if blob is None else blob, 1_700_000_000)

    def test_a_clean_cas_pair_reports_ok(self):
        """``check`` must work on a migrated pair at all — the first blocker."""
        self._write_cas_index()
        self._write_cas_vectors([
            self._cas_row(*row) for row in self.CAS_INDEX
        ])
        report = ragcheck.check(self.cfg)
        self.assertTrue(report["ok"], report["note"])
        self.assertIs(report["chunk_level"], True)
        self.assertIn("chunk_id", report["note"], "note 要说清用的是哪个身份")
        self.assertEqual((report["missing"], report["orphan"]), (0, 0))
        self.assertEqual((report["chunks_missing"], report["chunks_orphan"]), (0, 0))

    def test_chunk_identity_is_chunk_id_not_the_line_number(self):
        """A block that moved keeps its id; a block that vanished loses it.

        This is the property the CAS key exists for, and it is invisible to the
        positional comparison: both chunks here sit at the *same* startlines, so
        ``(path, startline)`` would call the store perfectly healthy.
        """
        self._write_cas_index([
            ("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
            ("b" * 16, "z0-public/alpha.md", 16, "alpha two"),
            ("c" * 16, "z1-restricted/bravo.md", 1, "bravo one"),
        ])
        self._write_cas_vectors([
            # same path+line as the first index chunk, but a *stale* id: the
            # block was edited in place, so its content (and id) changed.
            self._cas_row("f" * 16, "z0-public/alpha.md", 1, "alpha one EDITED"),
            self._cas_row("b" * 16, "z0-public/alpha.md", 16, "alpha two"),
            self._cas_row("c" * 16, "z1-restricted/bravo.md", 1, "bravo one"),
        ])
        report = ragcheck.check(self.cfg)
        # page-level is blind to it: every page still has vectors
        self.assertEqual((report["missing"], report["orphan"]), (0, 0))
        self.assertEqual(report["chunks_missing"], 1, "the new content has no vector")
        self.assertEqual(report["chunks_orphan"], 1, "the old content's vector is stale")
        self.assertEqual(report["chunks_missing_sample"], ["z0-public/alpha.md:1"],
                         "样例要能落到人能找到的位置")
        self.assertFalse(report["ok"])

    def test_dim_mismatch_uses_the_row_dim_not_the_config_default(self):
        """CAS stores ``dim`` per row, so that column is the authority.

        A store built at 8 dims while the config says 1024 is *internally*
        consistent — it is the wrong model width, not a corrupt blob, and the
        per-row ``dim`` is what tells the two apart.
        """
        self._write_cas_index([("a" * 16, "z0-public/alpha.md", 1, "alpha one")])
        self._write_cas_vectors([
            self._cas_row("a" * 16, "z0-public/alpha.md", 1, "alpha one", dim=8),
            # a genuinely truncated blob: length != dim*4 even against its own dim
            self._cas_row("b" * 16, "z0-public/alpha.md", 16, "alpha two",
                          dim=8, blob=b"\x00" * 12),
        ])
        report = ragcheck.check(self.cfg)
        self.assertEqual(report["dim_mismatch"], 1, "only the truncated blob counts")
        self.assertFalse(report["ok"])

    def test_model_mismatch_is_still_detected(self):
        self._write_cas_index([("a" * 16, "z0-public/alpha.md", 1, "alpha one")])
        self._write_cas_vectors([
            self._cas_row("a" * 16, "z0-public/alpha.md", 1, "alpha one", model="stale-model"),
        ])
        report = ragcheck.check(self.cfg)
        self.assertEqual(report["model_mismatch"], 1)
        self.assertIn("not model", report["note"])
        self.assertNotIn("dim", report["note"].split("—")[0], "dim 是干净的，别混报")

    def test_prune_deletes_by_chunk_id(self):
        """``delete ... where id=?`` raises on a CAS store; this is its fix.

        The fixture is five rows with two orphans (40%), deliberately under
        ``BULK_DELETE_LIMIT`` so this exercises the delete itself rather than
        the guard — the guard has its own test below.
        """
        self._write_cas_index([
            ("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
            ("b" * 16, "z0-public/alpha.md", 16, "alpha two"),
            ("c" * 16, "z1-restricted/bravo.md", 1, "bravo one"),
        ])
        gone_one, gone_two = "9" * 16, "8" * 16
        self._write_cas_vectors([
            self._cas_row("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
            self._cas_row("b" * 16, "z0-public/alpha.md", 16, "alpha two"),
            self._cas_row("c" * 16, "z1-restricted/bravo.md", 1, "bravo one"),
            self._cas_row(gone_one, "z0-public/gone.md", 1, "gone"),
            self._cas_row(gone_two, "z0-public/gone-two.md", 1, "gone"),
        ])
        before = _snapshot(self.vdb)

        dry = ragcheck.prune(self.cfg)
        self.assertEqual((dry["removed"], dry["kept"]), (2, 3))
        self.assertEqual(_snapshot(self.vdb), before, "dry-run 不许动库")

        result = ragcheck.prune(self.cfg, dry_run=False)
        self.assertFalse(result["refused"], result["note"])
        self.assertTrue(result["ok"], result["note"])
        self.assertEqual((result["removed"], result["kept"]), (2, 3))

        con = sqlite3.connect(self.vdb)
        try:
            left = sorted(con.execute("select chunk_id from vectors").fetchall())
        finally:
            con.close()
        self.assertEqual(left, [("a" * 16,), ("b" * 16,), ("c" * 16,)],
                         "只删孤儿，按 chunk_id")
        self.assertEqual(ragcheck.prune(self.cfg, dry_run=False)["removed"], 0, "幂等")

    def test_the_bulk_guard_still_applies_to_a_cas_store(self):
        """Depth defence is schema-independent — verify it kept working.

        2 of 3 rows (67%) is over ``BULK_DELETE_LIMIT``: the first version of
        this fixture tripped it, which is exactly the behaviour a migration must
        not lose.
        """
        self._write_cas_index([("a" * 16, "z0-public/alpha.md", 1, "alpha one")])
        self._write_cas_vectors([
            self._cas_row("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
            self._cas_row("9" * 16, "z0-public/gone.md", 1, "gone"),
            self._cas_row("8" * 16, "z0-public/gone-two.md", 1, "gone"),
        ])
        before = _snapshot(self.vdb)

        refused = ragcheck.prune(self.cfg, dry_run=False)
        self.assertTrue(refused["refused"])
        self.assertEqual(refused["reason"], ragcheck.REFUSED_BULK)
        self.assertEqual(refused["removed"], 0, "拒绝路径不许执行任何 DELETE")
        self.assertEqual(_snapshot(self.vdb), before)

        forced = ragcheck.prune(self.cfg, dry_run=False, force=True)
        self.assertFalse(forced["refused"], forced["note"])
        self.assertEqual(forced["removed"], 2)
        self.assertEqual(forced["bypassed"], [ragcheck.REFUSED_BULK])

    def test_prune_removes_only_whole_gone_pages_not_single_chunks(self):
        """The page-level rule is unchanged: a page that still exists keeps its rows."""
        self._write_cas_index([("a" * 16, "z0-public/alpha.md", 1, "alpha one")])
        self._write_cas_vectors([
            self._cas_row("b" * 16, "z0-public/alpha.md", 16, "a chunk the index lost"),
        ])
        report = ragcheck.check(self.cfg)
        self.assertEqual(report["chunks_orphan"], 1, "check 看得见")
        result = ragcheck.prune(self.cfg, dry_run=False)
        self.assertEqual(result["removed"], 0, "页面还在 ⇒ prune 不碰它")
        con = sqlite3.connect(self.vdb)
        try:
            self.assertEqual(con.execute("select count(*) from vectors").fetchone()[0], 1)
        finally:
            con.close()

    def test_a_legacy_store_under_a_cas_index_degrades_instead_of_crashing(self):
        """Mixed pair (index migrated, vectors not yet): report, never traceback.

        This is the normal shape *during* an upgrade, and it is the reason the
        comparison requires the **same** identity on both sides: ``chunk_id``
        against ``(path, startline)`` yields an all-noise difference (every hash
        differs from every tuple), so an unguarded comparison would report every
        chunk as missing and every row as an orphan.
        """
        self._write_cas_index()
        self._write_vectors()                      # the pre-CAS 7-row fixture
        report = ragcheck.check(self.cfg)
        self.assertEqual(report["chunks"], 5)
        self.assertEqual(report["vectors"], 7)
        self.assertEqual(report["orphan"], 2, "页级口径照常工作")
        self.assertIs(report["chunk_level"], False, "身份不同 ⇒ 块级比对不成立")
        self.assertEqual((report["chunks_missing"], report["chunks_orphan"]), (0, 0),
                         "不许把「身份不同」报成「块全缺」")
        self.assertIn("different identities", report["note"])
        self.assertIn("migrate-cas.py", report["note"], "note 要给出下一步")
        self.assertFalse(report["ok"])
        self.assertTrue(report["note"].strip())

    def test_a_cas_store_under_a_legacy_index_degrades_too(self):
        """The other mixed direction: old index, migrated vector store."""
        self._write_index(INDEX_ROWS)              # pre-CAS meta (id/path/startline/text)
        self._write_cas_vectors([
            self._cas_row("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
        ])
        report = ragcheck.check(self.cfg)
        self.assertIs(report["chunk_level"], False)
        self.assertEqual((report["chunks_missing"], report["chunks_orphan"]), (0, 0))
        self.assertIn("different identities", report["note"])
        self.assertIn("startline", report["note"])
        self.assertIn("chunk_id", report["note"])

    def test_a_cas_store_with_a_pathless_row_still_counts_orphans(self):
        """``_path_of_key`` resolves a hash back through meta; a bad row stays a bad row.

        ``path`` is ``NOT NULL`` in the CAS DDL, so a pathless row cannot be
        written through the real schema — it is reproduced here by dropping the
        constraint, which is the shape a hand-edited store would have.  What
        matters is that a NULL path is counted as ``bad_path`` and does not
        crash the sort or the JSON dump.
        """
        self._write_cas_index([("a" * 16, "z0-public/alpha.md", 1, "alpha one")])
        con = sqlite3.connect(self.vdb)
        try:
            with con:
                con.execute("drop table vectors")
                con.execute(
                    "create table vectors(chunk_id text primary key, model text, dim integer,"
                    " path text, startline integer, text text, vec blob, updated integer) without rowid"
                )
                con.executemany(
                    "insert into vectors values (?,?,?,?,?,?,?,?)", [
                        self._cas_row("a" * 16, "z0-public/alpha.md", 1, "alpha one"),
                        self._cas_row("7" * 16, None, 1, "no path"),
                    ])
        finally:
            con.close()

        report = ragcheck.check(self.cfg)          # 不许抛

        self.assertEqual(report["bad_path"], 1)
        self.assertEqual(report["orphan"], 0, "坏行单独计数，不混进孤儿")
        json.dumps(report)


class TestRagConfig(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name) / "home"
        self.home.mkdir()

    def _load(self, **env: str) -> Config:
        with mock.patch.dict(os.environ):
            for key in [k for k in os.environ if k.startswith("DSH_KB_")]:
                os.environ.pop(key, None)
            os.environ.update(env)
            return load_config(home=self.home)

    def test_defaults_match_the_documented_layout(self):
        cfg = self._load()
        self.assertEqual(DEFAULT_RAG_VECTORS_DB, "")
        self.assertEqual(cfg.rag.vectors_db, str(self.home / "index" / "vectors.sqlite"))
        self.assertEqual(cfg.rag.model, DEFAULT_RAG_MODEL)
        self.assertEqual(cfg.rag.model, "bge-m3")
        self.assertEqual(cfg.rag.dim, DEFAULT_RAG_DIM)
        self.assertEqual(cfg.rag.dim, 1024)

    def test_environment_overrides_the_defaults(self):
        cfg = self._load(
            DSH_KB_RAG_VECTORS_DB="/tmp/elsewhere.sqlite",
            DSH_KB_RAG_MODEL="e5-mistral",
            DSH_KB_RAG_DIM="384",
        )
        self.assertEqual(cfg.rag.vectors_db, "/tmp/elsewhere.sqlite")
        self.assertEqual(cfg.rag.model, "e5-mistral")
        self.assertEqual(cfg.rag.dim, 384)

    def test_config_file_rag_block_is_honoured(self):
        (self.home / "config.json").write_text(
            '{"rag": {"vectors_db": "/tmp/from-file.sqlite", "model": "custom", "dim": 768}}',
            encoding="utf-8",
        )
        cfg = self._load()
        self.assertEqual(cfg.rag.vectors_db, "/tmp/from-file.sqlite")
        self.assertEqual(cfg.rag.model, "custom")
        self.assertEqual(cfg.rag.dim, 768)

    def test_flat_rag_keys_are_honoured(self):
        (self.home / "config.json").write_text(
            '{"rag_model": "flat-model", "rag_dim": "512"}', encoding="utf-8"
        )
        cfg = self._load()
        self.assertEqual(cfg.rag.model, "flat-model")
        self.assertEqual(cfg.rag.dim, 512)

    def test_environment_beats_the_config_file(self):
        (self.home / "config.json").write_text('{"rag": {"model": "from-file"}}', encoding="utf-8")
        cfg = self._load(DSH_KB_RAG_MODEL="from-env")
        self.assertEqual(cfg.rag.model, "from-env")

    def test_existing_config_keys_keep_working(self):
        """Adding rag settings must not change how the old keys are read."""
        (self.home / "config.json").write_text(
            '{"source": "local", "local_root": "/tmp/bundle", "stale_minutes": 42,'
            ' "never": ["private"], "zones": [{"name": "z0-public", "index": true}],'
            ' "rag": {"model": "custom"}}',
            encoding="utf-8",
        )
        cfg = self._load()
        self.assertEqual(cfg.source, "local")
        self.assertEqual(cfg.local_root, "/tmp/bundle")
        self.assertEqual(cfg.stale_minutes, 42)
        self.assertEqual(cfg.never, ("private",))
        self.assertEqual([z.name for z in cfg.zones], ["z0-public"])
        self.assertEqual(cfg.rag.model, "custom")
        self.assertEqual(cfg.unknown_keys, ())

    def test_unknown_keys_are_still_reported(self):
        (self.home / "config.json").write_text(
            '{"future_key": 1, "rag": {"dims": 3}}', encoding="utf-8"
        )
        cfg = self._load()
        self.assertEqual(set(cfg.unknown_keys), {"future_key", "rag.dims"})

    def test_allow_external_vectors_db_is_a_typed_field_not_a_lookup(self):
        """放行开关现在是 ``RagConfig`` 的字段，三条来源都要能落到它上面。

        在此之前 ``ragcheck`` 只能自己去翻 ``config.json``（它刻意不动
        ``config.py``），于是"配置读到什么"取决于翻文件的那段代码 —— 这次
        把它变成真字段，翻文件那条路降级成兜底。
        """
        self.assertIsNone(self._load().rag.allow_external_vectors_db,
                          "默认必须是 None（没表态）—— 空 home + 带外库要被拒绝")
        (self.home / "config.json").write_text(
            '{"rag": {"allow_external_vectors_db": true}}', encoding="utf-8")
        self.assertIs(self._load().rag.allow_external_vectors_db, True)
        (self.home / "config.json").write_text(
            '{"rag_allow_external_vectors_db": "no"}', encoding="utf-8")
        self.assertIs(self._load().rag.allow_external_vectors_db, False)
        self.assertIs(self._load(
            DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB="yes").rag.allow_external_vectors_db, True)
        self.assertTrue(ragcheck._allow_external_vectors_db(
            self._load(DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB="1")))
        # 环境变量优先于配置文件（与其它 rag 键的优先级一致）
        (self.home / "config.json").write_text(
            '{"rag": {"allow_external_vectors_db": false}}', encoding="utf-8")
        self.assertTrue(ragcheck._allow_external_vectors_db(
            self._load(DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB="1")))

    def test_enabled_defaults_to_file_presence(self):
        cfg = self._load(DSH_KB_RAG_VECTORS_DB=str(self.home / "nope.sqlite"))
        self.assertIsNone(cfg.rag.enabled)
        self.assertFalse(ragcheck.enabled(cfg))
        present = self.home / "yes.sqlite"
        present.write_bytes(b"")
        cfg = self._load(DSH_KB_RAG_VECTORS_DB=str(present))
        self.assertTrue(ragcheck.enabled(cfg))

    def test_enabled_can_be_forced(self):
        cfg = self._load(
            DSH_KB_RAG_VECTORS_DB=str(self.home / "nope.sqlite"),
            DSH_KB_RAG_ENABLED="1",
        )
        self.assertIs(cfg.rag.enabled, True)
        self.assertTrue(ragcheck.enabled(cfg))
        cfg = self._load(DSH_KB_RAG_ENABLED="off")
        self.assertIs(cfg.rag.enabled, False)
        self.assertFalse(ragcheck.enabled(cfg))


class TestVectorsDbPath(RagFixture):
    def test_vectors_db_resolves_the_configured_path(self):
        self.assertEqual(ragcheck.vectors_db(self.cfg), self.vdb)

    def test_tilde_is_expanded(self):
        cfg = self._load(DSH_KB_RAG_VECTORS_DB="~/rag.sqlite")
        self.assertEqual(ragcheck.vectors_db(cfg), Path.home() / "rag.sqlite")


if __name__ == "__main__":
    unittest.main()
