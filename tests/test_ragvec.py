"""ragvec 的单元测试：开关、fill/rrf 两种融合、降级、zone 过滤。"""

from __future__ import annotations

import sqlite3
import struct
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import ragvec  # noqa: E402
from dsh_kb.index import Hit  # noqa: E402

try:                      # numpy is optional: the vector arm is opt-in.
    ragvec._numpy()       # ask the module, so DSH_KB_NUMPY_PATH counts too
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


def _vec(*values: float) -> bytes:
    return struct.pack("<%df" % len(values), *values)


class _Cfg(SimpleNamespace):
    """最小 Config 替身：只需要 rag / mirror_dir / query_prefixes。"""

    def query_prefixes(self, zone: str):
        return ("/",) if zone == "all" else (zone, "z0-public/", "z1-restricted/")


def make_cfg(tmp: Path, *, zone_prefixes=("z1-restricted/", "z0-public/")) -> _Cfg:
    db = tmp / "vectors.sqlite"
    if db.exists():            # 同一个 tmp 目录被用第二次（zone 过滤那组）
        db.unlink()
    con = sqlite3.connect(db)
    with con:
        con.execute(
            "create table vectors(path text, startline integer, vec blob,"
            " model text, dim integer)"
        )
        con.executemany(
            "insert into vectors values(?,?,?,?,?)",
            [
                ("z1-restricted/a.md", 1, _vec(1.0, 0.0), "bge-m3", 8),
                ("z0-public/b.md", 1, _vec(0.0, 1.0), "bge-m3", 8),
            ],
        )
    con.close()
    mirror = tmp / "mirror"
    (mirror / "z1-restricted").mkdir(parents=True, exist_ok=True)
    (mirror / "z1-restricted" / "a.md").write_text("第一行\n第二行\n第三行\n", encoding="utf-8")
    (mirror / "z0-public").mkdir(parents=True, exist_ok=True)
    (mirror / "z0-public" / "b.md").write_text("b 页\n", encoding="utf-8")
    cfg = _Cfg(
        rag=SimpleNamespace(vectors_db=str(db), vector_enabled=True, vector_mode="fill"),
        mirror_dir=mirror,
    )
    cfg.query_prefixes = lambda zone: zone_prefixes
    return cfg


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed (the vector arm is optional)")
class TestVectorArm(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = make_cfg(self.tmp)
        # 固定查询向量 = 第一维，必然最像 a.md
        self._orig = ragvec.VectorArm.embed
        ragvec.VectorArm.embed = lambda self, text: ragvec._numpy().asarray([1.0, 0.0], dtype="<f4")

    def tearDown(self):
        ragvec.VectorArm.embed = self._orig
        ragvec._CACHE.clear()

    def test_fill_keeps_keyword_order_and_appends(self):
        cfg = make_cfg(self.tmp, zone_prefixes=("z1-restricted/",))   # 只放行 z1
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(cfg, "查询", 3, kw, 40, zone="z01", mode="fill")
        self.assertEqual(merged[0].path, "z1-restricted/kw.md")  # 关键词顺序不变
        self.assertIn("fill", note)
        self.assertGreater(len(merged), 1)  # 向量候选补在后面
        self.assertNotIn("z0-public/b.md", [h.path for h in merged], "zone 过滤必须生效")

    def test_rrf_reorders_but_same_members(self):
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(self.cfg, "查询", 3, kw, 40, zone="z01", mode="rrf")
        self.assertIn("RRF", note)
        paths = [h.path for h in merged]
        self.assertIn("z1-restricted/a.md", paths)
        self.assertIn("z1-restricted/kw.md", paths)
        # 并列时的既定行为：两路各自 rank1 的候选 RRF 分相同，**关键词优先**（插入顺序）。
        # 这条断言是行为契约，不是巧合 —— 改融合公式时若这里红了，说明并列裁决变了。
        self.assertEqual(merged[0].path, "z1-restricted/kw.md")

    def test_unreachable_endpoint_raises_not_crashes(self):
        ragvec.VectorArm.embed = self._orig  # 走真实 urllib
        import os

        os.environ["DSH_KB_EMBED_ENDPOINT"] = "http://127.0.0.1:1/api/embed"
        try:
            with self.assertRaises(Exception):
                ragvec.augment(self.cfg, "查询", 3, [], 40, zone="z01")
        finally:
            os.environ.pop("DSH_KB_EMBED_ENDPOINT", None)


# --------------------------------------------------------------------------- #
# 孤儿向量：向量库里有、当前索引里没有的块
#
# 背景（2026-11-06）：``VectorArm.load()`` 原来整表扫 ``vectors``，不校验
# ``chunk_id`` 是否还在 ``meta`` 里。重分块后旧行照常加载，``self.meta`` 里的
# ``(path, startline)`` 指向新索引里**完全不同**的内容，向量臂于是静默返回错位
# 行号 —— 错位比没有更糟，因为它不报错。下面这几条把「错位在结构上不可能」
# 钉住：孤儿必须在建矩阵**之前**被丢掉。
# --------------------------------------------------------------------------- #

def _cas_store(db: Path, entries, *, index_ids=None) -> Path:
    """建一对同目录文件：``vectors.sqlite``（CAS 形状）+ 可选 ``kb.sqlite``。

    ``entries`` 是 ``(chunk_id, path, startline, vec_bytes)``；``index_ids`` 是
    索引里真实存在的 chunk_id 集合（``None`` = 不建 kb.sqlite）。
    """
    con = sqlite3.connect(db)
    with con:
        con.execute(
            "create table vectors(chunk_id text primary key, model text, dim integer,"
            " path text, startline integer, text text, vec blob, updated integer)")
        con.executemany(
            "insert into vectors(chunk_id, model, dim, path, startline, text, vec, updated)"
            " values(?,?,?,?,?,?,?,0)",
            [(cid, "bge-m3", len(vec) // 4, path, line, "", vec)
             for cid, path, line, vec in entries])
    con.close()
    if index_ids is not None:
        _write_index(db.parent / "kb.sqlite", index_ids)
    return db


def _write_index(kb: Path, ids) -> Path:
    """写一个只有 ``meta(chunk_id, …)`` 的最小 CAS 索引（够校验身份即可）。"""
    con = sqlite3.connect(kb)
    with con:
        con.execute(
            "create table if not exists meta(rid integer primary key, chunk_id text"
            " not null unique, path text not null, startline integer not null,"
            " text text not null)")
        con.execute("delete from meta")
        con.executemany(
            "insert into meta(rid, chunk_id, path, startline, text) values(?,?,?,?,?)",
            [(i + 1, cid, "z1-restricted/x.md", 1, "") for i, cid in enumerate(ids)])
    con.close()
    return kb


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed (the vector arm is optional)")
class TestOrphanVectors(unittest.TestCase):
    """向量行必须属于当前索引；不属于的不能进矩阵、更不能变成命中。"""

    def setUp(self):
        import tempfile

        self.tmp = Path(tempfile.mkdtemp())
        self._orig = ragvec.VectorArm.embed
        ragvec.VectorArm.embed = lambda self, text: ragvec._numpy().asarray(
            [1.0, 0.0], dtype="<f4")

    def tearDown(self):
        ragvec.VectorArm.embed = self._orig
        ragvec._CACHE.clear()

    def test_orphan_rows_are_dropped_before_the_matrix(self):
        """① 有一条孤儿（指向已不存在的块）时，只留下有效块，查询只返回有效块。"""
        db = _cas_store(
            self.tmp / "vectors.sqlite",
            [("live000000000001", "z1-restricted/keep.md", 1, _vec(1.0, 0.0)),
             ("dead000000000001", "z1-restricted/gone.md", 7, _vec(0.0, 1.0))],
            index_ids={"live000000000001"},
        )
        arm = ragvec.VectorArm(db).load()
        self.assertEqual(arm.dropped, 1)
        self.assertEqual(arm.meta, [("z1-restricted/keep.md", 1)])
        self.assertEqual(arm.matrix.shape, (1, 2))
        hits = arm.query("查询", prefixes=())
        self.assertEqual([h[0] for h in hits], ["z1-restricted/keep.md"])
        self.assertIn("已忽略 1 条", arm.health_note())

    def test_all_orphans_raise_and_search_falls_back_to_lexical(self):
        """② 全是孤儿：加载直接抛错，``index.search`` 把它变成一条 note、照常返回词法命中。"""
        from dsh_kb import index
        from dsh_kb.config import Config, RagConfig

        home = self.tmp / "home"
        page = home / "mirror" / "z1-restricted" / "page.md"
        page.parent.mkdir(parents=True)
        page.write_text("CANARY-KEYWORD-4242 出现一次\n第二行\n", encoding="utf-8")
        vectors = home / "index" / "vectors.sqlite"
        vectors.parent.mkdir(parents=True)
        _cas_store(vectors,
                   [("orphan0000000001", "z1-restricted/page.md", 1, _vec(1.0, 0.0))],
                   index_ids=set())        # 索引里没有任何块 ⇒ 这一行是孤儿
        cfg = Config(home=home, rag=RagConfig(
            vectors_db=str(vectors), vector_enabled=True, vector_always=True))
        index.build(cfg)                    # 真实索引：词法臂必须还有命中

        with self.assertRaises(RuntimeError) as ctx:
            ragvec.VectorArm(vectors).load()
        self.assertIn("都不在当前索引中", str(ctx.exception))

        result = index.search(cfg, "CANARY-KEYWORD-4242", zone="z01", limit=3)
        self.assertTrue(result.hits, "词法命中必须照常返回")
        self.assertEqual(result.hits[0].path, "z1-restricted/page.md")
        self.assertIn("vector arm skipped", result.note)
        self.assertIn("都不在当前索引中", result.note)

    def test_missing_index_keeps_old_behaviour(self):
        """③a 没有同目录 kb.sqlite（临时 home / 拷出来的向量库）：按旧行为全部加载 + 说明。"""
        db = _cas_store(
            self.tmp / "vectors.sqlite",
            [("aaa0000000000001", "z1-restricted/a.md", 1, _vec(1.0, 0.0)),
             ("bbb0000000000001", "z0-public/b.md", 1, _vec(0.0, 1.0))],
            index_ids=None,
        )
        arm = ragvec.VectorArm(db).load()
        self.assertEqual(len(arm.meta), 2, "校验不了 ≠ 拒绝加载")
        self.assertEqual(arm.dropped, 0)
        self.assertIn("未能校验索引一致性", arm.unverified)

    def test_unreadable_index_keeps_old_behaviour_without_raising(self):
        """③b kb.sqlite 存在但打不开：同样退回旧行为，不抛异常，且 note 说出来。"""
        db = _cas_store(
            self.tmp / "vectors.sqlite",
            [("aaa0000000000001", "z1-restricted/a.md", 1, _vec(1.0, 0.0))],
            index_ids=None,
        )
        (self.tmp / "kb.sqlite").write_bytes(b"this is not a sqlite database\n" * 8)
        arm = ragvec.VectorArm(db).load()
        self.assertEqual(len(arm.meta), 1)
        self.assertIn("不可读", arm.unverified)
        self.assertIn("未能校验索引一致性", arm.unverified)
        # 端到端：note 里必须说出来，而且不能改变命中
        cfg = _Cfg(rag=SimpleNamespace(vectors_db=str(db), vector_enabled=True,
                                       vector_mode="fill"),
                   mirror_dir=self.tmp / "mirror")
        cfg.query_prefixes = lambda zone: ("z1-restricted/", "z0-public/")
        merged, note = ragvec.augment(cfg, "查询", 3, [], 40, zone="z01", mode="fill")
        self.assertIn("未能校验索引一致性", note)
        self.assertEqual([h.path for h in merged], ["z1-restricted/a.md"])

    def test_empty_index_drops_everything(self):
        """空索引是「校验成功且没有块」，不是「校验不了」——两种必须区别对待。"""
        db = _cas_store(
            self.tmp / "vectors.sqlite",
            [("aaa0000000000001", "z1-restricted/a.md", 1, _vec(1.0, 0.0))],
            index_ids=set(),
        )
        with self.assertRaises(RuntimeError) as ctx:
            ragvec.VectorArm(db).load()
        self.assertIn("都不在当前索引中", str(ctx.exception))

    def test_legacy_vectors_without_chunk_id_still_load(self):
        """CAS 迁移前的向量库（无 chunk_id 列）：不能因为加了校验就突然读不出来。"""
        cfg = make_cfg(self.tmp)          # 这份夹具就是旧格式 + 无 kb.sqlite
        arm = ragvec.VectorArm(cfg.rag.vectors_db).load()
        self.assertEqual(len(arm.meta), 2)
        self.assertEqual(arm.dropped, 0)
        self.assertIn("旧格式", arm.unverified)

    def test_cache_key_follows_the_index(self):
        """重分块只改 kb.sqlite：缓存键必须跟着索引走，否则旧（含孤儿的）矩阵会被复用。"""
        db = _cas_store(
            self.tmp / "vectors.sqlite",
            [("live000000000001", "z1-restricted/keep.md", 1, _vec(1.0, 0.0)),
             ("late000000000002", "z1-restricted/later.md", 1, _vec(0.0, 1.0))],
            index_ids={"live000000000001"},
        )
        first = ragvec.VectorArm(db).load()
        self.assertEqual(len(first.meta), 1)
        kb = self.tmp / "kb.sqlite"
        _write_index(kb, {"live000000000001", "late000000000002"})
        import os
        later = kb.stat().st_mtime_ns + 5_000_000_000   # 模拟「一次重建发生在之后」
        os.utime(kb, ns=(later, later))
        second = ragvec.VectorArm(db).load()
        self.assertEqual(len(second.meta), 2, "索引变了，缓存必须失效")
        self.assertEqual(second.dropped, 0)


if __name__ == "__main__":
    unittest.main()