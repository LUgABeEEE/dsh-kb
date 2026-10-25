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


if __name__ == "__main__":
    unittest.main()