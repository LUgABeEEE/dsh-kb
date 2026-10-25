"""``kb-rag/kbask.py`` 的 zone 过滤与仓库内实现必须**同义**。

两条检索路径读了同一份 ``config.json``：``dsh-kb search``（关键词）走
``Config.query_prefixes``，``kbask.py``（向量）自己展开一遍。自己展开一遍本身
不是错 —— 错的是**展开规则不一样**，而且没人会发现：向量侧原本连 zone 参数
都没有，一个 ``z0`` 查询可以把 ``z1-restricted`` 的块当证据交给生成模型。

所以这里不去断言"实现了某些字符串"，而是把三处放在一起对答案：

* ``Config.query_prefixes``  —— 关键词臂（生产）
* ``kbask.zone_prefixes``    —— 向量臂（生产）
* ``bench.zone_path_allows`` —— 度量台（判断单条路径放不放行）

``kbask.py`` 不在本仓库里（它是 kb-rag 的运行目录），所以文件缺失时跳过；
存在时必须一致。numpy 用桩替代：被测函数是纯字符串逻辑，不需要真的向量。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import closing
from pathlib import Path

from dsh_kb.config import load_config

#: Optional parity check against an external ``kbask.py``.  Skipped unless
#: ``DSH_KB_TEST_KBASK_PY`` points at one; nothing in this repo depends on it.
KBASK_PY = Path(os.environ["DSH_KB_TEST_KBASK_PY"]) if os.environ.get("DSH_KB_TEST_KBASK_PY") else None

ZONES = [
    {"name": "z0-public", "index": True},
    {"name": "z1-restricted", "index": True},
    {"name": "z2-sensitive", "index": False},
    {"name": "z3-skills", "index": False},
]
SELECTORS = ("z0", "z1", "z2", "z3", "z01", "default", "all")


def _load_kbask(config_path: Path):
    """按路径加载 kbask.py，numpy 用桩顶住（模块顶层就会 import）。"""
    stub = types.ModuleType("numpy")
    stub.float32 = float
    stub.asarray = lambda value, dtype=None: value
    stub.frombuffer = lambda *a, **k: []
    saved = sys.modules.get("numpy")
    sys.modules["numpy"] = stub
    try:
        spec = importlib.util.spec_from_file_location("kbask_under_test", KBASK_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.CONFIG_JSON = str(config_path)
        return module
    finally:
        if saved is None:
            sys.modules.pop("numpy", None)
        else:
            sys.modules["numpy"] = saved


class KbaskZoneParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if KBASK_PY is None or not KBASK_PY.exists():
            raise unittest.SkipTest("no external kbask.py configured (DSH_KB_TEST_KBASK_PY)")
        cls._tmp = tempfile.TemporaryDirectory()
        cfg_path = Path(cls._tmp.name) / "config.json"
        cfg_path.write_text(json.dumps({"zones": ZONES}), encoding="utf-8")
        cls.cfg = load_config(home=Path(cls._tmp.name))
        cls.kbask = _load_kbask(cfg_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_prefixes_agree_with_the_keyword_arm(self):
        for selector in SELECTORS:
            self.assertEqual(
                self.kbask.zone_prefixes(selector), self.cfg.query_prefixes(selector),
                f"selector {selector!r} 两侧展开不一致 —— 这正是要修的不对称")

    def test_z0_never_admits_a_restricted_prefix(self):
        prefixes = self.kbask.zone_prefixes("z0")
        self.assertIn("z0-public/", prefixes)
        self.assertNotIn("z1-restricted/", prefixes)

    def test_layout_dirs_are_searchable_under_every_selector(self):
        for selector in SELECTORS:
            prefixes = self.kbask.zone_prefixes(selector)
            self.assertIn("_meta/", prefixes)
            self.assertIn("_incoming/", prefixes)

    def test_where_clause_filters_exactly_what_the_keyword_arm_filters(self):
        """把 SQL 片段真的跑一遍：结论必须与 ``Config.query_prefixes`` 一致。"""
        rows = ["index.md", "log.md", "z0-public/a.md", "z1-restricted/b.md",
                "z2-sensitive/c.md", "z3-skills/d.md", "_meta/e.md", "_incoming/f.md"]
        with closing(sqlite3.connect(":memory:")) as con:
            con.execute("create table vectors(path text)")
            con.executemany("insert into vectors values (?)", [(p,) for p in rows])
            for selector in SELECTORS:
                clause, params = self.kbask.zone_where(selector)
                got = {p for (p,) in con.execute(
                    f"select path from vectors where {clause}", params)}
                prefixes = self.cfg.query_prefixes(selector)
                want = {p for p in rows if "/" not in p or p.startswith(prefixes)}
                self.assertEqual(got, want, f"selector {selector!r} 的 SQL 语义不一致")

    def test_unknown_selector_degrades_to_root_docs_only(self):
        """选择器谁都匹配不上时退化成"只看根级文档"，而不是"看全部"。"""
        with closing(sqlite3.connect(":memory:")) as con:
            con.execute("create table vectors(path text)")
            con.executemany("insert into vectors values (?)",
                            [("index.md",), ("z1-restricted/b.md",)])
            clause, params = self.kbask.zone_where("z9")
            got = {p for (p,) in con.execute(
                f"select path from vectors where {clause}", params)}
            self.assertEqual(got, {"index.md"})
