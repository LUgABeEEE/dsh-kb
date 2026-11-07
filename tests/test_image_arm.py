"""图片多模态侧车（P2）的单元测试。

覆盖四件必须成立的事：

1. 图片库不存在 / 图片路失败时，``augment()`` 仍返回纯文本结果并留下 note；
2. RRF 融合后 ``score`` 是 RRF 值（不是余弦），图片路权重独立可配；
3. 构建端**一条输入一个请求**（多元素 ``input`` 会静默只返回第一条向量）；
4. 凭据只认 ``CREDENTIAL_REF`` 那一个键（宿主可改名），且绝不进统计/错误/日志。

大多数用例把 ``VectorArm.query`` / ``ImageArm.query`` 打桩，所以**不需要 numpy**；
真正读库的排序用例在 ``TestImageArmIntegration`` 里，没有 numpy 时跳过（与
``test_ragvec`` 同一约定）。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import image_embed, ragvec  # noqa: E402
from dsh_kb.config import Config, RagConfig  # noqa: E402
from dsh_kb.index import Hit, build as index_build, search as index_search  # noqa: E402

#: 本机把 numpy 借给可选路的位置（仅当直接 import 不到时用）。测试本身不依赖
#: 这个路径：没有 numpy 也能跑除 ``TestImageArmIntegration`` 以外的全部用例。
_NUMPY_FALLBACKS = ("/data/agent/tools/kb-rerank/lib",)


def _probe_numpy() -> bool:
    try:
        ragvec._numpy()
        return True
    except Exception:
        return False


def _borrow_numpy() -> bool:
    """有 numpy 就用；没有就借本机已知的一份（仅此进程，测试后清掉）。

    刻意**不在 import 时**改环境：``tests/test_ragvec.py`` 也靠
    ``DSH_KB_NUMPY_PATH`` 决定 skip/fail，模块间不该互相影响。
    """
    if _probe_numpy():
        return True
    for candidate in _NUMPY_FALLBACKS:
        if os.environ.get("DSH_KB_NUMPY_PATH") or not Path(candidate).is_dir():
            continue
        os.environ["DSH_KB_NUMPY_PATH"] = candidate
        if _probe_numpy():
            return True
        os.environ.pop("DSH_KB_NUMPY_PATH", None)
    return False

SENTINEL_KEY = "sentinel-payg-key-do-not-log-0123456789"


# ------------------------------------------------------------------ helpers --


class _Cfg(SimpleNamespace):
    """最小 Config 替身：rag / mirror_dir / query_prefixes。"""

    def query_prefixes(self, zone: str):
        return ("z1-restricted/", "z0-public/")


def make_cfg(tmp: Path, *, image_enabled: bool | None = None,
             image_weight=None, image_db=None, vector_always: bool = False,
             image_prefixes=None,
             prefixes=("z1-restricted/", "z0-public/")) -> _Cfg:
    rag = SimpleNamespace(
        vectors_db=str(tmp / "vectors.sqlite"),
        vector_enabled=True,
        vector_mode="rrf",
        vector_always=vector_always,
    )
    if image_enabled is not None:
        rag.image_arm_enabled = image_enabled
    if image_weight is not None:
        rag.image_arm_weight = image_weight
    if image_db is not None:
        rag.image_db = str(image_db)
    if image_prefixes is not None:
        rag.image_arm_prefixes = image_prefixes
    cfg = _Cfg(rag=rag, mirror_dir=tmp / "mirror")
    cfg.query_prefixes = lambda zone: prefixes
    return cfg


def write_credentials(path: Path, refs: dict) -> None:
    lines = ["version: 1", "refs:"]
    for key, value in refs.items():
        lines.append(f"  {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_image_db(path: Path, rows, dim: int, model: str = "doubao-embedding-vision-251215") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        "create table images(path text primary key, sha text not null, dim integer not null, "
        "model text not null, vec blob not null, updated integer not null)"
    )
    for rel, vec in rows:
        con.execute(
            "insert into images values(?,?,?,?,?,?)",
            (rel, "sha-" + rel, dim, model, struct.pack("<%df" % len(vec), *vec), 1700000000),
        )
    con.commit()
    con.close()


def image_bytes(size: int = 3072, seed: int = 0) -> bytes:
    """够大、但不真的是图片：构建端不解码，只看字节。"""
    return b"\x89PNG\r\n\x1a\n" + bytes([seed % 251]) * (size - 8)


class _FakeAPI:
    """替代 ``image_embed._post``：记录每次请求体，返回固定维度向量。"""

    def __init__(self, dim: int = 4, fail_times: int = 0,
                 always_fail: bool = False, key: str = SENTINEL_KEY):
        self.dim = dim
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.key = key
        self.calls: list[dict] = []

    def __call__(self, payload, key, timeout):
        self.calls.append(payload)
        if self.always_fail or self.fail_times > 0:
            if self.fail_times:
                self.fail_times -= 1
            raise RuntimeError("HTTP 500: upstream sad, auth=" + (self.key or ""))
        return {"data": {"embedding": [1.0] + [0.0] * (self.dim - 1), "object": "embedding"}}


# --------------------------------------------------- augment / RRF 融合 -----


class TestImageFusion(unittest.TestCase):
    """图片第三路并入 ``augment`` 的行为契约（全部打桩，无需 numpy）。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._orig_vec = ragvec.VectorArm.query
        self._orig_img = ragvec.ImageArm.query
        self._vec_hits: list = []
        self._img_hits: list = []
        ragvec.VectorArm.query = (
            lambda _self, text, prefixes, depth=32, hyde_text=None: list(self._vec_hits)
        )
        ragvec.ImageArm.query = lambda _self, text, prefixes, depth=32: list(self._img_hits)

    def tearDown(self):
        ragvec.VectorArm.query = self._orig_vec
        ragvec.ImageArm.query = self._orig_img
        ragvec._CACHE.clear()
        ragvec._IMAGE_CACHE.clear()

    # -- ① 关闭 / 库不存在 / 失败降级 -------------------------------------

    def test_disabled_by_default_and_never_touches_image_arm(self):
        cfg = make_cfg(self.tmp)                       # 没有 image_arm_enabled 字段
        called = []

        def boom(*args, **kwargs):
            called.append(True)
            raise AssertionError("图片路在默认关闭时不该被调用")

        ragvec.ImageArm.query = boom
        self._vec_hits = [("z1-restricted/a.md", 1, 0.9)]
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(cfg, "查询", 5, kw, 40, mode="rrf")
        self.assertFalse(called)
        self.assertNotIn("图片路", note)
        self.assertIn("z1-restricted/a.md", [h.path for h in merged])

    def test_missing_db_degrades_to_text_results(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.tmp / "nope.sqlite")
        ragvec.ImageArm.query = self._orig_img            # 走真实的 load()
        self._vec_hits = [("z1-restricted/a.md", 1, 0.9)]
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(cfg, "查询", 5, kw, 40, mode="rrf")
        self.assertIn("图片路跳过：库不存在", note)
        self.assertIn("z1-restricted/a.md", [h.path for h in merged])
        self.assertIn("z1-restricted/kw.md", [h.path for h in merged])

    def test_image_failure_is_silent_and_noted(self):
        cfg = make_cfg(self.tmp, image_enabled=True)
        db = self.tmp / "vectors-image.sqlite"
        db.write_bytes(b"")                              # 存在但不是 sqlite

        def boom(*args, **kwargs):
            raise RuntimeError("boom")

        ragvec.ImageArm.query = boom
        self._vec_hits = [("z1-restricted/a.md", 1, 0.9)]
        merged, note = ragvec.augment(cfg, "查询", 5, [], 40, mode="rrf")
        self.assertIn("图片路跳过", note)
        self.assertIn("boom", note)
        self.assertEqual([h.path for h in merged], ["z1-restricted/a.md"])

    def test_missing_numpy_degrades(self):
        db = self.tmp / "numpy.sqlite"
        make_image_db(db, [("z1-restricted/a.png", [1.0, 0.0])], dim=2)
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=db)
        ragvec.ImageArm.query = self._orig_img
        self._vec_hits = [("z1-restricted/a.md", 1, 0.9)]
        with mock.patch.object(ragvec, "_numpy", side_effect=ImportError("no numpy here")):
            merged, note = ragvec.augment(cfg, "查询", 5, [], 40, mode="rrf")
        self.assertIn("图片路跳过", note)
        self.assertIn("numpy", note)
        self.assertEqual([h.path for h in merged], ["z1-restricted/a.md"])

    def test_schema_mismatch_degrades_to_text(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.tmp / "bad.sqlite")
        con = sqlite3.connect(str(self.tmp / "bad.sqlite"))
        con.execute("create table images(path text)")
        con.commit()
        con.close()
        ragvec.ImageArm.query = self._orig_img                  # 走真实 load()
        self._vec_hits = [("z1-restricted/a.md", 1, 0.9)]
        merged, note = ragvec.augment(cfg, "查询", 5, [], 40, mode="rrf")
        self.assertIn("图片路跳过", note)
        self.assertIn("schema", note)
        self.assertEqual([h.path for h in merged], ["z1-restricted/a.md"])

    def test_no_candidates_at_all_still_returns_existing(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.tmp / "nope.sqlite")
        ragvec.ImageArm.query = self._orig_img
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(cfg, "查询", 5, kw, 40, mode="fill")
        self.assertEqual([h.path for h in merged], ["z1-restricted/kw.md"])
        self.assertIn("无候选", note)
        self.assertIn("图片路跳过", note)

    # -- ② RRF 分值与权重 -------------------------------------------------

    def test_rrf_score_is_rrf_value_not_cosine(self):
        cfg = make_cfg(self.tmp, image_enabled=True)      # 权重缺省 → 0.6
        self._vec_hits = [("z1-restricted/a.md", 1, 0.9)]
        self._img_hits = [("z1-restricted/x.png", 1, 0.8)]
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(cfg, "查询", 5, kw, 40, mode="rrf")
        by_path = {h.path: h for h in merged}
        # 关键词 rank1 / 文本向量 rank1 / 图片 rank1 各自只拿到一路的 RRF 分。
        self.assertAlmostEqual(by_path["z1-restricted/kw.md"].score, 1.0 / (ragvec.RRF_K + 1), places=12)
        self.assertAlmostEqual(by_path["z1-restricted/a.md"].score, 1.0 / (ragvec.RRF_K + 1), places=12)
        self.assertAlmostEqual(by_path["z1-restricted/x.png"].score,
                               ragvec.DEFAULT_IMAGE_WEIGHT / (ragvec.RRF_K + 1), places=12)
        self.assertAlmostEqual(ragvec.DEFAULT_IMAGE_WEIGHT, 0.6)
        self.assertIn("图片路 1 候选", note)
        # 命中片段不是二进制乱码
        self.assertEqual(by_path["z1-restricted/x.png"].snippet, "[图片] x.png")

    def test_image_weight_is_configurable(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_weight=1.5)
        self._vec_hits = []
        self._img_hits = [("z1-restricted/x.png", 1, 0.8)]
        merged, note = ragvec.augment(cfg, "查询", 5, [], 40, mode="rrf")
        self.assertEqual([h.path for h in merged], ["z1-restricted/x.png"])
        self.assertAlmostEqual(merged[0].score, 1.5 / (ragvec.RRF_K + 1), places=12)
        self.assertIn("权重 1.5", note)

    def test_bad_weight_falls_back_to_default(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_weight="not-a-number")
        enabled, weight, _db = ragvec._image_settings(cfg)
        self.assertTrue(enabled)
        self.assertEqual(weight, ragvec.DEFAULT_IMAGE_WEIGHT)

    def test_settings_defaults_without_config_fields(self):
        cfg = make_cfg(self.tmp)                          # 完全没有图片字段
        enabled, weight, db = ragvec._image_settings(cfg)
        self.assertFalse(enabled)
        self.assertEqual(weight, ragvec.DEFAULT_IMAGE_WEIGHT)
        self.assertEqual(db, self.tmp / "vectors-image.sqlite")

        cfg2 = make_cfg(self.tmp, image_enabled=True, image_db=self.tmp / "custom.sqlite")
        enabled2, _weight2, db2 = ragvec._image_settings(cfg2)
        self.assertTrue(enabled2)
        self.assertEqual(db2, self.tmp / "custom.sqlite")

    def test_env_overrides_config(self):
        cfg = make_cfg(self.tmp)                       # 配置文件里没有图片字段
        with mock.patch.dict(os.environ, {
            "DSH_KB_RAG_IMAGE_ARM_ENABLED": "1",
            "DSH_KB_RAG_IMAGE_ARM_WEIGHT": "2.5",
            "DSH_KB_RAG_IMAGE_DB": str(self.tmp / "env.sqlite"),
        }):
            enabled, weight, db = ragvec._image_settings(cfg)
        self.assertTrue(enabled)
        self.assertEqual(weight, 2.5)
        self.assertEqual(db, self.tmp / "env.sqlite")
        # 环境变量显式 0 也能把配置文件里的 true 关掉（与其余 rag 项同优先级）。
        with mock.patch.dict(os.environ, {"DSH_KB_RAG_IMAGE_ARM_ENABLED": "0"}):
            self.assertFalse(ragvec._image_settings(make_cfg(self.tmp, image_enabled=True))[0])

    # -- ③ fill 模式 -------------------------------------------------------

    def test_fill_mode_appends_image_after_keyword_hits(self):
        cfg = make_cfg(self.tmp, image_enabled=True)
        self._vec_hits = []
        self._img_hits = [("z1-restricted/x.png", 1, 0.8)]
        kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
        merged, note = ragvec.augment(cfg, "查询", 5, kw, 40, mode="fill")
        self.assertEqual(merged[0].path, "z1-restricted/kw.md")
        self.assertEqual(merged[1].path, "z1-restricted/x.png")
        self.assertEqual(merged[1].snippet, "[图片] x.png")
        self.assertAlmostEqual(merged[1].score, round(-1.0 / (ragvec.RRF_K + 1), 6), places=12)
        self.assertIn("fill", note)
        self.assertIn("图片路 1 候选", note)


# ------------------------------------------------------ ImageArm（真库） ----


class TestImageArmIntegration(unittest.TestCase):
    """读真库：归一化、排序、zone 前缀过滤、与 augment 的拼装。"""

    def setUp(self):
        if not _borrow_numpy():
            self.skipTest("numpy is not installed (the image arm is optional)")
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "vectors-image.sqlite"
        make_image_db(self.db, [("z1-restricted/a.png", [1.0, 0.0]),
                                ("z0-public/b.png", [0.0, 1.0])], dim=2)
        self._orig_embed = image_embed.embed_text
        image_embed.embed_text = lambda text, **kw: [1.0, 0.0]   # 查询向量固定朝 a.png

    def tearDown(self):
        image_embed.embed_text = self._orig_embed
        ragvec._CACHE.clear()
        ragvec._IMAGE_CACHE.clear()

    def test_load_vectors_normalizes(self):
        paths, matrix, dim, model = image_embed.load_vectors(self.db)
        self.assertEqual(paths, ["z1-restricted/a.png", "z0-public/b.png"])
        self.assertEqual(dim, 2)
        self.assertEqual(model, "doubao-embedding-vision-251215")
        self.assertAlmostEqual(float(matrix[0][0]), 1.0, places=6)

    def test_query_orders_and_filters_zone(self):
        arm = ragvec.ImageArm(self.db)
        hits = arm.query("架构图", ("z1-restricted/",), depth=5)
        self.assertEqual([h[0] for h in hits], ["z1-restricted/a.png"])
        self.assertEqual(hits[0][1], 1)                 # 图片命中行号恒为 1
        self.assertAlmostEqual(hits[0][2], 1.0, places=6)
        both = ragvec.ImageArm(self.db).query("架构图", (), depth=5)
        self.assertEqual([h[0] for h in both], ["z1-restricted/a.png", "z0-public/b.png"])

    def test_zone_miss_note_and_prefix_opt_in(self):
        db3 = self.tmp / "z3-only.sqlite"
        make_image_db(db3, [("z3-skills/x.png", [1.0, 0.0])], dim=2)
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=db3,
                       prefixes=("z1-restricted/", "z0-public/"))
        orig_vec = ragvec.VectorArm.query
        ragvec.VectorArm.query = lambda *args, **kwargs: []
        embed_calls: list = []
        orig_embed = image_embed.embed_text
        image_embed.embed_text = lambda text, **kw: embed_calls.append(text) or [1.0, 0.0]
        try:
            _merged, note = ragvec.augment(cfg, "配色", 5, [], 40, mode="rrf")
            self.assertIn("图片路 0 候选", note)          # 库里有图，但 z01 前缀下没有
            self.assertIn("库内 1 张", note)
            # 前缀完全不命中时不该再花钱/花时间去嵌入查询（欠费时更不该重试）
            self.assertEqual(embed_calls, [])
            with mock.patch.dict(os.environ,
                                 {"DSH_KB_RAG_IMAGE_ARM_PREFIXES": "z3-skills/"}):
                merged2, note2 = ragvec.augment(cfg, "配色", 5, [], 40, mode="rrf")
        finally:
            ragvec.VectorArm.query = orig_vec
            image_embed.embed_text = orig_embed
        self.assertEqual(embed_calls, ["配色"])
        self.assertIn("图片路 1 候选", note2)              # 显式 opt-in 后才可见
        self.assertEqual(merged2[0].path, "z3-skills/x.png")

    def test_account_overdue_403_degrades(self):
        """2026-11-06 21:20 实测的 403 AccountOverdueError 必须只降级、不抛。"""
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.db)
        image_embed.embed_text = self._orig_embed         # 真实 embed_text 会先读 key 再发请求
        creds = self.tmp / "creds.yaml"
        write_credentials(creds, {image_embed.CREDENTIAL_REF: SENTINEL_KEY})
        def _overdue(*args, **kwargs):
            # 每次都新建并抛出：HTTPError 的 body 读一次就空了，重试共用同一个
            # 实例会让第 2/3 次拿到空 body（这正是真实代码重试时的形状）。
            raise urllib.error.HTTPError(
                "https://ark.example/embeddings/multimodal", 403, "Forbidden", {},
                io.BytesIO(b'{"error":{"code":"AccountOverdueError","message":"overdue"}}'),
            )
        orig_vec = ragvec.VectorArm.query
        ragvec.VectorArm.query = lambda *args, **kwargs: []
        image_embed._clear_key_cache()
        try:
            with mock.patch.dict(os.environ, {"DSH_KB_IMAGE_CREDENTIALS": str(creds)}), \
                 mock.patch.object(image_embed.urllib.request, "urlopen", side_effect=_overdue), \
                 mock.patch("time.sleep", return_value=None):
                image_embed._clear_key_cache()
                merged, note = ragvec.augment(cfg, "架构图", 5, [], 40, mode="rrf")
        finally:
            ragvec.VectorArm.query = orig_vec
            image_embed._clear_key_cache()
        self.assertEqual(merged, [])
        self.assertIn("图片路跳过", note)
        self.assertIn("403", note)
        self.assertIn("AccountOverdueError", note)
        self.assertNotIn(SENTINEL_KEY, note)

    def test_missing_key_degrades_without_http(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.db)
        image_embed.embed_text = self._orig_embed            # 换回真实函数：先要 key
        creds = self.tmp / "empty-creds.yaml"
        write_credentials(creds, {})                          # refs 段是空的
        orig_vec = ragvec.VectorArm.query
        ragvec.VectorArm.query = lambda self, text, prefixes, depth=32, hyde_text=None: []
        image_embed._clear_key_cache()
        try:
            with mock.patch.dict(os.environ, {"DSH_KB_IMAGE_CREDENTIALS": str(creds)}):
                image_embed._clear_key_cache()
                merged, note = ragvec.augment(cfg, "架构图", 5, [], 40, mode="rrf")
        finally:
            ragvec.VectorArm.query = orig_vec
            image_embed._clear_key_cache()
        self.assertIn("图片路跳过", note)
        self.assertIn(image_embed.CREDENTIAL_REF, note)
        self.assertEqual(merged, [])

    def test_augment_end_to_end_fill_with_real_store(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.db)
        orig_vec = ragvec.VectorArm.query
        ragvec.VectorArm.query = lambda self, text, prefixes, depth=32, hyde_text=None: []
        try:
            kw = [Hit(path="z1-restricted/kw.md", line=1, score=-9.0, snippet="kw")]
            merged, note = ragvec.augment(cfg, "架构图", 5, kw, 40, mode="fill")
        finally:
            ragvec.VectorArm.query = orig_vec
        self.assertEqual(merged[0].path, "z1-restricted/kw.md")
        self.assertIn("z1-restricted/a.png", [h.path for h in merged])
        self.assertIn("[图片] a.png", [h.snippet for h in merged])
        self.assertIn("图片路 2 候选", note)


# ------------------------------------------------------------ 发现 / 凭据 --


class TestDiscoverAndCredentials(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.creds = self.tmp / "creds.yaml"
        self._orig_creds_env = os.environ.get("DSH_KB_IMAGE_CREDENTIALS")
        os.environ["DSH_KB_IMAGE_CREDENTIALS"] = str(self.creds)
        image_embed._clear_key_cache()

    def tearDown(self):
        if self._orig_creds_env is None:
            os.environ.pop("DSH_KB_IMAGE_CREDENTIALS", None)
        else:
            os.environ["DSH_KB_IMAGE_CREDENTIALS"] = self._orig_creds_env
        image_embed._clear_key_cache()

    def test_discover_skips_tiny_and_non_images(self):
        mirror = self.tmp / "mirror"
        (mirror / "z1-restricted").mkdir(parents=True)
        (mirror / "z0-public" / "deep").mkdir(parents=True)
        (mirror / ".hidden").mkdir()
        (mirror / "z1-restricted" / "arch.png").write_bytes(image_bytes(3072, 1))
        (mirror / "z0-public" / "shot.JPG").write_bytes(image_bytes(4096, 2))
        (mirror / "z0-public" / "deep" / "nested.webp").write_bytes(image_bytes(3072, 3))
        (mirror / "z3-skills").mkdir()
        (mirror / "z3-skills" / "tiny.png").write_bytes(image_bytes(131, 4))
        (mirror / "z0-public" / "readme.md").write_text("not an image", encoding="utf-8")
        (mirror / ".hidden" / "secret.png").write_bytes(image_bytes(3072, 5))

        items, skipped = image_embed.discover_images(mirror)
        self.assertEqual([it.rel for it in items], [
            "z0-public/deep/nested.webp", "z0-public/shot.JPG", "z1-restricted/arch.png",
        ])
        self.assertEqual(skipped["small"], 1)
        self.assertEqual(skipped["unreadable"], 0)
        self.assertEqual(items[2].sha, image_embed.sha256_file(mirror / "z1-restricted" / "arch.png"))
        self.assertEqual(items[2].size, 3072)

    def test_discover_missing_mirror_raises(self):
        with self.assertRaises(FileNotFoundError):
            image_embed.discover_images(self.tmp / "nope")

    def test_reads_only_the_configured_ref(self):
        write_credentials(self.creds, {"SUBSCRIPTION_KEY": "plan-secret", image_embed.CREDENTIAL_REF: SENTINEL_KEY})
        self.assertEqual(image_embed.api_key(), SENTINEL_KEY)
        self.assertEqual(image_embed.api_key(required=True), SENTINEL_KEY)

    def test_plan_key_is_not_a_fallback(self):
        write_credentials(self.creds, {"SUBSCRIPTION_KEY": "plan-secret"})
        self.assertIsNone(image_embed.api_key())
        with self.assertRaises(image_embed.ImageAuthError) as ctx:
            image_embed.api_key(required=True)
        message = str(ctx.exception)
        self.assertIn(image_embed.CREDENTIAL_REF, message)
        self.assertNotIn("plan-secret", message)

    def test_missing_credentials_file(self):
        self.assertIsNone(image_embed.api_key())
        with self.assertRaises(image_embed.ImageAuthError):
            image_embed.api_key(required=True)

    def test_loose_parser_handles_quotes_and_stops_at_next_section(self):
        text = ("version: 1\nrefs:\n"
                '  TEST_KEY: "' + SENTINEL_KEY + '"\n'
                "  OTHER: plain\n"
                "records:\n  NESTED: nope\n")
        refs = image_embed._parse_refs_loose(text)
        self.assertEqual(refs["TEST_KEY"], SENTINEL_KEY)
        self.assertEqual(refs["OTHER"], "plain")
        self.assertNotIn("NESTED", refs)

    def test_extract_embedding_accepts_object_and_array(self):
        self.assertEqual(image_embed._extract_embedding({"data": {"embedding": [0.5, 0.25]}}), [0.5, 0.25])
        self.assertEqual(image_embed._extract_embedding({"data": [{"embedding": [1.0]}]}), [1.0])
        with self.assertRaises(image_embed.ImageStoreError):
            image_embed._extract_embedding({"data": {"embedding": []}})


# ------------------------------------------------------------------ 构建 --


class TestBuild(unittest.TestCase):
    """构建端：逐条请求、幂等、重试、schema、统计。HTTP 全部打桩。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mirror = self.tmp / "mirror"
        (self.mirror / "z1-restricted").mkdir(parents=True)
        (self.mirror / "z0-public").mkdir(parents=True)
        self.files = {
            "z1-restricted/arch.png": self.mirror / "z1-restricted" / "arch.png",
            "z0-public/shot.jpg": self.mirror / "z0-public" / "shot.jpg",
            "z0-public/palette.webp": self.mirror / "z0-public" / "palette.webp",
        }
        for i, path in enumerate(self.files.values()):
            path.write_bytes(image_bytes(3072, i + 1))
        self.db = self.tmp / "index" / "vectors-image.sqlite"
        self.creds = self.tmp / "creds.yaml"
        write_credentials(self.creds, {image_embed.CREDENTIAL_REF: SENTINEL_KEY})
        self._orig_creds_env = os.environ.get("DSH_KB_IMAGE_CREDENTIALS")
        os.environ["DSH_KB_IMAGE_CREDENTIALS"] = str(self.creds)
        image_embed._clear_key_cache()
        self.api = _FakeAPI(dim=4)
        self._real_post = image_embed._post          # 403 用例要绕开假 _post
        self._post_patch = mock.patch.object(image_embed, "_post", self.api)
        self._post_patch.start()

    def tearDown(self):
        self._post_patch.stop()
        if self._orig_creds_env is None:
            os.environ.pop("DSH_KB_IMAGE_CREDENTIALS", None)
        else:
            os.environ["DSH_KB_IMAGE_CREDENTIALS"] = self._orig_creds_env
        image_embed._clear_key_cache()

    def _rows(self):
        con = sqlite3.connect(str(self.db))
        try:
            return con.execute("select path, sha, dim, model, vec, updated from images").fetchall()
        finally:
            con.close()

    def test_one_request_per_image_and_rows_written(self):
        stats = image_embed.build(self.db, self.mirror, dim=4)
        self.assertEqual(stats["discovered"], 3)
        self.assertEqual(stats["embedded"], 3)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(len(self.api.calls), 3)
        for payload in self.api.calls:
            # 致命坑：多元素 input 不报错但只返回第一条向量。必须永远只有一条。
            self.assertEqual(len(payload["input"]), 1)
            self.assertEqual(payload["model"], image_embed.DEFAULT_MODEL)
            self.assertEqual(payload["dimensions"], image_embed.DEFAULT_DIM)
            self.assertEqual(payload["input"][0]["type"], "image_url")
            self.assertTrue(payload["input"][0]["image_url"]["url"].startswith("data:image/"))

        rows = self._rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual({r[2] for r in rows}, {4})
        self.assertEqual({r[3] for r in rows}, {image_embed.DEFAULT_MODEL})
        for rel, sha, dim, _model, blob, updated in rows:
            self.assertEqual(dim, 4)
            self.assertEqual(len(blob), 16)          # 4 维 * 4 字节
            self.assertGreater(updated, 0)
            self.assertEqual(sha, image_embed.sha256_file(self.mirror / rel))
        self.assertNotIn(SENTINEL_KEY, json.dumps(stats))

    def test_second_build_is_idempotent_and_needs_no_http(self):
        image_embed.build(self.db, self.mirror, dim=4)
        self.assertEqual(len(self.api.calls), 3)
        stats = image_embed.build(self.db, self.mirror, dim=4)
        self.assertEqual(stats["unchanged"], 3)
        self.assertEqual(stats["todo"], 0)
        self.assertEqual(stats["embedded"], 0)
        self.assertEqual(len(self.api.calls), 3, "未变的文件不该再发请求")

    def test_changed_file_is_reembedded(self):
        image_embed.build(self.db, self.mirror, dim=4)
        target = self.files["z1-restricted/arch.png"]
        target.write_bytes(image_bytes(3072, 99))
        stats = image_embed.build(self.db, self.mirror, dim=4)
        self.assertEqual(stats["embedded"], 1)
        self.assertEqual(stats["unchanged"], 2)
        self.assertEqual(len(self.api.calls), 4)

    def test_limit_then_resume(self):
        first = image_embed.build(self.db, self.mirror, dim=4, limit=1)
        self.assertEqual(first["embedded"], 1)
        self.assertEqual(first["limited_to"], 1)
        second = image_embed.build(self.db, self.mirror, dim=4)
        self.assertEqual(second["embedded"], 2)
        self.assertEqual(second["unchanged"], 1)

    def test_retry_three_attempts_then_success(self):
        self.api.fail_times = 2
        with mock.patch("time.sleep", return_value=None):
            stats = image_embed.build(self.db, self.mirror, dim=4, limit=1)
        self.assertEqual(stats["embedded"], 1)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(len(self.api.calls), 3)      # 1 + 2 次重试

    def test_retry_exhausted_redacts_key(self):
        self.api.always_fail = True
        with mock.patch("time.sleep", return_value=None):
            stats = image_embed.build(self.db, self.mirror, dim=4, limit=1)
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["embedded"], 0)
        self.assertEqual(len(self.api.calls), image_embed.RETRIES)
        dumped = json.dumps(stats, ensure_ascii=False)
        self.assertNotIn(SENTINEL_KEY, dumped)
        self.assertIn("***", dumped)

    def test_http_403_is_not_retried(self):
        """欠费 403 不该重试：账户状态不会因为再等 1.5 秒变好。"""
        calls: list = []

        def _raise(*args, **kwargs):
            calls.append(1)
            raise urllib.error.HTTPError(
                "https://ark.example/x", 403, "Forbidden", {},
                io.BytesIO(b'{"error":{"code":"AccountOverdueError"}}'),
            )

        with mock.patch.object(image_embed, "_post", self._real_post), \
             mock.patch.object(image_embed.urllib.request, "urlopen", side_effect=_raise):
            with self.assertRaises(RuntimeError) as ctx:
                image_embed.embed_text("x", key=SENTINEL_KEY)
        self.assertEqual(len(calls), 1, "4xx 只允许一次请求")
        self.assertIn("403", str(ctx.exception))
        self.assertIn("AccountOverdueError", str(ctx.exception))
        self.assertNotIn(SENTINEL_KEY, str(ctx.exception))

    def test_http_5xx_is_retried(self):
        state = {"n": 0}

        def _post(payload, key, timeout):
            state["n"] += 1
            if state["n"] < 3:
                raise image_embed.ImageHTTPError(500, "HTTP 500: upstream boom")
            return {"data": {"embedding": [1.0, 0.0]}}

        with mock.patch.object(image_embed, "_post", _post), \
             mock.patch("time.sleep", return_value=None):
            vec = image_embed.embed_text("x", key=SENTINEL_KEY)
        self.assertEqual(vec, [1.0, 0.0])
        self.assertEqual(state["n"], 3)

    def test_missing_key_fails_before_any_http(self):
        write_credentials(self.creds, {"SUBSCRIPTION_KEY": "plan-secret"})
        image_embed._clear_key_cache()
        with self.assertRaises(image_embed.ImageAuthError):
            image_embed.build(self.db, self.mirror, dim=4)
        self.assertEqual(self.api.calls, [])

    def test_schema_mismatch_is_rejected(self):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db))
        con.execute("create table images(path text)")
        con.commit()
        con.close()
        with self.assertRaises(image_embed.ImageStoreError) as ctx:
            image_embed.build(self.db, self.mirror, dim=4)
        self.assertIn("schema", str(ctx.exception))

    def test_load_vectors_rejects_bad_schema(self):
        self.db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(self.db))
        con.execute("create table images(path text, vec blob)")
        con.commit()
        con.close()
        with self.assertRaises(image_embed.ImageStoreError):
            image_embed.load_vectors(self.db)

    def test_prune_removes_vanished_rows(self):
        image_embed.build(self.db, self.mirror, dim=4)
        self.files["z0-public/shot.jpg"].unlink()
        stats = image_embed.build(self.db, self.mirror, dim=4, prune=True)
        self.assertEqual(stats["pruned"], 1)
        self.assertEqual(len(self._rows()), 2)

    def test_status_reports_freshness(self):
        image_embed.build(self.db, self.mirror, dim=4)
        info = image_embed.status(self.db, self.mirror)
        self.assertTrue(info["exists"])
        self.assertEqual(info["total"], 3)
        self.assertEqual(info["dims"], [4])
        self.assertEqual(info["models"], [image_embed.DEFAULT_MODEL])
        self.assertEqual(info["zones"], {"z0-public/": 2, "z1-restricted/": 1})
        self.assertEqual(info["orphan"], [])
        self.assertEqual(info["stale"], [])
        self.assertEqual(info["untracked"], [])

        self.files["z0-public/shot.jpg"].unlink()
        info2 = image_embed.status(self.db, self.mirror)
        self.assertEqual(info2["orphan"], ["z0-public/shot.jpg"])

    def test_embed_text_and_image_use_single_item(self):
        image_embed.embed_text("一张架构图", key=SENTINEL_KEY)
        payload = self.api.calls[-1]
        self.assertEqual(len(payload["input"]), 1)
        self.assertEqual(payload["input"][0]["type"], "text")
        self.assertTrue(payload["instructions"].endswith("Query:"))

        image_embed.embed_image(self.files["z1-restricted/arch.png"], key=SENTINEL_KEY)
        payload = self.api.calls[-1]
        self.assertEqual(len(payload["input"]), 1)
        self.assertEqual(payload["input"][0]["type"], "image_url")

    def test_cli_build_and_status(self):
        self.api.dim = image_embed.DEFAULT_DIM       # CLI 走默认维度 2048
        out = io.StringIO()
        with redirect_stdout(out):
            code = image_embed.main(["--build", "--mirror", str(self.mirror),
                                     "--db", str(self.db), "--json",
                                     "--limit", "2", "--workers", "2"])
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["embedded"], 2)
        self.assertEqual(payload["db"], str(self.db))

        out = io.StringIO()
        with redirect_stdout(out):
            code = image_embed.main(["--status", "--db", str(self.db), "--json"])
        self.assertEqual(code, 0)
        info = json.loads(out.getvalue())
        self.assertEqual(info["total"], 2)


class TestImagePrefixes(unittest.TestCase):
    """major-3 的开关：默认跟随 zone，显式 opt-in 才能看到 z3 的图。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_default_follows_zone_prefixes(self):
        cfg = make_cfg(self.tmp)
        self.assertEqual(ragvec._image_prefixes(cfg, ("z0-public/", "z1-restricted/")),
                         ("z0-public/", "z1-restricted/"))

    def test_env_opt_in_normalises_trailing_slash(self):
        cfg = make_cfg(self.tmp)
        with mock.patch.dict(os.environ,
                             {"DSH_KB_RAG_IMAGE_ARM_PREFIXES": "z3-skills, z0-public"}):
            self.assertEqual(ragvec._image_prefixes(cfg, ("z0-public/",)),
                             ("z3-skills/", "z0-public/"))

    def test_config_attr_opt_in(self):
        cfg = make_cfg(self.tmp, image_prefixes=["z3-skills/"])
        self.assertEqual(ragvec._image_prefixes(cfg, ("z0-public/",)), ("z3-skills/",))

    def test_empty_value_falls_back_to_zone(self):
        cfg = make_cfg(self.tmp)
        with mock.patch.dict(os.environ, {"DSH_KB_RAG_IMAGE_ARM_PREFIXES": ""}):
            self.assertEqual(ragvec._image_prefixes(cfg, ("z0-public/",)), ("z0-public/",))


class TestReadiness(unittest.TestCase):
    """image_arm_readiness：不联网、不 import numpy，直接回答「有没有机会被用上」。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_no_source_in_production_zone(self):
        db = self.tmp / "z3.sqlite"
        make_image_db(db, [("z3-skills/a.png", [1.0, 0.0])], dim=2)
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=db)
        info = ragvec.image_arm_readiness(cfg)
        self.assertTrue(info["enabled"])
        self.assertEqual(info["rows"], 1)
        self.assertEqual(info["matched"], 0)
        self.assertEqual(info["rows_by_zone"], {"z3-skills/": 1})
        self.assertFalse(info["usable"])
        self.assertIn("无可用图源", info["verdict"])
        self.assertIn("z3-skills", info["verdict"])

    def test_gate_off_is_reported(self):
        db = self.tmp / "z1.sqlite"
        make_image_db(db, [("z1-restricted/a.png", [1.0, 0.0])], dim=2)
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=db, vector_always=False)
        info = ragvec.image_arm_readiness(cfg)
        self.assertEqual(info["matched"], 1)
        self.assertFalse(info["gate"]["reachable_when_fts_full"])
        self.assertFalse(info["usable"])
        self.assertIn("vector_always", info["verdict"])

    def test_all_gates_open_is_usable(self):
        db = self.tmp / "z1b.sqlite"
        make_image_db(db, [("z1-restricted/a.png", [1.0, 0.0])], dim=2)
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=db, vector_always=True)
        info = ragvec.image_arm_readiness(cfg)
        self.assertTrue(info["gate"]["reachable_when_fts_full"])
        self.assertTrue(info["usable"])
        self.assertIn("可用", info["verdict"])

    def test_missing_db_is_reported_without_raising(self):
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=self.tmp / "nope.sqlite")
        info = ragvec.image_arm_readiness(cfg)
        self.assertFalse(info["db_exists"])
        self.assertFalse(info["usable"])
        self.assertIn("库不存在", info["verdict"])

    def test_disabled_is_reported(self):
        cfg = make_cfg(self.tmp)
        info = ragvec.image_arm_readiness(cfg)
        self.assertFalse(info["enabled"])
        self.assertIn("未启用", info["verdict"])

    def test_broken_schema_is_reported_without_raising(self):
        db = self.tmp / "broken.sqlite"
        con = sqlite3.connect(str(db))
        con.execute("create table images(path text)")
        con.commit()
        con.close()
        cfg = make_cfg(self.tmp, image_enabled=True, image_db=db)
        info = ragvec.image_arm_readiness(cfg)
        self.assertFalse(info["usable"])
        self.assertIn("schema", info["verdict"])


class TestUpstreamGate(unittest.TestCase):
    """major-4：图片臂在 FTS 命中满额时到底会不会被调用（真实 index.search）。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        (self.home / "mirror" / "z0-public").mkdir(parents=True)
        (self.home / "mirror" / "z0-public" / "mount.md").write_text(
            "---\ntype: lesson\ntitle: 挂载点\n---\n\n"
            "挂载点 挂载点 挂载点 挂载点 说明。\n", encoding="utf-8")
        self.imgdb = self.tmp / "vectors-image.sqlite"
        make_image_db(self.imgdb, [("z0-public/pic.png", [1.0, 0.0])], dim=2)
        self.rag = RagConfig(
            vectors_db=str(self.tmp / "vectors.sqlite"),
            vector_enabled=True,
            vector_always=False,
            image_arm_enabled=True,
            image_arm_db=str(self.imgdb),
        )
        self.cfg = Config(home=self.home, source="local",
                          local_root=str(self.home / "mirror"), rag=self.rag)
        index_build(self.cfg)
        self.calls: list = []
        self._orig_vec = ragvec.VectorArm.query
        self._orig_img = ragvec.ImageArm.query
        ragvec.VectorArm.query = lambda *args, **kwargs: []

        def _img_query(_arm, text, prefixes, depth=32):
            self.calls.append(tuple(prefixes))
            return [("z0-public/pic.png", 1, 0.9)]

        ragvec.ImageArm.query = _img_query

    def tearDown(self):
        ragvec.VectorArm.query = self._orig_vec
        ragvec.ImageArm.query = self._orig_img
        ragvec._CACHE.clear()
        ragvec._IMAGE_CACHE.clear()

    def test_fts_full_query_skips_image_arm_without_vector_always(self):
        result = index_search(self.cfg, "挂载点", zone="z01", limit=1)
        self.assertEqual(len(result.hits), 1)          # FTS 命中已满额
        self.assertEqual(self.calls, [], "满额且没有 vector_always 时不该进 augment")
        self.assertNotIn("图片路", result.note or "")

    def test_fts_full_query_reaches_image_arm_with_vector_always(self):
        cfg = replace(self.cfg, rag=replace(self.rag, vector_always=True))
        result = index_search(cfg, "挂载点", zone="z01", limit=1)
        self.assertEqual(len(result.hits), 1)
        self.assertTrue(self.calls, "vector_always=true 时满额查询也必须进 augment")
        self.assertIn("图片路 1 候选", result.note or "")
        self.assertIn(("z0-public/", "z1-restricted/", "_meta/", "_incoming/"),
                      self.calls)

    def test_under_limit_query_reaches_image_arm_anyway(self):
        result = index_search(self.cfg, "挂载点", zone="z01", limit=8)
        self.assertLess(len(result.hits), 8)
        self.assertTrue(self.calls)
        self.assertIn("图片路 1 候选", result.note or "")


if __name__ == "__main__":
    unittest.main()


class TestKeyNameIsTheHostsChoice(unittest.TestCase):
    """键名由宿主决定 —— 库里写死某个服务的键名，换一个服务就得改代码。

    这条同时钉住「未配置凭据路径」的行为：没有默认路径，必须显式给，
    否则换个 home 就会读到别人的凭据、或者报成「模型不可用」。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.creds = self.tmp / "creds.yaml"
        self._env = {k: v for k, v in os.environ.items() if k.startswith("DSH_KB_IMAGE")}
        for k in [k for k in os.environ if k.startswith("DSH_KB_IMAGE")]:
            os.environ.pop(k)
        self.addCleanup(self._restore)

    def _restore(self):
        for k in [k for k in os.environ if k.startswith("DSH_KB_IMAGE")]:
            os.environ.pop(k)
        os.environ.update(self._env)

    def test_a_renamed_key_is_honoured(self):
        os.environ["DSH_KB_IMAGE_CREDENTIALS"] = str(self.creds)
        os.environ["DSH_KB_IMAGE_KEY_NAME"] = "MY_MULTIMODAL_KEY"
        write_credentials(self.creds, {"MY_MULTIMODAL_KEY": SENTINEL_KEY,
                                       "IMAGE_EMBED_API_KEY": "wrong-one"})
        import importlib
        mod = importlib.reload(image_embed)
        try:
            self.assertEqual(mod.CREDENTIAL_REF, "MY_MULTIMODAL_KEY")
            self.assertEqual(mod.api_key(required=True), SENTINEL_KEY)
        finally:
            importlib.reload(image_embed)

    def test_no_default_credentials_path(self):
        """没有 DSH_KB_IMAGE_CREDENTIALS 时不能悄悄用某个内置路径。"""
        self.assertEqual(image_embed.DEFAULT_CREDENTIALS, "",
                         "库里不该有部署路径默认值")
        self.assertIsNone(image_embed.credentials_path())
        self.assertIsNone(image_embed.api_key())
        with self.assertRaises(image_embed.ImageAuthError):
            image_embed.api_key(required=True)
