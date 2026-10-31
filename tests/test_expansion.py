"""Query expansion: every stage must degrade to "do nothing", never to an error."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import expansion  # noqa: E402


class Cfg:
    """Only what the expansion module reads: home and the mirror directory."""

    def __init__(self, home: Path, mirror: Path | None = None):
        self.home = str(home)
        self.mirror_dir = str(mirror or (home / "mirror"))


class Hit:
    def __init__(self, path, line=1, score=0.0, snippet=""):
        self.path = path
        self.line = line
        self.score = score
        self.snippet = snippet


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cfg = Cfg(self.tmp)
        self._env = {k: v for k, v in os.environ.items() if k.startswith("DSH_KB_")}

    def tearDown(self):
        for key in [k for k in os.environ if k.startswith("DSH_KB_")]:
            os.environ.pop(key, None)
        os.environ.update(self._env)
        self._tmp.cleanup()


class TestPaths(Base):
    def test_state_defaults_to_home_not_to_a_machine_path(self):
        os.environ.pop("DSH_KB_REWRITE_CACHE_DB", None)
        self.assertEqual(expansion.cache_db(self.cfg),
                         self.tmp / "state" / "rewrite-cache.sqlite")
        os.environ.pop("DSH_KB_DICT_PATH", None)
        self.assertEqual(expansion.dict_path(self.cfg), self.tmp / "dict" / "synonyms.json")

    def test_both_can_be_pointed_elsewhere(self):
        os.environ["DSH_KB_REWRITE_CACHE_DB"] = "/tmp/elsewhere.sqlite"
        os.environ["DSH_KB_DICT_PATH"] = "/tmp/syn.json"
        self.assertEqual(expansion.cache_db(self.cfg), Path("/tmp/elsewhere.sqlite"))
        self.assertEqual(expansion.dict_path(self.cfg), Path("/tmp/syn.json"))


class TestConfidence(Base):
    def test_no_hits_is_the_none_band(self):
        fields = expansion.confidence_fields(self.cfg, "q", [])
        self.assertEqual(fields["confidence"], "none")
        self.assertIsNone(fields["rerank_top1"])

    def test_bands_follow_the_thresholds(self):
        hits = [Hit("a.md")]
        cases = [(10.0, "high"), (0.0, "medium"), (-9.0, "low")]
        for score, band in cases:
            with mock.patch("handrolled.rerank.score_pairs", return_value=[score]):
                fields = expansion.confidence_fields(self.cfg, "q", hits)
            self.assertEqual(fields["confidence"], band, f"score {score}")
            self.assertEqual(fields["rerank_top1"], round(float(score), 3))

    def test_an_unavailable_reranker_is_unknown_not_a_crash(self):
        with mock.patch("handrolled.rerank.score_pairs", return_value=None):
            fields = expansion.confidence_fields(self.cfg, "q", [Hit("a.md")])
        self.assertEqual(fields["confidence"], "unknown")

    def test_the_signal_can_be_switched_off(self):
        with mock.patch.object(expansion, "CONFIDENCE_ENABLED", False):
            self.assertEqual(expansion.confidence_fields(self.cfg, "q", [Hit("a.md")]), {})


class TestNeedsRewrite(Base):
    def test_off_means_never(self):
        with mock.patch.object(expansion, "REWRITE_MODE", "off"), \
             mock.patch.object(expansion, "DICT_ON", False):
            self.assertFalse(expansion.needs_rewrite("某个查询", [], {"confidence": "none"}))

    def test_weak_first_pass_triggers_and_strong_pass_does_not(self):
        with mock.patch.object(expansion, "REWRITE_MODE", "auto"):
            self.assertTrue(expansion.needs_rewrite("怎么配机场", [], {"confidence": "none"}))
            self.assertTrue(expansion.needs_rewrite("怎么配机场", [Hit("a")], {"confidence": "low"}))
            self.assertFalse(expansion.needs_rewrite("怎么配机场", [Hit("a")], {"confidence": "high"}))
            self.assertFalse(expansion.needs_rewrite("怎么配机场", [Hit("a")], {"confidence": "unknown"}))


class TestIdentifierGuard(Base):
    """An identifier lookup must never be rewritten into prose.

    The lexical layer guarantees "exactly or not at all" for identifiers; a
    rewrite turns `UNIQ-CLI-CANARY-88888` into generic Chinese, which then
    matches unrelated pages — a lookup that should return nothing starts
    returning noise.  Observed happening against a live local model.
    """

    def test_identifiers_are_recognised(self):
        for query in ("UNIQ-CLI-CANARY-88888", "v0.3.4", "ds_agent", "errno2", "src/index.py"):
            self.assertTrue(expansion.looks_like_identifier(query), query)

    def test_prose_is_not_an_identifier(self):
        for query in ("怎么配机场", "mount stale", "dsh-kb 是什么", ""):
            self.assertFalse(expansion.looks_like_identifier(query), query)

    def test_an_identifier_with_zero_hits_is_not_rewritten(self):
        with mock.patch.object(expansion, "REWRITE_MODE", "auto"):
            self.assertFalse(
                expansion.needs_rewrite("UNIQ-CLI-CANARY-88888", [], {"confidence": "none"}))
            # and prose with zero hits still is
            self.assertTrue(
                expansion.needs_rewrite("怎么配机场", [], {"confidence": "none"}))


class TestDictionary(Base):
    def _write(self, groups):
        path = self.tmp / "dict" / "synonyms.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"groups": groups}, ensure_ascii=False), encoding="utf-8")
        return path

    def test_disabled_by_default_returns_nothing(self):
        self._write([["机场", "代理"]])
        self.assertEqual(expansion.dict_variants(self.cfg, "机场怎么配"), [])

    def test_enabled_substitutes_one_term_per_group(self):
        self._write([["机场", "代理"], ["订阅", "节点"]])
        with mock.patch.object(expansion, "DICT_ON", True):
            variants = expansion.dict_variants(self.cfg, "机场 订阅 怎么配")
        self.assertIn("代理 订阅 怎么配", variants)
        self.assertIn("机场 节点 怎么配", variants)

    def test_latin_terms_match_whole_tokens_only(self):
        self.assertTrue(expansion.term_hit("k8s", "部署 k8s 集群"))
        # A hyphen is not an alphanumeric, so a hyphenated identifier still counts
        # as a whole token — that is the intended behaviour, not an accident.
        self.assertTrue(expansion.term_hit("k8s", "deployments-k8s-helper"))
        # But a term glued into a longer word must not match.
        self.assertFalse(expansion.term_hit("k8s", "myk8scluster"))
        self.assertTrue(expansion.term_hit("机场", "机场怎么配"))

    def test_the_table_reloads_when_the_file_changes(self):
        path = self._write([["a", "b"]])
        with mock.patch.object(expansion, "DICT_ON", True):
            self.assertEqual(expansion.dict_variants(self.cfg, "a"), ["b"])
            import time

            time.sleep(0.01)
            path.write_text(json.dumps({"groups": [["a", "c"]]}), encoding="utf-8")
            os.utime(path, (time.time() + 5, time.time() + 5))
            self.assertEqual(expansion.dict_variants(self.cfg, "a"), ["c"])

    def test_a_broken_file_is_not_fatal(self):
        path = self.tmp / "dict" / "synonyms.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        with mock.patch.object(expansion, "DICT_ON", True):
            self.assertEqual(expansion.dict_variants(self.cfg, "a"), [])


class TestMerging(unittest.TestCase):
    def test_fill_keeps_the_original_order_and_only_fills(self):
        primary = [Hit("a"), Hit("b")]
        extra = [[Hit("c"), Hit("a")]]
        merged = expansion.merge_hits(primary, extra, 3)
        self.assertEqual([h.path for h in merged], ["a", "b", "c"])

    def test_fill_never_displaces_a_primary_hit(self):
        primary = [Hit("a"), Hit("b"), Hit("c")]
        extra = [[Hit("z"), Hit("y")]]
        merged = expansion.merge_hits(primary, extra, 3)
        self.assertEqual([h.path for h in merged], ["a", "b", "c"])

    def test_top2_pins_the_first_two_and_lets_the_rest_compete(self):
        primary = [Hit("a"), Hit("b"), Hit("c")]
        extra = [[Hit("z"), Hit("y")], [Hit("z"), Hit("x")]]
        merged = expansion.merge_top2(primary, extra, 4)
        self.assertEqual([h.path for h in merged][:2], ["a", "b"])
        self.assertEqual(merged[2].path, "z", "z ranks first in both variants")

    def test_the_strategy_is_selected_by_configuration(self):
        primary = [Hit("a"), Hit("b"), Hit("c")]
        extra = [[Hit("z")]]
        self.assertEqual([h.path for h in expansion.merge_variants(primary, extra, 3, mode="fill")],
                         ["a", "b", "c"])
        self.assertEqual([h.path for h in expansion.merge_variants(primary, extra, 3, mode="top2")],
                         ["a", "b", "z"])


class TestRewriteCache(Base):
    def test_store_then_lookup_is_an_exact_hit(self):
        expansion.cache_store(self.cfg, "怎么配机场", ["代理怎么设置"])
        got = expansion.cache_lookup(self.cfg, "怎么配机场")
        self.assertEqual(got["variants"], ["代理怎么设置"])
        self.assertEqual(got["hit"], "exact")

    def test_a_miss_is_none_and_creates_no_row(self):
        self.assertIsNone(expansion.cache_lookup(self.cfg, "never stored"))
        self.assertTrue((self.tmp / "state").exists(), "the table is created lazily")

    def test_the_cache_can_be_switched_off(self):
        with mock.patch.object(expansion, "REWRITE_CACHE_ON", False):
            expansion.cache_store(self.cfg, "q", ["v"])
            self.assertIsNone(expansion.cache_lookup(self.cfg, "q"))

    def test_cache_keys_differ_per_query(self):
        self.assertNotEqual(expansion.cache_key("a"), expansion.cache_key("b"))
        self.assertEqual(expansion.cache_key("a "), expansion.cache_key("a"))


class TestRewriteDegrades(Base):
    def test_an_unreachable_model_returns_no_variants(self):
        with mock.patch.object(expansion, "REWRITE_MODE", "auto"), \
             mock.patch.object(expansion, "REWRITE_ENDPOINT", "http://127.0.0.1:1"), \
             mock.patch.object(expansion, "REWRITE_TIMEOUT", 1.0):
            self.assertEqual(expansion.rewrite_variants(self.cfg, "随便问点什么"), [])


class TestPageRerank(Base):
    def test_off_is_an_identity_transform(self):
        hits = [Hit("a.md"), Hit("b.md")]
        self.assertEqual(expansion.page_rerank(self.cfg, "q", hits, 2), (hits, None))

    def test_passages_covers_a_long_page_and_caps_the_count(self):
        text = "x" * 5000
        pieces = expansion.passages(text, 800, 600)
        self.assertGreater(len(pieces), 1)
        self.assertLessEqual(len(pieces), 24)

    def test_a_short_page_is_one_passage(self):
        self.assertEqual(expansion.passages("short", 800, 600), ["short"])

    def test_an_empty_page_yields_nothing(self):
        self.assertEqual(expansion.passages("   ", 800, 600), [])


if __name__ == "__main__":
    unittest.main()
