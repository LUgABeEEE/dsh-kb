"""The reranker's memory contract: optional, off by default, gives memory back.

Nothing here needs onnxruntime, tokenizers or a model file — the point is that
the module is safe to import and inspect on a machine that has none of them.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import rerank  # noqa: E402


class TestIdleSeconds(unittest.TestCase):
    def test_default_is_ten_minutes(self):
        env = {k: v for k, v in os.environ.items() if k != "DSH_KB_RERANK_IDLE_S"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(rerank._idle_seconds(), 600)

    def test_zero_disables_eviction(self):
        with mock.patch.dict(os.environ, {"DSH_KB_RERANK_IDLE_S": "0"}):
            self.assertEqual(rerank._idle_seconds(), 0)

    def test_garbage_falls_back_to_the_default(self):
        with mock.patch.dict(os.environ, {"DSH_KB_RERANK_IDLE_S": "soon"}):
            self.assertEqual(rerank._idle_seconds(), rerank.DEFAULT_IDLE_S)

    def test_negative_is_clamped_to_zero(self):
        with mock.patch.dict(os.environ, {"DSH_KB_RERANK_IDLE_S": "-5"}):
            self.assertEqual(rerank._idle_seconds(), 0)


class TestWithoutAModel(unittest.TestCase):
    @unittest.skipIf(os.environ.get("DSH_KB_RERANK_ROOT"),
                     "this machine configures a model root; CI covers the unset case")
    def test_reports_itself_unavailable(self):
        self.assertFalse(rerank.available())

    def test_status_is_safe_when_nothing_is_loaded(self):
        status = rerank.status()
        self.assertFalse(status["loaded"])
        self.assertIsNone(status["idle_for"])
        self.assertIn("idle_seconds", status)

    def test_unloading_when_nothing_is_loaded_is_a_no_op(self):
        self.assertFalse(rerank.unload("test"))

    def test_scoring_without_a_model_returns_none(self):
        """Degradation is the contract: callers keep the original order."""
        self.assertIsNone(rerank.score_pairs("query", ["passage"]))


if __name__ == "__main__":
    unittest.main()
