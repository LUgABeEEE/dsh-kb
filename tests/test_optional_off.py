"""The optional layers must be off in a stock environment.

The semantic arm is the one part of this project that can pull in heavy
third-party code, models and network egress.  Everything about it is gated by configuration,
and this file is the counterweight to "it exists": a clone that has never heard
of ``DSH_KB_RERANK_ROOT`` or ``DSH_KB_DEEP_ENDPOINT`` has to behave exactly like
0.1.0 and say so, instead of guessing at a path.

The checks run in a subprocess with every ``DSH_KB_*`` variable stripped, so the
developer's own deployment configuration cannot make them pass.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

PROBE = r"""
import json
import sys

import dsh_kb
from dsh_kb import config
from handrolled import deeprank, embed_onnx, rerank

rag = config.RagConfig()
print(json.dumps({
    "rerank_available": rerank.available(),
    "onnx_available": embed_onnx.available(),
    "deep_endpoint": deeprank._endpoint(),
    "deep_model": rag.deep_model,
    "rerank_enabled": rag.rerank_enabled,
    "vector_enabled": rag.vector_enabled,
    "deep_enabled": rag.deep_enabled,
    "third_party": sorted({"numpy", "onnxruntime", "tokenizers"} & set(sys.modules)),
}))
"""


class TestOptionalLayersAreOff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = {k: v for k, v in os.environ.items() if not k.startswith("DSH_KB_")}
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-c", PROBE],
            capture_output=True, text=True, env=env, timeout=180,
        )
        if result.returncode != 0:                      # pragma: no cover - diagnostics
            raise AssertionError(f"probe failed:\n{result.stdout}\n{result.stderr}")
        cls.state = json.loads(result.stdout.strip().splitlines()[-1])

    def test_importing_the_package_pulls_in_no_heavy_third_party_code(self):
        """`dsh_kb` has one dependency (handrolled, itself stdlib-only); it must
        never drag in numpy/onnxruntime/tokenizers behind your back."""
        self.assertEqual(self.state["third_party"], [])

    def test_model_backed_layers_report_themselves_absent(self):
        """No model path configured ⇒ unavailable, not "try the author's disk"."""
        self.assertFalse(self.state["rerank_available"])
        self.assertFalse(self.state["onnx_available"])

    def test_deep_rerank_has_no_endpoint_and_no_model(self):
        """A vendor URL or a model name baked into the code would be a bug."""
        self.assertEqual(self.state["deep_endpoint"], "")
        self.assertEqual(self.state["deep_model"], "")

    def test_the_switches_default_to_off(self):
        self.assertFalse(self.state["rerank_enabled"])
        self.assertFalse(self.state["vector_enabled"])
        self.assertFalse(self.state["deep_enabled"])


if __name__ == "__main__":
    unittest.main()
