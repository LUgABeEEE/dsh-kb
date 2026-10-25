"""限幅重排（maxshift）的确定性单测。

2026-10-24 实测背景：生产 23/33 → 25/33（+2/−0），但只有「位移受限」时才不伤并列页。
这些用例锁住三件事：近邻允许交换、远处不许空降、结果必须是原集合的排列。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb.index import Hit, _clamp_shift


def _hits(paths: list[str]) -> list:
    return [Hit(path=p, line=1, score=0.0, snippet="", zone="") for p in paths]


class TestClampShift(unittest.TestCase):
    def test_zero_sigma_is_identity(self):
        hits = _hits(["a", "b", "c"])
        self.assertEqual(_clamp_shift(hits, [3.0, 2.0, 1.0], 0), hits)

    def test_single_swap_is_allowed(self):
        hits = _hits(["a", "b", "c"])
        out = _clamp_shift(hits, [1.0, 5.0, 1.0], 1)
        self.assertEqual([h.path for h in out], ["b", "a", "c"])

    def test_far_jump_is_clamped(self):
        hits = _hits(["a", "b", "c", "d", "e"])
        out = _clamp_shift(hits, [0.0, 0.0, 0.0, 0.0, 9.0], 1)
        paths = [h.path for h in out]
        # e 原名次第 5，σ=1 最多到第 4，绝不能进前 3
        self.assertNotIn("e", paths[:3])

    def test_result_is_a_permutation(self):
        hits = _hits(["a", "b", "c", "d"])
        out = _clamp_shift(hits, [1.0, 9.0, 2.0, 3.0], 2)
        self.assertEqual(sorted(h.path for h in out), sorted(h.path for h in hits))
        self.assertEqual(len(out), len(hits))

    def test_empty_and_single(self):
        self.assertEqual(_clamp_shift([], [], 2), [])
        one = _hits(["a"])
        self.assertEqual(_clamp_shift(one, [1.0], 2), one)


if __name__ == "__main__":
    unittest.main()