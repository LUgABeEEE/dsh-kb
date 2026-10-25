"""深池重排（deep 慢车道）的离线单测 —— 不打网络、不依赖凭据。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import deeprank


class TestParseOrder(unittest.TestCase):
    def test_reads_indices_in_order(self):
        self.assertEqual(deeprank._parse_order("3,1,7", 5, 3), [3, 1])

    def test_drops_out_of_range_and_garbage(self):
        self.assertEqual(deeprank._parse_order("9, x, 2", 5, 3), [2])

    def test_dedupes_and_caps(self):
        self.assertEqual(deeprank._parse_order("1,1,2,3,4", 5, 2), [1, 2])

    def test_empty(self):
        self.assertEqual(deeprank._parse_order("", 5, 3), [])


class TestReorder(unittest.TestCase):
    def test_picked_first_rest_keeps_order(self):
        self.assertEqual(deeprank.reorder(["a", "b", "c", "d"], [3, 1]), ["c", "a", "b", "d"])

    def test_is_permutation(self):
        out = deeprank.reorder(list("abcdef"), [6, 2])
        self.assertEqual(sorted(out), list("abcdef"))

    def test_bad_indices_ignored(self):
        self.assertEqual(deeprank.reorder(["a", "b"], [9, 0]), ["a", "b"])


class TestPickPages(unittest.TestCase):
    def test_no_key_returns_none(self):
        with mock.patch.object(deeprank, "_api_key", return_value=None):
            self.assertIsNone(deeprank.pick_pages("q", [("p", "body")]))

    def test_empty_pages_returns_empty(self):
        self.assertEqual(deeprank.pick_pages("q", []), [])

    def test_network_failure_returns_none(self):
        with mock.patch.object(deeprank, "_api_key", return_value="sk-test"), \
             mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
            self.assertIsNone(deeprank.pick_pages("q", [("p", "body")]))


if __name__ == "__main__":
    unittest.main()