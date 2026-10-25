"""Regression tests for the chunk/page unit mix-up in ``_selective_tokens``.

The defect (verified on the live index, 2026-10-19)
--------------------------------------------------
``vocab`` is declared ``fts5vocab(chunks, 'row')``, so ``vocab.doc`` counts the
**chunks** that contain a term — but the generic-word threshold was computed
from ``select count(distinct path) from meta``, i.e. **pages**.  The two are not
interchangeable: a page is several chunks (live index: 1,320 pages vs 6,129
chunks), so the effective cut was 660/6,129 ≈ **10.8% of chunks** instead of the
documented "half the corpus" — about **4.6x more aggressive** than advertised.
It discarded mid-frequency terms that still discriminate: ``目录`` (doc=771),
``kb`` (doc=1450), ``dsh`` (doc=2944).

What is pinned here
-------------------
* the corpus unit is a **chunk** — a fixture whose page count and chunk count
  differ, so a page-based total cannot pass by accident;
* a ~60%-of-chunks word is dropped under *both* the old and the new rule (the
  original intent — "drop words that appear in half the corpus or more" — is
  preserved, not relaxed away);
* a mid-frequency word (below 50% of chunks, above the old bogus cut) is kept
  after the fix and was dropped before it: the fail-before/pass-after case;
* a 10%-of-chunks word is kept either way (it was never the problem);
* ``SELECTIVE_CUT_RATIO`` really drives the threshold, and the "every token is
  generic ⇒ return the original list" fallback still holds.

Note on the polarity of the counter-example
-------------------------------------------
The natural reading "a 60% word is dropped *after* the fix but not before" is
arithmetically impossible on a consistent index: every page holds at least one
chunk, so ``pages <= chunks`` and therefore ``pages // 2 <= chunks // 2 < 0.6 x
chunks`` — the legacy threshold always dropped a 60% word too.  The fix *loosens*
the cut (660 -> 3,064 on the live index), so the behaviour that flips is the
**mid-frequency** word in ``(pages // 2, chunks // 2)`` — exactly the ``目录`` /
``kb`` class of real casualties.  ``test_legacy_rule_is_the_counter_example``
runs the verbatim pre-fix implementation next to the current one so the
before/after difference is machine-checked rather than described.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb import index  # noqa: E402
from dsh_kb.config import Config  # noqa: E402

#: ``Config.chunk_lines`` default — the fixture must chunk the way production does.
CHUNK_LINES = 15

#: Optional live-index spot check.  Point ``DSH_KB_TEST_LIVE_DB`` at a real
#: index to run it; without that variable (CI, fresh clone) the case skips.
LIVE_DB = Path(os.environ["DSH_KB_TEST_LIVE_DB"]) if os.environ.get("DSH_KB_TEST_LIVE_DB") else None

#: The pre-fix denominator: pages, not chunks.  Named so the counter-example
#: reads as "the old query", not as a magic string.
_PAGES_SQL = "select count(distinct path) from meta"

#: Terms the defect report measured on the live corpus.
LIVE_PROBES = ("目录", "2026", "dsh", "kb")


def _legacy_selective_tokens(con, tokens: list[str]) -> list[str]:
    """Verbatim **pre-fix** implementation, kept as the counter-example.

    This is what the shipped code did before the fix: chunk-level ``doc`` versus
    a page-level total.  Do not "fix" it — its only job is to prove that the
    assertions below would have failed on the old code.
    """
    freq = index._document_frequency(con, tokens)
    total = con.execute("select count(distinct path) from meta").fetchone()[0]
    if not freq or total <= 1:
        return tokens
    cut = max(1, total // 2)
    kept = [t for t in tokens if freq.get(t.lower(), 0) < cut]
    return kept or tokens


#: 20 chunks spread over 6 pages (4+4+4+4+2+2).  The asymmetry is the point:
#: ``pages // 2 == 3`` (legacy cut) while ``chunks // 2 == 10`` (correct cut).
LAYOUT = (4, 4, 4, 4, 2, 2)
COMMON_CHUNKS = frozenset(range(12))        # 12/20 = 60% of chunks
MIDDLING_CHUNKS = frozenset({0, 1, 2, 3})   #  4/20 = 20% of chunks
RARE_CHUNKS = frozenset({18, 19})           #  2/20 = 10% of chunks


def _build_fixture_mirror(root: Path) -> None:
    """Write markdown whose *chunk* distribution is known exactly."""
    chunk_id = 0
    for page, n_chunks in enumerate(LAYOUT):
        lines: list[str] = []
        for _ in range(n_chunks):
            marks = []
            if chunk_id in COMMON_CHUNKS:
                marks.append("common")
            if chunk_id in MIDDLING_CHUNKS:
                marks.append("middling")
            if chunk_id in RARE_CHUNKS:
                marks.append("rare")
            lines.append(" ".join(marks) if marks else "no-marker-header")
            lines.extend(
                f"filler line {i} lorem ipsum dolor sit amet consectetur"
                for i in range(1, CHUNK_LINES)
            )
            chunk_id += 1
        (root / f"page-{page:03d}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class SelectiveTokensFixture(unittest.TestCase):
    """A small, self-consistent index built by the real ``index.build``."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        home = Path(cls._tmp.name) / "kb"
        cls.cfg = Config(home=home)
        cls.cfg.mirror_dir.mkdir(parents=True, exist_ok=True)
        _build_fixture_mirror(cls.cfg.mirror_dir)
        cls.stats = index.build(cls.cfg)
        cls.con = sqlite3.connect(f"file:{cls.cfg.index_db}?mode=ro", uri=True)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls._tmp.cleanup()

    def doc(self, term: str) -> int:
        return index._document_frequency(self.con, [term]).get(term, 0)

    def total(self, sql: str) -> int:
        return self.con.execute(sql).fetchone()[0]

    def test_fixture_has_more_chunks_than_pages(self):
        """Guards the guard: with pages == chunks this fixture proves nothing."""
        self.assertEqual(self.stats.chunks, sum(LAYOUT))
        self.assertEqual(self.stats.pages, len(LAYOUT))
        self.assertNotEqual(self.stats.pages, self.stats.chunks)

    def test_vocab_doc_counts_chunks_not_pages(self):
        """The premise of the defect, asserted instead of assumed."""
        self.assertEqual(self.doc("common"), len(COMMON_CHUNKS))      # 12 chunks
        self.assertEqual(self.doc("middling"), len(MIDDLING_CHUNKS))  # 4 chunks
        self.assertEqual(self.doc("rare"), len(RARE_CHUNKS))          # 2 chunks
        # ... while the page-level total is a different number entirely (6).
        self.assertEqual(self.total("select count(distinct path) from meta"), len(LAYOUT))

    def test_mid_frequency_token_is_no_longer_dropped(self):
        """The fail-before/pass-after case (the ``目录``/``kb`` class).

        Pre-fix this returned ``["rare"]``: ``middling`` (doc=4) was discarded
        because the bogus cut was ``pages // 2 == 3``.
        """
        # spec side: 4/20 = 20% is well under half the chunks, so it must survive
        self.assertGreaterEqual(self.doc("middling"), max(1, self.stats.pages // 2))
        self.assertLess(self.doc("middling"), max(1, int(self.stats.chunks * index.SELECTIVE_CUT_RATIO)))
        self.assertEqual(
            index._selective_tokens(self.con, ["common", "middling", "rare"]),
            ["middling", "rare"],
        )

    def test_generic_word_is_still_dropped(self):
        """The original intent survives: >= 50% of chunks is still generic."""
        self.assertGreaterEqual(
            self.doc("common"),
            max(1, int(self.stats.chunks * index.SELECTIVE_CUT_RATIO)),
        )
        self.assertNotIn("common", index._selective_tokens(self.con, ["common", "middling"]))

    def test_rare_word_is_kept_by_both_rules(self):
        """2/20 = 10% was never the problem — it must stay kept either way."""
        self.assertIn("rare", index._selective_tokens(self.con, ["rare"]))
        self.assertIn("rare", _legacy_selective_tokens(self.con, ["rare"]))

    def test_legacy_rule_is_the_counter_example(self):
        """Run the pre-fix code next to the new one: they must disagree.

        If someone reverts ``_CORPUS_SIZE_SQL`` to the page-level query the two
        functions collapse into each other and this test fails.
        """
        tokens = ["common", "middling", "rare"]
        self.assertEqual(_legacy_selective_tokens(self.con, tokens), ["rare"])
        self.assertEqual(index._selective_tokens(self.con, tokens), ["middling", "rare"])
        self.assertNotEqual(
            index._selective_tokens(self.con, tokens),
            _legacy_selective_tokens(self.con, tokens),
        )

    def test_unknown_token_is_kept(self):
        """No behaviour change for a term the index has never seen."""
        self.assertEqual(index._selective_tokens(self.con, ["absent-from-corpus"]),
                         ["absent-from-corpus"])

    def test_everything_generic_falls_back_to_the_original_list(self):
        """A query made only of generic words still gets a usable (broad) query."""
        with mock.patch.object(index, "SELECTIVE_CUT_RATIO", 0.05):
            self.assertEqual(
                index._selective_tokens(self.con, ["common", "middling"]),
                ["common", "middling"],
            )

    def test_ratio_constant_is_wired_to_the_threshold(self):
        """Lowering the ratio drops more; the knob is the only variable."""
        with mock.patch.object(index, "SELECTIVE_CUT_RATIO", 1.0):
            self.assertEqual(
                index._selective_tokens(self.con, ["common", "middling", "rare"]),
                ["common", "middling", "rare"],
            )

    def test_default_ratio_is_half(self):
        """A silent change of the default would move every threshold."""
        self.assertEqual(index.SELECTIVE_CUT_RATIO, 0.5)


class SelectiveTokensLiveCorpus(unittest.TestCase):
    """Read-only spot check against the real index; skips when it is unreadable.

    Never writes: opened ``mode=ro``, and the production rebuild commands are
    deliberately not run here.  A concurrent ``dsh-kb index`` (another process
    rebuilds this file) can make a query fail mid-flight; that is reported as a
    skip, never as a failure — the point is to check the rule, not to fight over
    the file.
    """

    con = None

    @classmethod
    def setUpClass(cls):
        if LIVE_DB is None or not LIVE_DB.exists():
            return
        try:
            con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
            con.execute("select count(*) from meta").fetchone()
            con.execute("select count(*) from vocab").fetchone()
        except sqlite3.Error:
            return
        cls.con = con

    @classmethod
    def tearDownClass(cls):
        if cls.con is not None:
            cls.con.close()

    def setUp(self):
        if self.con is None:
            self.skipTest(f"live index not readable: {LIVE_DB}")

    @contextlib.contextmanager
    def readable(self):
        """Turn a mid-rebuild SQLite error into a skip instead of a failure."""
        try:
            yield
        except sqlite3.Error as exc:
            self.skipTest(f"live index unreadable right now: {exc}")

    def chunks(self) -> int:
        """Corpus size in the documented unit, read straight from ``meta``.

        Deliberately *not* ``index._CORPUS_SIZE_SQL``: the expectations below
        must describe the documented rule ("50% of chunks"), so reverting the
        implementation makes these tests fail instead of agreeing with it.
        """
        return self.con.execute("select count(*) from meta").fetchone()[0]

    def pages(self) -> int:
        return self.con.execute(_PAGES_SQL).fetchone()[0]

    def cut(self, total: int, ratio: float = index.SELECTIVE_CUT_RATIO) -> int:
        return max(1, int(total * ratio))

    def test_corpus_total_is_the_chunk_count_not_the_page_count(self):
        """A direct check that would have failed on the old one-line query."""
        with self.readable():
            chunks, pages = self.chunks(), self.pages()
            self.assertEqual(self.con.execute(index._CORPUS_SIZE_SQL).fetchone()[0], chunks)
            self.assertGreater(chunks, pages)      # a page holds several chunks
            self.assertNotEqual(chunks, pages)
            self.assertGreater(self.cut(chunks), self.cut(pages))

    def test_probe_terms_follow_the_documented_rule(self):
        """Each probe term's fate must match "dropped iff doc >= 50% of chunks"."""
        with self.readable():
            freq = index._document_frequency(self.con, list(LIVE_PROBES))
            present = {t: d for t, d in freq.items() if d}
            if not present:
                self.skipTest("no probe term present in the live index")
            cut = self.cut(self.chunks())
            # A term the index has never seen counts as doc=0, hence kept.
            expected = [t for t in LIVE_PROBES if present.get(t, 0) < cut]
            self.assertEqual(index._selective_tokens(self.con, list(LIVE_PROBES)), expected)

    def test_mid_frequency_probe_terms_survive(self):
        """The regression itself, on real data: terms above the old page cut.

        ``目录`` (doc=771), ``dsh`` (doc=2944) and ``kb`` (doc=1450) all sit in
        ``(pages // 2, chunks // 2)`` — the old code dropped them, the new one
        must not.  ``2026`` (doc=3520) is above 50% of chunks even after the
        fix, so it is legitimately still dropped (intent preserved).
        """
        with self.readable():
            freq = index._document_frequency(self.con, list(LIVE_PROBES))
            present = {t: d for t, d in freq.items() if d}
            if not present:
                self.skipTest("no probe term present in the live index")
            legacy_cut = self.cut(self.pages())
            cut = self.cut(self.chunks())
            flipped = [t for t in LIVE_PROBES if legacy_cut <= present.get(t, 0) < cut]
            if not flipped:
                self.skipTest(f"none of {LIVE_PROBES} is in the flip window {legacy_cut}..{cut}")
            kept = index._selective_tokens(self.con, list(LIVE_PROBES))
            for term in flipped:
                self.assertIn(term, kept, f"{term} (doc={present[term]}) was dropped again")
            # ... and the genuinely generic one is still filtered out.
            for term in LIVE_PROBES:
                if present.get(term, 0) >= cut:
                    self.assertNotIn(term, kept)


if __name__ == "__main__":
    unittest.main(verbosity=2)
