"""Chunking strategies: ``lines`` (default, unchanged) and ``structural``.

Why this file exists
--------------------
Chunking used to be one fixed sliding window of ``chunk_lines`` (15) lines.  It
is a reasonable unit for source code but arbitrary for Markdown: it cuts
sentences in half, separates a heading from the body it introduces, and lets a
chunk grow until luck decides otherwise.  ``structural`` makes the boundary
follow the document (ATX headings, fenced code, paragraph breaks) and puts a
ceiling on the length.

The load-bearing constraint is **backwards compatibility**: a chunk is named by
its content (``chunk_id = sha256(path + 0x1f + text)[:16]``), so changing how a
page is cut renames every chunk of every page and forces a full re-embed of
``vectors.sqlite``.  That must be an explicit, one-setting decision — hence the
default stays ``lines`` and the first tests here pin the old behaviour against a
verbatim copy of the old loop, both at the ``_chunk_page`` level and end to end
through ``build()`` (chunk ids included).

The second contract is reconstructability: the structural partition is an
ordered, gapless run of line slices, so joining the chunks with ``"\\n"``
rebuilds the page line for line.  Whitespace-only blocks are therefore *folded*
into a neighbour, never deleted — the test that reassembles every fixture is the
one that would catch a lost line.

Everything runs in a temporary directory: no network, no production paths.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb import index  # noqa: E402
from dsh_kb.config import Config, load_config  # noqa: E402

#: Markdown page that exercises every structural rule at once: YAML
#: frontmatter, three heading levels that *do* split, a level-4 heading that does
#: not, a fenced block with a heading-looking line and blank lines inside it, and
#: a short section that must stay a single chunk.
MARKDOWN_PAGE = """---
type: lesson
title: 分块策略夹具
tags: [kb, chunk]
---

前言段落，位于第一个标题之前。

# 一级标题
alpha 正文第一行。
alpha 正文第二行。

## 二级标题
bravo 正文。

```python
# 这一行看起来像标题，但它在围栏里
def f():
    return 1

还有空行也在围栏里
```

### 三级标题
charlie 正文。

#### 四级标题
delta 正文：四级标题不另起一块。

tail 结尾行。
"""

#: A page with no headings at all: the only structure available is the blank
#: line between paragraphs, which is exactly the fallback path.
PARAGRAPH_PAGE = "\n\n".join(
    f"段落{i}：" + "字" * 40 for i in range(6)
) + "\n"

#: Leading blanks, trailing blanks, and a fence at the very end.
MESSY_PAGE = "\n\n\nleading blanks above\n\n```\nfence at the end\n```\n\n\n"

CORPORA = {
    "markdown": MARKDOWN_PAGE,
    "paragraphs": PARAGRAPH_PAGE,
    "messy": MESSY_PAGE,
}


def chunks(raw: str, **overrides) -> list[tuple[int, str]]:
    """Chunk one page through the module under test (no filesystem involved)."""
    cfg = Config(home=Path("/nonexistent-home"), **overrides)
    return index._chunk_page(raw, cfg)


def legacy_chunks(raw: str, chunk_lines: int = 15) -> list[tuple[int, str]]:
    """Verbatim copy of the pre-refactor loop — the behaviour to stay identical to.

    Kept as a copy rather than imported on purpose: an import would move with the
    code, and the point of this function is that it must **not** move.
    """
    lines = raw.splitlines()
    out = []
    for start in range(0, len(lines), chunk_lines):
        block = "\n".join(lines[start:start + chunk_lines])
        if not block.strip():
            continue
        out.append((start + 1, block))
    return out


def rebuild(raw: str, blocks: list[tuple[int, str]]) -> str:
    """Join chunk texts back the way the partition was made."""
    return "\n".join(text for _line, text in blocks)


class DefaultStrategyTest(unittest.TestCase):
    """``chunk_strategy`` defaults must keep today's index byte-identical."""

    def test_defaults_are_lines(self):
        cfg = Config(home=Path("/nonexistent-home"))
        self.assertEqual(cfg.chunk_strategy, "lines")
        self.assertEqual(cfg.chunk_lines, 15)
        self.assertEqual(cfg.chunk_max_chars, 1000)

    def test_default_and_explicit_lines_agree(self):
        explicit = chunks(MARKDOWN_PAGE, chunk_strategy="lines")
        self.assertEqual(chunks(MARKDOWN_PAGE), explicit)
        self.assertEqual(explicit, legacy_chunks(MARKDOWN_PAGE))

    def test_lines_chunks_match_legacy_formula(self):
        for name, raw in CORPORA.items():
            with self.subTest(page=name):
                self.assertEqual(chunks(raw), legacy_chunks(raw))

    def test_lines_strategy_still_honours_chunk_lines(self):
        raw = "\n".join(f"line {i}" for i in range(1, 11))
        got = chunks(raw, chunk_lines=3)
        self.assertEqual(got, legacy_chunks(raw, chunk_lines=3))
        self.assertEqual([line for line, _text in got], [1, 4, 7, 10])

    def test_unknown_strategy_falls_back_to_lines(self):
        # A typo must not switch on the re-chunking path (it renames every chunk
        # id and forces a full re-embed); falling back is the safe direction.
        for value in ("struct", "", "章节", None, 3):
            with self.subTest(value=value):
                cfg = Config(home=Path("/nonexistent-home"), chunk_strategy=value)
                self.assertEqual(cfg.chunk_strategy, "lines")
                self.assertEqual(index._chunk_page(MARKDOWN_PAGE, cfg),
                                 legacy_chunks(MARKDOWN_PAGE))

    def test_build_with_default_config_matches_legacy_chunk_ids(self):
        """End to end: default build writes the same (startline, text, chunk_id) triples."""
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            page = home / "mirror" / "z0-public" / "page.md"
            page.parent.mkdir(parents=True)
            page.write_text(MARKDOWN_PAGE, encoding="utf-8")
            cfg = Config(home=home)
            index.build(cfg)

            with sqlite3.connect(str(cfg.index_db)) as con:
                rows = con.execute(
                    "select path, startline, text, chunk_id from meta order by startline"
                ).fetchall()
        for rel, startline, text, chunk_id in rows:
            self.assertEqual(rel, "z0-public/page.md")
            self.assertEqual(chunk_id, index.chunk_id_of(rel, text))
        self.assertEqual([(startline, text) for _rel, startline, text, _cid in rows],
                         legacy_chunks(MARKDOWN_PAGE))


class HeadingSplitTest(unittest.TestCase):
    """#/##/### start a new chunk; #### does not; heading lines stay in the chunk."""

    def setUp(self):
        self.blocks = chunks(MARKDOWN_PAGE, chunk_strategy="structural")

    def _block_starting_with(self, prefix: str) -> tuple[int, str]:
        for line, text in self.blocks:
            if text.lstrip("\n").startswith(prefix):
                return line, text
        self.fail(f"no chunk starts with {prefix!r}")

    def test_each_of_h1_h2_h3_starts_a_chunk(self):
        for heading in ("# 一级标题", "## 二级标题", "### 三级标题"):
            with self.subTest(heading=heading):
                line, text = self._block_starting_with(heading)
                self.assertTrue(text.lstrip("\n").startswith(heading))
                self.assertEqual(MARKDOWN_PAGE.splitlines()[line - 1], heading)

    def test_h4_does_not_split(self):
        # ``####`` is deliberately not a boundary (spec: #/##/###), so it must
        # live inside the same chunk as the body that follows it.
        for _line, text in self.blocks:
            if "#### 四级标题" in text:
                self.assertIn("delta 正文", text)
                self.assertIn("tail 结尾行。", text)
                break
        else:                                    # pragma: no cover - fixture guard
            self.fail("fixture lost its level-4 heading")

    def test_startline_is_one_based_and_points_at_the_heading(self):
        lines = MARKDOWN_PAGE.splitlines()
        for line, text in self.blocks:
            first = text.lstrip("\n").splitlines()[0]
            self.assertEqual(lines[line - 1], first)

    def test_heading_syntax_follows_commonmark(self):
        # "#标题" (no space) and "#hashtag" are not ATX headings in CommonMark,
        # and an indented "# 代码" (4 spaces) is a code block.  Splitting on
        # those would turn tags and pasted snippets into fake sections.
        raw = "#not a heading\nbody one\n\n#hashtag 列表\nbody two\n\n    # 缩进代码\nbody three\n"
        got = chunks(raw, chunk_strategy="structural")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][0], 1)

    def test_short_section_stays_one_chunk(self):
        # Every heading section here is well under the 1000-char cap, so the
        # structural page yields exactly one chunk per section — 4 of them:
        # preamble, h1, h2, and h3 (which also swallows the level-4 heading and
        # the tail, since #### does not split).  The lines strategy produced 2
        # windows for the same page, cut at arbitrary places.
        self.assertEqual(len(self.blocks), 4)
        self.assertGreater(len(legacy_chunks(MARKDOWN_PAGE)), 1)
        self.assertNotEqual([t for _l, t in self.blocks],
                            [t for _l, t in legacy_chunks(MARKDOWN_PAGE)])


class LongBlockTest(unittest.TestCase):
    """Over-long sections fall back to blank lines, then to line boundaries."""

    def test_over_long_section_falls_back_to_blank_lines(self):
        cap = 200
        got = chunks(PARAGRAPH_PAGE, chunk_strategy="structural", chunk_max_chars=cap)
        self.assertGreater(len(got), 1)
        for _line, text in got:
            self.assertLessEqual(len(text), cap)
            # A paragraph break is the preferred cut: no chunk starts on a
            # continuation line of a split paragraph.
            self.assertTrue(text.lstrip("\n").startswith("段落"))
            self.assertFalse(text.lstrip("\n").startswith("字"))

    def test_single_paragraph_falls_back_to_line_boundaries(self):
        raw = "\n".join(f"line {i} " + "x" * 40 for i in range(20))
        got = chunks(raw, chunk_strategy="structural", chunk_max_chars=120)
        self.assertGreater(len(got), 1)
        for _line, text in got:
            self.assertLessEqual(len(text), 120)
        self.assertEqual(rebuild(raw, got), "\n".join(raw.splitlines()))

    def test_single_over_long_line_is_atomic(self):
        # No cut can make one line fit; splitting it would lose a line.
        raw = "y" * 500 + "\nshort tail\n"
        got = chunks(raw, chunk_strategy="structural", chunk_max_chars=100)
        self.assertEqual(len(got[0][1]), 500)
        self.assertEqual(got[0][0], 1)
        self.assertEqual(got[1], (2, "short tail"))

    def test_cap_counts_characters_not_utf8_bytes(self):
        # 3 CJK lines x 300 characters = 900 characters but 2702 UTF-8 bytes.
        # Counting bytes would split a block that comfortably fits the cap.
        raw = "\n".join("字" * 300 for _ in range(3)) + "\n"
        self.assertGreater(len(raw.encode("utf-8")), 1000)
        got = chunks(raw, chunk_strategy="structural", chunk_max_chars=1000)
        self.assertEqual(len(got), 1)
        self.assertLessEqual(len(got[0][1]), 1000)

    def test_cap_slack_for_a_blank_line_that_cannot_be_dropped(self):
        # 99 content chars + a blank line already sits on the 100-char cap; the
        # next blank line has nowhere to go — dropping it would break the
        # line-for-line rebuild — so it stays and the block ends 1 char over.
        # Measured on the production corpus: 2 such blocks out of 131 028.
        raw = "z" * 99 + "\n\n\n"
        got = chunks(raw, chunk_strategy="structural", chunk_max_chars=100)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0][1], "z" * 99 + "\n\n")
        self.assertEqual(rebuild(raw, got), "\n".join(raw.splitlines()))


class CodeFenceTest(unittest.TestCase):
    """A fenced block is one unit: neither headings nor the cap may cut it."""

    def test_fence_is_never_cut(self):
        fence = "```python\n" + "\n".join(f"code line {i}" for i in range(30)) + "\n```"
        raw = "# 标题\nbefore fence\n\n" + fence + "\nafter fence\n"
        got = chunks(raw, chunk_strategy="structural", chunk_max_chars=80)
        holders = [text for _l, text in got if "```python" in text]
        self.assertEqual(len(holders), 1, "the fence must live in exactly one chunk")
        block = holders[0]
        self.assertIn("code line 0", block)
        self.assertIn("code line 29", block)         # body intact, end included
        self.assertIn(fence, block)                  # one contiguous fence, uncut
        self.assertNotIn("after fence", block)

    def test_heading_inside_fence_does_not_split(self):
        got = chunks(MARKDOWN_PAGE, chunk_strategy="structural")
        block = [text for _l, text in got if "# 这一行看起来像标题" in text][0]
        self.assertIn("def f():", block)
        self.assertIn("还有空行也在围栏里", block)

    def test_unterminated_fence_is_not_cut(self):
        raw = "```\n" + "\n".join(f"line {i}" for i in range(20)) + "\n"
        got = chunks(raw, chunk_strategy="structural", chunk_max_chars=40)
        self.assertEqual(len(got), 1)
        self.assertEqual(rebuild(raw, got), "\n".join(raw.splitlines()))

    def test_oversized_fence_stays_whole(self):
        # The cap yields to the fence: cutting it would contradict "never cut".
        fence = "```\n" + "\n".join("z" * 40 for _ in range(10)) + "\n```"
        got = chunks("# h\n" + fence + "\n", chunk_strategy="structural",
                     chunk_max_chars=100)
        holder = [text for _l, text in got if "```" in text][0]
        self.assertGreater(len(holder), 100)
        self.assertTrue(holder.rstrip().endswith("```"))


class ReassemblyTest(unittest.TestCase):
    """The structural partition must be exact and free of whitespace-only chunks."""

    def test_blocks_rebuild_page_line_for_line(self):
        for name, raw in CORPORA.items():
            with self.subTest(page=name):
                got = chunks(raw, chunk_strategy="structural", chunk_max_chars=60)
                self.assertEqual(rebuild(raw, got), "\n".join(raw.splitlines()))

    def test_startlines_are_monotonic_and_contiguous(self):
        got = chunks(MARKDOWN_PAGE, chunk_strategy="structural", chunk_max_chars=60)
        self.assertEqual(got[0][0], 1)
        for (line, text), (next_line, _t) in zip(got, got[1:]):
            # A block may end with a blank line, whose newline is part of the
            # text — count separators, not ``splitlines()``.
            self.assertEqual(next_line, line + text.count("\n") + 1)

    def test_no_whitespace_only_blocks(self):
        for name, raw in CORPORA.items():
            with self.subTest(page=name):
                for line, text in chunks(raw, chunk_strategy="structural"):
                    self.assertTrue(text.strip(), f"blank chunk at line {line}")

    def test_blank_page_has_no_chunks(self):
        for raw in ("", "\n", "\n\n   \n\n", "   "):
            with self.subTest(raw=raw):
                self.assertEqual(chunks(raw, chunk_strategy="structural"), [])
                self.assertEqual(chunks(raw), [])

    def test_messy_page_keeps_leading_and_trailing_blank_lines(self):
        got = chunks(MESSY_PAGE, chunk_strategy="structural")
        self.assertEqual(rebuild(MESSY_PAGE, got), "\n".join(MESSY_PAGE.splitlines()))
        self.assertTrue(got[-1][1].endswith("\n"))


class ConfigTest(unittest.TestCase):
    """The new knobs follow the existing env-over-file-over-default precedence."""

    def test_environment_overrides(self):
        env = {
            "DSH_KB_CHUNK_STRATEGY": "structural",
            "DSH_KB_CHUNK_MAX_CHARS": "640",
            "DSH_KB_CHUNK_LINES": "9",
        }
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
            cfg = load_config(home=tmp)
        self.assertEqual(cfg.chunk_strategy, "structural")
        self.assertEqual(cfg.chunk_max_chars, 640)
        self.assertEqual(cfg.chunk_lines, 9)

    def test_config_file_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.json").write_text(json.dumps({
                "chunk_strategy": "STRUCTURAL",     # case-insensitive on purpose
                "chunk_max_chars": "800",           # a string is coerced, not fatal
            }), encoding="utf-8")
            cfg = load_config(home=home)
        self.assertEqual(cfg.chunk_strategy, "structural")
        self.assertEqual(cfg.chunk_max_chars, 800)
        self.assertEqual(cfg.unknown_keys, ())

    def test_invalid_strategy_falls_back_to_lines_and_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.json").write_text(
                json.dumps({"chunk_strategy": "段落"}), encoding="utf-8")
            cfg = load_config(home=home)
        self.assertEqual(cfg.chunk_strategy, "lines")
        self.assertIn("chunk_strategy", cfg.unknown_keys)

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(home=tmp, chunk_strategy="structural",
                              chunk_max_chars=256)
        self.assertEqual(cfg.chunk_strategy, "structural")
        self.assertEqual(cfg.chunk_max_chars, 256)


class ImageArmConfigTest(unittest.TestCase):
    """P2 supplementary: the image sidecar's keys must be real config fields.

    The image arm reads its settings through ``getattr(cfg.rag, ..., default)``,
    so it worked before these fields existed.  The gap was on the **write** side:
    a deployment configured through ``config.json`` (production runs three MCP
    processes, so an environment variable is easy to miss on one of them) had its
    keys land in ``unknown_keys`` and silently do nothing — a silent downgrade.
    Pinned here: file form is known, typed, and readable under both names.
    """

    def test_config_file_keys_are_known_and_typed(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "config.json").write_text(json.dumps({"rag": {
                "image_arm_enabled": True,
                "image_arm_weight": "0.75",          # a string must be coerced
                "image_arm_db": "/srv/kb/index/vectors-image.sqlite",
            }}), encoding="utf-8")
            cfg = load_config(home=home)
        self.assertEqual(cfg.unknown_keys, ())
        self.assertIs(cfg.rag.image_arm_enabled, True)
        self.assertIsInstance(cfg.rag.image_arm_weight, float)
        self.assertEqual(cfg.rag.image_arm_weight, 0.75)
        self.assertEqual(cfg.rag.image_arm_db, "/srv/kb/index/vectors-image.sqlite")
        # ``ragvec._image_settings`` reads the historical name; the alias has to
        # follow the field, otherwise the file form stays ineffective.
        self.assertEqual(cfg.rag.image_db, cfg.rag.image_arm_db)

    def test_defaults_keep_the_arm_off(self):
        cfg = Config(home=Path("/nonexistent-home"))
        self.assertFalse(cfg.rag.image_arm_enabled)
        self.assertEqual(cfg.rag.image_arm_weight, 0.6)
        self.assertEqual(cfg.rag.image_arm_db, "")
        self.assertEqual(cfg.rag.image_db, "")

    def test_environment_form_still_works(self):
        env = {"DSH_KB_RAG_IMAGE_ARM_ENABLED": "1",
               "DSH_KB_RAG_IMAGE_ARM_WEIGHT": "0.8",
               "DSH_KB_RAG_IMAGE_ARM_DB": "/tmp/vectors-image.sqlite"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, env):
            cfg = load_config(home=tmp)
        self.assertTrue(cfg.rag.image_arm_enabled)
        self.assertEqual(cfg.rag.image_arm_weight, 0.8)
        self.assertEqual(cfg.rag.image_arm_db, "/tmp/vectors-image.sqlite")

    def test_bad_values_fall_back_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"DSH_KB_RAG_IMAGE_ARM_WEIGHT": "heavy"}):
            cfg = load_config(home=tmp)
        self.assertEqual(cfg.rag.image_arm_weight, 0.6)
        with tempfile.TemporaryDirectory() as tmp:      # non-string path in the file
            home = Path(tmp)
            (home / "config.json").write_text(
                json.dumps({"rag": {"image_arm_db": 42}}), encoding="utf-8")
            cfg = load_config(home=home)
        self.assertEqual(cfg.rag.image_arm_db, "")


class StructuralBuildTest(unittest.TestCase):
    """A structural rebuild writes exactly the structural blocks, and they search."""

    def test_build_writes_structural_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            page = home / "mirror" / "z0-public" / "page.md"
            page.parent.mkdir(parents=True)
            page.write_text(MARKDOWN_PAGE, encoding="utf-8")
            cfg = Config(home=home, chunk_strategy="structural")
            index.build(cfg)
            expected = chunks(MARKDOWN_PAGE, chunk_strategy="structural")
            with sqlite3.connect(str(cfg.index_db)) as con:
                rows = con.execute(
                    "select startline, text, chunk_id from meta order by startline"
                ).fetchall()
            self.assertEqual([(line, text) for line, text, _cid in rows], expected)
            for _line, text, chunk_id in rows:
                self.assertEqual(chunk_id, index.chunk_id_of("z0-public/page.md", text))

            hit = index.search(cfg, "charlie", zone="z01")
            self.assertEqual(len(hit.hits), 1)
            lines = MARKDOWN_PAGE.splitlines()
            self.assertIn("charlie", lines[hit.hits[0].line - 1])


if __name__ == "__main__":                       # pragma: no cover
    unittest.main()
