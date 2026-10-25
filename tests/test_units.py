"""Unit tests for the pure helpers: text handling and configuration policy.

These run anywhere — no network, no filesystem fixtures.
"""
from __future__ import annotations

import sys
import unittest
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb.config import Config, Zone, load_config, source_available  # noqa: E402
from dsh_kb.text import (  # noqa: E402
    expand_for_index,
    make_snippet,
    match_expression,
    parse_frontmatter,
    split_frontmatter,
    query_tokens,
)


class TestTextExpansion(unittest.TestCase):
    def test_latin_survives_expansion(self):
        """Regression: an early version dropped all latin text, so identifiers
        and paths became unsearchable while Chinese still worked."""
        out = expand_for_index("docker 与 fstab 的配置")
        for needle in ("docker", "fstab", "配置"):
            self.assertIn(needle, out)

    def test_cjk_bigrams_are_appended(self):
        out = expand_for_index("知识库")
        self.assertIn("知识", out)
        self.assertIn("识库", out)

    def test_pure_latin_is_untouched(self):
        self.assertEqual(expand_for_index("only latin here"), "only latin here")

    def test_query_tokens_split_identifiers(self):
        tokens = query_tokens("z2-sensitive")
        self.assertEqual(tokens[0], "z2-sensitive")
        self.assertIn("z2", tokens)
        self.assertIn("sensitive", tokens)

    def test_query_tokens_expand_chinese(self):
        tokens = query_tokens("挂载")
        self.assertIn("挂载", tokens)

    def test_query_tokens_are_deduplicated(self):
        tokens = query_tokens("docker docker DOCKER")
        self.assertEqual([t.lower() for t in tokens], ["docker"])

    def test_match_expression_quotes_every_token(self):
        self.assertEqual(match_expression(["a", "b"]), '"a" OR "b"')

    def test_match_expression_neutralises_fts_syntax(self):
        expr = match_expression(['OR "x"'])
        self.assertTrue(expr.startswith('"'))
        self.assertNotIn(" OR x", expr)


class TestSnippets(unittest.TestCase):
    def test_reports_line_of_first_hit(self):
        body = "line one\nline two\nneedle here\nline four"
        snippet, line = make_snippet(body, ["needle"], width=80)
        self.assertEqual(line, 3)
        self.assertIn("needle", snippet)

    def test_falls_back_to_head_when_no_hit(self):
        snippet, line = make_snippet("nothing to see", ["absent"], width=80)
        self.assertEqual(line, 1)
        self.assertTrue(snippet)

    def test_window_is_bounded(self):
        body = "x" * 500 + "needle" + "y" * 500
        snippet, _ = make_snippet(body, ["needle"], width=100)
        self.assertLessEqual(len(snippet), 110)


class TestFrontmatter(unittest.TestCase):
    def test_reads_scalars_and_inline_lists(self):
        text = "---\ntype: lesson\ntitle: 标题\ntags: [a, b]\n---\nbody\n"
        meta = parse_frontmatter(text)
        self.assertEqual(meta["type"], "lesson")
        self.assertEqual(meta["title"], "标题")
        self.assertEqual(meta["tags"], ["a", "b"])

    def test_missing_frontmatter_is_empty(self):
        self.assertEqual(parse_frontmatter("# just a heading"), {})

    def test_inline_mapping_is_read(self):
        """``generated.at`` 是"谁更新"的判据，所以内联 map 必须读出来。

        改写自旧断言（旧版刻意跳过 `{...}`）。跳过它意味着 `pages.fm_at`
        永远是空的，而 supersede 的默认 `valid_until` 与"谁取代谁"的排序都要
        它。仍然不支持的只有**更深的块状嵌套**，那是另一条断言。
        """
        meta = parse_frontmatter("---\ntype: x\ngenerated: { by: a, at: b }\n---\n")
        self.assertEqual(meta["type"], "x")
        self.assertEqual(meta["generated"], {"by": "a", "at": "b"})

    def test_block_list_and_deep_nesting(self):
        """块状列表（supersedes 的常见写法）要读；更深的嵌套只吃掉、不报错。"""
        text = ("---\ntype: x\nsupersedes:\n  - \"z1/a.md\"\n  - z1/b.md\n"
                "sources:\n  - id: a\n    resource: /b\n---\nbody\n")
        meta = parse_frontmatter(text)
        self.assertEqual(meta["supersedes"], ["z1/a.md", "z1/b.md"])
        self.assertEqual(meta["sources"], [{"id": "a"}])

    def test_quoted_comma_survives_the_inline_split(self):
        meta = parse_frontmatter('---\ntags: ["a,b", c]\n---\n')
        self.assertEqual(meta["tags"], ["a,b", "c"])

    def test_valueless_key_is_an_empty_list(self):
        self.assertEqual(parse_frontmatter("---\nsupersedes:\n---\nx\n"), {"supersedes": []})


class TestSplitFrontmatter(unittest.TestCase):
    """前端块剥离 + **行号回算**：报错行号比不剥更糟。"""

    def test_header_lines_are_counted_so_file_lines_survive(self):
        text = "---\ntype: x\ntitle: T\n---\nbody one\nbody two\n"
        body, header = split_frontmatter(text)
        self.assertEqual(body, "body one\nbody two\n")
        self.assertEqual(header, 4, "前 4 行是头部（含收尾的 ---）")
        # body 第 2 行 → 文件第 6 行
        self.assertEqual(6, header + 2)

    def test_no_header_is_a_noop(self):
        text = "# just a heading\nmore\n"
        self.assertEqual(split_frontmatter(text), (text, 0))

    def test_unterminated_header_is_left_alone(self):
        text = "---\ntype: x\nno closing rule\n"
        self.assertEqual(split_frontmatter(text), (text, 0))

    def test_header_without_body(self):
        """没有正文页也要算对：头部占了 1..3 行（含收尾那条 ---）。"""
        body, header = split_frontmatter("---\ntype: x\n---")
        self.assertEqual(body, "")
        self.assertEqual(header, 3)

    def test_inline_rule_inside_the_body_is_not_a_terminator_twice(self):
        text = "---\ntype: x\n---\nbody\n---\nafter\n"
        body, header = split_frontmatter(text)
        self.assertEqual(body, "body\n---\nafter\n")
        self.assertEqual(header, 3)


class TestIndexNever(unittest.TestCase):
    """``index_never``：只挡索引、不挡镜像 —— 与 ``never`` 的差别就是它的全部意义。"""

    def setUp(self):
        self.cfg = Config(home=Path("/tmp/dsh-kb-index-never"),
                          index_never=("/backup/", "/out/", "_待入库/"))

    def test_substring_and_case_and_backslash(self):
        self.assertTrue(self.cfg.is_index_never("z1/a/backup/x.md"))
        self.assertTrue(self.cfg.is_index_never("z1/A/BACKUP/x.md"))
        self.assertTrue(self.cfg.is_index_never("z1\\a\\out\\x.md"))
        self.assertFalse(self.cfg.is_index_never("z1/a/layout/x.md"), "/out/ 不该误伤 layout")
        self.assertFalse(self.cfg.is_index_never("z1/a/README.md"))

    def test_index_skips_but_mirror_keeps(self):
        rel = "z1-restricted/workspace/p/backup/index.md"
        self.assertFalse(self.cfg.index_allows(rel), "不进索引")
        self.assertTrue(self.cfg.mirror_allows(rel), "**仍然镜像** —— 本地副本不该被 prune 掉")

    def test_never_still_beats_everything(self):
        cfg = Config(home=Path("/tmp/x"), never=("credentials",),
                     index_never=("/backup/",))
        self.assertFalse(cfg.mirror_allows("z1/credentials/a.md"))
        self.assertFalse(cfg.index_allows("z1/credentials/a.md"))

    def test_default_is_empty_so_nothing_changes_unless_asked(self):
        self.assertEqual(Config(home=Path("/tmp/x")).index_never, ())

    def test_config_file_is_honoured(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "kb"
            home.mkdir()
            (home / "config.json").write_text(
                json.dumps({"index_never": ["/out/"]}), encoding="utf-8")
            cfg = load_config(home=home)
            self.assertEqual(cfg.index_never, ("/out/",))
            self.assertFalse(cfg.index_allows("z1/a/out/x.md"))
            self.assertTrue(cfg.mirror_allows("z1/a/out/x.md"))


class TestZonePolicy(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(home=Path("/tmp/dsh-kb-test"))

    def test_default_zones(self):
        names = [z.name for z in self.cfg.zones]
        self.assertEqual(names, ["z0-public", "z1-restricted", "z2-sensitive"])

    def test_sensitive_zone_is_never_mirrored_by_default(self):
        """The privacy fix: sensitive content must not even land on disk."""
        self.assertFalse(self.cfg.mirror_allows("z2-sensitive/meds.md"))
        self.assertTrue(self.cfg.mirror_allows("z0-public/notes.md"))
        self.assertTrue(self.cfg.mirror_allows("z1-restricted/profile.md"))

    def test_sensitive_zone_is_not_indexed(self):
        self.assertFalse(self.cfg.index_allows("z2-sensitive/meds.md"))
        self.assertTrue(self.cfg.index_allows("z0-public/notes.md"))

    def test_never_list_beats_everything(self):
        cfg = self.cfg
        self.assertFalse(cfg.mirror_allows("z0-public/credentials/token.md"))
        self.assertFalse(cfg.index_allows("z0-public/credentials/token.md"))
        self.assertFalse(cfg.mirror_allows("密钥/key.md"))

    def test_non_markdown_is_not_indexed(self):
        self.assertFalse(self.cfg.index_allows("z0-public/data.json"))

    def test_root_documents_are_allowed(self):
        self.assertTrue(self.cfg.mirror_allows("index.md"))
        self.assertTrue(self.cfg.index_allows("index.md"))

    def test_zone_selector_expansion(self):
        self.assertEqual(self.cfg.zones_for_query("z01"), ("z0-public", "z1-restricted"))
        self.assertEqual(self.cfg.zones_for_query("z0"), ("z0-public",))
        self.assertEqual(len(self.cfg.zones_for_query("all")), 3)

    def test_mirror_true_index_false_is_possible(self):
        cfg = Config(
            home=Path("/tmp/x"),
            zones=(Zone("z2-sensitive", index=False, inject=False, mirror=True),),
        )
        self.assertTrue(cfg.mirror_allows("z2-sensitive/a.md"))
        self.assertFalse(cfg.index_allows("z2-sensitive/a.md"))


class TestSourceResolution(unittest.TestCase):
    def test_auto_prefers_mount_over_webdav(self):
        cfg = Config(home=Path("/tmp/x"), source="auto", mount="/tmp", webdav_root="https://example.invalid/dav")
        self.assertEqual(source_available(cfg)[0], "mount")

    def test_auto_falls_back_to_webdav(self):
        cfg = Config(home=Path("/tmp/x"), source="auto", mount="/definitely/not/here",
                     webdav_root="https://example.invalid/dav")
        self.assertEqual(source_available(cfg)[0], "webdav")

    def test_nothing_configured_is_reported_not_raised(self):
        kind, reason = source_available(Config(home=Path("/tmp/x"), source="auto"))
        self.assertEqual(kind, "")
        self.assertIn("no source", reason)

    def test_env_override_wins(self):
        import os

        old = os.environ.get("DSH_KB_HOME")
        os.environ["DSH_KB_HOME"] = "/tmp/dsh-kb-env-home"
        try:
            self.assertEqual(str(load_config().home), "/tmp/dsh-kb-env-home")
        finally:
            if old is None:
                os.environ.pop("DSH_KB_HOME", None)
            else:
                os.environ["DSH_KB_HOME"] = old


class TestConfigFileFollowsHome(unittest.TestCase):
    """`--home X` must read `X/config.json`, not the default directory's.

    Regression, found the honest way on a real deployment: the config file was
    loaded from the environment-derived home *before* overrides were applied, so
    a host configured with `--home` (the recommended, scrub-proof way for MCP
    hosts) ran on the built-in defaults — three `never` path exclusions and
    every zone's `inject` flag were silently ignored.
    """

    def _load_without_env(self, **overrides):
        import os

        old = os.environ.pop("DSH_KB_HOME", None)
        try:
            return load_config(**overrides)
        finally:
            if old is not None:
                os.environ["DSH_KB_HOME"] = old

    def test_home_override_relocates_the_config_file(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "state"
            home.mkdir()
            (home / "config.json").write_text(
                json.dumps({"never": ["only-in-this-file"], "stale_minutes": 99}),
                encoding="utf-8",
            )
            cfg = self._load_without_env(home=home)
            self.assertEqual(cfg.home, home)
            self.assertEqual(cfg.never, ("only-in-this-file",))
            self.assertEqual(cfg.stale_minutes, 99)

    def test_zones_come_from_the_overridden_home(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "state"
            home.mkdir()
            (home / "config.json").write_text(
                json.dumps({"zones": [{"name": "z0-public", "index": True,
                                       "inject": False, "mirror": True}]}),
                encoding="utf-8",
            )
            cfg = self._load_without_env(home=home)
            self.assertEqual([(z.name, z.inject) for z in cfg.zones], [("z0-public", False)])

    def test_a_missing_config_file_still_falls_back_to_defaults(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "no-config-here"
            empty.mkdir()
            cfg = self._load_without_env(home=empty)
            self.assertEqual(cfg.home, empty)
            # the dataclass default, not a number typed twice
            self.assertEqual(cfg.stale_minutes, Config(home=empty).stale_minutes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
