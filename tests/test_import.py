"""The import engine: deterministic rendering, guards, and additive writes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import importer  # noqa: E402
from dsh_kb.config import Config, Zone  # noqa: E402
from dsh_kb.importers import Record  # noqa: E402


def cfg(**overrides) -> Config:
    base = dict(
        home=Path("/tmp/kb-home"),
        zones=(Zone("z0-public", index=True), Zone("z1-restricted", index=True),
               Zone("z2-sensitive", index=False, mirror=False)),
        never=("credentials", "secrets"),
    )
    base.update(overrides)
    return Config(**base)


class FakeSource:
    """The three source methods ``apply`` needs, in memory."""

    def __init__(self, files: dict[str, str] | None = None):
        self.files = dict(files or {})
        self.writes: list[str] = []

    def walk(self):
        class Entry:
            def __init__(self, rel):
                self.rel = rel
                self.is_dir = False

        return [Entry(rel) for rel in sorted(self.files)]

    def read(self, rel: str) -> bytes:
        return self.files[rel].encode("utf-8")

    def write(self, rel: str, data: bytes) -> None:
        self.files[rel] = data.decode("utf-8")
        self.writes.append(rel)


def records(*items: Record) -> list[Record]:
    return list(items)


class TestSlug(unittest.TestCase):
    def test_case_is_folded_because_windows_would_merge_the_files(self):
        self.assertEqual(importer.slug("Docker"), importer.slug("docker"))
        self.assertEqual(importer.slug("Docker"), "docker")

    def test_cjk_is_kept(self):
        self.assertEqual(importer.slug("论文 项目"), "论文-项目")

    def test_unsafe_characters_are_removed(self):
        self.assertEqual(importer.slug('a/b:c*d?"e'), "a-b-c-d-e")

    def test_empty_becomes_a_usable_stem(self):
        self.assertEqual(importer.slug("///"), "page")

    def test_a_markdown_suffix_is_not_doubled(self):
        self.assertEqual(importer.slug("agents.md"), "agents")
        self.assertEqual(importer.slug("notes.markdown"), "notes")


class TestPlan(unittest.TestCase):
    def _plan(self, recs, **kwargs):
        options = dict(prefix="memory", zone="z1-restricted", generator="test/1",
                       resource="/tmp/records.json", at="2026-10-29T12:00:00+08:00")
        options.update(kwargs)
        return importer.plan(cfg(), recs, **options)

    def test_first_tag_is_the_default_group(self):
        plan = self._plan(records(
            Record(content="a", tags=("git",), type="fact"),
            Record(content="b", tags=("git",), type="decision"),
            Record(content="c"),
        ))
        self.assertEqual([p.rel for p in plan.pages],
                         ["memory/git.md", "memory/untagged.md"])
        self.assertEqual(plan.pages[0].records, 2)

    def test_group_by_none_gives_one_page_per_record(self):
        plan = self._plan(records(
            Record(content="a", id="one"), Record(content="b", id="two"),
        ), group_by="none")
        self.assertEqual([p.rel for p in plan.pages], ["memory/one.md", "memory/two.md"])

    def test_group_by_field_uses_a_record_attribute(self):
        plan = self._plan(records(
            Record(content="a", type="fact"), Record(content="b", type="decision"),
        ), group_by="field:type")
        self.assertEqual([p.rel for p in plan.pages],
                         ["memory/decision.md", "memory/fact.md"])

    def test_colliding_group_names_do_not_overwrite_each_other(self):
        """`【常守规则` and `常守规则` slugify identically — both must survive."""
        recs = [Record(content="a", tags=("【常守规则",)),
                Record(content="b", tags=("常守规则",))]
        plan = self._plan(recs)
        rels = sorted(p.rel for p in plan.pages)
        self.assertEqual(len(rels), 2, "two groups must produce two pages")
        self.assertEqual(len(set(rels)), 2, "the two paths must differ")

    def test_collision_paths_are_stable_across_runs(self):
        recs = [Record(content="a", tags=("【常守规则",)),
                Record(content="b", tags=("常守规则",))]
        first = sorted(p.rel for p in self._plan(recs).pages)
        second = sorted(p.rel for p in self._plan(list(reversed(recs))).pages)
        self.assertEqual(first, second)

    def test_a_public_zone_needs_an_explicit_opt_in(self):
        with self.assertRaises(ValueError) as ctx:
            self._plan(records(Record(content="a")), zone="z0-public")
        self.assertIn("--allow-public", str(ctx.exception))

    def test_an_unknown_zone_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._plan(records(Record(content="a")), zone="z9-nope")
        self.assertIn("not configured", str(ctx.exception))

    def test_the_never_list_aborts_the_whole_run(self):
        with self.assertRaises(ValueError) as ctx:
            self._plan(records(Record(content="a", tags=("credentials",))))
        self.assertIn("never", str(ctx.exception))

    def test_an_oversized_page_is_refused_instead_of_written(self):
        with self.assertRaises(ValueError) as ctx:
            self._plan(records(Record(content="x" * 500)), max_page_bytes=100)
        self.assertIn("max-page-bytes", str(ctx.exception))

    def test_no_records_is_an_error_not_an_empty_run(self):
        with self.assertRaises(ValueError):
            self._plan([])


class TestRender(unittest.TestCase):
    def test_front_matter_and_body_are_stable(self):
        page = importer.plan(cfg(), records(
            Record(content="body text", title="A note", tags=("git",), type="fact",
                   id="one", updated_at="2026-10-20T10:00:00+08:00", origin="obsidian"),
        ), prefix="memory", zone="z1-restricted", generator="test/1",
            resource="/tmp/r.json", at="2026-10-29T12:00:00+08:00").pages[0]

        self.assertIn('title: "git"', page.content)
        self.assertIn("zone: z1", page.content)
        self.assertIn("generated: { by: test/1, at: 2026-10-29T12:00:00+08:00 }", page.content)
        self.assertIn("## A note · 2026-10-20", page.content)
        self.assertIn("id: `one` · type: fact · origin: obsidian · tags: git", page.content)
        self.assertTrue(page.content.endswith("\n"))

    def test_front_matter_tags_are_a_summary_not_an_index(self):
        recs = [Record(content="a", tags=("git", "src-id:mem_1", "workspace:/x", "qc:stale")),
                Record(content="b", tags=("git", "docker"))]
        page = importer.plan(cfg(), recs, prefix="m", zone="z1-restricted",
                             generator="g", resource="r").pages[0]
        front = page.content.split("---")[1]
        self.assertIn("git", front)
        self.assertIn("docker", front)
        self.assertNotIn("src-id:", front, "internal tags stay on the record, not the page")
        self.assertNotIn("workspace:", front)
        self.assertNotIn("qc:", front)

    def test_front_matter_tags_are_capped(self):
        recs = [Record(content=str(i), tags=(f"tag{i:02d}",)) for i in range(30)]
        page = importer.plan(cfg(), recs, prefix="m", zone="z1-restricted",
                             generator="g", resource="r").pages[0]
        front = page.content.split("---")[1]
        tags_line = next(line for line in front.splitlines() if line.startswith("tags:"))
        self.assertLessEqual(tags_line.count('"') // 2, importer.MAX_FRONT_MATTER_TAGS + 5)

    def test_no_wall_clock_by_default(self):
        """Re-importing the same records must write nothing — that needs a stable page."""
        page = importer.plan(cfg(), records(Record(content="a", tags=("git",))),
                             prefix="m", zone="z1-restricted", generator="g",
                             resource="r").pages[0]
        self.assertNotIn("at:", page.content)
        self.assertIn("generated: { by: g }", page.content)

    def test_an_explicit_as_of_is_recorded_verbatim(self):
        page = importer.plan(cfg(), records(Record(content="a", tags=("git",))),
                             prefix="m", zone="z1-restricted", generator="g",
                             resource="r", at="2026-10-29T00:00:00+08:00").pages[0]
        self.assertIn("generated: { by: g, at: 2026-10-29T00:00:00+08:00 }", page.content)

    def test_the_same_records_render_the_same_bytes(self):
        recs = records(Record(content="a", tags=("t",)), Record(content="b", tags=("t",)))
        first = importer.plan(cfg(), recs, prefix="m", zone="z1-restricted",
                              generator="g", resource="r", at="2026-10-29T12:00:00+08:00")
        second = importer.plan(cfg(), list(reversed(recs)), prefix="m", zone="z1-restricted",
                               generator="g", resource="r", at="2026-10-29T12:00:00+08:00")
        self.assertEqual(first.pages[0].content, second.pages[0].content)


class TestApply(unittest.TestCase):
    def _plan(self, recs, **kwargs):
        options = dict(prefix="memory", zone="z1-restricted", generator="t", resource="r",
                       at="2026-10-29T12:00:00+08:00")
        options.update(kwargs)
        return importer.plan(cfg(), recs, **options)

    def test_dry_run_writes_nothing(self):
        plan = self._plan(records(Record(content="a", tags=("git",))))
        source = FakeSource()
        outcome = importer.apply(source, plan, write=False)
        self.assertEqual((outcome.new, outcome.written), (1, 0))
        self.assertEqual(source.writes, [])

    def test_a_second_run_is_a_no_op(self):
        plan = self._plan(records(Record(content="a", tags=("git",))))
        source = FakeSource()
        importer.apply(source, plan, write=True)
        again = importer.apply(source, plan, write=True)
        self.assertEqual((again.new, again.changed, again.same), (0, 0, 1))
        self.assertEqual(len(source.writes), 1)

    def test_a_changed_record_rewrites_the_page(self):
        plan = self._plan(records(Record(content="a", tags=("git",))))
        source = FakeSource()
        importer.apply(source, plan, write=True)
        changed = self._plan(records(Record(content="b", tags=("git",))))
        outcome = importer.apply(source, changed, write=True)
        self.assertEqual((outcome.changed, outcome.written), (1, 1))
        self.assertIn("b", source.files["memory/git.md"])


if __name__ == "__main__":
    unittest.main()
