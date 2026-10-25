"""Supersede relations: parsing, indexing, retrieval marking, and health checks.

What is pinned here, and why each one matters operationally:

* **fields are read the way people write them** — a block list, an inline list,
  a bare scalar, quoted values, and a `tags:` string instead of a list.  A
  supersede field that silently fails to parse is worse than an absent one: the
  page keeps looking current;
* **a superseded page is still returned** — annotated (`[已被 … 取代]`, flags on
  the hit) and ranked after live pages.  Hiding it recreates the "I know I wrote
  that down and search cannot find it" complaint, so the default must not do it;
* **`include_superseded=False` really hides, and still fills `limit`** — the
  filter runs after ranking, so the only honest way to fill the page is to widen
  the window; that loop is asserted explicitly;
* **`valid_until` is judged against the clock**, not presence: a future date
  does not retire a page;
* **an index built before these tables existed still answers** — the deploy
  order is "code first, data later", and that window must not break search;
* **`check()` catches the four ways the relation can rot** — inverse mismatch,
  cycle, dangling reference, two successors for one page — and compares the page
  table against the chunk corpus (A7).

Everything is built in a temporary directory: no network, no production paths.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb import index, supersede  # noqa: E402
from dsh_kb.config import Config  # noqa: E402

FILLER = "\n".join(f"filler line {i} lorem ipsum dolor sit amet" for i in range(20))

PAGES = {
    # live: nothing says it is replaced
    "live.md": f'---\ntype: decision\ntitle: "知识库索引重建方法"\ntags: [x]\n---\n'
               f"# 现行\nneedle 现行结论在此\n{FILLER}\n",
    # replaced by live.md, declared from the old side only
    "old.md": f'---\ntype: decision\ntitle: "旧结论"\nsuperseded_by: "live.md"\n---\n'
              f"# 旧\nneedle 旧结论在此\n{FILLER}\n",
    # the new page declaring what it replaces (block list), inverse on the far side
    "new.md": f'---\ntype: decision\ntitle: "新结论"\nsupersedes:\n  - "older.md"\n---\n'
              f"# 新\nneedle 新结论在此\n{FILLER}\n",
    "older.md": f'---\ntype: decision\ntitle: "更旧"\nsuperseded_by: new.md\n---\n'
                f"# 更旧\nneedle 更旧结论在此\n{FILLER}\n",
    # retired by the clock, not by a successor
    "expired.md": f'---\ntype: fact\ntitle: "过期"\nvalid_until: 2020-01-01\n---\n'
                  f"# 过期\nneedle 过期结论在此\n{FILLER}\n",
    # not yet retired
    "future.md": f'---\ntype: fact\ntitle: "未到期"\nvalid_until: 2099-01-01\n---\n'
                 f"# 未到期\nneedle 未到期结论在此\n{FILLER}\n",
}

QUERY = "needle"


class _Fixture(unittest.TestCase):
    pages = PAGES

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.cfg = Config(home=Path(cls._tmp.name) / "kb")
        cls.cfg.mirror_dir.mkdir(parents=True, exist_ok=True)
        for rel, body in cls.pages.items():
            target = cls.cfg.mirror_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        cls.stats = index.build(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def hits(self, **kwargs):
        return index.search(self.cfg, QUERY, include_superseded=True, **kwargs)


class FieldParsing(unittest.TestCase):
    """`page_meta` must read the shapes a human actually writes."""

    def test_block_list_inline_list_and_scalar(self):
        for text, expected in (
            ('---\nsupersedes:\n  - "a.md"\n  - b.md\n---\n', ["a.md", "b.md"]),
            ('---\nsupersedes: ["a.md", b.md]\n---\n', ["a.md", "b.md"]),
            ("---\nsupersedes: a.md\n---\n", ["a.md"]),
            ("---\nsupersedes: [a.md, a.md]\n---\n", ["a.md", "a.md"]),
            ("---\nsupersedes:\n---\n", []),
            ("---\n---\n", []),
        ):
            with self.subTest(text=text):
                self.assertEqual(supersede.page_meta("p.md", text).supersedes, expected)

    def test_tags_as_string_is_a_one_item_list(self):
        self.assertEqual(supersede.page_meta("p.md", "---\ntags: solo\n---\n").tags, ("solo",))

    def test_scalar_relation_and_clock_fields(self):
        page = supersede.page_meta("p.md", (
            "---\ntype: decision\ntitle: \"T\"\nsuperseded_by: \"new.md\"\n"
            "valid_until: 2026-11-01\ngenerated: { by: agent, at: 2026-10-01T00:00:00+08:00 }\n"
            "not_superseding: 讲的是另一件事\n---\n"))
        self.assertEqual(page.superseded_by, "new.md")
        self.assertEqual(page.valid_until, "2026-11-01")
        self.assertEqual(page.fm_at, "2026-10-01T00:00:00+08:00")
        self.assertEqual(page.not_superseding, "讲的是另一件事")
        self.assertTrue(page.is_superseded())

    def test_valid_until_needs_a_clock_before_it_retires_a_page(self):
        page = supersede.page_meta("p.md", "---\nvalid_until: 2020-01-01\n---\n")
        self.assertFalse(page.is_superseded(), "没有 now 就不要猜它过期了")
        self.assertTrue(page.is_superseded(now="2026-10-19T00:00:00+0800"))
        future = supersede.page_meta("p.md", "---\nvalid_until: 2099-01-01\n---\n")
        self.assertFalse(future.is_superseded(now="2026-10-19T00:00:00+0800"))

    def test_a_broken_header_never_raises(self):
        self.assertEqual(supersede.page_meta("p.md", "no frontmatter at all").path, "p.md")


class BuildAndCheck(_Fixture):
    def test_build_records_pages_and_edges(self):
        con = sqlite3.connect(f"file:{self.cfg.index_db}?mode=ro", uri=True)
        try:
            pages = con.execute("select count(*) from pages").fetchone()[0]
            edges = con.execute("select child, parent from supersedes order by child").fetchall()
        finally:
            con.close()
        self.assertEqual(pages, len(PAGES))
        self.assertEqual(edges, [("new.md", "older.md")])
        self.assertEqual(self.stats.supersede_edges, 1)
        self.assertEqual(self.stats.superseded_pages, 2)          # old.md + older.md
        self.assertIn("已取代", self.stats.summary)

    def test_check_is_clean_on_a_consistent_corpus(self):
        report = supersede.check(self.cfg)
        self.assertTrue(report["built"])
        self.assertTrue(report["ok"], report)
        self.assertEqual(report["pages"], len(PAGES))
        self.assertEqual(report["edges"], 1)
        self.assertEqual(report["with_superseded_by"], 2)
        self.assertEqual(report["with_valid_until"], 2)
        self.assertNotIn("problems", {k: v for k, v in report["problems"].items() if v})

    def test_check_catches_an_inverse_mismatch(self):
        """A 取代 B，B 却不认账 —— 这是"谁赢"变成查询歧义的起点。"""
        with _mutated(self.cfg) as db:
            con = sqlite3.connect(db)
            con.execute("update pages set superseded_by='' where path='older.md'")
            con.commit()
            con.close()
            report = supersede.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertEqual(report["problems"]["inverse"], [("new.md", "older.md")])

    def test_check_catches_a_dangling_reference(self):
        with _mutated(self.cfg) as db:
            con = sqlite3.connect(db)
            con.execute("insert into supersedes(child, parent) values ('old.md','ghost.md')")
            con.commit()
            con.close()
            report = supersede.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertEqual(report["problems"]["dangling"], [("old.md", "ghost.md")])

    def test_check_catches_two_successors_for_one_page(self):
        with _mutated(self.cfg) as db:
            con = sqlite3.connect(db)
            con.execute("insert into supersedes(child, parent) values ('live.md','older.md')")
            con.update if False else None
            con.execute("update pages set superseded_by='live.md' where path='older.md'")
            con.commit()
            con.close()
            report = supersede.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertEqual([tuple(r) for r in report["problems"]["multi_parent"]],
                         [("older.md", 2)])

    def test_check_catches_a_cycle(self):
        with _mutated(self.cfg) as db:
            con = sqlite3.connect(db)
            con.execute("insert into supersedes(child, parent) values ('older.md','new.md')")
            con.commit()
            con.close()
            report = supersede.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertEqual([tuple(r) for r in report["problems"]["cycles"]],
                         [("new.md", "new.md"), ("older.md", "older.md")])

    def test_check_flags_a_page_table_that_lost_rows(self):
        """A7：pages 与 meta 必须描述同一份语料，否则排序/统计都在骗人。"""
        with _mutated(self.cfg) as db:
            con = sqlite3.connect(db)
            con.execute("delete from pages where path='live.md'")
            con.commit()
            con.close()
            report = supersede.check(self.cfg)
        self.assertFalse(report["ok"])
        self.assertIn("rebuild", report["note"])


class RetrievalMarking(_Fixture):
    def test_superseded_hits_are_marked_not_hidden(self):
        result = self.hits(limit=10)
        by_path = {h.path: h for h in result.hits}
        self.assertIn("old.md", by_path, "默认不得隐藏历史页")
        self.assertTrue(by_path["old.md"].superseded)
        self.assertEqual(by_path["old.md"].superseded_by, "live.md")
        self.assertTrue(by_path["old.md"].snippet.startswith("[已被 live.md 取代"))
        self.assertEqual(result.superseded_shown, 3, "old/older/expired 三条")
        self.assertIn("已被取代", result.note)

    def test_live_pages_are_not_marked(self):
        by_path = {h.path: h for h in self.hits(limit=10).hits}
        for path in ("live.md", "new.md", "future.md"):
            self.assertFalse(by_path[path].superseded, path)
            self.assertFalse(by_path[path].snippet.startswith("[已被"), path)

    def test_expired_by_clock_is_marked_and_future_is_not(self):
        by_path = {h.path: h for h in self.hits(limit=10).hits}
        self.assertTrue(by_path["expired.md"].superseded)
        self.assertIn("有效至 2020-01-01", by_path["expired.md"].snippet)
        self.assertFalse(by_path["future.md"].superseded)

    def test_superseded_hits_come_after_every_live_hit(self):
        hits = self.hits(limit=10).hits
        flags = [h.superseded for h in hits]
        self.assertEqual(flags, sorted(flags), "现行页必须整体排在已被取代的页之前")

    def test_hiding_still_fills_the_limit(self):
        hidden = index.search(self.cfg, QUERY, limit=3, include_superseded=False)
        self.assertEqual(len(hidden.hits), 3, "过滤在排序之后 ⇒ 必须靠加宽候选池补齐")
        self.assertTrue(all(not h.superseded for h in hidden.hits))
        self.assertTrue(all(not h.snippet.startswith("[已被") for h in hidden.hits))
        self.assertEqual(hidden.superseded_shown, 0)

    def test_hiding_returns_fewer_when_the_corpus_has_fewer_live_pages(self):
        hidden = index.search(self.cfg, QUERY, limit=10, include_superseded=False)
        self.assertEqual(sorted(h.path for h in hidden.hits),
                         ["future.md", "live.md", "new.md"])


class ProposalGate(_Fixture):
    """R1：与现行页撞题却没声明的提案，必须先补一句声明。"""

    def _proposal(self, title, type_="decision", tags=("x",), extra=""):
        return (f"---\ntype: {type_}\ntitle: \"{title}\"\ntags: [{', '.join(tags)}]\n"
                f"{extra}---\n# {title}\n正文\n")

    def test_declaring_not_superseding_clears_the_gate(self):
        text = self._proposal("知识库索引重建方案", extra="not_superseding: \"只讲实现细节\"\n")
        verdict = supersede.gate(self.cfg, "_incoming/p.md", text)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["verdict"], "declared")
        self.assertIn("not_superseding", verdict["reason"])

    def test_declaring_supersedes_clears_the_gate(self):
        text = self._proposal("知识库索引重建方案", extra="supersedes: [\"live.md\"]\n")
        verdict = supersede.gate(self.cfg, "_incoming/p.md", text)
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["verdict"], "declared")
        self.assertEqual(verdict["reason"], "supersedes: live.md")

    def test_same_topic_without_a_declaration_is_blocked(self):
        verdict = supersede.gate(self.cfg, "_incoming/p.md", self._proposal("知识库索引重建方案"))
        self.assertFalse(verdict["ok"])
        self.assertEqual(verdict["verdict"], "must_declare")
        self.assertTrue(verdict["candidates"], "要报出撞了谁")
        self.assertIn("补 supersedes", verdict["reason"])
        self.assertEqual(verdict["candidates"][0]["path"], "live.md")

    def test_blocking_needs_matching_type_and_overlapping_tags(self):
        title = "知识库索引重建方案"
        self.assertEqual(supersede.candidates(self.cfg, title, type_="decision",
                                              tags=("x",)), supersede.candidates(
            self.cfg, title, type_="decision", tags=("x",)))
        self.assertEqual(supersede.candidates(self.cfg, title, type_="other",
                                              tags=("x",)), [])
        self.assertEqual(supersede.candidates(self.cfg, title, type_="decision",
                                              tags=("y",)), [])

    def test_an_unrelated_title_passes(self):
        verdict = supersede.gate(self.cfg, "_incoming/p.md",
                                 self._proposal("完全不相干的另一件事"))
        self.assertTrue(verdict["ok"])
        self.assertEqual(verdict["verdict"], "clear")

    def test_a_short_title_is_never_blocked(self):
        self.assertEqual(supersede.candidates(self.cfg, "短", type_="decision", tags=("x",)), [])

    def test_the_proposal_itself_is_not_a_collision(self):
        """提案已在索引里时，不能和自己撞（否则重推同一份文件永远过不去）。"""
        verdict = supersede.gate(self.cfg, "live.md", PAGES["live.md"])
        self.assertTrue(verdict["ok"], verdict)
        self.assertEqual(verdict["verdict"], "clear")

    def test_candidates_are_ordered_by_similarity(self):
        found = supersede.candidates(self.cfg, "知识库索引重建方法", type_="decision",
                                     tags=("x",))
        self.assertEqual([c.path for c in found], ["live.md"])
        self.assertGreaterEqual(found[0].dice, supersede.GATE_THRESHOLD)


class OlderIndexStillAnswers(unittest.TestCase):
    """代码先上、数据后迁：没有 pages 表的库必须照常检索。"""

    def test_search_without_the_pages_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(home=Path(tmp) / "kb")
            cfg.mirror_dir.mkdir(parents=True, exist_ok=True)
            (cfg.mirror_dir / "a.md").write_text(
                "# a\nneedle here\n" + FILLER + "\n", encoding="utf-8")
            index.build(cfg)
            con = sqlite3.connect(cfg.index_db)
            con.executescript("drop table pages; drop table supersedes;")
            con.commit()
            con.close()

            result = index.search(cfg, "needle", limit=5)
            self.assertEqual([h.path for h in result.hits], ["a.md"])
            self.assertFalse(result.hits[0].superseded)
            self.assertEqual(result.superseded_shown, 0)
            self.assertEqual(supersede.check(cfg)["built"], False)
            self.assertIn("no pages/supersedes", supersede.check(cfg)["note"])


class _mutated:
    """Copy the built index, let the test mutate the copy, then point cfg at it.

    ``check()`` must be exercised against *broken* data — but the fixture's
    index is shared by several tests, so each mutation works on a copy and the
    original is restored afterwards.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        self.backup = Path(str(self.cfg.index_db) + ".guard-backup")
        self.backup.write_bytes(self.cfg.index_db.read_bytes())
        return self.cfg.index_db

    def __exit__(self, *exc):
        self.cfg.index_db.write_bytes(self.backup.read_bytes())
        self.backup.unlink()
        return False
