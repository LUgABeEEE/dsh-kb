"""Adapters: each one reads its own shape and must fail loudly on a bad one."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dsh_kb import importers  # noqa: E402


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestRegistry(unittest.TestCase):
    def test_the_advertised_names_all_load(self):
        self.assertEqual(
            importers.names(),
            ["basic-memory", "jsonl", "mem0", "notion", "obsidian", "records"],
        )

    def test_unknown_format_is_rejected_with_the_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "x.json", "{}")
            with self.assertRaises(ValueError) as ctx:
                importers.load("zep", path)
            self.assertIn("records", str(ctx.exception))

    def test_missing_path_is_rejected(self):
        with self.assertRaises(ValueError):
            importers.load("records", Path("/nonexistent/nope.json"))


class TestRecords(unittest.TestCase):
    def test_envelope_and_bare_list_are_both_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envelope = write(root, "a.json", json.dumps({
                "version": 1, "kind": "records", "generator": "fixture/1",
                "items": [{"content": "one", "id": "1", "tags": ["t"], "type": "fact"}],
            }))
            bare = write(root, "b.json", json.dumps([{"content": "two"}]))
            first = importers.load("records", envelope)
            second = importers.load("records", bare)
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].content, "one")
            self.assertEqual(first[0].tags, ("t",))
            self.assertEqual(first[0].type, "fact")
            self.assertEqual(second[0].content, "two")

    def test_a_future_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "x.json", json.dumps({"version": 2, "items": []}))
            with self.assertRaises(ValueError) as ctx:
                importers.load("records", path)
            self.assertIn("version", str(ctx.exception))

    def test_empty_content_is_dropped_not_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "x.json", json.dumps(
                [{"content": "   "}, {"content": "kept"}]))
            self.assertEqual([r.content for r in importers.load("records", path)], ["kept"])


class TestJsonl(unittest.TestCase):
    def test_objects_strings_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "x.jsonl",
                         '# a header comment\n'
                         '{"content": "one", "tags": ["a"]}\n'
                         '\n'
                         '"two"\n')
            records = importers.load("jsonl", path)
            self.assertEqual([r.content for r in records], ["one", "two"])
            self.assertEqual(records[0].tags, ("a",))

    def test_a_broken_line_names_its_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), "x.jsonl", '{"content": "ok"}\n{broken\n')
            with self.assertRaises(ValueError) as ctx:
                importers.load("jsonl", path)
            self.assertIn(":2:", str(ctx.exception))


class TestMem0(unittest.TestCase):
    def test_results_envelope_and_bare_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrapped = write(root, "a.json", json.dumps({"results": [
                {"id": "m1", "memory": "remembered", "metadata": {"tags": ["x"]},
                 "updated_at": "2026-10-20T10:00:00+08:00"},
            ]}))
            bare = write(root, "b.json", json.dumps([{"memory": "second"}]))
            first = importers.load("mem0", wrapped)[0]
            self.assertEqual(first.content, "remembered")
            self.assertEqual(first.tags, ("x",))
            self.assertEqual(first.updated_at, "2026-10-20T10:00:00+08:00")
            self.assertEqual(importers.load("mem0", bare)[0].content, "second")


class TestObsidian(unittest.TestCase):
    def test_front_matter_wikilinks_and_application_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "notes/a.md",
                  "---\ntype: decision\ntags: [dsh-kb, git]\n---\n"
                  "# A note\n\nsee [[other|the other page]] here\n")
            write(root, ".obsidian/workspace.md", "# app state\n")
            records = importers.load("obsidian", root)
            self.assertEqual(len(records), 1, "the .obsidian directory is not knowledge")
            record = records[0]
            self.assertEqual(record.type, "decision")
            self.assertEqual(record.tags, ("dsh-kb", "git"))
            self.assertEqual(record.title, "A note")
            self.assertNotIn("[[", record.content)
            self.assertIn("the other page", record.content)
            self.assertNotIn("# A note", record.content, "the H1 is kept as the record title")

    def test_a_page_without_a_body_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "empty.md", "---\ntags: [x]\n---\n")
            self.assertEqual(importers.load("obsidian", root), [])


class TestNotion(unittest.TestCase):
    def test_page_ids_are_stripped_from_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "Design notes 0123456789abcdef0123456789abcdef.md", "body only\n")
            write(root, "Design notes 0123456789abcdef0123456789abcdef.csv", "a,b\n")
            records = importers.load("notion", root)
            self.assertEqual(len(records), 1, "the CSV twin is redundant")
            self.assertEqual(records[0].title, "Design notes")
            self.assertEqual(records[0].content, "body only")


class TestBasicMemory(unittest.TestCase):
    def test_observations_become_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "note.md",
                  "---\ntitle: 项目笔记\ntype: project\ntags: [proj]\n---\n"
                  "- [decision] 用 SQLite\n- [fact] 语料 1.2 GB\n")
            records = importers.load("basic-memory", root)
            self.assertEqual([r.type for r in records], ["decision", "fact"])
            self.assertEqual([r.content for r in records], ["用 SQLite", "语料 1.2 GB"])
            self.assertTrue(all(r.title == "项目笔记" for r in records))
            self.assertEqual(records[0].id, "note.md#1")

    def test_a_page_without_observations_imports_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root, "plain.md", "# Plain\n\njust a note\n")
            records = importers.load("basic-memory", root)
            self.assertEqual(len(records), 1)
            self.assertIn("just a note", records[0].content)


if __name__ == "__main__":
    unittest.main()
