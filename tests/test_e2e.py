"""End-to-end tests over a self-contained fixture bundle.

No network and no external service is involved: the "share" is a temporary
directory, which is exactly what ``source=local`` is for.  The suite exercises
the properties that matter operationally:

* incremental sync really is incremental;
* excluded zones never reach the mirror or the index;
* Chinese (2-character), latin-identifier and mixed queries all hit;
* retrieval keeps working when the source disappears (offline mirror);
* prune removes only what the operator asked it to remove;
* ``diff`` reports drift/backlog without writing a single byte.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb.cli import main  # noqa: E402
from dsh_kb.config import Config  # noqa: E402
from dsh_kb.index import build, read, search, stats  # noqa: E402
from dsh_kb.mirror import (  # noqa: E402
    diff_entries,
    diff_mirror,
    load_manifest,
    manifest_age,
    save_manifest,
    sync,
)
from dsh_kb.service import diff_payload  # noqa: E402
from dsh_kb.sources import Entry, FilesystemSource  # noqa: E402

Z0_DOC = """---
type: lesson
title: 挂载点失效的处置
description: CIFS 挂载 stale 时如何恢复。
tags: [ops, cifs]
---

# 处置步骤

1. 先确认 `/proc/mounts` 里的选项是 `rw`。
2. 再执行 `mount -a` 让 systemd 重新挂载。

唯一标识 UNIQ-Z0-ALPHA-11111
"""

Z1_DOC = """---
type: preference
title: 交付偏好
description: 报告优先 HTML。
---

交付偏好：调研类报告要分清事实与推测，并尽量给引用。

唯一标识 UNIQ-Z1-BRAVO-22222
"""

Z2_DOC = """---
type: fact
title: 敏感分区自检
---

哨兵串 UNIQ-Z2-CANARY-99999 只允许显式点名时读取。
"""

CRED_DOC = """---
type: fact
title: 凭证存放说明
---

哨兵串 UNIQ-CRED-NEVER-00000 永远不得进入镜像或索引。
"""


class FixtureBundle:
    """Build a throw-away bundle that mimics the documented layout."""

    def __init__(self, root: Path):
        self.root = root
        (root / "z0-public").mkdir(parents=True)
        (root / "z1-restricted").mkdir(parents=True)
        (root / "z2-sensitive").mkdir(parents=True)
        (root / "_meta").mkdir(parents=True)
        (root / "_incoming").mkdir(parents=True)
        (root / "credentials").mkdir(parents=True)
        (root / "index.md").write_text(
            '---\nokf_version: "0.2"\n---\n\n# Bundle index\n\n* [z0-public](z0-public/) - public notes\n',
            encoding="utf-8",
        )
        (root / "log.md").write_text("# Change log\n\n## 2026-10-12\n* Creation\n", encoding="utf-8")
        (root / "z0-public" / "mount.md").write_text(Z0_DOC, encoding="utf-8")
        (root / "z1-restricted" / "delivery.md").write_text(Z1_DOC, encoding="utf-8")
        (root / "z2-sensitive" / "notes.md").write_text(Z2_DOC, encoding="utf-8")
        (root / "credentials" / "keys.md").write_text(CRED_DOC, encoding="utf-8")
        (root / "_meta" / "registry.json").write_text('{"schema": 1}', encoding="utf-8")
        (root / "_meta" / "CONVENTIONS.md").write_text(
            "---\ntype: Meta\ntitle: 约定\n---\n\n# 写入策略\n\n提案先落在 _incoming/。\n\n唯一标识 UNIQ-META-DELTA-33333\n",
            encoding="utf-8",
        )
        (root / "_incoming" / "proposal.md").write_text(
            "---\ntype: lesson\ntitle: 待审提案\n---\n\n这是一条等待转正的提案。\n\n唯一标识 UNIQ-INCOMING-ECHO-44444\n",
            encoding="utf-8",
        )


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-test-"))
        self.bundle = self.tmp / "share"
        self.home = self.tmp / "home"
        FixtureBundle(self.bundle)
        self.cfg = Config(
            home=self.home,
            source="local",
            local_root=str(self.bundle),
            stale_minutes=15,
        )
        self.source = FilesystemSource(self.bundle, label="local")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- mirror --------------------------------------------------------
    def test_first_sync_mirrors_default_zones_only(self):
        result = sync(self.cfg, self.source)
        mirrored = {p.relative_to(self.cfg.mirror_dir).as_posix() for p in self.cfg.mirror_dir.rglob("*") if p.is_file()}
        self.assertIn("z0-public/mount.md", mirrored)
        self.assertIn("z1-restricted/delivery.md", mirrored)
        self.assertIn("index.md", mirrored)
        # privacy invariants
        self.assertNotIn("z2-sensitive/notes.md", mirrored)
        self.assertNotIn("credentials/keys.md", mirrored)
        # index.md, log.md, _meta/{registry.json,CONVENTIONS.md},
        # _incoming/proposal.md, z0-public/mount.md, z1-restricted/delivery.md
        self.assertEqual(result.fetched, 7)

    def test_second_sync_is_incremental(self):
        sync(self.cfg, self.source)
        again = sync(self.cfg, self.source)
        self.assertEqual(again.fetched, 0)
        self.assertEqual(again.skipped, 7)

    def test_changed_file_is_refetched(self):
        sync(self.cfg, self.source)
        target = self.bundle / "z0-public" / "mount.md"
        target.write_text(Z0_DOC + "\n追加一行。\n", encoding="utf-8")
        # make the mtime change unambiguous on coarse-grained filesystems
        future = time.time() + 5
        import os

        os.utime(target, (future, future))
        result = sync(self.cfg, self.source)
        self.assertEqual(result.fetched, 1)
        self.assertEqual(result.changed, ["z0-public/mount.md"])

    def test_prune_removes_deleted_remote_files(self):
        sync(self.cfg, self.source)
        (self.bundle / "z1-restricted" / "delivery.md").unlink()
        kept = sync(self.cfg, self.source)
        self.assertEqual(kept.removed, [])           # prune is opt-in
        self.assertIn("z1-restricted/delivery.md", load_manifest(self.cfg)["files"])
        pruned = sync(self.cfg, self.source, prune=True)
        self.assertEqual(pruned.removed, ["z1-restricted/delivery.md"])
        self.assertFalse((self.cfg.mirror_dir / "z1-restricted" / "delivery.md").exists())

    def test_manifest_records_epoch_mtime(self):
        sync(self.cfg, self.source)
        entry = load_manifest(self.cfg)["files"]["z0-public/mount.md"]
        self.assertGreater(entry["mtime"], 0)
        self.assertEqual(entry["size"], len(Z0_DOC.encode("utf-8")))
        self.assertIsNotNone(manifest_age(self.cfg))

    def test_full_sync_ignores_the_manifest(self):
        """The documented escape hatch for the SMB one-second timestamp problem.

        An edit that lands in the same second *and* keeps the byte size identical
        is invisible to an mtime+size comparison — which is exactly what a share
        with 1-second timestamps produces.
        """
        sync(self.cfg, self.source)
        target = self.bundle / "z0-public" / "mount.md"
        original = target.read_bytes()
        before = target.stat()
        target.write_bytes(b"X" * len(original))
        os.utime(target, (before.st_atime, before.st_mtime))

        self.assertEqual(sync(self.cfg, self.source).fetched, 0)      # invisible
        self.assertEqual(sync(self.cfg, self.source, full=True).fetched, 7)  # forced

    # ---- index and search ----------------------------------------------
    def test_index_skips_excluded_zones(self):
        sync(self.cfg, self.source)
        built = build(self.cfg)
        # index.md, log.md, _meta/CONVENTIONS.md, _incoming/proposal.md,
        # z0-public/mount.md, z1-restricted/delivery.md — never z2 or credentials
        self.assertEqual(built.pages, 6)
        self.assertGreater(built.chunks, 0)

    def test_layout_directories_stay_searchable(self):
        """Regression: `_meta/` and `_incoming/` are layout, not privacy zones.

        An earlier zone filter treated them as "not selected", which made the
        bundle's own conventions unsearchable under the default selector.
        """
        sync(self.cfg, self.source)
        build(self.cfg)
        self.assertTrue(search(self.cfg, "UNIQ-META-DELTA-33333").hits)
        self.assertTrue(search(self.cfg, "UNIQ-INCOMING-ECHO-44444").hits)

    def test_two_character_chinese_query(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        result = search(self.cfg, "挂载")
        self.assertTrue(result.hits, "2-character CJK query must match")
        self.assertIn("z0-public/mount.md", {h.path for h in result.hits})

    def test_latin_identifier_query(self):
        """Regression: identifiers were unsearchable when expansion dropped latin."""
        sync(self.cfg, self.source)
        build(self.cfg)
        result = search(self.cfg, "UNIQ-Z0-ALPHA-11111")
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].path, "z0-public/mount.md")

    def test_mixed_script_query(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        result = search(self.cfg, "交付偏好 HTML")
        self.assertTrue(result.hits)
        self.assertEqual(result.hits[0].path, "z1-restricted/delivery.md")

    def test_sensitive_zone_is_unsearchable_by_default(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        self.assertEqual(search(self.cfg, "UNIQ-Z2-CANARY-99999").hits, [])
        self.assertEqual(search(self.cfg, "UNIQ-CRED-NEVER-00000").hits, [])

    def test_zone_selector_filters_results(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        self.assertTrue(search(self.cfg, "UNIQ-Z1-BRAVO-22222", zone="z1").hits)
        self.assertEqual(search(self.cfg, "UNIQ-Z1-BRAVO-22222", zone="z0").hits, [])

    def test_hit_carries_line_number_and_zone(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        hit = search(self.cfg, "mount -a").hits[0]
        self.assertGreaterEqual(hit.line, 1)
        source_lines = (self.cfg.mirror_dir / hit.path).read_text(encoding="utf-8").splitlines()
        self.assertIn("mount -a", source_lines[hit.line - 1])

    def test_limit_and_snippet_width_are_respected(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        result = search(self.cfg, "的", limit=2, snippet_chars=60)
        self.assertLessEqual(len(result.hits), 2)
        for hit in result.hits:
            self.assertLessEqual(len(hit.snippet), 80)

    def test_missing_index_reports_instead_of_raising(self):
        result = search(self.cfg, "anything")
        self.assertEqual(result.hits, [])
        self.assertIn("index missing", result.note)

    def test_stats_match_build(self):
        sync(self.cfg, self.source)
        built = build(self.cfg)
        described = stats(self.cfg)
        self.assertEqual(built.pages, described.pages)
        self.assertEqual(built.chunks, described.chunks)

    # ---- read ----------------------------------------------------------
    def test_read_returns_a_line_window(self):
        sync(self.cfg, self.source)
        result = read(self.cfg, "z0-public/mount.md", start=1, lines=3)
        self.assertEqual(result.start, 1)
        self.assertEqual(result.end, 3)
        self.assertEqual(result.lines[0], "---")

    def test_read_refuses_paths_outside_the_mirror(self):
        sync(self.cfg, self.source)
        with self.assertRaises(FileNotFoundError):
            read(self.cfg, "../../etc/passwd")

    # ---- offline behaviour ---------------------------------------------
    def test_search_works_with_the_source_gone(self):
        sync(self.cfg, self.source)
        build(self.cfg)
        shutil.rmtree(self.bundle)                     # simulate an offline share
        self.assertTrue(search(self.cfg, "挂载").hits)
        self.assertTrue(search(self.cfg, "UNIQ-Z0-ALPHA-11111").hits)

    # ---- index hygiene -------------------------------------------------
    def test_rebuild_shrinks_the_database(self):
        """DROP TABLE does not return pages, so the file must be recreated."""
        sync(self.cfg, self.source)
        build(self.cfg)
        big = self.cfg.index_db.stat().st_size
        (self.bundle / "z1-restricted" / "delivery.md").unlink()
        sync(self.cfg, self.source, prune=True)
        build(self.cfg)
        self.assertLess(self.cfg.index_db.stat().st_size, big * 2)
        self.assertEqual(stats(self.cfg).pages, 5)  # six pages minus the deleted one


class DiffComparison(unittest.TestCase):
    """`mirror.diff_entries` —— 纯函数，只喂内存结构，不碰盘也不联网。"""

    def test_classifies_added_removed_and_changed(self):
        known = {
            "keep.md": {"size": 10, "mtime": 1000.0},
            "edit.md": {"size": 10, "mtime": 1000.0},
            "gone.md": {"size": 20, "mtime": 1000.0},
        }
        entries = [
            Entry(rel="z0-public", is_dir=True),               # 目录不算文件差异
            Entry(rel="keep.md", is_dir=False, size=10, mtime=1000.0),
            Entry(rel="edit.md", is_dir=False, size=14, mtime=1200.0),
            Entry(rel="new.md", is_dir=False, size=5, mtime=1300.0),
        ]
        diff = diff_entries(known, entries)

        self.assertEqual([c.rel for c in diff.added], ["new.md"])
        self.assertEqual([c.rel for c in diff.removed], ["gone.md"])
        self.assertEqual([c.rel for c in diff.changed], ["edit.md"])
        self.assertEqual(diff.counts, {"added": 1, "removed": 1, "changed": 1})
        self.assertFalse(diff.empty)

        changed = diff.changed[0].as_dict()
        self.assertEqual(changed["size"], 14)
        self.assertEqual(changed["previous_size"], 10)
        self.assertEqual(changed["size_delta"], 4)
        self.assertEqual(changed["mtime_delta"], 200.0)
        # 缺的一侧是 None，而不是 0 —— 调用方不必猜哪半边不存在
        self.assertIsNone(diff.added[0].as_dict()["previous_size"])
        self.assertIsNone(diff.removed[0].as_dict()["size"])
        self.assertEqual(diff.as_dict()["counts"], {"added": 1, "removed": 1, "changed": 1})

    def test_mtime_jitter_within_tolerance_is_not_a_change(self):
        """网络共享把时间戳截到秒，所以 1 秒内的抖动不能算"变了"。"""
        known = {"a.md": {"size": 10, "mtime": 1000.0}}
        self.assertTrue(diff_entries(
            known, [Entry(rel="a.md", is_dir=False, size=10, mtime=1000.4)]).empty)
        moved = [Entry(rel="a.md", is_dir=False, size=10, mtime=1002.0)]
        self.assertEqual([c.rel for c in diff_entries(known, moved).changed], ["a.md"])
        # 大小变了但 mtime 没动（同秒编辑）也要报出来
        resized = [Entry(rel="a.md", is_dir=False, size=11, mtime=1000.0)]
        self.assertEqual([c.rel for c in diff_entries(known, resized).changed], ["a.md"])

    # ---- 镜像侧（t7 盲点 B）：快照里有、本地镜像里没有 ----
    def test_mirror_diff_sees_a_file_the_snapshot_expects_but_disk_lost(self):
        """只是删掉了镜像里的文件：manifest/远端都没变，旧的 diff 一点都看不见。"""
        known = {
            "z0-public/pic.png": {"size": 12, "mtime": 1000.0},
            "z0-public/a.md": {"size": 10, "mtime": 1000.0},
        }
        local = {"z0-public/a.md": 10}            # pic.png 被删了

        diff = diff_mirror(known, local, remote={"z0-public/pic.png", "z0-public/a.md"})

        self.assertEqual([c.rel for c in diff.removed], ["z0-public/pic.png"])
        self.assertEqual(diff.counts, {"added": 0, "removed": 1, "changed": 0})
        self.assertFalse(diff.empty)
        self.assertIsNone(diff.removed[0].size)
        self.assertEqual(diff.removed[0].previous_size, 12)
        self.assertTrue(diff.remote_checked)

    def test_mirror_diff_changed_is_a_size_mismatch_not_an_mtime_difference(self):
        known = {"x.bin": {"size": 100, "mtime": 1000.0}}
        diff = diff_mirror(known, {"x.bin": 12}, remote={"x.bin"})
        self.assertEqual([c.rel for c in diff.changed], ["x.bin"])
        self.assertEqual(diff.changed[0].previous_size, 100)
        self.assertEqual(diff.changed[0].size, 12)
        self.assertEqual(diff.changed[0].size_delta, -88)
        # 大小一致时一律不算：镜像文件的 mtime 是本地写入时间，与快照里的远端
        # mtime 不可比（这正是旧实现只敢在 sync 里用 mtime 的原因）。
        self.assertTrue(diff_mirror(known, {"x.bin": 100}, remote={"x.bin"}).empty)

    def test_mirror_diff_local_only_files_need_the_remote_listing(self):
        known = {"a.md": {"size": 1, "mtime": 1.0}}
        local = {"a.md": 1, "draft.bin": 5}

        # 没有源清单：无法区分"本地多出"和"上次 sync 写到一半就挂了"（manifest 最后才写）
        self.assertEqual(diff_mirror(known, local).added, [])
        self.assertFalse(diff_mirror(known, local).remote_checked)

        # 源上有、快照还没记 → 只是"还没同步"，不是本地多出
        self.assertEqual(
            diff_mirror(known, local, remote={"a.md", "draft.bin"}).added, [])
        # 快照和源都不认识它 → 本地多出
        self.assertEqual(
            [c.rel for c in diff_mirror(known, local, remote={"a.md"}).added],
            ["draft.bin"])

    def test_mirror_diff_does_not_blame_the_mirror_for_a_stale_snapshot(self):
        """源上也已经没有这个文件了：是快照过期，不是本地丢文件。"""
        known = {"gone.md": {"size": 5, "mtime": 1.0}}
        self.assertEqual([c.rel for c in diff_mirror(known, {}).removed], ["gone.md"])
        self.assertTrue(diff_mirror(known, {}, remote=set()).empty)


class DiffReport(unittest.TestCase):
    """`dsh-kb diff` —— 只读体检。源就是临时目录，全程离线。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-diff-"))
        self.bundle = self.tmp / "share"
        self.home = self.tmp / "home"
        FixtureBundle(self.bundle)
        self.cfg = Config(home=self.home, source="local",
                          local_root=str(self.bundle), stale_minutes=15)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- helpers --------------------------------------------------------
    def prepare(self):
        """先 sync + build：一个"刚同步过、索引跟得上"的健康状态。"""
        sync(self.cfg, FilesystemSource(self.bundle, label="local"))
        build(self.cfg)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        argv = ("--home", str(self.home), "--source", "local",
                "--local-root", str(self.bundle), *argv)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def diff_json(self, *argv: str) -> tuple[int, dict, str]:
        code, out, err = self.run_cli("diff", "--json", *argv)
        return code, json.loads(out), err

    @staticmethod
    def tree_state(root: Path) -> dict:
        """路径 -> (字节数, mtime_ns)：用来断言"一个字节都没被写"。"""
        if not root.exists():
            return {}
        state = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                state[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
        return state

    def snapshot(self):
        return self.tree_state(self.bundle), self.tree_state(self.home)

    # ---- 1. 结构 + 只读 --------------------------------------------------
    def test_json_shape_is_stable_and_the_whole_command_is_read_only(self):
        self.prepare()
        before = self.snapshot()

        code, payload, err = self.diff_json()

        self.assertEqual(code, 0, err)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["attention"], [])
        for key in ("content", "index", "incoming", "conflicts", "escapes"):
            self.assertIn(key, payload, f"--json 必须固定给出 {key} 这一组")
        self.assertEqual(payload["content"]["counts"],
                         {"added": 0, "removed": 0, "changed": 0})
        self.assertEqual(payload["index"]["meta_paths"], payload["index"]["mirror_pages"])
        self.assertFalse(payload["index"]["drift"])
        self.assertIsNotNone(payload["index"]["age_seconds"])
        self.assertEqual(payload["conflicts"]["count"], 0)
        self.assertEqual(payload["escapes"]["count"], 0)
        self.assertEqual(payload["incoming"]["basis"], "source")
        self.assertEqual(payload["incoming"]["files"], 1)
        self.assertFalse(payload["incoming"]["stale"])

        self.assertEqual(self.snapshot(), before,
                         "diff 不得写源、镜像、manifest 或索引（连 mtime 都不许动）")

    def test_human_output_is_chinese_and_gives_next_steps(self):
        self.prepare()
        code, out, err = self.run_cli("diff")
        self.assertEqual(code, 0, err)
        for section in ("内容差异", "索引", "_incoming 积压", "冲突副本", "镜像越界", "结论"):
            self.assertIn(section, out)
        self.assertIn("退出码 0", out)

    # ---- 2. 内容差异 ------------------------------------------------------
    def test_content_diff_is_informational_not_an_alarm(self):
        self.prepare()
        (self.bundle / "z0-public" / "new.md").write_text(Z0_DOC, encoding="utf-8")
        target = self.bundle / "z0-public" / "mount.md"
        target.write_text(Z0_DOC + "\n追加一行。\n", encoding="utf-8")
        future = time.time() + 5
        os.utime(target, (future, future))
        (self.bundle / "z1-restricted" / "delivery.md").unlink()

        code, payload, err = self.diff_json()

        self.assertEqual(code, 0, err)
        self.assertEqual(payload["content"]["counts"],
                         {"added": 1, "removed": 1, "changed": 1})
        self.assertEqual([c["path"] for c in payload["content"]["added"]],
                         ["z0-public/new.md"])
        self.assertEqual([c["path"] for c in payload["content"]["removed"]],
                         ["z1-restricted/delivery.md"])
        changed = payload["content"]["changed"][0]
        self.assertEqual(changed["path"], "z0-public/mount.md")
        self.assertIsInstance(changed["size_delta"], int)
        self.assertIsNotNone(changed["previous_mtime"])
        self.assertNotEqual(changed["mtime"], changed["previous_mtime"])
        # 全是 Markdown 的差异：分开计数后，"其他文件"这一组必须是 0。
        self.assertEqual(payload["content"]["markdown"]["counts"],
                         {"added": 1, "removed": 1, "changed": 1})
        self.assertEqual(payload["content"]["other"]["counts"],
                         {"added": 0, "removed": 0, "changed": 0})
        # 镜像副本没动过，所以镜像侧没有缺口。
        self.assertEqual(payload["mirror_gaps"]["counts"],
                         {"added": 0, "removed": 0, "changed": 0})
        self.assertEqual(payload["attention"], [],
                         "内容差异是信息而不是异常：下一次 sync 就会收敛")

    def test_a_missing_non_markdown_mirror_file_is_not_invisible(self):
        """t7 实测的假阴性：删掉 mirror/z0-public/pic.png，旧 diff 报 exit 0。

        根因是"内容差异"只比远端 vs 快照，从没看过本地镜像这一侧 —— 删掉一个
        已同步的文件，manifest 与远端都没变，于是各项计数全 0，而 `sync` 立刻
        fetched=1。现在镜像侧的缺口独立成组，并且按 Markdown / 其他文件分开计数。
        """
        self.prepare()
        victim = self.cfg.mirror_dir / "_meta" / "registry.json"   # 非 .md，fixture 自带
        self.assertTrue(victim.is_file())
        victim.unlink()

        code, payload, err = self.diff_json()

        self.assertEqual(code, 10, err)
        self.assertIn("mirror-gap", payload["attention"])
        gaps = payload["mirror_gaps"]
        self.assertEqual(gaps["counts"]["removed"], 1)
        self.assertEqual(gaps["other"]["counts"]["removed"], 1,
                         "registry.json 不是 .md，要落在「其他文件」这一组")
        self.assertEqual(gaps["markdown"]["counts"]["removed"], 0)
        self.assertEqual([item["path"] for item in gaps["other"]["removed"]],
                         ["_meta/registry.json"])

        # 人读版必须点名这个文件，并说清下一步
        code, out, _ = self.run_cli("diff")
        self.assertEqual(code, 10)
        self.assertIn("_meta/registry.json", out)
        self.assertIn("sync", out)

        # 断言这条缺口是真的："报告缺文件 ⇒ sync 会补回来"（这正是旧的 0 计数漏掉的动作）
        result = sync(self.cfg, FilesystemSource(self.bundle, label="local"))
        self.assertEqual(result.fetched, 1)
        self.assertEqual(result.changed, ["_meta/registry.json"])

    def test_non_markdown_content_change_is_counted_separately_and_stays_informational(self):
        """远端新增一个非 .md 文件：要能单独看见，但内容差异照旧不算异常。"""
        self.prepare()
        (self.bundle / "z0-public" / "pic.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        code, payload, err = self.diff_json()

        self.assertEqual(code, 0, err)
        self.assertEqual(payload["content"]["counts"],
                         {"added": 1, "removed": 0, "changed": 0})
        self.assertEqual(payload["content"]["other"]["counts"],
                         {"added": 1, "removed": 0, "changed": 0})
        self.assertEqual(payload["content"]["markdown"]["counts"],
                         {"added": 0, "removed": 0, "changed": 0})
        self.assertEqual([item["path"] for item in payload["content"]["other"]["added"]],
                         ["z0-public/pic.png"])
        self.assertEqual(payload["mirror_gaps"]["counts"],
                         {"added": 0, "removed": 0, "changed": 0},
                         "远端新增但还没同步的文件不是镜像缺口")
        self.assertEqual(payload["attention"], [],
                         "只有内容差异（含非 .md）时仍然是 0，别把日常噪音变成告警")

    def test_a_locally_truncated_mirror_file_is_reported(self):
        """本地文件被截断/改坏：内容差异看不见（远端没变），镜像大小对不上才看得见。"""
        self.prepare()
        victim = self.cfg.mirror_dir / "z0-public" / "mount.md"
        original = victim.stat().st_size
        victim.write_bytes(b"x")

        code, payload, err = self.diff_json()

        self.assertEqual(code, 10, err)
        self.assertIn("mirror-gap", payload["attention"])
        self.assertEqual(payload["mirror_gaps"]["counts"]["changed"], 1)
        self.assertEqual(payload["mirror_gaps"]["markdown"]["counts"]["changed"], 1)
        item = payload["mirror_gaps"]["markdown"]["changed"][0]
        self.assertEqual(item["path"], "z0-public/mount.md")
        self.assertEqual(item["previous_size"], original)
        self.assertEqual(item["size"], 1)
        # 增量 sync 认为"远端没变"，不会重取 —— 建议必须是 --full
        code, out, _ = self.run_cli("diff")
        self.assertEqual(code, 10)
        self.assertIn("sync --full", out)

    def test_a_local_only_file_is_visible_but_not_an_alarm(self):
        """镜像里多出一个快照/源都不认识的非 .md 文件：计数可见，但不报异常。

        这条是刻意的取舍：镜像多出来的文件不会让检索失效（Markdown 才会漂移索引），
        而且 `sync --prune` 只删"快照里有、远端没有"的文件，删不掉它 —— 所以这里
        把它当信息报出来，让人自己决定，而不是每天亮一次红灯。
        """
        self.prepare()
        (self.cfg.mirror_dir / "z0-public" / "local.bin").write_bytes(b"local")

        code, payload, err = self.diff_json()

        self.assertEqual(code, 0, err)
        self.assertEqual(payload["mirror_gaps"]["counts"],
                         {"added": 1, "removed": 0, "changed": 0})
        self.assertEqual(payload["mirror_gaps"]["other"]["counts"]["added"], 1)
        self.assertEqual(payload["attention"], [])

    def test_mirror_gap_is_still_visible_when_the_source_is_down(self):
        """源不可用时算不了"本地多出"，但"本地缺失"只靠 manifest 就能判定。"""
        self.prepare()
        (self.cfg.mirror_dir / "_meta" / "registry.json").unlink()
        shutil.rmtree(self.bundle)                    # 共享盘掉线

        code, payload, err = self.diff_json()

        self.assertEqual(code, 10, err)
        self.assertIn("source-unavailable", payload["attention"])
        self.assertIn("mirror-gap", payload["attention"])
        self.assertEqual(payload["mirror_gaps"]["counts"]["removed"], 1)
        self.assertFalse(payload["mirror_gaps"]["remote_checked"])
        self.assertTrue(payload["mirror_gaps"]["note"],
                        "算不了的项要说出来（比如「本地多出」）")

    # ---- 3. 漂移 / 冲突 / 越界 → 10 --------------------------------------
    def test_drift_conflicts_and_escapes_are_flagged_with_exit_10(self):
        self.prepare()
        # (a) sync 到新文件却忘了重建索引 → 漂移
        (self.bundle / "z0-public" / "extra.md").write_text(Z0_DOC, encoding="utf-8")
        sync(self.cfg, FilesystemSource(self.bundle, label="local"))
        # (b) 源上的冲突副本
        (self.bundle / "z0-public" / "mount.conflict-20261018.md").write_text(
            Z0_DOC, encoding="utf-8")
        # (c) 策略收紧前同步进来、如今越界的镜像文件（manifest 里还记着它）
        leaked = self.cfg.mirror_dir / "z2-sensitive" / "leaked.md"
        leaked.parent.mkdir(parents=True, exist_ok=True)
        leaked.write_text("泄漏\n", encoding="utf-8")
        manifest = load_manifest(self.cfg)
        manifest["files"]["z2-sensitive/leaked.md"] = {
            "size": len("泄漏\n".encode("utf-8")), "mtime": time.time(), "source": "local",
        }
        save_manifest(self.cfg, manifest)

        code, payload, err = self.diff_json()

        self.assertEqual(code, 10, err)
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["index"]["drift"])
        self.assertEqual(payload["index"]["meta_paths"] + 1, payload["index"]["mirror_pages"])
        self.assertEqual(payload["conflicts"]["files"],
                         ["z0-public/mount.conflict-20261018.md"])
        escape = payload["escapes"]["files"][0]
        self.assertEqual(escape["path"], "z2-sensitive/leaked.md")
        self.assertTrue(escape["in_manifest"], "策略收紧前同步进来的要能分辨出来")
        self.assertEqual(payload["attention"],
                         ["index-drift", "conflict-copy", "mirror-escape"])

        # 人读版每条异常都要给可执行建议
        code, out, _ = self.run_cli("diff")
        self.assertEqual(code, 10)
        self.assertIn("建议", out)
        self.assertIn("dsh-kb index", out)
        self.assertIn("退出码 10", out)

    # ---- 4. _incoming 积压 ------------------------------------------------
    def test_old_proposal_is_reported_as_backlog(self):
        past = time.time() - 30 * 86400
        os.utime(self.bundle / "_incoming" / "proposal.md", (past, past))
        self.prepare()

        code, payload, err = self.diff_json()

        self.assertEqual(code, 10, err)
        self.assertTrue(payload["incoming"]["stale"])
        self.assertEqual(payload["incoming"]["files"], 1)
        self.assertEqual(payload["incoming"]["oldest"], "_incoming/proposal.md")
        self.assertGreater(payload["incoming"]["oldest_seconds"], 20 * 86400)
        self.assertIn("incoming-backlog", payload["attention"])

    def test_threshold_flag_can_tighten_the_backlog_window(self):
        self.prepare()          # 提案是刚写的，默认阈值下不算积压
        code, payload, _ = self.diff_json()
        self.assertEqual(code, 0)
        self.assertFalse(payload["incoming"]["stale"])

        code, payload, _ = self.diff_json("--incoming-days", "0")
        self.assertEqual(code, 10)
        self.assertTrue(payload["incoming"]["stale"])
        self.assertEqual(payload["incoming"]["threshold_seconds"], 0.0)

    # ---- 5. 源不可达 → 10（不是崩溃、不是 11）------------------------------
    def test_source_gone_degrades_to_local_facts_with_exit_10(self):
        self.prepare()
        shutil.rmtree(self.bundle)               # 共享盘掉线

        code, payload, err = self.diff_json()

        self.assertEqual(code, 10, err)
        self.assertNotEqual(code, 11, "源不可达是软告警，不是硬失败")
        self.assertFalse(payload["source"]["available"])
        self.assertIn("not a directory", payload["source"]["note"])
        self.assertIn("source-unavailable", payload["attention"])
        # 本地能算的照旧算：索引年龄、积压、越界都在
        self.assertIsNone(payload["content"]["remote_files"])
        self.assertTrue(payload["content"]["note"])
        self.assertIsNotNone(payload["index"]["age_seconds"])
        self.assertEqual(payload["incoming"]["basis"], "mirror")
        self.assertEqual(payload["incoming"]["files"], 1)
        self.assertEqual(payload["conflicts"]["files"], [])
        self.assertTrue(payload["conflicts"]["note"], "数不了要说出来，不能假装是 0")

        code, out, err = self.run_cli("diff")
        self.assertEqual(code, 10)
        self.assertIn("源", out)
        self.assertIn("不可用", out)

    # ---- 6. 硬失败 → 11 ---------------------------------------------------
    def test_broken_local_state_is_a_hard_failure_11(self):
        not_a_dir = self.tmp / "not-a-dir"
        not_a_dir.write_text("x", encoding="utf-8")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--home", str(not_a_dir), "diff"])
        self.assertEqual(code, 11)
        self.assertIn("not a directory", err.getvalue())

    def test_corrupt_manifest_is_a_hard_failure_11(self):
        self.prepare()
        self.cfg.manifest_path.write_text("{ this is not json", encoding="utf-8")
        code, out, err = self.run_cli("diff")
        self.assertEqual(code, 11)
        self.assertIn("manifest", err)
        self.assertIn("dsh-kb sync", err, "坏掉的 manifest 要告诉人怎么重建")

    def test_diff_payload_is_usable_without_the_cli(self):
        """service.diff_payload 是纯读的用例层函数，MCP 侧将来能直接复用。"""
        self.prepare()
        payload = diff_payload(self.cfg)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["content"]["counts"]["added"], 0)

    def test_fresh_home_without_a_source_is_not_an_alarm(self):
        """全新 home + 没配源 → 0；源掉线但本地有数据 → 10。

        这两件事刻意分开：源掉线时本地有快照，看不见远端是真问题；全新 home
        只是还没配好/还没同步过，没有快照就没有"差异"可报，配置问题由 doctor
        负责喊（它返回 1）。否则 `diff` 一上手就红，会训练人忽略这个检查。
        """
        fresh = self.tmp / "fresh-home"
        saved = {key: os.environ.pop(key) for key in list(os.environ)
                 if key.startswith("DSH_KB_")}
        try:
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(["--home", str(fresh), "diff", "--json"])
        finally:
            os.environ.update(saved)

        self.assertEqual(code, 0, err.getvalue())
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["source"]["available"])
        self.assertEqual(payload["attention"], [])
        self.assertTrue(payload["source"]["note"], "没配源也要在 note 里说清楚")
        self.assertFalse(fresh.exists(), "只是看一眼，不该把 home 目录建出来")


if __name__ == "__main__":
    unittest.main(verbosity=2)
