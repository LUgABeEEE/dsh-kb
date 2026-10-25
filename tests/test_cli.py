"""CLI smoke tests.

These exist because a missing import in the CLI wiring (``describe_path``) was
only caught by running the command by hand.  Every command is invoked through
``main()`` here, so a broken import or an argparse mistake fails the suite.

The environment is configured entirely through ``DSH_KB_*`` variables, which is
also how a deployment with no config file would run.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb.cli import main  # noqa: E402

FIXTURE = {
    "index.md": "---\nokf_version: \"0.2\"\n---\n\n# Index\n\n* [z0](z0-public/) - notes\n",
    "log.md": "# Change log\n\n## 2026-10-12\n* Creation\n",
    "z0-public/mount.md": ("---\ntype: lesson\ntitle: 挂载失效的排查方法\n"
                           "tags: [\"挂载\", \"排查\"]\n---\n\n"
                           "先看 /proc/mounts，再 mount -a。\n\nUNIQ-CLI-ALPHA-77777\n"),
    "z1-restricted/pref.md": "---\ntype: preference\ntitle: 偏好\n---\n\n报告优先 HTML。\n",
    "z2-sensitive/med.md": "---\ntype: fact\ntitle: 敏感\n---\n\nUNIQ-CLI-CANARY-88888\n",
    "_incoming/proposal.md": "---\ntype: lesson\ntitle: 待审\n---\n\n提案内容。\n",
}

#: 推送测试里"本地改过一版"的正文。刻意与 fixture 不同长度，便于断言字节数。
LOCAL_EDIT = "---\ntype: lesson\ntitle: 待审\n---\n\n提案内容（第二版，本地改过）。\n"

#: fixture 里会进索引的页面（z2-sensitive 是 index=false 的分区）。
INDEXED_PATHS = ("index.md", "log.md", "z0-public/mount.md",
                 "z1-restricted/pref.md", "_incoming/proposal.md")


def vec_row(rid: int, path: str, *, model: str | None = "bge-m3",
            blob: bytes | None = None, line: int = 1) -> tuple:
    """One vector row; the table shape matches what the documented builder writes."""
    return (rid, path, line, f"text-{rid}", f"hash-{rid}", model,
            b"\x00" * 4096 if blob is None else blob)


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-cli-"))
        self.bundle = self.tmp / "share"
        for rel, text in FIXTURE.items():
            path = self.bundle / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.home = self.tmp / "home"
        # RAG 侧也要备份：向量库默认落在 <home>/index/ 下，显式覆盖是为了让测试
        # 绝不读到真实知识库的库文件，也不因为这台机器恰好有/没有向量而变绿变红。
        self._env_backup = {
            key: os.environ.get(key)
            for key in ("DSH_KB_HOME", "DSH_KB_SOURCE", "DSH_KB_LOCAL_ROOT",
                        "DSH_KB_RAG_VECTORS_DB", "DSH_KB_RAG_OLLAMA", "DSH_KB_RAG_ENABLED")
        }
        os.environ["DSH_KB_HOME"] = str(self.home)
        os.environ["DSH_KB_SOURCE"] = "local"
        os.environ["DSH_KB_LOCAL_ROOT"] = str(self.bundle)
        # 默认把向量库指到一个不存在的路径、并关掉 ollama 探测：测试既不该读真库，
        # 也不该因为这台机器恰好有/没有向量而变绿或变红。
        os.environ["DSH_KB_RAG_VECTORS_DB"] = str(self.tmp / "no-such-vectors.sqlite")
        os.environ["DSH_KB_RAG_OLLAMA"] = "-"

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    # ---- push fixtures --------------------------------------------------
    def _incoming(self) -> Path:
        """本地待推目录（mirror/_incoming）。"""
        return self.home / "mirror" / "_incoming"

    def _remote_incoming(self) -> Path:
        """远端 bundle 的 _incoming/ —— 也就是 push 的写入目标。"""
        return self.bundle / "_incoming"

    def _receipts_path(self) -> Path:
        return self.home / "state" / "push-log.jsonl"

    def _receipts(self) -> list[dict]:
        path = self._receipts_path()
        if not path.is_file():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _bundle_tree(self) -> dict:
        """远端 bundle 整棵树的 {相对路径: 字节} —— 证明"一个字节都没被写"。"""
        return {path.relative_to(self.bundle).as_posix(): path.read_bytes()
                for path in sorted(self.bundle.rglob("*")) if path.is_file()}

    # ---- rag fixtures ---------------------------------------------------
    def _vectors_db(self) -> Path:
        return self.home / "index" / "vectors.sqlite"

    def write_vectors(self, rows: list[tuple], path: Path | None = None) -> Path:
        """造一个最小可用的向量库；默认写进 home 并把环境指过去。

        ``path`` 用于"外部向量库"场景（home 之外的库 —— 2026-10-19 事故的形状）。
        """
        path = path or self._vectors_db()
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        try:
            con.execute(
                "create table vectors(id integer primary key, path text, startline integer,"
                " text text, hash text, model text, vec blob)"
            )
            con.executemany("insert into vectors values(?,?,?,?,?,?,?)", rows)
            con.commit()
        finally:
            con.close()
        os.environ["DSH_KB_RAG_VECTORS_DB"] = str(path)
        return path

    @staticmethod
    def _file_state(path: Path) -> tuple[str, int, int]:
        """(sha256, size, mtime_ns)：证明一个文件"一个字节都没被动过"。"""
        stat = path.stat()
        return hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_size, stat.st_mtime_ns

    def index_mirror(self) -> Path:
        """sync + index：让镜像与索引都就绪（rag-check 要对比的就是这两侧）。"""
        self.run_cli("sync", "--json")
        code, out, err = self.run_cli("index", "--json")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["pages"], len(INDEXED_PATHS))
        return self.home / "index" / "kb.sqlite"

    # ---- every command must at least run -------------------------------
    def test_status_runs_and_reports_source(self):
        code, out, _ = self.run_cli("status", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["source"]["kind"], "local")
        self.assertEqual(payload["mirror"]["files"], 0)

    def test_doctor_runs_and_returns_checks(self):
        code, out, _ = self.run_cli("doctor", "--json")
        payload = json.loads(out)
        self.assertIn("checks", payload)
        names = {c["name"] for c in payload["checks"]}
        self.assertIn("source reachable", names)
        self.assertIn("no excluded paths in mirror", names)
        # nothing failed on a healthy fixture
        self.assertEqual(code, 0, out)

    def test_sync_then_search_round_trip(self):
        code, out, err = self.run_cli("sync", "--json")
        self.assertEqual(code, 0, err)
        # index.md, log.md, z0-public/mount.md, z1-restricted/pref.md,
        # _incoming/proposal.md — and never z2-sensitive
        self.assertEqual(json.loads(out)["fetched"], 5)

        code, out, _ = self.run_cli("index", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["pages"], 5)

        code, out, _ = self.run_cli("search", "挂载", "--json", "--no-auto-refresh")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["hits"][0]["path"], "z0-public/mount.md")

    def test_sensitive_zone_is_not_searchable_by_default(self):
        self.run_cli("sync", "--json")
        self.run_cli("index", "--json")
        _, out, _ = self.run_cli("search", "UNIQ-CLI-CANARY-88888", "--json", "--no-auto-refresh")
        self.assertEqual(json.loads(out)["count"], 0)

    def test_read_returns_a_window(self):
        self.run_cli("sync", "--json")
        code, out, _ = self.run_cli("read", "index.md", "--start", "1", "--lines", "2", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["start"], 1)
        self.assertEqual(len(payload["lines"]), 2)

    def test_refresh_is_sync_plus_index(self):
        code, out, err = self.run_cli("refresh", "--json")
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(payload["sync"]["fetched"], 5)
        self.assertGreater(payload["index"]["chunks"], 0)

    def test_second_refresh_does_not_rewrite_the_index(self):
        """没变化就不该重建索引 —— 定时刷新 4 次/天不该每次都重写整库。"""
        self.run_cli("refresh", "--json")
        db = self.home / "index" / "kb.sqlite"
        before = db.stat().st_mtime_ns
        code, out, _ = self.run_cli("refresh", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["sync"]["fetched"], 0)
        self.assertEqual(db.stat().st_mtime_ns, before,
                         "an empty refresh must not rebuild the whole index")

    def test_refresh_rebuilds_when_content_changes(self):
        self.run_cli("refresh", "--json")
        db = self.home / "index" / "kb.sqlite"
        before = db.stat().st_mtime_ns
        (self.bundle / "z0-public" / "new.md").write_text("---\ntype: fact\n---\n\n新内容\n",
                                                          encoding="utf-8")
        code, out, _ = self.run_cli("refresh", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["sync"]["fetched"], 1)
        self.assertNotEqual(db.stat().st_mtime_ns, before, "changed content must rebuild the index")

    def test_push_reports_and_does_not_crash(self):
        self.run_cli("sync", "--json")
        code, out, err = self.run_cli("push")
        self.assertIn(code, (0, 1))
        self.assertIn("_incoming", out + err)

    # ---- push: no silent overwrite --------------------------------------
    def test_push_refuses_to_overwrite_a_changed_remote_file(self):
        """默认路径上同名提案不能静默覆盖 —— push 是这个工具唯一的写动作。"""
        self.run_cli("sync", "--json")
        local = self._incoming() / "proposal.md"
        remote = self._remote_incoming() / "proposal.md"
        local.write_text(LOCAL_EDIT, encoding="utf-8")
        before = remote.read_bytes()
        local_sha = hashlib.sha256(local.read_bytes()).hexdigest()
        remote_sha = hashlib.sha256(before).hexdigest()

        code, out, err = self.run_cli("push")
        text = out + err
        self.assertEqual(code, 1, text)
        # 远端逐字节未变：拒绝必须是"真的没写"
        self.assertEqual(remote.read_bytes(), before)
        # 两端 sha256 前 12 位 + 各自字节数都要打出来，人才判断得了
        self.assertIn(local_sha[:12], text)
        self.assertIn(remote_sha[:12], text)
        self.assertIn(f"({len(local.read_bytes())} B)", text)
        self.assertIn(f"({len(before)} B)", text)
        self.assertIn("--force", text)

        receipts = self._receipts()
        self.assertEqual(len(receipts), 1, "被拒绝的尝试也要记一行回执")
        self.assertEqual(receipts[0]["result"], "refused")
        self.assertTrue(receipts[0]["remoteExisted"])
        self.assertEqual(receipts[0]["sha256"], local_sha)
        self.assertEqual(receipts[0]["bytes"], len(local.read_bytes()))

    def test_push_identical_bytes_is_unchanged_and_writes_nothing(self):
        """同字节就是幂等：报 unchanged、退出码 0、不碰远端文件。"""
        self.run_cli("sync", "--json")
        remote = self._remote_incoming() / "proposal.md"
        before = remote.read_bytes()
        before_mtime = remote.stat().st_mtime_ns

        code, out, err = self.run_cli("push", "--json")
        self.assertEqual(code, 0, out + err)
        payload = json.loads(out)
        self.assertEqual(payload["result"], "unchanged")
        self.assertEqual(payload["counts"]["unchanged"], 1)
        self.assertTrue(payload["remoteExisted"])
        self.assertTrue(payload["ok"])
        self.assertEqual(remote.read_bytes(), before)
        self.assertEqual(remote.stat().st_mtime_ns, before_mtime,
                         "unchanged must not rewrite the remote file")
        self.assertEqual(self._receipts()[-1]["result"], "unchanged")

    def test_push_force_overwrites_and_records_forced(self):
        self.run_cli("sync", "--json")
        local = self._incoming() / "proposal.md"
        local.write_text(LOCAL_EDIT, encoding="utf-8")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()

        code, out, err = self.run_cli("push", "--force", "--json")
        self.assertEqual(code, 0, out + err)
        payload = json.loads(out)
        self.assertEqual(payload["result"], "forced")
        self.assertTrue(payload["remoteExisted"])
        self.assertEqual(payload["sha256"], digest)
        self.assertEqual((self._remote_incoming() / "proposal.md").read_bytes(),
                         local.read_bytes())

        receipt = self._receipts()[-1]
        self.assertEqual(receipt["result"], "forced")
        self.assertEqual(receipt["sha256"], digest)
        self.assertEqual(receipt["bytes"], len(local.read_bytes()))
        # forced 的回执要能说明"覆盖掉了什么"，不只是"写了什么"
        self.assertIn("remoteSha256", receipt)

    def test_push_expect_mismatch_refuses_and_writes_nothing(self):
        self.run_cli("sync", "--json")
        # 先把远端同名文件删掉，确保拒绝的原因只可能是 --expect
        (self._remote_incoming() / "proposal.md").unlink()
        local = self._incoming() / "proposal.md"
        local.write_text(LOCAL_EDIT, encoding="utf-8")

        code, out, err = self.run_cli("push", "--expect", "0" * 64, "--json")
        self.assertEqual(code, 1, out + err)
        payload = json.loads(out)
        self.assertEqual(payload["result"], "refused")
        self.assertEqual(payload["reason"], "expect-mismatch")
        self.assertFalse((self._remote_incoming() / "proposal.md").exists(),
                         "--expect mismatch must write nothing")
        receipt = self._receipts()[-1]
        self.assertEqual(receipt["result"], "refused")
        self.assertEqual(receipt["reason"], "expect-mismatch")

    def test_push_expect_match_pushes_normally(self):
        self.run_cli("sync", "--json")
        (self._remote_incoming() / "proposal.md").unlink()
        local = self._incoming() / "proposal.md"
        local.write_text(LOCAL_EDIT, encoding="utf-8")
        digest = hashlib.sha256(local.read_bytes()).hexdigest()

        # 大写与 sha256: 前缀都该被接受 —— 人是照着 sha256sum 复制过来的
        code, out, err = self.run_cli("push", "--expect", f"sha256:{digest.upper()}", "--json")
        self.assertEqual(code, 0, out + err)
        payload = json.loads(out)
        self.assertEqual(payload["result"], "ok")
        self.assertFalse(payload["remoteExisted"])
        self.assertEqual((self._remote_incoming() / "proposal.md").read_bytes(),
                         local.read_bytes())

    def test_push_log_lists_only_the_last_n_receipts(self):
        self.run_cli("sync", "--json")
        local = self._incoming() / "proposal.md"
        self.run_cli("push")                     # unchanged
        local.write_text(LOCAL_EDIT, encoding="utf-8")
        self.run_cli("push")                     # refused
        self.run_cli("push", "--force")          # forced
        self.assertEqual([r["result"] for r in self._receipts()],
                         ["unchanged", "refused", "forced"],
                         "every attempt gets exactly one receipt line")

        code, out, err = self.run_cli("push", "--log", "2")
        self.assertEqual(code, 0, err)
        self.assertIn("last 2 of 3", out)
        self.assertIn("forced", out)
        self.assertIn("refused", out)
        self.assertNotIn("unchanged", out)

        # 不带 N 时默认 10 条；--json 给出结构化数组
        code, out, _ = self.run_cli("push", "--log", "--json")
        self.assertEqual(code, 0)
        tail = json.loads(out)
        self.assertEqual([r["result"] for r in tail], ["unchanged", "refused", "forced"])

    def test_push_refuses_names_that_are_not_plain_file_names(self):
        """提案名必须是 _incoming/ 下的单个文件名，不许当路径用。

        回归背景（t4 验证发现）：`push ../index.md` 会让写入目标变成
        `_incoming/../index.md`，落点直接跑到 bundle 顶层；而存在性检查查的是**字面**
        路径，永远判"远端不存在" —— 于是既不比对内容也不要 --force，直接覆盖远端
        index.md，还回一句 ok，回执里的 path 也和实际落点不符。
        """
        self.run_cli("sync", "--json")
        tree_before = self._bundle_tree()
        # 本地确实存在 ../index.md 指向的那个文件 —— 所以"没被拒"就会真的写出去
        mirror_index = self.home / "mirror" / "index.md"
        self.assertTrue(mirror_index.is_file())
        mirror_index.write_bytes(b"PWNED-BY-TRAVERSAL\n")

        bad_names = ["../index.md", "a/b.md", "/abs.md", "..", ".", "",
                     "..\\index.md", "C:x.md"]
        code, out, err = self.run_cli("push", *bad_names)

        text = out + err
        self.assertEqual(code, 1, text)
        self.assertIn("plain file name", text)
        # 远端整棵树逐字节没动：既没覆盖 index.md，也没被建出 a/ 或 abs.md
        self.assertEqual(self._bundle_tree(), tree_before,
                         "越界的名字不许在远端留下任何痕迹")
        self.assertFalse((self.bundle / "a").exists())
        self.assertFalse((self.bundle / "abs.md").exists())
        self.assertFalse((self.bundle / "x.md").exists())
        # 每个坏名字一行回执：failed + 说明，且不谎报目标路径
        receipts = self._receipts()
        self.assertEqual([r["name"] for r in receipts], bad_names)
        for record in receipts:
            self.assertEqual(record["result"], "failed")
            self.assertIn("plain file name", record["error"])
            self.assertIsNone(record["path"], "没写就不该报一个目标路径")

        # 单独再跑一次缺陷点名的场景
        code, out, err = self.run_cli("push", "../index.md")
        self.assertEqual(code, 1)
        self.assertIn("../index.md", out + err)
        self.assertEqual(self._bundle_tree(), tree_before)

    def test_push_still_accepts_a_plain_name(self):
        """约束不能误伤正常用法：单个文件名照旧能推。"""
        self.run_cli("sync", "--json")
        proposal = self._incoming() / "second.md"
        proposal.write_text("---\ntype: note\n---\n\n第二份提案\n", encoding="utf-8")

        code, out, err = self.run_cli("push", "second.md")

        self.assertEqual(code, 0, out + err)
        self.assertEqual((self._remote_incoming() / "second.md").read_bytes(),
                         proposal.read_bytes())

    def test_push_keeps_going_after_a_failing_file(self):
        self.run_cli("sync", "--json")
        second = self._incoming() / "second.md"
        second.write_text("---\ntype: note\n---\n\n第二份提案\n", encoding="utf-8")

        code, out, err = self.run_cli("push", "missing.md", "second.md")
        text = out + err
        self.assertEqual(code, 1, text)
        self.assertIn("missing.md", text)
        self.assertEqual((self._remote_incoming() / "second.md").read_bytes(),
                         second.read_bytes(),
                         "one bad file must not abort the rest")
        self.assertEqual([r["result"] for r in self._receipts()], ["failed", "ok"])

    def test_push_empty_or_missing_incoming_is_still_success(self):
        """没东西可推是正常状态（调度器按天跑）：退出码 0，提示语不变。"""
        self.run_cli("sync", "--json")
        for path in self._incoming().iterdir():
            path.unlink()
        code, _, err = self.run_cli("push")
        self.assertEqual(code, 0)
        self.assertIn("_incoming/ is empty", err)

        shutil.rmtree(self._incoming())
        code, _, err = self.run_cli("push")
        self.assertEqual(code, 0)
        self.assertIn("has no _incoming/ directory", err)
        self.assertFalse(self._receipts_path().exists(), "nothing attempted, no receipts")

    def test_push_read_only_source_exits_2(self):
        self.run_cli("sync", "--json")

        class _NoWrite:
            label = "fake-readonly"

            def walk(self):
                return []

            def read(self, rel):  # pragma: no cover - must not be reached
                raise AssertionError("read-only push must not read the remote")

        with mock.patch("dsh_kb.cli.open_configured_source", return_value=_NoWrite()):
            code, _, err = self.run_cli("push")
        self.assertEqual(code, 2)
        self.assertIn("read-only", err)
        self.assertFalse(self._receipts_path().exists())

    # ---- supersede-check ------------------------------------------------
    def test_supersede_check_reports_a_pre_feature_index_as_not_built(self):
        """老索引没有 pages 表 ⇒ 报 warn 而不是 fail（功能可选，库没坏）。

        构造方式就是真实的部署窗口：索引是**旧代码**建的（没有 pages/supersedes
        两张表），而代码已经换成新的。
        """
        db = self.index_mirror()
        con = sqlite3.connect(db)
        con.executescript("drop table pages; drop table supersedes;")
        con.commit()
        con.close()
        code, out, err = self.run_cli("supersede-check")
        text = out + err
        self.assertEqual(code, 0, text)
        self.assertIn("rebuild", text, "要给出可操作的下一步")
        _, doc, _ = self.run_cli("doctor", "--json")
        names = {c["name"]: c for c in json.loads(doc)["checks"]}
        self.assertIn("supersede relations", names)
        self.assertEqual(names["supersede relations"]["status"], "warn")

    def test_supersede_check_passes_on_a_rebuilt_fixture(self):
        self.index_mirror()
        code, out, err = self.run_cli("supersede-check", "--json")
        payload = json.loads(out)
        self.assertEqual(code, 0, err)
        self.assertTrue(payload["built"])
        self.assertTrue(payload["ok"], payload)
        self.assertEqual(payload["pages"], len(INDEXED_PATHS))

    # ---- 提案闸门（R1）接在唯一的写入口上 ----
    TWIN = ('---\ntype: lesson\ntitle: "挂载失效的排查方法（补充）"\ntags: ["挂载"]\n---\n'
            '# 补充\n又踩了一次。\n')

    def _write_local_proposal(self, text, name="twin.md"):
        path = self.home / "mirror" / "_incoming" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_supersede_check_gate_blocks_an_undeclared_twin(self):
        """提案闸门：与现行页撞题却没声明 ⇒ 退出码 10 + 说清补哪一句。"""
        self.index_mirror()
        proposal = self.tmp / "twin.md"
        proposal.write_text(self.TWIN, encoding="utf-8")
        code, out, err = self.run_cli("supersede-check", "--proposal", str(proposal))
        text = out + err
        self.assertEqual(code, 10, text)
        self.assertIn("not_superseding", text)
        self.assertIn("z0-public/mount.md", text, "要说清撞了谁")

    def test_push_refuses_an_undeclared_twin_and_writes_nothing(self):
        """闸门在**任何写入之前**拦：远端一个字节都不该多。"""
        self.index_mirror()
        self._write_local_proposal(self.TWIN)
        remote_before = sorted(p.name for p in self._remote_incoming().iterdir())
        code, out, err = self.run_cli("push", "twin.md")
        text = out + err
        self.assertEqual(code, 1, text)
        self.assertIn("提案闸门拦下", text)
        self.assertIn("--allow-undeclared", text)
        self.assertEqual(sorted(p.name for p in self._remote_incoming().iterdir()),
                         remote_before, "被拦下的提案不得落盘")

    def test_push_accepts_a_declared_proposal(self):
        self.index_mirror()
        self._write_local_proposal(
            self.TWIN.replace('---\n', '---\nnot_superseding: "只讲补充步骤，不取代原文"\n', 1))
        code, out, err = self.run_cli("push", "twin.md")
        self.assertEqual(code, 0, out + err)

    def test_push_accepts_the_explicit_override(self):
        self.index_mirror()
        self._write_local_proposal(self.TWIN)
        code, out, err = self.run_cli("push", "twin.md", "--allow-undeclared")
        self.assertEqual(code, 0, out + err)

    def test_supersede_check_gate_rejects_a_missing_file(self):
        code, out, err = self.run_cli("supersede-check", "--proposal", "/nope/missing.md")
        self.assertEqual(code, 2)
        self.assertIn("找不到提案文件", out + err)

    # ---- rag-check ------------------------------------------------------
    def test_rag_check_missing_store_exits_10_with_a_build_hint(self):
        self.index_mirror()
        code, out, err = self.run_cli("rag-check")
        text = out + err
        self.assertEqual(code, 10, text)
        self.assertIn("向量库不存在", out)
        self.assertIn("向量构建器", text, "要说清怎么生成向量")
        self.assertIn("DSH_KB_RAG_VECTORS_DB", text)

    def test_rag_check_json_shape_and_dry_run_estimate(self):
        rows = [vec_row(i, path) for i, path in enumerate(INDEXED_PATHS, 1)]
        rows += [vec_row(6, "z9-gone/old.md"), vec_row(7, "z9-gone/older.md")]
        db = self.write_vectors(rows)          # 2/7 ≈ 29% 孤儿：未触发大批量保护
        self.index_mirror()
        before = hashlib.sha256(db.read_bytes()).hexdigest()

        code, out, err = self.run_cli("rag-check", "--json")

        self.assertEqual(code, 10, err)
        payload = json.loads(out)
        for key in ("ok", "chunks", "vectors", "missing", "orphan", "model_mismatch",
                    "dim_mismatch", "note", "orphan_sample", "vectors_db", "home", "model",
                    "dim", "enabled", "store_missing", "prune"):
            self.assertIn(key, payload, f"--json 必须给出 {key}")
        self.assertEqual(payload["chunks"], len(INDEXED_PATHS))
        self.assertEqual(payload["vectors"], 7)
        self.assertEqual(payload["orphan"], 2)
        self.assertEqual(payload["missing"], 0)
        self.assertEqual(payload["orphan_sample"], ["z9-gone/old.md", "z9-gone/older.md"])
        self.assertTrue(payload["note"])
        # 默认 dry-run：removed 是"待删"的预估，且数据库一个字节都没动
        prune = payload["prune"]
        self.assertEqual(
            {k: prune[k] for k in ("requested", "dry_run", "removed", "kept", "label",
                                   "refused", "reason", "external", "total", "orphan_rows")},
            {"requested": False, "dry_run": True, "removed": 2, "kept": 5,
             "label": "would remove", "refused": False, "reason": "", "external": False,
             "total": 7, "orphan_rows": 2})
        self.assertEqual(payload["home"], str(self.home))
        self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), before)

        # 人读版必须让人一眼看出是"待删"而不是"已删"
        code, out, _ = self.run_cli("rag-check")
        self.assertEqual(code, 10)
        self.assertIn("dry-run", out)
        self.assertIn("待删 2 行孤儿", out)
        self.assertIn("未改动", out)

    # ---- the accident this repair exists for -----------------------------
    def test_rag_check_prune_refuses_a_foreign_vector_store(self):
        """复现 2026-10-19 00:39 的事故：sandbox home + 外部（默认生产路径）向量库。

        当时漏设 DSH_KB_RAG_VECTORS_DB，prune 拿 sandbox 的 5 页索引当对照面，把外部
        库 5,521 行全删了。这里断言：默认拒绝（exit 11）、外部库 sha256/大小/mtime
        完全未变、人读与 --json 都给原因/比例/两条路径/解锁方式。
        """
        external_dir = Path(tempfile.mkdtemp(prefix="dsh-kb-external-"))
        self.addCleanup(shutil.rmtree, external_dir, True)
        external = external_dir / "vectors.sqlite"
        rows = [vec_row(1, "z0-public/mount.md")]
        rows += [vec_row(i, f"z9-gone/page-{i:04d}.md") for i in range(2, 60)]
        self.write_vectors(rows, path=external)
        self.index_mirror()
        before = self._file_state(external)

        code, out, err = self.run_cli("rag-check", "--prune", "--json")

        self.assertEqual(code, 11, out + err)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        prune = payload["prune"]
        self.assertTrue(prune["refused"])
        self.assertEqual(prune["label"], "refused")
        self.assertEqual(prune["reason"], "external-vectors-db")
        self.assertEqual(prune["removed"], 0, "拒绝路径不许执行任何 DELETE")
        self.assertTrue(prune["external"])
        self.assertIn(str(external), prune["note"])
        self.assertIn(str(self.home), prune["note"])
        self.assertIn("allow_external_vectors_db", prune["note"])
        self.assertEqual(self._file_state(external), before,
                         "外部向量库 sha256/大小/mtime 必须完全未变")

        # 人读版：原因、比例、两条路径、解锁方式，一样都不能少
        code, out, _ = self.run_cli("rag-check", "--prune")
        self.assertEqual(code, 11)
        for needle in ("拒绝执行", "external-vectors-db", "疑似 home 与 vectors_db",
                       str(self.home), str(external), "allow_external_vectors_db",
                       "退出码 11"):
            self.assertIn(needle, out)

    def test_rag_check_prune_force_overrides_the_bulk_guard(self):
        """大批量保护：home 内的库也会被拦；--force 是唯一解锁方式。"""
        rows = [vec_row(1, "z0-public/mount.md")]
        rows += [vec_row(i, f"z9-gone/old-{i}.md") for i in range(2, 11)]
        db = self.write_vectors(rows)          # 9/10 = 90% 孤儿
        self.index_mirror()
        before = self._file_state(db)

        code, out, _ = self.run_cli("rag-check", "--prune", "--json")
        self.assertEqual(code, 11)
        prune = json.loads(out)["prune"]
        self.assertEqual(prune["reason"], "bulk-delete")
        self.assertGreater(prune["ratio"], 0.5)
        self.assertIn("不属于同一知识库", prune["note"])
        self.assertEqual(self._file_state(db), before)

        code, out, _ = self.run_cli("rag-check", "--prune", "--force", "--json")
        self.assertEqual(code, 10, "删干净了，但索引侧还有 missing → 10")
        prune = json.loads(out)["prune"]
        self.assertFalse(prune["refused"])
        self.assertEqual(prune["label"], "removed")
        self.assertEqual(prune["removed"], 9)
        self.assertEqual(prune["bypassed"], ["bulk-delete"])
        con = sqlite3.connect(db)
        try:
            left = con.execute("select path from vectors").fetchall()
        finally:
            con.close()
        self.assertEqual(left, [("z0-public/mount.md",)])

    def test_rag_check_force_without_prune_is_a_usage_error(self):
        self.write_vectors([vec_row(1, "z0-public/mount.md")])
        self.index_mirror()
        code, out, err = self.run_cli("rag-check", "--force")
        self.assertEqual(code, 1)
        self.assertIn("--prune", err)

    def test_rag_check_prune_deletes_only_orphans(self):
        db = self.write_vectors([vec_row(1, "z0-public/mount.md"),
                                 vec_row(2, "z9-gone/old.md")])
        index_db = self.index_mirror()
        index_before = hashlib.sha256(index_db.read_bytes()).hexdigest()

        code, out, err = self.run_cli("rag-check", "--prune", "--json")

        payload = json.loads(out)
        self.assertEqual(payload["prune"]["dry_run"], False)
        self.assertEqual(payload["prune"]["removed"], 1)
        self.assertEqual(payload["prune"]["label"], "removed")
        self.assertEqual(payload["orphan"], 0, "删完要再取一次状态")
        self.assertEqual(code, 10, "prune 只治孤儿；missing 还在，所以仍是 10")

        con = sqlite3.connect(db)
        try:
            rows = con.execute("select id, path from vectors order by id").fetchall()
        finally:
            con.close()
        self.assertEqual(rows, [(1, "z0-public/mount.md")], "非孤儿行不许动")
        self.assertEqual(hashlib.sha256(index_db.read_bytes()).hexdigest(), index_before,
                         "kb.sqlite 永远只读")

        code, out, _ = self.run_cli("rag-check", "--prune", "--json")
        payload = json.loads(out)
        self.assertEqual(payload["prune"]["removed"], 0)
        self.assertEqual(payload["prune"]["label"], "no-orphans", "二次调用幂等")

        # 人读版：真删要写"已删"，不能写成"待删"
        con = sqlite3.connect(db)
        try:
            con.execute("insert into vectors values(9,'z9-gone/again.md',1,'t','h','bge-m3',?)",
                        (b"\x00" * 4096,))
            con.commit()
        finally:
            con.close()
        code, out, _ = self.run_cli("rag-check", "--prune")
        self.assertIn("已删 1 行孤儿", out)
        self.assertNotIn("待删", out)

    def test_rag_check_clean_store_is_exit_0(self):
        self.write_vectors([vec_row(i, path) for i, path in enumerate(INDEXED_PATHS, 1)])
        self.index_mirror()

        code, out, err = self.run_cli("rag-check", "--json")

        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["missing"], 0)
        self.assertEqual(payload["orphan"], 0)
        self.assertEqual(payload["model_mismatch"], 0)
        self.assertEqual(payload["dim_mismatch"], 0)
        self.assertEqual(payload["prune"]["label"], "no-orphans")

    def test_rag_check_counts_model_and_dim_mismatch(self):
        self.write_vectors([
            vec_row(1, "index.md"),
            vec_row(2, "log.md", model="text-embedding-3"),
            vec_row(3, "z0-public/mount.md", blob=b"\x00" * 10),
            (4, "z1-restricted/pref.md", 1, "t", "h4", None, b"\x00" * 4096),
        ])
        self.index_mirror()

        code, out, _ = self.run_cli("rag-check", "--json")

        payload = json.loads(out)
        self.assertEqual(payload["model_mismatch"], 2, "模型名不符 + NULL 都算")
        self.assertEqual(payload["dim_mismatch"], 1, "blob 长度不是 dim*4")
        self.assertEqual(payload["orphan"], 0)
        self.assertEqual(code, 10)
        self.assertIn("model", payload["note"])

    def test_rag_check_broken_store_exits_10_with_a_readable_note(self):
        db = self._vectors_db()
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_text("this is not a database", encoding="utf-8")
        os.environ["DSH_KB_RAG_VECTORS_DB"] = str(db)
        self.index_mirror()

        code, out, err = self.run_cli("rag-check", "--json")

        self.assertEqual(code, 10, err)
        payload = json.loads(out)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["store_missing"], "文件在，只是读不了")
        self.assertIn("unusable", payload["note"])
        self.assertIn("vectors(", payload["note"], "note 要给出期望的表结构")

        code, out, _ = self.run_cli("rag-check")
        self.assertEqual(code, 10)
        self.assertIn("unusable", out)

    # ---- doctor: RAG line -----------------------------------------------
    def test_doctor_reports_rag_absent_as_warn_not_fail(self):
        code, out, err = self.run_cli("doctor", "--json")
        checks = {c["name"]: c for c in json.loads(out)["checks"]}
        self.assertIn("rag consistency", checks)
        line = checks["rag consistency"]
        self.assertEqual(line["status"], "warn")
        self.assertIn("absent", line["detail"])
        self.assertEqual(code, 0, "没有向量只是 warn：读与检索都还正常")

    def test_doctor_rag_line_shows_chunks_vs_vectors(self):
        self.write_vectors([vec_row(1, "z0-public/mount.md")])
        self.index_mirror()

        code, out, _ = self.run_cli("doctor", "--json")

        line = {c["name"]: c for c in json.loads(out)["checks"]}["rag consistency"]
        self.assertIn(f"{len(INDEXED_PATHS)} chunks / 1 vectors", line["detail"])
        self.assertIn("missing 4", line["detail"])
        self.assertIn("orphan 0", line["detail"])
        self.assertIn("model≠bge-m3 0", line["detail"])
        self.assertIn("dim≠1024 0", line["detail"])
        self.assertEqual(line["status"], "warn", "有 missing 就该 warn")
        self.assertEqual(code, 0, "warn 不改判定")

        code, out, _ = self.run_cli("doctor")
        self.assertEqual(code, 0)
        self.assertIn("rag consistency", out)

    def test_doctor_warns_when_ollama_is_unreachable(self):
        self.write_vectors([vec_row(i, path) for i, path in enumerate(INDEXED_PATHS, 1)])
        os.environ["DSH_KB_RAG_OLLAMA"] = "http://127.0.0.1:1"      # 必然拒连
        self.index_mirror()

        code, out, _ = self.run_cli("doctor", "--json")

        line = {c["name"]: c for c in json.loads(out)["checks"]}["rag consistency"]
        self.assertEqual(line["status"], "warn")
        self.assertIn("ollama", line["detail"])
        self.assertEqual(code, 0, "ollama 不可达只是 warn：search 照常工作")

    def test_doctor_rag_check_stays_count_level(self):
        """计数级证明：doctor 对 vectors 只发 count/distinct 查询，且从不删。

        把 sqlite3.connect 换成记录语句的连接工厂 —— 一旦有人把 ragcheck.prune()
        或"select … vec …"塞进 doctor，这里立刻失败。
        """
        self.write_vectors([vec_row(i, path) for i, path in enumerate(INDEXED_PATHS, 1)])
        self.index_mirror()
        statements: list[str] = []
        real_connect = sqlite3.connect

        class Recording(sqlite3.Connection):
            def execute(self, sql, *args, **kwargs):        # noqa: D102 - 只为记录
                statements.append(sql)
                return super().execute(sql, *args, **kwargs)

        def connect(*args, **kwargs):
            kwargs["factory"] = Recording
            return real_connect(*args, **kwargs)

        with mock.patch("dsh_kb.ragcheck.sqlite3.connect", side_effect=connect):
            code, out, err = self.run_cli("doctor", "--json")

        self.assertEqual(code, 0, err)
        vector_sql = [s for s in statements if "vectors" in s.lower()]
        self.assertTrue(vector_sql, "RAG 行必须真的去查向量表")
        for sql in vector_sql:
            lowered = " ".join(sql.lower().split())
            self.assertTrue(lowered.startswith("select count(") or "distinct path" in lowered,
                            f"doctor 不该全量遍历向量表：{sql}")
        self.assertFalse([s for s in statements if "delete" in s.lower()],
                         "doctor 绝不删除任何东西")
        self.assertIn("rag consistency",
                      {c["name"] for c in json.loads(out)["checks"]})

    # ---- argument handling ---------------------------------------------
    def test_shared_flags_work_before_and_after_the_subcommand(self):
        code_a, out_a, _ = self.run_cli("--json", "status")
        code_b, out_b, _ = self.run_cli("status", "--json")
        self.assertEqual(code_a, 0)
        self.assertEqual(code_b, 0)
        self.assertEqual(json.loads(out_a)["home"], json.loads(out_b)["home"])

    def test_sync_dry_run_lists_exclusions_without_writing(self):
        code, out, _ = self.run_cli("sync", "--dry-run", "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["files"], 6)
        self.assertEqual(payload["would_mirror"], 5)
        self.assertEqual(payload["excluded_count"], 1)
        self.assertFalse(self.home.joinpath("mirror").exists())

    def test_full_flag_ignores_the_manifest(self):
        self.run_cli("sync", "--json")
        code, out, _ = self.run_cli("sync", "--full", "--json")
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["fetched"], 5)

    def test_unknown_config_key_does_not_break_anything(self):
        """配置文件里多一个键不该让 CLI 崩掉（实测：加了说明键直接 TypeError）。"""
        self.home.mkdir(parents=True, exist_ok=True)
        (self.home / "config.json").write_text(json.dumps({
            "source": "local", "local_root": str(self.bundle), "totally_unknown": 1,
        }, ensure_ascii=False), encoding="utf-8")
        code, out, err = self.run_cli("status", "--json")
        self.assertEqual(code, 0, err)
        code, out, _ = self.run_cli("doctor", "--json")
        checks = {c["name"]: c for c in json.loads(out)["checks"]}
        self.assertIn("config keys", checks)
        self.assertEqual(checks["config keys"]["status"], "warn")
        self.assertIn("totally_unknown", checks["config keys"]["detail"])

    def test_unknown_source_is_reported_not_raised(self):
        os.environ["DSH_KB_LOCAL_ROOT"] = str(self.tmp / "does-not-exist")
        code, out, err = self.run_cli("sync")
        self.assertNotEqual(code, 0)
        self.assertIn("not a directory", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
