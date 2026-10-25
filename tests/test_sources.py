"""Tests for transports: mount inspection, filesystem sources and SMB.

Nothing here needs a real share.  The SMB tests drive a **stub ``smbclient``**
that speaks the exact output format of the real binary (captured from a live
server), so the parser is exercised against reality rather than an assumption —
an earlier version of it guessed the column order and was wrong.
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb.mounts import (  # noqa: E402
    decode_mount_field,
    describe_path,
    is_unc,
    mount_for,
    parse_mounts,
    stale_hint,
)
from dsh_kb.sources import (  # noqa: E402
    FilesystemSource,
    SmbSource,
    SourceReadOnly,
    SourceUnavailable,
    normalise_root,
)

#: A realistic mount table: a local root, a read-only CIFS view of a share and a
#: nested writable sub-share.
MOUNTS = textwrap.dedent(
    """\
    /dev/sda2 / ext4 rw,relatime 0 0
    //192.0.2.10/kb /mnt/kb cifs rw,relatime,vers=3.0,username=me 0 0
    //192.0.2.10/kb/outbox /mnt/kb-out cifs rw,relatime,vers=3.0 0 0
    //192.0.2.10/archive /mnt/archive cifs ro,relatime,vers=3.0 0 0
    192.0.2.20:/export /mnt/nfs nfs4 rw,relatime 0 0
    /dev/sdb1 /mnt/with\\040space ext4 rw,relatime 0 0
    """
)

#: Exact shape of `smbclient ... -c "recurse ON; ls"`, captured from a live NAS.
SMB_LISTING = textwrap.dedent(
    """\
      .                                   D        0  Mon Oct 12 22:23:52 2026
      ..                                  D        0  Mon Oct 12 22:23:51 2026
      z0-public                           D        0  Mon Oct 12 22:23:51 2026
      index.md                            N      980  Mon Oct 12 22:23:51 2026
      log.md                              N      336  Mon Oct 12 22:23:52 2026

    \\知识库\\z0-public
      .                                   D        0  Mon Oct 12 22:23:51 2026
      ..                                  D        0  Mon Oct 12 22:23:52 2026
      mount stale.md                      N     1234  Mon Oct 12 22:23:52 2026

    \\知识库\\z0-public\\deep
      .                                   D        0  Mon Oct 12 22:23:52 2026
      note.md                             N       10  Mon Oct 12 22:23:52 2026

    \\t\\t7772088952 blocks of size 1024. 5891711316 blocks available
    """
)


class TestMountTable(unittest.TestCase):
    def test_octal_escapes_are_decoded(self):
        self.assertEqual(decode_mount_field(r"/mnt/with\040space"), "/mnt/with space")
        self.assertEqual(decode_mount_field(r"a\134b"), "a\\b")
        self.assertEqual(decode_mount_field("plain"), "plain")

    def test_parse_reads_all_rows(self):
        rows = parse_mounts(MOUNTS)
        self.assertEqual(len(rows), 6)
        self.assertIn("cifs", {r.fstype for r in rows})

    def test_longest_mount_point_wins(self):
        inside = mount_for("/mnt/kb-out/reports", mounts_text=MOUNTS)
        self.assertIsNotNone(inside)
        self.assertEqual(inside.mountpoint, "/mnt/kb-out")
        outer = mount_for("/mnt/kb/other", mounts_text=MOUNTS)
        self.assertEqual(outer.mountpoint, "/mnt/kb")

    def test_network_and_readonly_flags(self):
        self.assertTrue(mount_for("/mnt/kb/x", mounts_text=MOUNTS).is_network)
        self.assertTrue(mount_for("/mnt/nfs/x", mounts_text=MOUNTS).is_network)
        self.assertFalse(mount_for("/etc/hosts", mounts_text=MOUNTS).is_network)
        self.assertTrue(mount_for("/mnt/archive/x", mounts_text=MOUNTS).readonly)
        self.assertFalse(mount_for("/mnt/kb/x", mounts_text=MOUNTS).readonly)

    def test_device_reported_for_cifs(self):
        self.assertEqual(mount_for("/mnt/kb/x", mounts_text=MOUNTS).server, "//192.0.2.10/kb")

    def test_autofs_trigger_does_not_shadow_the_real_mount(self):
        """`/proc/mounts` lists an autofs trigger and the cifs mount on the same
        mount point; the real filesystem must win."""
        table = (
            "systemd-1 /mnt/nas/kb autofs rw,relatime,fd=63,pgrp=1 0 0\n"
            "//192.0.2.10/kb /mnt/nas/kb cifs rw,relatime,vers=3.0 0 0\n"
        )
        info = mount_for("/mnt/nas/kb/知识库", mounts_text=table)
        self.assertEqual(info.fstype, "cifs")
        self.assertTrue(info.is_network)

    def test_describe_path_marks_unc_and_local(self):
        self.assertTrue(describe_path(r"\\nas\share\kb")["unc"])
        self.assertTrue(is_unc("//nas/share"))
        self.assertFalse(is_unc("/mnt/kb"))
        described = describe_path("/mnt/archive/x", mounts_text=MOUNTS)
        self.assertTrue(described["network"])
        self.assertTrue(described["readonly"])
        self.assertEqual(described["fstype"], "cifs")

    def test_stale_hint_is_actionable(self):
        import errno

        stale = stale_hint(OSError(errno.ESTALE, "Stale file handle"))
        self.assertIn("stale", stale)
        self.assertIn("mount -a", stale)
        denied = stale_hint(OSError(errno.EACCES, "Permission denied"))
        self.assertIn("read-only", denied)
        gone = stale_hint(OSError(errno.ENOENT, "No such file or directory"))
        self.assertIn("not reachable", gone)


class TestRootNormalisation(unittest.TestCase):
    def test_posix_path_is_untouched(self):
        self.assertEqual(str(normalise_root("/mnt/kb/knowledge")), "/mnt/kb/knowledge")

    def test_tilde_is_expanded(self):
        self.assertEqual(str(normalise_root("~/kb")), os.path.expanduser("~/kb"))

    def test_quotes_are_stripped(self):
        self.assertEqual(str(normalise_root('"/mnt/kb"')), "/mnt/kb")

    @unittest.skipUnless(os.sep == "/", "only meaningful on POSIX")
    def test_windows_spellings_are_translated_on_posix(self):
        self.assertEqual(str(normalise_root("C:\\Users\\me\\kb")), "C:/Users/me/kb")
        self.assertEqual(str(normalise_root(r"\\nas\share\kb")), "//nas/share/kb")


class TestFilesystemSource(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-src-"))
        (self.tmp / "z0-public").mkdir()
        (self.tmp / "z0-public" / "a.md").write_text("hello\n", encoding="utf-8")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _readonly_mounts(self) -> str:
        return f"//192.0.2.10/archive {self.tmp} cifs ro,relatime 0 0\n"

    def test_walk_lists_files_and_dirs(self):
        source = FilesystemSource(self.tmp)
        rels = {e.rel for e in source.walk()}
        self.assertEqual(rels, {"z0-public", "z0-public/a.md"})

    def test_read_returns_bytes(self):
        self.assertEqual(FilesystemSource(self.tmp).read("z0-public/a.md"), b"hello\n")

    def test_missing_directory_is_an_actionable_error(self):
        source = FilesystemSource(self.tmp / "nope")
        with self.assertRaises(SourceUnavailable):
            source.walk()

    def test_write_is_refused_on_a_read_only_mount(self):
        source = FilesystemSource(self.tmp, mounts_text=self._readonly_mounts())
        self.assertTrue(source.mount["readonly"])
        with self.assertRaises(SourceReadOnly) as ctx:
            source.write("_incoming/x.md", b"data")
        self.assertIn("read-only", str(ctx.exception))

    def test_write_probe_reports_read_only(self):
        source = FilesystemSource(self.tmp, mounts_text=self._readonly_mounts())
        ok, why = source.probe_writable()
        self.assertFalse(ok)
        self.assertIn("read-only", why)

    def test_write_probe_succeeds_and_leaves_nothing(self):
        source = FilesystemSource(self.tmp)
        ok, why = source.probe_writable("_incoming")
        self.assertTrue(ok, why)
        self.assertEqual(list((self.tmp / "_incoming").iterdir()), [])

    def test_write_creates_parents_and_replaces_atomically(self):
        source = FilesystemSource(self.tmp)
        source.write("_incoming/deep/x.md", b"one")
        source.write("_incoming/deep/x.md", b"two")
        self.assertEqual((self.tmp / "_incoming" / "deep" / "x.md").read_bytes(), b"two")
        self.assertEqual([p.name for p in (self.tmp / "_incoming" / "deep").iterdir()], ["x.md"])

    def test_describe_reports_network_flag(self):
        source = FilesystemSource(self.tmp, mounts_text=f"//h/s {self.tmp} cifs rw 0 0\n")
        self.assertTrue(source.describe()["network"])


SMB_STUB = '''#!/usr/bin/env python3
"""Minimal smbclient stand-in used by the test-suite."""
import os, re, shutil, sys
from pathlib import Path

argv = sys.argv[1:]
cmds = ""
for i, a in enumerate(argv):
    if a == "-c":
        cmds = argv[i + 1]
fixture = Path(os.environ["SMB_FIXTURE"])


def listing(root: Path) -> str:
    lines = []

    def emit(directory: Path, rel: str) -> None:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
        for e in entries:
            attr = "D" if e.is_dir() else "N"
            size = 0 if e.is_dir() else e.stat().st_size
            lines.append(f"  {e.name:<36}{attr}  {size:>8}  Mon Oct 12 22:00:00 2026")
        for e in entries:
            if e.is_dir():
                sub = f"{rel}/{e.name}" if rel else e.name
                lines.append("")
                lines.append("\\\\" + sub.replace("/", "\\\\"))
                emit(e, sub)

    emit(root, "")
    return "\\n".join(lines)


# Dispatch on the command verb, not on a substring of the whole command line:
# a temp path that happens to contain "ls" (e.g. TMPDIR=/tmp/ls-repro) used to
# send `get` down the `ls` branch and fail as a random 1-in-200 flake.
verbs = [part.split()[0] for part in (p.strip() for p in cmds.split(";")) if part]

if "get" in verbs:
    m = re.search(r'get "([^"]+)" "([^"]+)"', cmds)
    shutil.copy2(fixture / m.group(1), m.group(2))
elif "put" in verbs:
    m = re.search(r'put "([^"]+)" "([^"]+)"', cmds)
    target = fixture / m.group(2)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(m.group(1), target)
elif "ls" in verbs:
    print(listing(fixture))
    print()
    print("\\t\\t7772088952 blocks of size 1024. 5891711316 blocks available")
else:
    sys.exit(1)
'''


class TestSmbSource(unittest.TestCase):
    """Drive the real parser through a stub client with real output format."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-smb-"))
        self.share = self.tmp / "share"
        (self.share / "z0-public" / "deep").mkdir(parents=True)
        (self.share / "z0-public" / "mount stale.md").write_text("x" * 1234, encoding="utf-8")
        (self.share / "z0-public" / "deep" / "note.md").write_text("y" * 10, encoding="utf-8")
        (self.share / "index.md").write_text("root\n", encoding="utf-8")
        self.stub = self.tmp / "smbclient"
        self.stub.write_text(SMB_STUB, encoding="utf-8")
        self.stub.chmod(self.stub.stat().st_mode | stat.S_IXUSR)
        os.environ["SMB_FIXTURE"] = str(self.share)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("SMB_FIXTURE", None)

    def _source(self) -> SmbSource:
        return SmbSource("//server/share", client=str(self.stub))

    def test_walk_reconstructs_the_tree(self):
        entries = {e.rel: e for e in self._source().walk()}
        self.assertIn("index.md", entries)
        self.assertIn("z0-public", entries)
        self.assertTrue(entries["z0-public"].is_dir)
        self.assertIn("z0-public/deep", entries)
        self.assertIn("z0-public/deep/note.md", entries)

    def test_names_with_spaces_survive(self):
        entries = {e.rel: e for e in self._source().walk()}
        self.assertIn("z0-public/mount stale.md", entries)
        self.assertEqual(entries["z0-public/mount stale.md"].size, 1234)

    def test_trailing_blocks_line_is_not_an_entry(self):
        names = {e.rel for e in self._source().walk()}
        self.assertFalse(any("blocks" in n for n in names))

    def test_read_returns_content(self):
        self.assertEqual(self._source().read("index.md"), b"root\n")

    def test_write_reaches_the_share(self):
        self._source().write("_incoming/new.md", b"payload")
        self.assertEqual((self.share / "_incoming" / "new.md").read_bytes(), b"payload")

    def test_missing_client_binary_is_actionable(self):
        source = SmbSource("//server/share", client=str(self.tmp / "nope"))
        with self.assertRaises(SourceUnavailable) as ctx:
            source.ensure_available()
        self.assertIn("smbclient", str(ctx.exception))


class TestWebdavEncoding(unittest.TestCase):
    """Request targets must be ASCII, and encoding must be idempotent.

    A non-ASCII collection path raised ``UnicodeEncodeError`` inside the request
    — swallowed by the resilient walker, so the transport silently returned an
    empty tree.  These tests pin both the encoding and its idempotence.
    """

    def _client(self, url: str):
        from dsh_kb.webdav import WebDAVClient

        return WebDAVClient(url)

    def test_chinese_collection_path_is_encoded(self):
        client = self._client("https://nas.example:5000/webdav/知识库")
        self.assertEqual(client.base_path, "/webdav/%E7%9F%A5%E8%AF%86%E5%BA%93")
        self.assertTrue(client._url("").isascii())

    def test_already_encoded_url_is_not_double_encoded(self):
        encoded = "https://nas.example:5000/webdav/%E7%9F%A5%E8%AF%86%E5%BA%93"
        client = self._client(encoded)
        self.assertEqual(client.base_path, "/webdav/%E7%9F%A5%E8%AF%86%E5%BA%93")
        self.assertNotIn("%25", client.base_path)

    def test_relative_paths_are_encoded_once(self):
        client = self._client("https://nas.example:5000/webdav/知识库")
        self.assertEqual(
            client._url("_meta/约定.md"),
            "/webdav/%E7%9F%A5%E8%AF%86%E5%BA%93/_meta/%E7%BA%A6%E5%AE%9A.md",
        )
        # idempotent: feeding an encoded path back changes nothing
        self.assertEqual(client._url("_meta/%E7%BA%A6%E5%AE%9A.md"), client._url("_meta/约定.md"))

    def test_display_path_is_decoded(self):
        client = self._client("https://nas.example:5000/webdav/知识库")
        self.assertEqual(client.display_path("约定.md"), "/webdav/知识库/约定.md")

    def test_root_url_yields_a_slash(self):
        self.assertEqual(self._client("https://nas.example:5000")._url(""), "/")


class TestWebdavSource(unittest.TestCase):
    """The href→relative-path mapping, which silently dropped every file once."""

    class FakeClient:
        base_path = "/webdav/kb"

        def listdir(self, path=""):  # pragma: no cover - not used here
            return []

    def _source(self, root=""):
        from dsh_kb.sources import WebdavSource

        return WebdavSource(self.FakeClient(), root=root)

    def test_write_creates_missing_parent_collections(self):
        """A bare PUT into a missing collection is HTTP 500 errno 2.

        Measured: a 217-file import failed 217 times for exactly this reason.
        MKCOL does not recurse, so every level must be created.
        """
        from dsh_kb.sources import WebdavSource

        calls: list[tuple[str, str]] = []

        class RecordingClient:
            base_path = "/kb"

            def mkcol(self, path):
                calls.append(("MKCOL", path))
                return 201

            def put(self, path, data, content_type=None):
                calls.append(("PUT", path))
                return 201

        source = WebdavSource(RecordingClient())
        source.write("z0-public/manual/A.md", b"body")
        self.assertEqual(calls, [("MKCOL", "z0-public"), ("MKCOL", "z0-public/manual"),
                                 ("PUT", "z0-public/manual/A.md")])

    def test_write_does_not_repeat_parent_creation(self):
        from dsh_kb.sources import WebdavSource

        calls: list[tuple[str, str]] = []

        class RecordingClient:
            base_path = "/kb"

            def mkcol(self, path):
                calls.append(("MKCOL", path))
                return 201

            def put(self, path, data, content_type=None):
                calls.append(("PUT", path))
                return 201

        source = WebdavSource(RecordingClient())
        source.write("z0-public/a.md", b"1")
        source.write("z0-public/b.md", b"2")
        self.assertEqual([c for c in calls if c[0] == "MKCOL"], [("MKCOL", "z0-public")])
        self.assertEqual(len([c for c in calls if c[0] == "PUT"]), 2)

    def test_top_level_write_needs_no_mkcol(self):
        from dsh_kb.sources import WebdavSource

        calls: list[str] = []

        class RecordingClient:
            base_path = "/kb"

            def mkcol(self, path):  # pragma: no cover - must not be called
                calls.append("MKCOL")

            def put(self, path, data, content_type=None):
                calls.append("PUT")

        WebdavSource(RecordingClient()).write("index.md", b"x")
        self.assertEqual(calls, ["PUT"])

    def test_mkcol_failures_do_not_mask_the_put(self):
        """Already-exists and permission errors are the PUT's business."""
        from dsh_kb.sources import WebdavSource

        calls: list[str] = []

        class GrumpyClient:
            base_path = "/kb"

            def mkcol(self, path):
                raise OSError("405 Method Not Allowed")

            def put(self, path, data, content_type=None):
                calls.append("PUT")
                return 201

        WebdavSource(GrumpyClient()).write("a/b.md", b"x")
        self.assertEqual(calls, ["PUT"])

    def test_hrefs_are_made_relative_to_the_collection(self):
        source = self._source()
        self.assertEqual(source._strip("/webdav/kb/index.md"), "index.md")
        self.assertEqual(source._strip("/webdav/kb/z0-public/a.md"), "z0-public/a.md")

    def test_percent_encoded_hrefs_are_decoded(self):
        source = self._source()
        self.assertEqual(source._strip("/webdav/kb/%E7%9F%A5%E8%AF%86.md"), "知识.md")

    def test_paths_outside_the_collection_are_ignored(self):
        source = self._source()
        self.assertIsNone(source._strip("/webdav/other/index.md"))

    def test_extra_root_is_appended_to_the_prefix(self):
        source = self._source(root="z0-public")
        self.assertEqual(source._strip("/webdav/kb/z0-public/a.md"), "a.md")
        self.assertIsNone(source._strip("/webdav/kb/other/a.md"))

    def test_encoded_collection_path_is_compared_in_decoded_space(self):
        """The client stores its collection encoded; hrefs arrive encoded too."""
        from dsh_kb.sources import WebdavSource

        class EncodedClient:
            base_path = "/webdav/%E7%9F%A5%E8%AF%86%E5%BA%93"

        source = WebdavSource(EncodedClient())
        self.assertEqual(
            source._strip("/webdav/%E7%9F%A5%E8%AF%86%E5%BA%93/index.md"), "index.md"
        )
        self.assertIsNone(source._strip("/webdav/other/index.md"))

    def test_unreachable_collection_raises_instead_of_looking_empty(self):
        """A dead server must not be reported as "0 files, refresh succeeded"."""
        from dsh_kb.sources import SourceUnavailable, WebdavSource

        class DeadClient:
            base_path = "/kb"

            def listdir(self, path=""):
                raise OSError("Connection refused")

        with self.assertRaises(SourceUnavailable) as ctx:
            WebdavSource(DeadClient()).walk()
        self.assertIn("cannot list", str(ctx.exception))

    def test_a_failing_subdirectory_is_recorded_but_not_fatal(self):
        from dsh_kb.sources import WebdavSource
        from dsh_kb.webdav import DavEntry

        class HalfDeadClient:
            base_path = "/kb"

            def listdir(self, path=""):
                if path == "":
                    return [
                        DavEntry(href="/kb/index.md", is_dir=False, size=3, modified=""),
                        DavEntry(href="/kb/sub", is_dir=True, size=0, modified=""),
                    ]
                raise OSError("permission denied")

        source = WebdavSource(HalfDeadClient())
        self.assertEqual([e.rel for e in source.walk()], ["index.md", "sub"])
        self.assertTrue(source.last_errors)


class TestSmbParsing(unittest.TestCase):
    """Pure parser tests, including the `cd`-prefixed header form."""

    def test_headers_are_stripped_relative_to_the_subdir(self):
        from dsh_kb.sources import _parse_smbclient_ls

        entries = _parse_smbclient_ls(SMB_LISTING, "知识库")
        rels = {e.rel for e in entries}
        self.assertIn("z0-public/mount stale.md", rels)
        self.assertIn("z0-public/deep/note.md", rels)
        self.assertNotIn("知识库/z0-public", rels)

    def test_empty_listing_is_empty(self):
        from dsh_kb.sources import _parse_smbclient_ls

        self.assertEqual(_parse_smbclient_ls("", ""), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
