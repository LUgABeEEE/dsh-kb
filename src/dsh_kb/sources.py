"""Read/write sources: where the authoritative bundle lives.

Three interchangeable implementations behind one duck-typed interface::

    class Source:
        label: str
        def walk(self) -> list[Entry]
        def read(self, rel: str) -> bytes
        def write(self, rel: str, data: bytes) -> None   # optional

* :class:`FilesystemSource` — a local directory, a **Samba/CIFS mount**, an NFS
  mount or a Windows drive/UNC path.  Walking is ``rglob`` and reads are
  ``open()``: the fast path (a 10 000-file tree walks in milliseconds).
  A network mount behaves differently enough that the class inspects it (see
  :mod:`dsh_kb.mounts`) and turns raw OS errors into actionable sentences.
* :class:`SmbSource` — a Samba share reached through ``smbclient`` **without
  mounting**, for hosts where mounting needs root or the mount has gone stale.
* :class:`WebdavSource` — a remote collection over HTTP(S), one PROPFIND per
  directory (~55 ms on a consumer NAS).

One interface is what lets the same mirror logic serve all of them unchanged.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path

from .config import Config, source_available
from .mounts import describe_path, stale_hint

PROBE_NAME = ".dsh-kb-write-probe"

#: One listing row: ``name … attributes  size  date``.
_SMB_ROW_RE = re.compile(
    r"^\s+(?P<name>.+?)\s{2,}(?P<attr>[ADNHSR]+)\s+(?P<size>\d+)\s+(?P<date>\S.*)$"
)


class SourceError(RuntimeError):
    """Base class for source problems that an operator can act on."""


class SourceUnavailable(SourceError):
    """The source cannot be reached at all (stale mount, host down, ...)."""


class SourceReadOnly(SourceError):
    """The source is reachable but will not accept writes."""


@dataclass
class Entry:
    """One item in the bundle, independent of the transport."""

    rel: str          # bundle-relative POSIX path, decoded
    is_dir: bool
    size: int = 0
    mtime: float = 0.0  # epoch seconds; 0 when unknown


def normalise_root(value: str | os.PathLike) -> Path:
    """Accept POSIX, Windows drive and UNC spellings of the same directory.

    Backslashes are accepted on every platform (people paste Windows paths into
    a Linux config and vice versa), and ``~`` is expanded.
    """
    text = str(value).strip().strip('"')
    if os.sep == "/" and text.startswith("\\\\"):
        text = text.replace("\\", "/")          # UNC pasted on POSIX
    if os.sep == "/" and len(text) > 1 and text[1] == ":" and text[2:3] in ("\\", "/"):
        text = text.replace("\\", "/")          # C:\... pasted on POSIX
    return Path(os.path.expanduser(text))


class FilesystemSource:
    """A directory the OS can read: local, or a mounted network share."""

    def __init__(
        self,
        root: str | os.PathLike,
        label: str = "filesystem",
        *,
        mounts_text: str | None = None,
    ):
        self.root = normalise_root(root)
        self.label = label
        # `mounts_text` is a testability seam: it lets the suite declare a fake
        # read-only CIFS mount without owning a share.
        self.mount = describe_path(self.root, mounts_text=mounts_text)

    # ---- metadata ------------------------------------------------------
    def describe(self) -> dict:
        return {**self.mount, "label": self.label, "root": str(self.root)}

    @property
    def is_network(self) -> bool:
        return bool(self.mount.get("network"))

    def ensure_available(self) -> None:
        """Fail early with a sentence, rather than deep inside a walk."""
        try:
            if not self.root.is_dir():
                raise SourceUnavailable(f"not a directory: {self.root}")
            next(iter(self.root.iterdir()), None)   # touches the mount
        except SourceUnavailable:
            raise
        except OSError as exc:
            raise SourceUnavailable(f"{self.root}: {stale_hint(exc)}") from exc

    def probe_writable(self, subdir: str = "") -> tuple[bool, str]:
        """Try a real write so ``doctor`` can be authoritative, not optimistic."""
        if self.mount.get("readonly"):
            return False, f"mounted read-only at {self.mount.get('mountpoint')}"
        folder = self.root / subdir if subdir else self.root
        probe = folder / PROBE_NAME
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            probe.unlink()
            return True, "write probe succeeded"
        except OSError as exc:
            return False, stale_hint(exc)
        finally:
            try:
                probe.unlink()
            except OSError:
                pass

    # ---- reads ---------------------------------------------------------
    def walk(self) -> list[Entry]:
        self.ensure_available()
        out: list[Entry] = []
        try:
            paths = sorted(self.root.rglob("*"))
        except OSError as exc:
            raise SourceUnavailable(f"cannot walk {self.root}: {stale_hint(exc)}") from exc
        for path in paths:
            rel = path.relative_to(self.root).as_posix()
            try:
                if path.is_dir():
                    out.append(Entry(rel=rel, is_dir=True))
                    continue
                stat = path.stat()
            except OSError:
                # A single unreadable entry must not abort the whole walk.
                continue
            out.append(Entry(rel=rel, is_dir=False, size=stat.st_size, mtime=stat.st_mtime))
        return out

    def read(self, rel: str) -> bytes:
        try:
            return (self.root / rel).read_bytes()
        except OSError as exc:
            raise SourceUnavailable(f"cannot read {rel}: {stale_hint(exc)}") from exc

    # ---- writes --------------------------------------------------------
    def write(self, rel: str, data: bytes, content_type: str | None = None) -> None:
        """Write through to the share.

        Uses a temp file plus ``os.replace`` so a reader never sees a partial
        document.  Some SMB servers refuse the rename; that is retried as a
        direct write before giving up, and every failure is translated.
        """
        if self.mount.get("readonly"):
            raise SourceReadOnly(
                f"{self.root} is on a read-only mount ({self.mount.get('mountpoint')}); "
                "write through another source (e.g. DSH_KB_SOURCE=webdav)"
            )
        target = self.root / rel
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SourceReadOnly(f"cannot create {target.parent}: {stale_hint(exc)}") from exc

        tmp = target.with_name(target.name + ".dsh-kb-tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, target)
            return
        except OSError as exc:
            try:
                tmp.unlink()
            except OSError:
                pass
            try:
                target.write_bytes(data)
                return
            except OSError:
                raise SourceReadOnly(
                    f"cannot write {rel}: {stale_hint(exc)}"
                ) from exc


class SmbSource:
    """A Samba share reached with ``smbclient``, without mounting it.

    Useful when mounting needs root (a plain user session usually cannot call
    ``mount``) or when a mount has gone stale but the server is fine.  Requires
    the ``smbclient`` binary and a credentials file; see the docs for the
    auth-file format.
    """

    def __init__(
        self,
        share: str,
        *,
        subdir: str = "",
        client: str = "smbclient",
        credentials: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 60.0,
        label: str = "smb",
    ):
        if not share.startswith("//"):
            share = "//" + share.lstrip("\\/")
        self.share = share.rstrip("/")
        self.subdir = subdir.strip("/")
        self.client = client
        self.credentials = credentials
        self.username = username
        self.password = password
        self.timeout = timeout
        self.label = label
        self.root = self.share + (f"/{self.subdir}" if self.subdir else "")

    # ---- plumbing ------------------------------------------------------
    def _auth_args(self) -> list[str]:
        if self.credentials:
            return ["-A", self.credentials]
        if self.username:
            return ["-U", f"{self.username}%{self.password or ''}"]
        return ["-N"]

    def _run(self, commands: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [self.client, self.share, *self._auth_args(), "-c", "; ".join(commands)],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SourceUnavailable(
                f"{self.client!r} not found: install a Samba client (e.g. `apt install smbclient`)"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SourceUnavailable(f"{self.share}: smbclient timed out after {self.timeout}s") from exc

    def ensure_available(self) -> None:
        result = self._run(["ls"])
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            raise SourceUnavailable(
                f"cannot list {self.share}: {detail[-1] if detail else 'smbclient failed'}"
            )

    # ---- reads ---------------------------------------------------------
    def walk(self) -> list[Entry]:
        """List recursively using one ``smbclient`` invocation.

        ``recurse ON`` + ``ls`` prints a ``\\dir`` line before each directory's
        contents, which is enough to reconstruct the tree without a call per
        directory.
        """
        self.ensure_available()
        commands = []
        if self.subdir:
            commands.append(f'cd "{self.subdir}"')
        commands.append("recurse ON")
        commands.append("ls")
        result = self._run(commands)
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            raise SourceUnavailable(f"cannot walk {self.root}: {detail[-1] if detail else 'smbclient failed'}")
        return _parse_smbclient_ls(result.stdout, self.subdir)

    def read(self, rel: str) -> bytes:
        remote = f"{self.subdir}/{rel}" if self.subdir else rel
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "payload"
            result = self._run([f'get "{remote}" "{local}"'])
            if result.returncode != 0 or not local.exists():
                detail = (result.stderr or result.stdout or "").strip().splitlines()
                raise SourceUnavailable(
                    f"cannot read {rel}: {detail[-1] if detail else 'smbclient failed'}"
                )
            return local.read_bytes()

    def write(self, rel: str, data: bytes, content_type: str | None = None) -> None:
        remote = f"{self.subdir}/{rel}" if self.subdir else rel
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "payload"
            local.write_bytes(data)
            parent = str(Path(remote).parent).replace("\\", "/")
            # smbclient's mkdir does not create parents, so build each level.
            commands = [f'mkdir "{part}"' for part in _parent_chain(parent)]
            commands.append(f'put "{local}" "{remote}"')
            result = self._run(commands)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "").strip().splitlines()
                raise SourceReadOnly(
                    f"cannot write {rel}: {detail[-1] if detail else 'smbclient failed'}"
                )

    def describe(self) -> dict:
        return {
            "label": self.label,
            "root": self.root,
            "share": self.share,
            "subdir": self.subdir,
            "fstype": "smb",
            "device": self.share,
            "network": True,
            "readonly": False,
            "unc": True,
        }


def _parent_chain(directory: str) -> list[str]:
    """Every ancestor of a remote directory, so ``smbclient mkdir`` can build it."""
    if directory in ("", "."):
        return []
    parts = [p for p in directory.split("/") if p and p != "."]
    return ["/".join(parts[: index + 1]) for index in range(len(parts))]


def _parse_smbclient_ls(output: str, subdir: str = "") -> list[Entry]:
    """Reconstruct a tree from ``smbclient recurse ON; ls`` output.

    Verified against a real server — the format is::

          .                                   D        0  Mon Oct 12 22:23:52 2026
          index.md                            N      980  Mon Oct 12 22:23:51 2026

        \\知识库\\z0-public
          .                                   D        0  ...
          ..                                  D        0  ...

    i.e. ``name, attributes, size, date`` (name first, columns separated by two
    or more spaces), with a ``\\path`` header before each directory block, and a
    trailing ``... blocks of size 1024 ...`` line that must not be mistaken for
    an entry.
    """
    out: list[Entry] = []
    current: list[str] = []
    prefix = [p for p in subdir.strip("/").split("/") if p] if subdir else []

    for raw in output.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("\\"):
            parts = [p for p in line.strip().replace("\\", "/").split("/") if p and p != "."]
            if prefix and parts[: len(prefix)] == prefix:
                parts = parts[len(prefix):]
            current = parts
            continue
        row = _SMB_ROW_RE.match(line)
        if row is None:
            continue
        name = row.group("name").strip()
        if name in (".", "..") or not name:
            continue
        rel = "/".join([*current, name])
        if row.group("attr").upper().startswith("D"):
            out.append(Entry(rel=rel, is_dir=True))
        else:
            out.append(Entry(rel=rel, is_dir=False, size=int(row.group("size")), mtime=0.0))

    out.sort(key=lambda e: e.rel)
    return out


class WebdavSource:
    """A remote collection reached over WebDAV."""

    def __init__(self, client, root: str = "", label: str = "webdav"):
        self.client = client
        self.root = root.strip("/")
        self.label = label
        self.mount = {"network": True, "fstype": "webdav"}
        #: Sub-directories that failed during the last :meth:`walk`.
        self.last_errors: list[str] = []
        #: Collections already created (or attempted) by :meth:`_ensure_parents`.
        self._made_dirs: set[str] = set()

    def _join(self, rel: str) -> str:
        return f"{self.root}/{rel}" if self.root else rel

    def _base_path(self) -> str:
        """Collection prefix that server hrefs must be made relative to.

        The client already carries the collection in its ``base_path``, so
        stripping only ``root`` produced paths like
        ``webdav/kb/index.md`` and silently dropped every file — which is
        why this is a named method with a test.
        """
        base = str(getattr(self.client, "base_path", "") or "").rstrip("/")
        if self.root:
            base = f"{base}/{self.root}"
        return base

    def _strip(self, href_path: str) -> str | None:
        """Turn a server href into a bundle-relative path (decoded).

        Both sides are compared in *decoded* space: the client stores its
        collection percent-encoded (it must, to build request targets), while
        hrefs arrive percent-encoded too.  Comparing them without decoding was
        how every file silently vanished.
        """
        import urllib.parse

        decoded = urllib.parse.unquote(href_path).rstrip("/")
        base = urllib.parse.unquote(self._base_path())
        if base and decoded != base and not decoded.startswith(base + "/"):
            return None
        return decoded[len(base):].lstrip("/")

    def walk(self) -> list[Entry]:
        """List the collection recursively.

        A failure at the **root** means the source is unreachable and must
        propagate: silently returning "0 files" once made a dead server look
        like a successful refresh.  A failure deeper down is recorded and the
        walk continues, so one bad directory cannot hide the rest.
        """
        out: list[Entry] = []
        stack: list[str] = [""]
        self.last_errors = []
        while stack:
            current = stack.pop()
            try:
                children = self.client.listdir(self._join(current))
            except Exception as exc:  # noqa: BLE001 - translated below
                if current == "":
                    raise SourceUnavailable(
                        f"cannot list {self.root or '/'}: {type(exc).__name__}: {exc}"
                    ) from exc
                self.last_errors.append(f"{current}: {type(exc).__name__}: {exc}")
                continue
            for child in children:
                rel = self._strip(child.href)
                if rel is None or rel == "":
                    continue
                if child.is_dir:
                    out.append(Entry(rel=rel, is_dir=True))
                    stack.append(rel)
                else:
                    out.append(
                        Entry(rel=rel, is_dir=False, size=child.size, mtime=_http_date(child.modified))
                    )
        out.sort(key=lambda e: e.rel)
        return out

    def read(self, rel: str) -> bytes:
        return self.client.get(self._join(rel))

    def _ensure_parents(self, rel: str) -> None:
        """Create the collection chain holding ``rel`` — MKCOL does not recurse.

        A bare ``PUT`` into a missing collection is just
        ``HTTP 500 errno 2 No such file or directory``, which is how a 217-file
        import failed 217 times before this existed.  Collections already dealt
        with are remembered per instance, so a bulk import does not repeat them.
        """
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        if not parent or parent in self._made_dirs:
            return
        parts = [part for part in parent.split("/") if part]
        for depth in range(len(parts)):
            sub = "/".join(parts[: depth + 1])
            if sub in self._made_dirs:
                continue
            try:
                self.client.mkcol(self._join(sub))
            except Exception:  # noqa: BLE001 - the PUT below is the real arbiter
                pass
            self._made_dirs.add(sub)

    def write(self, rel: str, data: bytes, content_type: str = "text/markdown; charset=utf-8") -> None:
        self._ensure_parents(rel)
        self.client.put(self._join(rel), data, content_type=content_type)

    def describe(self) -> dict:
        return {"label": self.label, "root": self.root, "network": True, "fstype": "webdav",
                "readonly": False, "unc": False, "mountpoint": "", "device": self.client.base_path}


def _http_date(value: str) -> float:
    if not value:
        return 0.0
    try:
        return parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


def open_source(cfg: Config):
    """Instantiate the configured source.

    Returns ``(source, note)``.  ``source`` is ``None`` when nothing usable is
    configured; ``note`` always explains what happened, so the CLI can print a
    helpful message instead of a stack trace.
    """
    kind, location = source_available(cfg)
    if not kind:
        return None, f"no usable source ({location}) — set DSH_KB_MOUNT / DSH_KB_LOCAL_ROOT / DSH_KB_WEBDAV_ROOT"

    if kind in ("local", "mount"):
        root = normalise_root(location)
        if not root.is_dir():
            return None, f"{kind} path is not a directory: {location}"
        return FilesystemSource(root, label=kind), f"{kind}: {root}"

    if kind == "smb":
        return (
            SmbSource(
                location,
                subdir=os.environ.get("DSH_KB_SMB_SUBDIR", ""),
                credentials=os.environ.get("DSH_KB_SMB_CREDENTIALS") or None,
                username=os.environ.get("DSH_KB_SMB_USER") or None,
                password=os.environ.get("DSH_KB_SMB_PASSWORD") or None,
            ),
            f"smb: {location}",
        )

    client = _webdav_from_config(cfg, location)
    return WebdavSource(client, root="", label="webdav"), f"webdav: {location}"


def _webdav_from_config(cfg: Config, base_url: str):
    """Build a :class:`WebDAVClient` from config/env.

    Secrets are read from the environment only (``DSH_KB_WEBDAV_USER`` /
    ``DSH_KB_WEBDAV_PASSWORD``) or a credentials file — never from the config
    file, so a config that gets shared cannot leak them.
    """
    from .webdav import WebDAVClient

    password = os.environ.get("DSH_KB_WEBDAV_PASSWORD", "")
    cred_file = os.environ.get("DSH_KB_WEBDAV_CREDENTIALS", "")
    if cred_file and Path(cred_file).is_file():
        for line in Path(cred_file).read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            key = key.strip().upper()
            if key in ("DSH_KB_WEBDAV_PASSWORD", "MINAS_PASS") and not password:
                password = value.strip()
            if key in ("DSH_KB_WEBDAV_USER", "MINAS_USER"):
                os.environ.setdefault("DSH_KB_WEBDAV_USER", value.strip())

    return WebDAVClient(
        base_url,
        username=os.environ.get("DSH_KB_WEBDAV_USER", ""),
        password=password,
        cert=os.environ.get("DSH_KB_WEBDAV_CERT") or None,
        key=os.environ.get("DSH_KB_WEBDAV_KEY") or None,
        ca=os.environ.get("DSH_KB_WEBDAV_CA") or None,
        host_header=os.environ.get("DSH_KB_WEBDAV_HOST") or None,
        connect_host=os.environ.get("DSH_KB_WEBDAV_CONNECT") or None,
        verify=os.environ.get("DSH_KB_WEBDAV_VERIFY", "1") not in ("0", "false", "no"),
    )


__all__ = [
    "Entry",
    "FilesystemSource",
    "SmbSource",
    "SourceError",
    "SourceReadOnly",
    "SourceUnavailable",
    "WebdavSource",
    "normalise_root",
    "open_source",
]
