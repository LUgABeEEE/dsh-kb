"""Mount-point inspection: what kind of filesystem is this path on?

A Samba/CIFS mount is not the same animal as a local directory, and the
differences matter operationally:

* **read-only views.**  A share can be mounted ``rw`` and still be read-only for
  the account at the server end — or the operator may have mounted a
  deliberately read-only view of a larger share (a common safety pattern).
  ``push`` should fail with a sentence, not a traceback.
* **stale handles.**  When a network mount loses its session, almost every
  syscall fails with ``ESTALE``.  Walking the tree then raises from ``rglob``
  itself, so the error has to be caught and explained.
* **coarse timestamps.**  Measured on a consumer NAS: ``Last-Modified`` has
  **1-second** granularity.  Incremental sync therefore cannot rely on
  sub-second mtimes, and a same-second edit that keeps the size identical is
  invisible — hence the ``--full`` escape hatch.

Everything here is a pure function of ``/proc/mounts`` content (injectable, so
it is testable without any real share).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

#: Filesystem types that mean "this is somebody else's disk over a wire".
NETWORK_FSTYPES = frozenset(
    {
        "cifs",
        "smb3",
        "smb2",
        "smbfs",
        "nfs",
        "nfs4",
        "davfs",
        "davfs2",
        "sshfs",
        "fuse.sshfs",
        "fuse.rclone",
        "fuse.smbnetfs",
        "9p",
        "afs",
        "webdav",
    }
)

_ESCAPES = {"040": " ", "011": "\t", "012": "\n", "134": "\\"}


def decode_mount_field(value: str) -> str:
    """Undo the octal escaping ``/proc/mounts`` applies (``\\040`` is a space)."""
    out: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 3 < len(value) + 1:
            code = value[index + 1:index + 4]
            if code in _ESCAPES:
                out.append(_ESCAPES[code])
                index += 4
                continue
        out.append(value[index])
        index += 1
    return "".join(out)


@dataclass(frozen=True)
class MountInfo:
    """One row of the mount table."""

    mountpoint: str
    fstype: str
    device: str
    options: frozenset[str]

    @property
    def is_network(self) -> bool:
        return self.fstype.lower() in NETWORK_FSTYPES or self.device.startswith("//")

    @property
    def readonly(self) -> bool:
        return "ro" in self.options

    @property
    def server(self) -> str:
        """``//host/share`` for a CIFS device, otherwise the raw device."""
        return self.device


def parse_mounts(text: str) -> list[MountInfo]:
    """Parse ``/proc/mounts``-style content into :class:`MountInfo` rows."""
    rows: list[MountInfo] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        device, mountpoint, fstype, option_text = parts[0], parts[1], parts[2], parts[3]
        rows.append(
            MountInfo(
                mountpoint=decode_mount_field(mountpoint),
                fstype=fstype,
                device=decode_mount_field(device),
                options=frozenset(o for o in option_text.split(",") if o),
            )
        )
    return rows


def read_mounts(path: str = "/proc/mounts") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def mount_for(
    path: str | os.PathLike,
    *,
    mounts_text: str | None = None,
    mounts_file: str = "/proc/mounts",
) -> MountInfo | None:
    """Return the mount entry a path lives on (longest matching mount point).

    ``mounts_text`` short-circuits the platform lookup, which is how the tests
    exercise this without owning a real share.
    """
    text = mounts_text if mounts_text is not None else read_mounts(mounts_file)
    if not text:
        return None

    target = os.path.abspath(str(path))
    best: MountInfo | None = None
    best_key: tuple[int, int] = (-1, -1)
    for row in parse_mounts(text):
        point = row.mountpoint.rstrip("/") or "/"
        if not (target == point or target.startswith(point + os.sep) or point == "/"):
            continue
        # A network mount usually sits on top of an `autofs` trigger with the
        # same mount point (/proc/mounts lists both).  Score by depth first, and
        # treat a real filesystem as deeper than its trigger; on a full tie the
        # later row wins, which is the order /proc/mounts uses for stacked
        # mounts.
        depth = len(point)
        real = 0 if row.fstype == "autofs" else 1
        key = (depth, real)
        if key >= best_key:
            best, best_key = row, key
    return best


def is_unc(path: str | os.PathLike) -> bool:
    """True for a Windows UNC path such as ``\\\\server\\share``."""
    return str(path).startswith(("\\\\", "//"))


def describe_path(path: str | os.PathLike, *, mounts_text: str | None = None) -> dict:
    """Summarise where a path lives — used by ``status`` and ``doctor``."""
    info = mount_for(path, mounts_text=mounts_text)
    return {
        "path": str(path),
        "platform": sys.platform,
        "unc": is_unc(path),
        "mountpoint": info.mountpoint if info else "",
        "fstype": info.fstype if info else ("unc" if is_unc(path) else "local"),
        "device": info.server if info else "",
        "network": bool(info and info.is_network) or is_unc(path),
        "readonly": bool(info and info.readonly),
    }


def stale_hint(exc: OSError) -> str:
    """A sentence an operator can act on, for a network-mount failure."""
    errno = getattr(exc, "errno", None)
    text = str(exc).lower()
    if errno == 116 or "stale" in text:  # ESTALE
        return (
            "the share looks stale (ESTALE): the mount lost its session. "
            "Remount it (`sudo mount -a`) or fall back with DSH_KB_SOURCE=webdav"
        )
    if errno in (5, 13) or "permission denied" in text:
        return (
            "permission denied on the share: it may be mounted read-only, the account "
            "behind the mount may lack write access there, or the calling process is "
            "sandboxed (a sandboxed agent cannot write outside its own workspace)"
        )
    if errno in (2, 6, 19) or "no such device" in text:
        return "the share is not reachable (device/host gone): check that the NAS is up and mounted"
    return f"{type(exc).__name__}: {exc}"
