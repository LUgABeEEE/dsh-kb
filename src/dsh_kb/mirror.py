"""Mirror: a local, read-only copy of the authoritative bundle.

The share is the **single source of truth**; the mirror is a derived cache.
That asymmetry is deliberate:

* there is never a merge conflict, because the mirror is never written to as
  an authority (proposals go to the share, not to the mirror);
* retrieval costs nothing when the share is offline;
* refreshing is a pure function of ``(remote listing, manifest)``, so it is
  cheap to run often and safe to interrupt.

Incrementality comes from ``<home>/state/manifest.json``: one record per file
holding size and mtime.  A file whose size *and* mtime are unchanged is not
downloaded at all — no bytes cross the network or the mount.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Config
from .sources import Entry, FilesystemSource, WebdavSource

MANIFEST_SCHEMA = 1


@dataclass
class SyncResult:
    source: str = ""
    dirs: int = 0
    files: int = 0
    fetched: int = 0
    skipped: int = 0
    excluded: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duration: float = 0.0
    changed: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [
            f"{self.dirs} dirs / {self.files} files",
            f"fetched {self.fetched}",
            f"unchanged {self.skipped}",
        ]
        if self.excluded:
            parts.append(f"excluded {len(self.excluded)}")
        if self.removed:
            parts.append(f"pruned {len(self.removed)}")
        parts.append(f"{self.duration:.2f}s")
        return "；".join(parts)


def load_manifest(cfg: Config) -> dict:
    """Read the manifest, tolerating absence or corruption."""
    path = cfg.manifest_path
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("files"), dict):
                return data
        except (OSError, ValueError):
            pass
    return {"schema": MANIFEST_SCHEMA, "root": "", "files": {}, "lastSync": None, "source": ""}


def manifest_problem(cfg: Config) -> str | None:
    """Why the manifest is unusable, or ``None`` when it is fine (or absent).

    ``load_manifest`` deliberately tolerates corruption — sync must keep working
    with a broken snapshot.  A *reporting* caller is the opposite case: silently
    substituting an empty manifest would turn every ``diff`` into "the whole
    bundle is new", so it needs to say out loud what is wrong.
    """
    path = cfg.manifest_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return f"cannot read {path}: {exc}"
    except ValueError as exc:
        return f"{path} is not valid JSON ({exc}) — re-run `dsh-kb sync` to rebuild it"
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        return f"{path} has an unexpected shape — re-run `dsh-kb sync` to rebuild it"
    return None


def save_manifest(cfg: Config, manifest: dict) -> None:
    """Write the manifest atomically (temp file + rename)."""
    path = cfg.manifest_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def manifest_age(cfg: Config) -> float | None:
    """Seconds since the last successful sync, or ``None`` if never synced."""
    stamp = load_manifest(cfg).get("lastSync")
    if not stamp:
        return None
    try:
        from datetime import datetime

        return max(0.0, time.time() - datetime.fromisoformat(stamp).timestamp())
    except ValueError:
        return None


def sync(
    cfg: Config,
    source: FilesystemSource | WebdavSource,
    *,
    prune: bool = False,
    full: bool = False,
    on_event=None,
) -> SyncResult:
    """Refresh the mirror from ``source``.

    ``prune=True`` deletes local files that disappeared remotely.  It is off by
    default: a transient listing failure must never be able to delete data.

    ``full=True`` ignores the manifest and re-fetches everything.  Needed for
    the one case incremental sync cannot see: on a share whose timestamps have
    one-second granularity, an edit that lands in the same second *and* keeps
    the byte size identical is indistinguishable from "unchanged" (measured on
    a consumer NAS).
    """
    emit = on_event or (lambda _msg: None)
    started = time.perf_counter()
    result = SyncResult(source=getattr(source, "label", "?"))

    manifest = load_manifest(cfg)
    known: dict = {} if full else manifest.get("files", {})

    emit(f"listing {getattr(source, 'label', '?')} …")
    entries: list[Entry] = source.walk()
    dirs = [e for e in entries if e.is_dir]
    files = [e for e in entries if not e.is_dir]
    result.dirs, result.files = len(dirs), len(files)

    for entry in dirs:
        if cfg.mirror_allows(entry.rel):
            (cfg.mirror_dir / entry.rel).mkdir(parents=True, exist_ok=True)

    for entry in files:
        rel = entry.rel
        if not cfg.mirror_allows(rel):
            result.excluded.append(rel)
            continue

        previous = known.get(rel)
        target = cfg.mirror_dir / rel
        unchanged = (
            previous is not None
            and previous.get("size") == entry.size
            and abs(_epoch(previous.get("mtime")) - entry.mtime) < cfg.mtime_tolerance
            and target.exists()
        )
        if unchanged:
            result.skipped += 1
            continue

        try:
            payload = source.read(rel)
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the sync
            emit(f"WARN read failed {rel}: {type(exc).__name__}: {exc}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, target)

        known[rel] = {"size": entry.size, "mtime": entry.mtime, "source": result.source}
        result.fetched += 1
        result.changed.append(rel)

    if prune:
        remote = {e.rel for e in files if cfg.mirror_allows(e.rel)}
        for rel in [r for r in known if r not in remote]:
            victim = cfg.mirror_dir / rel
            if victim.exists():
                victim.unlink()
            known.pop(rel, None)
            result.removed.append(rel)

    manifest.update(
        {
            "schema": MANIFEST_SCHEMA,
            "root": str(getattr(source, "root", "") or ""),
            "source": result.source,
            "files": known,
            "lastSync": _now_iso(),
        }
    )
    save_manifest(cfg, manifest)
    result.duration = time.perf_counter() - started
    return result


def _epoch(value) -> float:
    """Coerce a manifest mtime to epoch seconds; unknown shapes mean "changed"."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _now_iso() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat(timespec="seconds")


def mirror_files(cfg: Config) -> list[Path]:
    """Every regular file currently in the mirror."""
    if not cfg.mirror_dir.is_dir():
        return []
    return sorted(p for p in cfg.mirror_dir.rglob("*") if p.is_file())


def mirror_size(cfg: Config) -> int:
    return sum(p.stat().st_size for p in mirror_files(cfg))


def manifest_as_dict(cfg: Config) -> dict:
    data = asdict(SyncResult())  # keeps the dataclass shape stable for callers
    data.update(load_manifest(cfg))
    return data


# --------------------------------------------------------------------------- #
# diff: what does the source look like next to the last snapshot?
# --------------------------------------------------------------------------- #
@dataclass
class FileChange:
    """One entry of a snapshot comparison.

    ``size``/``mtime`` describe the **current** (remote) state, ``previous_*``
    the value recorded in the manifest.  ``added`` has no previous value,
    ``removed`` has no current one — keeping both sides nullable means the
    caller never has to guess which half is missing.
    """

    rel: str
    size: int | None = None
    previous_size: int | None = None
    mtime: float | None = None
    previous_mtime: float | None = None

    @property
    def size_delta(self) -> int | None:
        if self.size is None or self.previous_size is None:
            return None
        return self.size - self.previous_size

    @property
    def mtime_delta(self) -> float | None:
        if self.mtime is None or self.previous_mtime is None:
            return None
        return self.mtime - self.previous_mtime

    def as_dict(self) -> dict:
        return {
            "path": self.rel,
            "size": self.size,
            "previous_size": self.previous_size,
            "mtime": self.mtime,
            "previous_mtime": self.previous_mtime,
            "size_delta": self.size_delta,
            "mtime_delta": self.mtime_delta,
        }


@dataclass
class ManifestDiff:
    added: list[FileChange] = field(default_factory=list)
    removed: list[FileChange] = field(default_factory=list)
    changed: list[FileChange] = field(default_factory=list)

    @property
    def counts(self) -> dict:
        return {"added": len(self.added), "removed": len(self.removed),
                "changed": len(self.changed)}

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def as_dict(self) -> dict:
        return {
            "added": [c.as_dict() for c in self.added],
            "removed": [c.as_dict() for c in self.removed],
            "changed": [c.as_dict() for c in self.changed],
            "counts": self.counts,
        }


def diff_entries(
    known: dict,
    entries,
    *,
    mtime_tolerance: float = 1.0,
) -> ManifestDiff:
    """Compare a manifest snapshot with a fresh listing — **pure function**.

    No disk, no network, no ``Config``: the caller filters policy (``mirror_allows``)
    and passes in both sides, so the comparison itself can be unit-tested with
    two dicts and a handful of :class:`Entry` objects.

    "Changed" follows the same size/mtime rule ``sync`` uses to decide whether a
    file has to be downloaded again, **relative to the snapshot**: different
    size, or mtime moved by at least ``mtime_tolerance`` (network shares truncate
    timestamps to a second, hence the tolerance).  A file whose mtime alone moved
    is still reported as changed — it is the one signal that the content *may*
    differ while the size stayed the same.

    It is a **subset**, not the whole set of what ``sync`` will do.  ``sync``
    also re-fetches a file whose local copy is missing (it checks
    ``target.exists()``), and it fetches files that are new in the listing; this
    comparison can see neither, because the manifest and the source are
    unchanged in that scenario.  That half of the truth lives in
    :func:`diff_mirror` — the two functions exist because the directions have
    different causes and different fixes.  (Measured on 2026-10-19: deleting
    ``mirror/z0-public/pic.png`` left every counter here at 0 while ``sync``
    immediately reported ``fetched=1``.)
    """
    remote: dict[str, Entry] = {}
    for entry in entries:
        if getattr(entry, "is_dir", False):
            continue
        remote[entry.rel] = entry

    out = ManifestDiff()
    for rel in sorted(remote):
        entry = remote[rel]
        previous = known.get(rel)
        if previous is None:
            out.added.append(FileChange(rel=rel, size=entry.size, mtime=entry.mtime))
            continue
        previous_size = previous.get("size")
        previous_mtime = _epoch(previous.get("mtime"))
        if (previous_size != entry.size
                or abs(previous_mtime - entry.mtime) >= mtime_tolerance):
            out.changed.append(FileChange(
                rel=rel, size=entry.size, previous_size=previous_size,
                mtime=entry.mtime, previous_mtime=previous_mtime or None,
            ))
    for rel in sorted(set(known) - set(remote)):
        previous = known[rel]
        out.removed.append(FileChange(
            rel=rel, previous_size=previous.get("size"),
            previous_mtime=_epoch(previous.get("mtime")) or None,
        ))
    return out


@dataclass
class MirrorDiff:
    """How the local mirror compares with the snapshot (and the source).

    Same item type as :class:`ManifestDiff` (:class:`FileChange`) so a caller
    can format both reports with one piece of code: ``size`` is the local size,
    ``previous_size`` the size recorded in the manifest.

    ``added``   in the mirror, but neither the snapshot nor the source knows it
    ``removed`` in the snapshot, missing from the mirror (sync would re-fetch it)
    ``changed`` in both, but the local size differs from the recorded size

    ``remote_checked`` says whether the source listing was available: ``added``
    can only be decided with it (see :func:`diff_mirror`).
    """

    added: list[FileChange] = field(default_factory=list)
    removed: list[FileChange] = field(default_factory=list)
    changed: list[FileChange] = field(default_factory=list)
    remote_checked: bool = False

    @property
    def counts(self) -> dict:
        return {"added": len(self.added), "removed": len(self.removed),
                "changed": len(self.changed)}

    @property
    def empty(self) -> bool:
        return not (self.added or self.removed or self.changed)

    def as_dict(self) -> dict:
        return {
            "added": [c.as_dict() for c in self.added],
            "removed": [c.as_dict() for c in self.removed],
            "changed": [c.as_dict() for c in self.changed],
            "counts": self.counts,
            "remote_checked": self.remote_checked,
        }


def diff_mirror(
    known: dict,
    local: dict,
    *,
    remote: set | None = None,
) -> MirrorDiff:
    """Compare the local mirror against the manifest snapshot — **pure function**.

    ``known`` is the manifest record (``rel -> {"size", "mtime", ...}``),
    ``local`` maps every file actually in the mirror to its size in bytes, and
    ``remote`` — when the source is reachable — is the set of paths the source
    lists.  The caller filters all three through the zone/``never`` policy first,
    so this function never has to know about ``Config``.

    ``removed`` (the direction that used to be invisible): the snapshot recorded
    a fetched file and the mirror no longer has it.  A file the *source* also
    dropped is excluded — that is a stale snapshot (``sync --prune`` will tidy
    it), not a damaged mirror.

    ``changed``: both sides present, local size differs from the recorded size.
    ``mtime`` is deliberately not compared: a mirror file's mtime is the moment
    it was written locally, while the manifest stores the *source's* mtime, so
    any comparison would be noise (this is also why ``sync`` compares mtimes
    against the manifest, never against the mirror).

    ``added``: in the mirror, but neither the snapshot nor the source knows it —
    only computed when ``remote`` is given.  Without a source listing a file
    written by an interrupted ``sync`` (the manifest is saved last) would be
    misreported as local-only, and a file added remotely but not yet fetched
    would be indistinguishable from it.
    """
    known_paths, local_paths = set(known), set(local)
    remote_paths = set(remote) if remote is not None else None
    out = MirrorDiff(remote_checked=remote_paths is not None)

    for rel in sorted(known_paths - local_paths):
        if remote_paths is not None and rel not in remote_paths:
            continue        # 源上也没有了：快照过期，不是本地丢文件
        previous = known.get(rel) or {}
        out.removed.append(FileChange(
            rel=rel,
            previous_size=previous.get("size"),
            previous_mtime=_epoch(previous.get("mtime")) or None,
        ))

    for rel in sorted(known_paths & local_paths):
        previous = known.get(rel) or {}
        try:
            snapshot_size = int(previous.get("size"))
        except (TypeError, ValueError):
            continue        # 手改过的 manifest 没有可信大小：不猜
        current_size = local.get(rel)
        if not isinstance(current_size, int) or isinstance(current_size, bool):
            continue
        if snapshot_size != current_size:
            out.changed.append(FileChange(
                rel=rel, size=current_size, previous_size=snapshot_size,
            ))

    if remote_paths is not None:
        for rel in sorted(local_paths - known_paths - remote_paths):
            size = local.get(rel)
            out.added.append(FileChange(
                rel=rel, size=size if isinstance(size, int) else None,
            ))
    return out
