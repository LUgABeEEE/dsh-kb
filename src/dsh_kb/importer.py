"""Render records into knowledge-base pages, then write them (or not).

The split is deliberate: adapters (``dsh_kb.importers``) only read, this module
only decides *where* a page goes and *what it looks like*, and the source object
only writes.  That is what lets ``import`` stay testable without a share, and it
is why nothing here knows the name of any particular memory product.

Guarantees, in order of how much they matter:

1. **Deterministic** — the same records always render the same bytes, so a
   re-import is a no-op and a diff means something.
2. **Additive** — ``import`` never deletes.  Removing pages stays with
   ``sync --prune``, where a human is in the loop.
3. **Zone-guarded** — a zone has to be named, and writing into a public zone
   needs ``allow_public``.  Personal memory must not reach ``z0`` by accident.
4. **Fail loudly** — a path that the ``never`` list excludes aborts the whole
   run instead of skipping a few files silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .importers import Record

#: Characters that cannot appear in a path on Windows, plus whitespace.
_UNSAFE = re.compile(r'[\\/:*?"<>|\s]+')

#: Fallback group when a record has no tags at all.
UNTAGGED = "untagged"


@dataclass(frozen=True)
class Page:
    rel: str
    content: str
    records: int


@dataclass(frozen=True)
class Plan:
    pages: tuple[Page, ...]
    items: int
    zone: str
    prefix: str


@dataclass(frozen=True)
class Outcome:
    new: int
    changed: int
    same: int
    failed: int
    written: int


def slug(text: str, limit: int = 48) -> str:
    """A path-safe, case-folded file name stem.

    Case folding is not cosmetic: the bundle is served to Windows over UNC,
    where ``Docker.md`` and ``docker.md`` are the same file and the second one
    wins silently.  Non-ASCII is kept (CJK tags stay readable) — Windows handles
    those fine, it is only case that collides.
    """
    cleaned = _UNSAFE.sub("-", str(text or "").strip()).strip("-.")
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch in "-_.")
    cleaned = cleaned.lower().strip("-.")
    for suffix in (".markdown", ".md"):
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)]
    return (cleaned[:limit].rstrip("-.")) or "page"


def _group_of(record: Record, group_by: str) -> str:
    if group_by == "first-tag":
        return record.tags[0] if record.tags else UNTAGGED
    if group_by == "none":
        return slug(record.id or record.content[:40]) or "page"
    if group_by.startswith("field:"):
        name = group_by.split(":", 1)[1].strip()
        value = getattr(record, name, "")
        if isinstance(value, (tuple, list)):
            value = value[0] if value else ""
        return str(value) or UNTAGGED
    raise ValueError(f"unknown group-by {group_by!r}; use first-tag, none or field:<name>")


def _date(value: str) -> str:
    """``2026-10-28T23:35:00+08:00`` → ``2026-10-28``; anything else → ``unknown``."""
    text = (value or "").strip()
    if not text:
        return "unknown"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else "unknown"


def _latest(records: list[Record]) -> str:
    stamps = sorted((_date(r.updated_at) for r in records if _date(r.updated_at) != "unknown"))
    return stamps[-1] if stamps else "unknown"


def _front_matter(*, type_: str, title: str, description: str, tags: list[str],
                  zone: str, generator: str, at: str, resource: str) -> str:
    quoted = "[" + ", ".join(f'"{tag}"' for tag in tags) + "]"
    lines = [
        "---",
        f"type: {type_}",
        f'title: "{title}"',
        f'description: "{description}"',
        f"tags: {quoted}",
        f"zone: {zone.split('-')[0]}",
        "status: stable",
        (f"generated: {{ by: {generator}, at: {at} }}" if at
         else f"generated: {{ by: {generator} }}"),
        "sources:",
        f"  - id: {generator}",
        f"    resource: {resource}",
        "---",
    ]
    return "\n".join(lines) + "\n"


def render(records: list[Record], *, title: str, generator: str, resource: str,
           zone: str, at: str) -> str:
    """One page: front matter, a provenance line, then one section per record."""
    ordered = sorted(records, key=lambda r: (_date(r.updated_at), r.id, r.content[:40]))
    types = sorted({r.type for r in ordered})
    tags: list[str] = []
    for record in ordered:
        for tag in record.tags:
            if tag not in tags:
                tags.append(tag)
    for type_ in types:
        if type_ not in tags:
            tags.append(type_)

    count = f"{len(ordered)} record" + ("" if len(ordered) == 1 else "s")
    description = f"{count} ({'/'.join(types)}), latest {_latest(ordered)}"
    parts = [
        _front_matter(type_=types[0], title=title, description=description, tags=tags,
                      zone=zone, generator=generator, at=at, resource=resource),
        f"\n# {title}\n",
        f"\n> Imported by `{generator}` from `{resource}` — {len(ordered)} records."
        " Each entry carries its id, so it can be traced back to the source.\n",
    ]
    for record in ordered:
        heading = record.title or record.type
        parts.append(f"\n## {heading} · {_date(record.updated_at)}\n")
        parts.append(record.content.strip() + "\n")
        meta = [f"id: `{record.id}`" if record.id else "", f"type: {record.type}"]
        if record.origin:
            meta.append(f"origin: {record.origin}")
        if record.tags:
            meta.append("tags: " + ", ".join(record.tags))
        parts.append(f"\n<sub>{' · '.join(item for item in meta if item)}</sub>\n")
    return "".join(parts)


def plan(cfg: Config, records: list[Record], *, prefix: str, zone: str,
         group_by: str = "first-tag", title_prefix: str = "", generator: str = "",
         resource: str = "", at: str = "", max_page_bytes: int = 2_000_000,
         allow_public: bool = False) -> Plan:
    """Decide the pages.  Raises ``ValueError`` rather than skipping silently."""
    if not records:
        raise ValueError("no records to import (the adapter returned nothing)")
    known = {z.name for z in cfg.zones}
    if zone not in known:
        raise ValueError(f"zone {zone!r} is not configured; configured zones: {', '.join(sorted(known))}")
    if zone.startswith("z0") and not allow_public:
        raise ValueError(f"refusing to write into {zone!r} without --allow-public")

    prefix = prefix.strip("/")
    buckets: dict[str, list[Record]] = {}
    for record in records:
        buckets.setdefault(_group_of(record, group_by), []).append(record)

    pages: list[Page] = []
    blocked: list[str] = []
    for group, group_records in sorted(buckets.items()):
        rel = f"{prefix}/{slug(group)}.md" if prefix else f"{slug(group)}.md"
        if cfg.is_never(rel):
            blocked.append(rel)
            continue
        title = f"{title_prefix}{group}".strip()
        body = render(group_records, title=title, generator=generator or "import",
                      resource=resource, zone=zone, at=at)
        if len(body.encode("utf-8")) > max_page_bytes:
            raise ValueError(
                f"{rel}: rendered page is {len(body.encode('utf-8'))} bytes, above "
                f"--max-page-bytes {max_page_bytes}; group by a narrower key"
            )
        pages.append(Page(rel=rel, content=body, records=len(group_records)))

    if blocked:
        raise ValueError(
            "refusing to import into paths the 'never' list excludes: " + ", ".join(blocked)
        )
    return Plan(pages=tuple(pages), items=len(records), zone=zone, prefix=prefix)


def apply(source, plan_: Plan, *, write: bool = False) -> Outcome:
    """Compare the plan against the source, and write when asked to."""
    existing = {entry.rel for entry in source.walk() if not entry.is_dir}
    new = changed = same = failed = written = 0
    for page in plan_.pages:
        if page.rel not in existing:
            new += 1
            if write:
                source.write(page.rel, page.content.encode("utf-8"))
                written += 1
            continue
        try:
            current = source.read(page.rel).decode("utf-8", "replace")
        except Exception:                       # noqa: BLE001 - one page must not abort the run
            failed += 1
            continue
        if current == page.content:
            same += 1
            continue
        changed += 1
        if write:
            source.write(page.rel, page.content.encode("utf-8"))
            written += 1
    return Outcome(new=new, changed=changed, same=same, failed=failed, written=written)
