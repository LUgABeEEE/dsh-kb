"""``records/v1`` — dsh-kb's own neutral JSON, and the shared item coercion.

The shape is documented in ``docs/import.md``: a ``version``/``kind`` envelope
around a flat ``items`` list.  A bare JSON array is accepted too, because that is
what most tools export and there is no ambiguity about what it means.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import Record

#: Field names different tools use for the text of one memory/note.
CONTENT_KEYS = ("content", "text", "memory", "body", "note")

#: Timestamp field names, in the order we prefer them.
TIME_KEYS = ("updated_at", "updatedAt", "created_at", "createdAt", "timestamp")


def coerce(item: dict, *, default_type: str = "note", origin: str = "",
           source: str = "") -> Record | None:
    """Turn one loosely-shaped object into a :class:`Record` (or ``None``).

    Unknown fields are ignored on purpose: an export that grows a new column
    must not start failing, and a record with no text is not knowledge.
    """
    if not isinstance(item, dict):
        return None
    content = ""
    for key in CONTENT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            content = value.strip()
            break
    if not content:
        return None

    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    tags = item.get("tags") or item.get("categories") or meta.get("tags") \
        or meta.get("categories") or ()
    if isinstance(tags, str):
        tags = (tags,)
    elif isinstance(tags, (list, tuple, set)):
        tags = tuple(tags)
    else:
        tags = ()

    updated = ""
    for key in TIME_KEYS:
        value = item.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            updated = str(value)
            break

    return Record(
        content=content,
        id=str(item.get("id") or item.get("uuid") or ""),
        tags=tuple(str(t).strip() for t in tags if str(t).strip()),
        type=str(item.get("type") or meta.get("type") or default_type),
        updated_at=updated,
        origin=str(item.get("origin") or meta.get("origin") or origin),
        source=source,
    )


def load(path: Path, **_options) -> list[Record]:
    """Read a ``records/v1`` document (or a bare list of items)."""
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc

    if isinstance(data, list):
        items, generator = data, "records/v1"
    elif isinstance(data, dict):
        version = data.get("version")
        if version is not None and version != 1:
            raise ValueError(f"{path}: unsupported records version {version!r} (this build reads 1)")
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError(f"{path}: expected an 'items' list")
        generator = str(data.get("generator") or "records/v1")
    else:
        raise ValueError(f"{path}: expected a JSON object or array")

    out: list[Record] = []
    for item in items:
        record = coerce(item, origin=generator, source=generator)
        if record is not None:
            out.append(record)
    return out
