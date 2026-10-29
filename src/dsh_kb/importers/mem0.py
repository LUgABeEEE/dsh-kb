"""mem0 exports.

mem0's ``get_all()`` returns either a bare list of memory objects or
``{"results": [...]}`` depending on version; each item carries ``memory`` (the
text), ``id``, ``metadata`` and timestamps.  Both shapes are accepted, and the
per-item fields go through the shared coercion so a metadata change does not
break the import.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import Record
from .records import coerce


def load(path: Path, *, default_type: str = "note", **_options) -> list[Record]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc

    if isinstance(data, dict):
        items = data.get("results")
        if items is None:
            items = data.get("memories")
        if items is None:
            items = [data]                      # a single memory object
    else:
        items = data
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a list of memories or {{'results': [...]}}")

    out: list[Record] = []
    for item in items:
        record = coerce(item, default_type=default_type, origin="mem0", source="mem0")
        if record is not None:
            out.append(record)
    return out
