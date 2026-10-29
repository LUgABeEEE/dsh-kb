"""JSON Lines — the lowest common denominator of "export everything".

One JSON object per line; blank lines and ``#`` comments are skipped so a file
can carry a header.  A line that is a bare string becomes the content, which
covers ``{"content": ...}``-less exports without inventing a second format.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import Record
from .records import coerce


def load(path: Path, *, default_type: str = "note", origin: str = "jsonl",
         **_options) -> list[Record]:
    out: list[Record] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
        if isinstance(item, str):
            item = {"content": item}
        record = coerce(item, default_type=default_type, origin=origin,
                        source=f"{path.name}:{number}")
        if record is not None:
            out.append(record)
    return out
