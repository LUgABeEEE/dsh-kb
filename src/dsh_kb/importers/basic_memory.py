"""basic-memory notes.

basic-memory stores a note as Markdown with ``title``/``type``/``permalink`` in
the front matter and the knowledge itself as observation bullets:

    - [decision] we chose SQLite over a server
    - [fact] the corpus is 1.2 GB

The observation is the unit worth indexing, not the page — one note usually
holds several unrelated statements.  When a page has no observations at all it
falls back to the whole body, so plain notes still import.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import Record
from .markdown import iter_markdown, one, page_record, strip_wikilinks, tags_of
from ..text import parse_frontmatter, split_frontmatter

_OBSERVATION = re.compile(r"^\s*[-*]\s*\[([^\]]+)\]\s*(.+?)\s*$", re.MULTILINE)


def load(path: Path, *, wikilinks: str = "strip", default_type: str = "note",
         **_options) -> list[Record]:
    out: list[Record] = []
    for page in iter_markdown(path):
        try:
            raw = page.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        front = parse_frontmatter(raw)
        body, _ = split_frontmatter(raw)
        if wikilinks == "strip":
            body = strip_wikilinks(body)
        try:
            rel = str(page.relative_to(path))
        except ValueError:
            rel = page.name

        observations = _OBSERVATION.findall(body)
        if not observations:
            record = page_record(page, path, default_type=default_type,
                                 wikilinks=wikilinks, origin="basic-memory")
            if record is not None:
                out.append(record)
            continue

        tags = tags_of(front.get("tags"))
        page_type = one(front.get("type")) or default_type
        for index, (category, text) in enumerate(observations, 1):
            text = text.strip()
            if not text:
                continue
            out.append(Record(
                content=text,
                title=one(front.get("title")) or page.stem,
                id=f"{rel}#{index}",
                tags=tags,
                type=category.strip() or page_type,
                updated_at=one(front.get("updated")) or one(front.get("created")),
                origin="basic-memory",
                source=rel,
            ))
    return out
