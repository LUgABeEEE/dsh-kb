"""Notion "Markdown & CSV" export.

Notion writes ``Page title <32 hex>.md`` next to a ``.csv`` of the same name, and
nests database rows in one directory per database.  The CSV is redundant for
this purpose (the Markdown carries the body), so only ``.md`` is read and the
page id is stripped from the title — otherwise every import would look like a
renamed page and the grouping would key on a random hash.
"""

from __future__ import annotations

from pathlib import Path

from . import Record
from .markdown import iter_markdown, page_record


def load(path: Path, *, wikilinks: str = "strip", default_type: str = "note",
         **_options) -> list[Record]:
    out: list[Record] = []
    for page in iter_markdown(path):
        if page.suffix.lower() == ".csv":          # iter_markdown already filters
            continue
        record = page_record(page, path, default_type=default_type, wikilinks=wikilinks,
                             strip_hash=True, origin="notion")
        if record is not None:
            out.append(record)
    return out
