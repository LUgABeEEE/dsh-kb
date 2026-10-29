"""Obsidian vault: one record per note.

An Obsidian vault is already Markdown, so the only real work is convention:
skip ``.obsidian/`` and ``.trash/``, read the YAML front matter, and reduce
``[[wikilinks]]`` to their label.
"""

from __future__ import annotations

from pathlib import Path

from . import Record
from .markdown import iter_markdown, page_record


def load(path: Path, *, wikilinks: str = "strip", default_type: str = "note",
         **_options) -> list[Record]:
    out: list[Record] = []
    for page in iter_markdown(path):
        record = page_record(page, path, default_type=default_type,
                             wikilinks=wikilinks, origin="obsidian")
        if record is not None:
            out.append(record)
    return out
