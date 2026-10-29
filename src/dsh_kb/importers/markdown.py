"""Shared reader for Markdown vaults.

Obsidian, Notion's Markdown export and basic-memory all hand you a directory of
``.md`` files whose front matter and headings differ only in convention.  One
reader handles the conventions as options instead of three near-copies:

* front matter is read with :func:`dsh_kb.text.parse_frontmatter` (the same
  zero-dependency YAML subset the rest of the project uses);
* the title is the front matter ``title``, else the first ``#`` heading, else
  the file name;
* ``[[wikilinks]]`` are reduced to their label by default — dsh-kb indexes plain
  text, and ``[[target|label]]`` would otherwise be indexed as markup;
* Notion appends a 32-hex page id to file names; ``strip_hash`` removes it.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from . import Record
from ..text import parse_frontmatter, split_frontmatter

#: Directories that hold application state rather than notes.
SKIP_DIRS = frozenset({
    ".obsidian", ".trash", ".git", ".stfolder", ".smart-env", ".obsidian-cache",
    "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
})

_WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_NOTION_ID = re.compile(r"\s+[0-9a-f]{32}$")
_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def iter_markdown(root: Path, *, skip: frozenset[str] = SKIP_DIRS):
    """Yield ``.md`` files under ``root``, skipping application-state dirs."""
    root = Path(root)
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if any(part in skip or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if path.name.startswith("."):
            continue
        yield path


def strip_wikilinks(text: str) -> str:
    """``[[page|label]]`` → ``label``; ``[[page]]`` → ``page``."""
    return _WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)


def one(value) -> str:
    """Scalar-ish access for a front matter value that may be a list."""
    if isinstance(value, (list, tuple)):
        return str(value[0]).strip().strip("'\"") if value else ""
    if isinstance(value, dict):
        return ""
    return str(value).strip().strip("'\"") if value is not None else ""


def tags_of(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in re.split(r"[,;]", value) if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _stem(path: Path, *, strip_hash: bool) -> str:
    stem = path.stem
    return _NOTION_ID.sub("", stem).strip() if strip_hash else stem.strip()


def _first_heading(body: str) -> str:
    match = _HEADING.search(body)
    return match.group(1).strip() if match else ""


def _drop_leading_heading(body: str, title: str) -> str:
    """Remove the page's own H1 when it just repeats the title we render anyway."""
    match = _HEADING.match(body)
    if match and match.group(1).strip().casefold() == title.strip().casefold():
        return body[match.end():].lstrip("\n")
    return body


def _updated_at(path: Path, front: dict) -> str:
    for key in ("updated", "updated_at", "created", "created_at", "date"):
        value = one(front.get(key))
        if value:
            return value
    generated = front.get("generated")
    if isinstance(generated, dict):
        value = one(generated.get("at"))
        if value:
            return value
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except OSError:
        return ""


def page_record(path: Path, root: Path, *, default_type: str = "note",
                wikilinks: str = "strip", strip_hash: bool = False,
                origin: str = "") -> Record | None:
    """One record per Markdown page; ``None`` when the page has no body."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    front = parse_frontmatter(raw)
    body, _ = split_frontmatter(raw)
    title = one(front.get("title")) or _first_heading(body) or _stem(path, strip_hash=strip_hash)
    body = _drop_leading_heading(body.strip(), title)
    if wikilinks == "strip":
        body = strip_wikilinks(body)
    body = body.strip()
    if not body:
        return None
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = path.name
    return Record(
        content=body,
        title=title,
        id=rel,
        tags=tags_of(front.get("tags")),
        type=one(front.get("type")) or default_type,
        updated_at=_updated_at(path, front),
        origin=origin or "markdown",
        source=rel,
    )
