"""Adapters that turn somebody else's export into dsh-kb records.

Every adapter is a function ``load(path, **options) -> list[Record]`` that reads
one file or directory and returns plain records.  Nothing here touches the
network, imports a third-party package, or knows where the knowledge base lives:
the renderer in :mod:`dsh_kb.importer` decides that.

Adding an adapter is one module plus one entry in :data:`ADAPTERS`.  The rule for
accepting one is the same everywhere: if the format cannot be read with the
standard library and a documented shape, it does not belong here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    """One unit of knowledge, before records are grouped into a page."""

    content: str
    #: The note's own title when the source has one (vault pages, basic-memory).
    title: str = ""
    id: str = ""
    tags: tuple[str, ...] = ()
    type: str = "note"
    updated_at: str = ""
    origin: str = ""
    #: Provenance: a relative path inside the export, or an upstream identifier.
    source: str = ""


#: adapter name -> (what it reads, module name)
ADAPTERS: dict[str, tuple[str, str]] = {
    "records": ("dsh-kb's own records/v1 JSON (the neutral format)", "records"),
    "jsonl": ("one JSON object per line (content/text/memory/body)", "jsonl"),
    "mem0": ("a mem0 memory export (JSON array or {\"results\": [...]})", "mem0"),
    "obsidian": ("an Obsidian vault (Markdown + YAML front matter)", "obsidian"),
    "notion": ("a Notion \"Markdown & CSV\" export directory", "notion"),
    "basic-memory": ("basic-memory notes (observations + relations)", "basic_memory"),
}


def names() -> list[str]:
    return sorted(ADAPTERS)


def describe() -> dict[str, str]:
    return {name: description for name, (description, _) in ADAPTERS.items()}


def load(name: str, path: Path, **options) -> list[Record]:
    """Run one adapter over ``path``.

    Raises ``ValueError`` for an unknown adapter or an input that does not match
    the shape it claims to read — silently importing nothing is the one outcome
    worse than failing.
    """
    if name not in ADAPTERS:
        raise ValueError(f"unknown format {name!r}; known formats: {', '.join(names())}")
    target = Path(path)
    if not target.exists():
        raise ValueError(f"no such file or directory: {target}")
    module = importlib.import_module(f"dsh_kb.importers.{ADAPTERS[name][1]}")
    return module.load(target, **options)
