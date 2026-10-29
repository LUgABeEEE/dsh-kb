# Importing from another tool

`dsh-kb import` turns somebody else's export into pages inside the bundle. It is
the inverse of `search`: one command, one input file or directory, deterministic
output, and **dry-run by default**.

```bash
# what would change? (the default — nothing is written)
dsh-kb import --from ~/notes/obsidian --format obsidian --prefix notes

# write it, then pick the new pages up
dsh-kb import --from ~/notes/obsidian --format obsidian --prefix notes --write
dsh-kb refresh
```

## Formats

| `--format` | Reads | Typical source |
|---|---|---|
| `records` | dsh-kb's own `records/v1` JSON (below) | your own exporter |
| `jsonl` | one JSON object per line (`content`/`text`/`memory`/`body`) | any log-shaped export |
| `mem0` | a mem0 export: JSON array or `{"results": [...]}` | `mem0.get_all()` |
| `obsidian` | a vault: Markdown + YAML front matter | an Obsidian folder |
| `notion` | a "Markdown & CSV" export directory | Notion → Export |
| `basic-memory` | notes whose body is `- [category] observation` bullets | basic-memory |

The list is meant to grow; each adapter is one module under
`src/dsh_kb/importers/` plus one entry in its `ADAPTERS` table. If a format
cannot be read with the standard library and a documented shape, it does not
belong here — that is the same rule as the rest of the project.

**Logseq** users: a Logseq graph is Markdown, so `--format obsidian` reads it;
the `property:: value` lines stay in the body as text.

## `records/v1`

The neutral format everything else converts into. It is deliberately boring:

```json
{
  "version": 1,
  "kind": "records",
  "generator": "my-exporter/1",
  "generated_at": "2026-10-29T12:00:00+08:00",
  "items": [
    {
      "id": "677e04c9",
      "content": "the text, in Markdown",
      "tags": ["dsh-kb", "git"],
      "type": "fact",
      "updated_at": "2026-10-28T23:35:00+08:00",
      "origin": "auto"
    }
  ]
}
```

- `content` is required; an item with no text is dropped rather than rendered empty.
- `id`, `tags`, `type`, `updated_at`, `origin` are optional; unknown fields are
  ignored, and `version` must be `1`.
- A bare JSON array is accepted too — it means the same thing.

## What gets written

One page per group, where the group is chosen by `--group-by`:

| Value | Group key |
|---|---|
| `first-tag` (default) | the first tag, or `untagged` |
| `none` | one page per record (keyed by id) |
| `field:<name>` | any record attribute, e.g. `field:type` or `field:source` |

The page path is `<prefix>/<slug>.md`. Slugs are **case-folded** on purpose: the
bundle is served to Windows over UNC, where `Docker.md` and `docker.md` are the
same file and one would silently overwrite the other. Non-ASCII is kept.

The page body carries front matter (`type`, `title`, `description`, `tags`,
`zone`, `status`, `generated`, `sources`), a provenance line, and one section per
record with its id — enough to trace any paragraph back to the source.

Re-importing the same input is a **no-op**: pages are compared byte for byte, and
only differences are written. The command reports `new / changed / same / failed`.

## Guard rails

- **A zone is mandatory and defaults to `z1-restricted`.** Writing into `z0-public`
  needs `--allow-public` — personal notes must not become public by a typo.
- **The `never` list wins.** If any target path matches it, the whole run aborts
  instead of skipping a few files quietly.
- **No deletion.** `import` only writes. Removing pages stays with
  `sync --prune`, where a human is in the loop.
- **No redaction.** Filtering belongs to the exporter: dsh-kb does not pretend to
  know what is private. It does promise not to amplify — no writing outside the
  prefix, no following symlinks.
- **Page size cap** (`--max-page-bytes`, default 2 MB): an oversized page is an
  error, not a page nobody can search.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | done (including "nothing changed") |
| `1` | usage or input error — bad format, bad JSON, zone refused, `never` hit |
| `10` | the source could not be read or written for some pages |
| `11` | the source is unavailable (offline share, read-only mount) |

## Writing your own adapter

```python
# src/dsh_kb/importers/mytool.py
from pathlib import Path
from . import Record
from .records import coerce

def load(path: Path, **_options) -> list[Record]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = coerce({"content": line}, origin="mytool")
        if record is not None:
            records.append(record)
    return records
```

Then add `"mytool": ("one-line description", "mytool")` to `ADAPTERS`. The
renderer, the guards and the CLI need no changes.
