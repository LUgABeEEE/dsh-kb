"""Text helpers: CJK-aware tokenisation, index expansion and snippet windows.

Why the double representation
-----------------------------
SQLite's ``unicode61`` tokenizer treats a run of CJK characters as a **single**
token, so a 2-character Chinese query cannot match a 6-character run.  We
therefore index two views of every chunk:

* the **original text** — so latin words, identifiers, paths and code behave
  exactly as expected (``z2-sensitive``, ``page-000123.md``, ``fts5`` …);
* **CJK adjacent bigrams** appended after it — so ``挂载`` matches ``重新挂载``.

The query side is expanded the same way, which keeps recall symmetric.

Everything here is pure: no I/O, no globals, trivially unit-testable.
"""
from __future__ import annotations

import re

#: CJK ideographs + Japanese kana + Hangul syllables.
CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
#: Latin/digit "words" as they appear in code, paths and identifiers.
WORD_RE = re.compile(r"[A-Za-z0-9_.\-]{2,}")
_SPLIT_RE = re.compile(r"[._\-]")


def cjk_bigrams(text: str) -> list[str]:
    """All adjacent character pairs of every CJK run in ``text``."""
    grams: list[str] = []
    for run in CJK_RE.findall(text):
        grams.extend(run[i:i + 2] for i in range(len(run) - 1))
    return grams


def expand_for_index(text: str) -> str:
    """Original text plus its CJK bigrams — the string stored in FTS5."""
    grams = cjk_bigrams(text)
    if not grams:
        return text
    return text + "\n" + " ".join(grams)


def query_tokens(query: str) -> list[str]:
    """Expand a user query into the token set used for ``MATCH``.

    Latin words are kept whole *and* split on ``. _ -`` so ``z2-sensitive``
    also tries ``z2`` / ``sensitive``; CJK runs are kept whole *and* expanded
    into bigrams.  Order is preserved and duplicates removed.
    """
    tokens: list[str] = []
    for word in WORD_RE.findall(query):
        tokens.append(word)
        for part in _SPLIT_RE.split(word):
            if len(part) >= 2 and part != word:
                tokens.append(part)
    for run in CJK_RE.findall(query):
        tokens.append(run)
        tokens.extend(run[i:i + 2] for i in range(len(run) - 1))

    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        low = token.lower()
        if low not in seen:
            seen.add(low)
            out.append(token)
    return out or [query]


def allows_relaxation(query: str) -> bool:
    """Whether a failed strict match should be retried as a relaxed OR.

    Relaxing is for *vague* queries: a phrase made of several words, or anything
    containing CJK (where requiring every adjacent bigram is brittle).  A single
    latin word or identifier is different — ``UNIQ-CLI-CANARY-88888`` is either
    present or it is not, and relaxing it would return every document that
    happens to contain ``UNIQ``.  That was a measured precision failure.
    """
    if CJK_RE.search(query):
        return True
    return len(query.split()) > 1


def match_expression(tokens: list[str], operator: str = "OR") -> str:
    """Build a safe FTS5 ``MATCH`` expression from literal tokens.

    Every token is wrapped in double quotes, so FTS5 syntax characters in user
    input are treated as data, never as query syntax.

    ``operator`` selects the intended trade-off:

    * ``"AND"`` — precision.  Every expanded token must be present, so an exact
      identifier such as ``UNIQ-Z2-CANARY-99999`` cannot be satisfied by a
      document that merely contains ``UNIQ``.
    * ``"OR"`` — recall.  Used as a second pass when the strict pass is empty,
      so a partially-remembered phrase still finds something.
    """
    if operator not in ("AND", "OR"):
        raise ValueError(f"unsupported operator: {operator!r}")
    return f" {operator} ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def make_snippet(
    text: str, tokens: list[str], width: int = 240, phrase: str | None = None
) -> tuple[str, int]:
    """Return ``(snippet, line_number)`` for the best match inside ``text``.

    Location preference, most specific first:

    1. the **whole query** as a literal substring — so ``mount -a`` reports the
       line that actually contains ``mount -a``;
    2. a token on a **word boundary** — so ``mount`` does not report the line of
       ``/proc/mounts``;
    3. a plain substring, which is what CJK needs (no word boundaries).

    ``line_number`` is 1-based and relative to ``text``; callers add the chunk's
    start line to obtain an absolute file line.
    """
    def pick(position: int | None) -> int:
        return -1 if position is None else position

    pos = -1
    if phrase:
        collapsed = " ".join(phrase.split())
        found = text.lower().find(collapsed.lower())
        if found >= 0:
            pos = found

    if pos < 0:
        for token in sorted(tokens, key=len, reverse=True):
            match = re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE
            )
            found = pick(match.start() if match else None)
            if found < 0:
                found = text.lower().find(token.lower())
            if found >= 0 and (pos < 0 or found < pos):
                pos = found

    if pos < 0:
        head = " ".join(text.split())[:width]
        return head, 1

    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    fragment = " ".join(text[start:end].split())
    line = text.count("\n", 0, pos) + 1
    return (("…" if start > 0 else "") + fragment + ("…" if end < len(text) else "")), line


def split_frontmatter(text: str) -> tuple[str, int]:
    """Split a page into ``(body, header_lines)``.

    ``header_lines`` is how many leading lines the frontmatter block occupied,
    so a line number found inside ``body`` can be mapped back to a **file** line
    (``file_line = chunk_start + header_lines + relative - 1``).  Getting this
    wrong would silently shift every reported line of every page that has a
    header — which is nearly all of them — so the arithmetic is asserted in the
    tests rather than assumed.
    """
    if not text.startswith("---"):
        return text, 0
    end = text.find("\n---", 3)
    if end < 0:
        return text, 0
    newline = text.find("\n", end + 1)
    if newline < 0:
        return "", text.count("\n") + 1              # header only, no body
    header = text[:newline]
    return text[newline + 1:], header.count("\n") + 1


def _split_quoted(body: str) -> list[str]:
    """Split on commas that are **not** inside quotes (``[a, "b,c"]`` → 2 items)."""
    out, buf, quote = [], [], ""
    for ch in body:
        if quote:
            if ch == quote:
                quote = ""
            buf.append(ch)
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch == ",":
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [item.strip() for item in out if item.strip()]


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _parse_value(value: str):
    """One frontmatter value: inline list, inline map, or scalar."""
    if value.startswith("[") and value.endswith("]"):
        return [_unquote(v) for v in _split_quoted(value[1:-1])]
    if value.startswith("{") and value.endswith("}"):
        pairs = {}
        for item in _split_quoted(value[1:-1]):
            key, sep, val = item.partition(":")
            if sep:
                pairs[key.strip()] = _unquote(val)
        return pairs
    return _unquote(value)


def parse_frontmatter(text: str) -> dict:
    """Best-effort YAML-subset frontmatter reader.

    Reads what the knowledge base actually writes and nothing more: scalars,
    inline lists (``tags: [a, "b"]``), inline maps (``generated: {by: x, at: y}``)
    and **one level** of block list (``supersedes:`` followed by ``- path``
    lines; a ``- key: value`` item becomes a one-key dict).  Anything nested
    deeper is skipped rather than guessed at.

    Deliberately not a YAML library: dsh-kb ships as a zero-dependency bundle,
    and the fields it needs back are a handful of lifecycle keys
    (``type``/``title``/``tags``/``status``/``generated.at`` plus the supersede
    triple).  A partial reader is only safe when it is honest about its limits,
    so this returns plain values and every caller must use ``.get()``.

    A key with **no** value becomes ``[]`` — that is how a block list starts.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    out: dict = {}
    block_key: str | None = None
    for line in text[3:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if line[:1] in (" ", "\t"):                      # continuation / block item
            if block_key is None or not stripped.startswith("-"):
                continue                                  # deeper nesting: skipped
            item = stripped[1:].strip()
            if not isinstance(out.get(block_key), list):
                continue
            if ":" in item and not item.startswith(("[", "{")):
                key, _, val = item.partition(":")
                out[block_key].append({key.strip(): _unquote(val)})
            else:
                out[block_key].append(_parse_value(item))
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if not value:
            block_key = key
            out[key] = []
        else:
            block_key = None
            out[key] = _parse_value(value)
    return out
