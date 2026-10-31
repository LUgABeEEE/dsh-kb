"""Supersede relations — "which page has replaced which".

Why this module exists
----------------------
The knowledge base accumulated *progressive states of the same topic*: a dozen
pages that were each true when written, with no field saying which one still
is.  Retrieval then happily returns a superseded conclusion as if it were
current — the failure this module fixes.  Design and the measurements behind it
are in ``.selfcheck/research/D3-supersede-and-contradiction.md``; the two
principles worth repeating here:

* **Two questions, two fields.**  ``status`` answers "is this page itself
  trustworthy" (quality); ``superseded_by`` answers "is it still in force"
  (time).  They are orthogonal on purpose, so both "``stable`` but replaced"
  (fine, merely outdated) and "``deprecated`` and nobody replaced it" (wrong
  and unsubstituted) stay representable.  **Machines read ``superseded_by``;
  ``deprecated`` is for humans.**
* **Annotate, don't hide.**  Default retrieval still *returns* a superseded page
  but marks it and ranks it last: hiding historically-true pages recreates the
  "I know I wrote that down but search cannot find it" complaint, and it eats
  into ``limit``.  ``include_superseded=False`` exists for the narrow case of
  wanting a current-conclusions list.

Field shapes (all optional; a page without them is simply "live"):

.. code-block:: yaml

    supersedes: ["z1-restricted/a.md", "z1-restricted/b.md"]  # 0..n
    superseded_by: "z1-restricted/c.md"                       # 0..1
    valid_until: 2026-11-01                                   # ISO8601 | null
    not_superseding: "讲的是另一件事"                          # 提案闸门用（见 push 预检）

A page may supersede several pages (that is the main scenario — one new page
absorbs a dozen old ones); a page may be superseded by **at most one** (two
successors means the topic has not converged, which is exactly the
contradiction we want surfaced, not silently resolved).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from handrolled.text import parse_frontmatter

#: Keys this module owns.  ``not_superseding`` is a *declaration* (the proposal
#: gate's escape hatch), not a relation, so it never reaches the index.
SUPERSEDE_KEYS = ("supersedes", "superseded_by", "valid_until", "not_superseding")


def _as_list(value) -> list[str]:
    """Normalize ``supersedes`` to a list of non-empty strings.

    Handles the three shapes a human may write: a block list (``- path``), an
    inline list, or a bare scalar (``supersedes: z1/a.md``) — rejecting the
    last one would be pedantry, and the relation is unambiguous.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):                     # `- path: x` typo guard
            item = next(iter(item.values()), "")
        text = str(item).strip().strip("'\"")
        if text:
            out.append(text)
    return out


def _one(value) -> str:
    """Normalize a single-valued field (``superseded_by`` / ``valid_until``)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if isinstance(value, dict):
        return ""
    return str(value).strip().strip("'\"")


@dataclass
class PageMeta:
    """The frontmatter fields the index keeps a row for."""

    path: str
    type: str = ""
    title: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    status: str = ""
    superseded_by: str = ""
    valid_until: str = ""
    fm_at: str = ""                                    # generated.at
    supersedes: list[str] = field(default_factory=list)
    not_superseding: str = ""

    @property
    def tags_text(self) -> str:
        """Tags as one space-separated field — small, and never queried by term."""
        return " ".join(self.tags)

    def is_superseded(self, now: str | None = None) -> bool:
        """Replaced, or past its ``valid_until`` — the two ways a page stops being current.

        Without a ``now`` a bare ``valid_until`` is **not** treated as expired:
        guessing here would silently drop a page whose window we simply did not
        evaluate.  ``superseded_by`` needs no clock, so it always counts.
        """
        if self.superseded_by:
            return True
        return bool(self.valid_until) and now is not None and self.valid_until <= now


def page_meta(rel: str, raw_text: str) -> PageMeta:
    """Parse one page's frontmatter into :class:`PageMeta` (never raises)."""
    try:
        fm = parse_frontmatter(raw_text)
    except Exception:                                  # noqa: BLE001 - a bad header must not break a build
        return PageMeta(path=rel)
    generated = fm.get("generated")
    fm_at = ""
    if isinstance(generated, dict):
        fm_at = _one(generated.get("at"))
    tags = fm.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    tag_list = tuple(str(t).strip().strip("'\"") for t in (tags or []) if str(t).strip())
    return PageMeta(
        path=rel,
        type=_one(fm.get("type")),
        title=_one(fm.get("title")),
        description=_one(fm.get("description")),
        tags=tag_list,
        status=_one(fm.get("status")),
        superseded_by=_one(fm.get("superseded_by")),
        valid_until=_one(fm.get("valid_until")),
        fm_at=fm_at,
        supersedes=_as_list(fm.get("supersedes")),
        not_superseding=_one(fm.get("not_superseding")),
    )


# --------------------------------------------------------------------------
# The proposal gate (spec §4, R1): force one line of declaration
# --------------------------------------------------------------------------
#: title Dice above this, *with* blocking (same type + overlapping tags + a
#: title of at least 8 characters) means the proposal sits on an existing topic.
#: Measured on the real corpus (§4): 0.7 → 7,302 pairs (too wide), 0.8 → 2,198
#: (usable), 0.9 → 149 (misses synonymous rewrites, which land at 0.6–0.8).
#: Erring wide is deliberate: a false positive costs one `not_superseding:` line,
#: while a false negative costs another progressive page nobody can retire.
GATE_THRESHOLD = 0.80
MIN_TITLE_CHARS = 8


def _bigrams(text: str) -> set[str]:
    body = "".join((text or "").lower().split())
    if not body:
        return set()
    if len(body) == 1:
        return {body}
    return {body[i:i + 2] for i in range(len(body) - 1)}


def title_dice(left: str, right: str) -> float:
    """Sørensen–Dice on character bigrams — CJK-friendly and cacheable."""
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


@dataclass
class Candidate:
    path: str
    title: str
    dice: float
    status: str = ""
    superseded_by: str = ""


def candidates(cfg, title: str, *, type_: str = "", tags: tuple[str, ...] = (),
               threshold: float = GATE_THRESHOLD) -> list[Candidate]:
    """Live pages that look like the same topic as ``title``.

    Blocking first (same ``type``, overlapping ``tags``, title long enough to be
    meaningful), then bigram Dice on the title.

    Read as *all* pages and filtered in Python on purpose: the corpus is ~1.3k
    rows, and expressing "tags share at least one element" over a
    space-separated column in SQL is exactly the kind of clever SQL that
    silently mismatches.  Exactness beats elegance at this size.
    """
    if len("".join(title.split())) < MIN_TITLE_CHARS:
        return []                                      # a 3-character title blocks nothing
    db = Path(cfg.index_db)
    if not db.is_file():
        return []
    import sqlite3
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        if not _has_table(con, "pages"):
            return []
        rows = con.execute(
            "select path, title, tags, type, status, coalesce(superseded_by,'')"
            " from pages where coalesce(superseded_by,'') = ''").fetchall()
    except Exception:                                  # noqa: BLE001 - advisory feature
        return []
    finally:
        con.close()
    want_tags = {t.strip().lower() for t in tags if str(t).strip()}
    out: list[Candidate] = []
    for path, other_title, other_tags, other_type, other_status, _succ in rows:
        if type_ and other_type and other_type.lower() != type_.lower():
            continue
        have_tags = {t.strip().lower() for t in str(other_tags or "").split() if t.strip()}
        if want_tags and have_tags and not (want_tags & have_tags):
            continue
        if not want_tags or not have_tags:
            continue                                   # blocking needs *both* sides' tags
        score = title_dice(title, other_title or "")
        if score >= threshold:
            out.append(Candidate(path=str(path), title=str(other_title or ""),
                                 dice=round(score, 4), status=str(other_status or "")))
    out.sort(key=lambda c: (-c.dice, c.path))
    return out


def _has_table(con, name: str) -> bool:
    return con.execute(
        "select count(*) from sqlite_master where type in ('table','view') and name=?",
        (name,)).fetchone()[0] > 0


def gate(cfg, rel: str, raw_text: str) -> dict:
    """R1: a proposal over an existing topic must declare *something*.

    Returns ``{"ok": bool, "verdict": str, "candidates": [...], "reason": str}``.
    ``ok=False`` means "add ``supersedes:`` or ``not_superseding:``" — it is a
    request for one line, never a ban: the proposal is the author's to make.
    """
    page = page_meta(rel, raw_text)
    title = page.title or Path(rel).stem
    found = candidates(cfg, title, type_=page.type, tags=page.tags)
    # 自己不算"撞车"：提案若已经在索引里（同一份草案被索引过、或某个 profile 没把
    # _incoming 排除在外），一个 1.0 的自匹配会把"重推同一份文件"变成永远过不去。
    own = {rel, f"_incoming/{rel}"} if "/" not in rel else {rel}
    found = [c for c in found if c.path not in own]
    if page.supersedes or page.not_superseding:
        return {"ok": True, "verdict": "declared",
                "candidates": [c.__dict__ for c in found],
                "reason": ("supersedes: " + ", ".join(page.supersedes)
                           if page.supersedes else f"not_superseding: {page.not_superseding}")}
    if not found:
        return {"ok": True, "verdict": "clear", "candidates": [],
                "reason": "没有 title/type/tags 都相近的现行页（≥0.80 的阻塞命中）"}
    listed = "；".join(f"{c.path}（{c.title}，Dice {c.dice}）" for c in found[:3])
    return {"ok": False, "verdict": "must_declare",
            "candidates": [c.__dict__ for c in found],
            "reason": (f"与现行页标题高度相似：{listed} —— 请补 supersedes: [...] 或 "
                       'not_superseding: "<理由>" 之一')}


# --------------------------------------------------------------------------
# The five assertions of the spec (A1–A5), runnable against any index
# --------------------------------------------------------------------------
def _rows(con, sql: str, params: tuple = ()) -> list[tuple]:
    try:
        return con.execute(sql, params).fetchall()
    except Exception:                                  # noqa: BLE001 - reported by the caller
        return []


def find_cycles(con) -> list[tuple[str, str]]:
    """``(child, reached)`` pairs where a page is its own ancestor.

    Edge direction follows the spec: for a page A declaring ``supersedes: [B]``
    the row is ``(child=A, parent=B)`` — walking ``parent`` therefore follows
    "what did this page replace", and a cycle means two pages each claim to
    have replaced the other.
    """
    return _rows(con, """
        with recursive anc(child, p) as (
            select child, parent from supersedes
          union
            select a.child, s.parent from supersedes s join anc a on s.child = a.p
        )
        select distinct child, p as reached from anc where child = p order by child
    """)


def check(cfg) -> dict:
    """Read-only health report of the supersede relations (spec §7, A1–A5 + A7).

    Never raises for a database that predates this feature: a missing ``pages``
    table means "this index has no supersede data yet", which is reported as
    ``built=False`` rather than as a failure — the same choice ``ragcheck``
    makes for an absent vector store.
    """
    db = Path(cfg.index_db)
    out: dict = {"built": False, "ok": True, "problems": {}, "note": ""}
    if not db.is_file():
        out.update(ok=False, note=f"index missing: {db}")
        return out
    import sqlite3

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")}
        if not {"pages", "supersedes"} <= tables:
            out["note"] = "index has no pages/supersedes tables — rebuild it (`dsh-kb index`)"
            return out
        out["built"] = True
        pages = con.execute("select count(*) from pages").fetchone()[0]
        chunks_pages = con.execute("select count(distinct path) from meta").fetchone()[0]
        edges = con.execute("select count(*) from supersedes").fetchone()[0]
        with_succ = con.execute(
            "select count(*) from pages where coalesce(superseded_by,'') <> ''").fetchone()[0]
        with_vu = con.execute(
            "select count(*) from pages where coalesce(valid_until,'') <> ''").fetchone()[0]
        with_edges = con.execute(
            "select count(distinct child) from supersedes").fetchone()[0]

        inverse = _rows(con, """
            select s.child, s.parent from supersedes s
            left join pages p on p.path = s.parent
            where coalesce(p.superseded_by,'') <> s.child limit 20
        """)
        dangling = _rows(con, """
            select s.child, s.parent from supersedes s
            left join pages p on p.path = s.parent
            where p.path is null limit 20
        """)
        multi = _rows(con, """
            select parent, count(*) c from supersedes group by parent having c > 1 limit 20
        """)
        cycles = find_cycles(con)
        live = con.execute("""
            select count(*) from pages
            where coalesce(superseded_by,'') = ''
              and (coalesce(valid_until,'') = '' or valid_until > datetime('now'))
        """).fetchone()[0]

        out.update({
            "pages": pages, "chunk_pages": chunks_pages, "edges": edges,
            "with_superseded_by": with_succ, "with_valid_until": with_vu,
            "with_edges": with_edges, "live": live, "superseded": pages - live,
            "problems": {"inverse": inverse, "dangling": dangling,
                         "multi_parent": multi, "cycles": cycles},
        })
        # A7: the page table and the chunk table must describe the same corpus.
        if pages != chunks_pages:
            out["ok"] = False
            out["note"] = (f"pages table has {pages} rows but the index has "
                           f"{chunks_pages} distinct paths — rebuild the index")
        else:
            out["ok"] = not any(out["problems"].values())
            bits = [f"{pages} pages", f"{edges} supersedes edge(s)",
                    f"{with_succ} replaced", f"{with_vu} with valid_until",
                    f"{live} live / {pages - live} superseded"]
            bad = {k: len(v) for k, v in out["problems"].items() if v}
            if bad:
                bits.append("problems: " + ", ".join(f"{k}={n}" for k, n in bad.items()))
            out["note"] = " · ".join(bits)
        return out
    finally:
        con.close()
