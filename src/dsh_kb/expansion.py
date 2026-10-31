"""Query expansion, confidence scoring and result merging.

The search itself is lexical and fast; this module is what turns a *weak* hit
set into a usable answer, and it only runs when the first pass was not good
enough.  Four stages, each independently switchable, each degrading to "do
nothing" rather than raising:

1. **Confidence** — a cross-encoder score on the top three pages, banded into
   ``high`` / ``medium`` / ``low``.  It is a *soft* signal returned to the caller
   (the model is told how much to trust the answer); it never filters hits.
   Banding rather than a hard threshold is deliberate: a hard cut had to trade
   "reject 82% of out-of-domain" against "drop 24% of answerable", which is a bad
   trade for a supplementary tool.
2. **Synonym expansion** — a domain dictionary maps a term the user typed onto
   the term the corpus uses, giving variants with zero model calls.
3. **Query rewriting** — a local model rewrites the question into a few
   differently-phrased queries.  Only reached when the first pass was weak, and
   the result is cached by query so a repeated question costs nothing.
4. **Page rerank** — cut the candidate pages into passages and score each page by
   its best passage, which recovers pages whose relevant part is not the snippet
   the index returned.

Everything here is opt-in by environment and off-by-default where it costs
latency or a model call.  Nothing in this module may raise into a search: a
failed rewrite is an empty list, an unreachable model is an empty list, a broken
dictionary is an empty list.

Where the data lives
--------------------
The rewrite cache and the synonym table are state, so they default to
``<home>/state/rewrite-cache.sqlite`` and ``<home>/dict/synonyms.json``.  Both
can be pointed elsewhere with ``DSH_KB_REWRITE_CACHE_DB`` /
``DSH_KB_DICT_PATH`` — there is no built-in absolute path, because a library has
no business guessing where a machine keeps its data.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------- #
# Confidence bands (cross-encoder scores).  Soft signal only — see module docs.
# --------------------------------------------------------------------------- #
CONF_HIGH = float(os.environ.get("DSH_KB_CONF_HIGH", "3.0"))
CONF_MEDIUM = float(os.environ.get("DSH_KB_CONF_MEDIUM", "-0.5"))
CONFIDENCE_ENABLED = os.environ.get("DSH_KB_CONFIDENCE", "1") != "0"

# --------------------------------------------------------------------------- #
# Query rewriting (any OpenAI-ish/local chat endpoint).  Off means "never call".
# --------------------------------------------------------------------------- #
REWRITE_MODE = os.environ.get("DSH_KB_REWRITE", "auto")          # auto | always | off
REWRITE_MODEL = os.environ.get("DSH_KB_REWRITE_MODEL", "qwen2.5:3b")
REWRITE_ENDPOINT = os.environ.get("DSH_KB_REWRITE_ENDPOINT", "http://127.0.0.1:11434")
REWRITE_TIMEOUT = float(os.environ.get("DSH_KB_REWRITE_TIMEOUT", "12"))
REWRITE_MAX_VARIANTS = int(os.environ.get("DSH_KB_REWRITE_VARIANTS", "2"))
REWRITE_TRIGGER_SCORE = float(os.environ.get("DSH_KB_REWRITE_TRIGGER", "1.0"))
REWRITE_KEEP_ALIVE = os.environ.get("DSH_KB_REWRITE_KEEP_ALIVE", "24h")
REWRITE_PROMPT_VERSION = os.environ.get("DSH_KB_REWRITE_PROMPT_VERSION", "p1")

REWRITE_PROMPT = (
    "把下面的检索问题改写成 3 条不同角度的检索查询，用于在技术知识库里查找答案。要求：\n"
    "1) 每行一条，不要编号、不要解释；\n"
    "2) 保留专有名词（产品名、命令、文件路径）；\n"
    "3) 换用同义词或更具体的说法，不要照抄原句。\n\n"
    "原问题：%s"
)

# --------------------------------------------------------------------------- #
# Rewrite cache: exact match by default; optional semantic match behind a flag.
# --------------------------------------------------------------------------- #
REWRITE_CACHE_ON = os.environ.get("DSH_KB_REWRITE_CACHE", "1") != "0"
REWRITE_SEMANTIC = float(os.environ.get("DSH_KB_REWRITE_SEMANTIC", "1.01"))
REWRITE_EMBED_MODEL = os.environ.get("DSH_KB_REWRITE_EMBED_MODEL", "bge-m3")
REWRITE_SEMANTIC_ENABLED = os.environ.get("DSH_KB_REWRITE_SEMANTIC_ENABLED", "0") == "1"
CACHE_VERSION = "v1"

# --------------------------------------------------------------------------- #
# Domain dictionary (off by default: it needs a curated file to be worth it).
# --------------------------------------------------------------------------- #
DICT_ON = os.environ.get("DSH_KB_DICT", "0").strip().lower() not in ("0", "off", "false", "no", "")
DICT_MAX_VARIANTS = int(os.environ.get("DSH_KB_DICT_VARIANTS", "3"))
_dict_cache: dict = {"path": None, "mtime": 0.0, "groups": []}

# --------------------------------------------------------------------------- #
# Page-level rerank (off by default: it costs one model call per passage).
# --------------------------------------------------------------------------- #
PAGE_RERANK = os.environ.get("DSH_KB_PAGE_RERANK", "off").strip().lower()   # off | auto | always
PAGE_TOP = int(os.environ.get("DSH_KB_PAGE_TOP", "5"))
PAGE_CHARS = int(os.environ.get("DSH_KB_PAGE_CHARS", "800"))
PAGE_STRIDE = int(os.environ.get("DSH_KB_PAGE_STRIDE", "600"))
PAGE_MAX_CHARS = int(os.environ.get("DSH_KB_PAGE_MAX_CHARS", "20000"))

#: How the extra result lists are combined: ``fill`` keeps the original order and
#: only fills empty slots; ``top2`` keeps the original top two and lets RRF
#: compete for the rest.
MERGE_MODE = os.environ.get("DSH_KB_MERGE", "fill").strip().lower()


# --------------------------------------------------------------------------- #
# Paths (state, not code — always relative to home unless overridden)
# --------------------------------------------------------------------------- #
def cache_db(cfg) -> Path:
    override = os.environ.get("DSH_KB_REWRITE_CACHE_DB")
    return Path(override) if override else Path(cfg.home) / "state" / "rewrite-cache.sqlite"


def dict_path(cfg) -> Path:
    override = os.environ.get("DSH_KB_DICT_PATH")
    return Path(override) if override else Path(cfg.home) / "dict" / "synonyms.json"


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
def page_excerpt(cfg, hit, chars: int = 1200) -> str:
    """Context around a hit, for the cross-encoder; falls back to the snippet."""
    try:
        from .service import read_payload

        payload = read_payload(cfg, hit.path, start=max(1, hit.line - 5), lines=18)
        text = "\n".join(line["text"] for line in payload["lines"])
        if text.strip():
            return text[:chars]
    except Exception:                                  # noqa: BLE001 - excerpt is best effort
        pass
    return (hit.snippet or "")[:chars]


def confidence_fields(cfg, query: str, hits) -> dict:
    """A "should you trust this" signal, without changing any existing field."""
    if not CONFIDENCE_ENABLED:
        return {}
    if not hits:
        return {"rerank_top1": None, "confidence": "none",
                "hint": "没有检索到相关页面。"}
    try:
        from handrolled.rerank import score_pairs

        scores = score_pairs(query, [page_excerpt(cfg, hit) for hit in hits[:3]])
    except Exception:                                  # noqa: BLE001
        scores = None
    if not scores:
        return {"rerank_top1": None, "confidence": "unknown",
                "hint": "重排不可用，无法评估相关性。"}
    top = round(float(max(scores)), 3)
    if top >= CONF_HIGH:
        return {"rerank_top1": top, "confidence": "high"}
    if top >= CONF_MEDIUM:
        return {"rerank_top1": top, "confidence": "medium",
                "hint": "相关性中等：可以用，但回答时要说明把握有限。"}
    return {"rerank_top1": top, "confidence": "low",
            "hint": "相关性低：不要当作确定答案；先说明可能没有对应资料，"
                    "再给最接近的线索（附 path:line）。"}


# --------------------------------------------------------------------------- #
# Rewrite cache
# --------------------------------------------------------------------------- #
def _cache_conn(cfg):
    path = cache_db(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=3)
    conn.execute(
        "create table if not exists rewrite_cache ("
        " key text primary key, query text not null, model text not null,"
        " variants text not null, embedding blob, created_at text not null)"
    )
    return conn


def cache_key(query: str) -> str:
    parts = "|".join((CACHE_VERSION, REWRITE_MODEL, REWRITE_PROMPT_VERSION, query.strip()))
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()


def _embed(text: str, timeout: float = 2.0):
    try:
        body = json.dumps({"model": REWRITE_EMBED_MODEL, "input": text}).encode("utf-8")
        req = urllib.request.Request(REWRITE_ENDPOINT.rstrip("/") + "/api/embed", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        vectors = data.get("embeddings") or []
        return vectors[0] if vectors else None
    except Exception:                                  # noqa: BLE001
        return None


def cosine(left, right) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = sum(a * a for a in left) ** 0.5
    norm_r = sum(b * b for b in right) ** 0.5
    return dot / (norm_l * norm_r) if norm_l and norm_r else 0.0


def cache_lookup(cfg, query: str, want_semantic: bool = True):
    if not REWRITE_CACHE_ON:
        return None
    try:
        conn = _cache_conn(cfg)
        try:
            row = conn.execute("select variants from rewrite_cache where key = ?",
                               (cache_key(query),)).fetchone()
            if row:
                return {"variants": json.loads(row[0]), "hit": "exact"}
            if not want_semantic or not REWRITE_SEMANTIC_ENABLED:
                return None
            vector = _embed(query)
            if not vector:
                return None
            best = (0.0, None)
            for stored, blob in conn.execute(
                    "select variants, embedding from rewrite_cache"
                    " where model = ? and embedding is not null", (REWRITE_MODEL,)):
                try:
                    raw = bytes(blob).decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
                    other = json.loads(raw)
                except Exception:                      # noqa: BLE001
                    continue
                score = cosine(vector, other)
                if score > best[0]:
                    best = (score, stored)
            if best[1] and best[0] >= REWRITE_SEMANTIC:
                return {"variants": json.loads(best[1]), "hit": "semantic",
                        "similarity": round(best[0], 4)}
        finally:
            conn.close()
    except Exception:                                  # noqa: BLE001 - cache is a nicety
        return None
    return None


def cache_store(cfg, query: str, variants: list[str], vector=None) -> None:
    if not REWRITE_CACHE_ON or not variants:
        return
    try:
        conn = _cache_conn(cfg)
        try:
            conn.execute(
                "insert or replace into rewrite_cache(key, query, model, variants, embedding, created_at)"
                " values (?,?,?,?,?,?)",
                (cache_key(query), query, REWRITE_MODEL,
                 json.dumps(variants, ensure_ascii=False),
                 json.dumps(vector) if vector else None,
                 datetime.now().astimezone().isoformat(timespec="seconds")),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:                                  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- #
# Synonym expansion
# --------------------------------------------------------------------------- #
def dict_groups(cfg) -> list[list[str]]:
    """Load the synonym table, reloading when the file changes on disk."""
    path = dict_path(cfg)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return []
    if _dict_cache["path"] != str(path) or _dict_cache["mtime"] != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            groups = [g for g in data.get("groups", []) if isinstance(g, list) and len(g) > 1]
            _dict_cache.update({"path": str(path), "groups": groups, "mtime": mtime})
        except (OSError, ValueError):
            return _dict_cache["groups"]
    return _dict_cache["groups"]


def term_hit(term: str, query: str) -> bool:
    """Whole-token match for latin terms, substring match for CJK."""
    if not term:
        return False
    if term.isascii():
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        return re.search(pattern, query, re.IGNORECASE) is not None
    return term in query


def dict_variants(cfg, query: str) -> list[str]:
    """Substitute domain synonyms into the query — no model call, milliseconds."""
    if not DICT_ON or not query.strip():
        return []
    variants: list[str] = []
    for group in dict_groups(cfg):
        matched = next((term for term in group if term_hit(term, query)), None)
        if matched is None:
            continue
        for alternative in group:
            if alternative == matched:
                continue
            variant = query.replace(matched, alternative, 1)
            if variant != query and variant not in variants:
                variants.append(variant)
            if len(variants) >= DICT_MAX_VARIANTS:
                return variants
    return variants


# --------------------------------------------------------------------------- #
# Page rerank
# --------------------------------------------------------------------------- #
def page_text(cfg, rel: str) -> str:
    try:
        return (Path(cfg.mirror_dir) / rel).read_text(encoding="utf-8", errors="replace")[:PAGE_MAX_CHARS]
    except Exception:                                  # noqa: BLE001
        return ""


def passages(text: str, chars: int, stride: int) -> list[str]:
    if len(text) <= chars:
        return [text] if text.strip() else []
    windows = []
    for start in range(0, max(1, len(text) - chars + 1), max(1, stride)):
        piece = text[start:start + chars].strip()
        if piece:
            windows.append(piece)
    return windows[:24]


def page_rerank(cfg, query: str, hits, limit: int):
    """Score each candidate page by its best passage, then order by that score."""
    if PAGE_RERANK == "off" or not hits:
        return list(hits), None
    try:
        from handrolled.rerank import score_pairs
    except Exception:                                  # noqa: BLE001
        return list(hits), None
    first: dict[str, object] = {}
    for hit in hits:
        first.setdefault(hit.path, hit)
    paths = list(first)[:PAGE_TOP]
    scores: dict[str, float] = {}
    passages_used = 0
    for rel in paths:
        pieces = passages(page_text(cfg, rel), PAGE_CHARS, PAGE_STRIDE)
        if not pieces:
            continue
        values = score_pairs(query, pieces)
        passages_used += len(pieces)
        if values:
            scores[rel] = max(values)
    if not scores:
        return list(hits), None
    ordered = sorted(first, key=lambda rel: -scores.get(rel, -1e9))
    return [first[rel] for rel in ordered][:limit], {"pages": len(scores), "passages": passages_used}


# --------------------------------------------------------------------------- #
# Query rewriting
# --------------------------------------------------------------------------- #
def rewrite_variants(cfg, query: str) -> list[str]:
    """Ask a local model for other phrasings; any failure is an empty list."""
    if REWRITE_MODE == "off" or not query.strip():
        return []
    cached = cache_lookup(cfg, query)
    if cached and cached.get("variants"):
        return cached["variants"]
    try:
        body = json.dumps({
            "model": REWRITE_MODEL, "stream": False,
            "messages": [{"role": "user", "content": REWRITE_PROMPT % query}],
            "options": {"temperature": 0.3, "num_predict": 220},
            "keep_alive": REWRITE_KEEP_ALIVE,
        }).encode("utf-8")
        request = urllib.request.Request(
            REWRITE_ENDPOINT.rstrip("/") + "/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=REWRITE_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:                                  # noqa: BLE001 - never block a search
        return []
    text = ((data.get("message") or {}).get("content") or "") if isinstance(data, dict) else ""
    variants: list[str] = []
    for line in text.splitlines():
        item = re.sub(r"^[\s\-\*\d.、)）]+", "", line).strip().strip('"').strip("。")
        if 3 <= len(item) <= 120 and item != query and item not in variants:
            variants.append(item)
    result = variants[:REWRITE_MAX_VARIANTS]
    if result:
        cache_store(cfg, query, result, vector=_embed(query))
    return result


#: A query that is one token and carries a digit or a separator is treated as an
#: identifier (``UNIQ-…``, ``errno 2``, ``v0.3.4``, ``ds_agent``).  Rewriting one
#: is always wrong: the model turns it into generic prose, the generic prose
#: matches unrelated pages, and a lookup that should return "nothing" starts
#: returning noise.  The lexical layer already guarantees identifiers match
#: exactly or not at all, so this stage must not undo that.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\-]*$")


def looks_like_identifier(query: str) -> bool:
    text = (query or "").strip()
    if not text or " " in text or not _IDENTIFIER.match(text):
        return False
    return bool(re.search(r"[0-9]", text) or "-" in text or "_" in text or "/" in text)


def needs_rewrite(query: str, hits, confidence: dict) -> bool:
    """Whether the first pass was weak enough to justify the extra work."""
    if REWRITE_MODE == "off" and not DICT_ON:
        return False
    if looks_like_identifier(query):
        return False
    if REWRITE_MODE == "always":
        return True
    if not hits:
        return True
    band = confidence.get("confidence")
    if band in ("none", "low"):
        return True
    if band in ("unknown", "high", "medium"):
        return False
    score = confidence.get("rerank_top1")
    return score is not None and score < REWRITE_TRIGGER_SCORE


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def merge_top2(primary, others, limit: int, k: int = 60):
    """Keep the original top two; let everything else compete by reciprocal rank.

    Fill-only never adds anything once top-k is full; top2 lets a rewrite replace
    ranks 3–5 while guaranteeing the two most trustworthy hits cannot be pushed
    out by noise.
    """
    head = list(primary)[:2]
    seen = {hit.path for hit in head}
    scores: dict[str, float] = {}
    pool: dict[str, object] = {}
    for rank, hit in enumerate(primary, start=1):
        if hit.path in seen:
            continue
        scores[hit.path] = scores.get(hit.path, 0.0) + 1.0 / (k + rank)
        pool.setdefault(hit.path, hit)
    for hits in others:
        for rank, hit in enumerate(hits, start=1):
            if hit.path in seen:
                continue
            scores[hit.path] = scores.get(hit.path, 0.0) + 1.0 / (k + rank)
            pool.setdefault(hit.path, hit)
    out = list(head)
    for path in sorted(scores, key=lambda item: -scores[item]):
        if len(out) >= limit:
            break
        out.append(pool[path])
    return out


def merge_hits(primary, others, limit: int, k: int = 60):
    """Keep the original order and only fill the empty slots from the variants.

    Deliberately not pure RRF: plain RRF lets a page ranked first by two variants
    beat the page ranked first by the original query — measured on a 50-question
    gold set, that dropped page@5 from 0.580 to 0.440.  Filling cannot make the
    original result worse.
    """
    seen: set[str] = set()
    out: list[object] = []
    for hit in primary:
        if len(out) >= limit:
            return out
        if hit.path not in seen:
            seen.add(hit.path)
            out.append(hit)
    scores: dict[str, float] = {}
    pool: dict[str, object] = {}
    for hits in others:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.path] = scores.get(hit.path, 0.0) + 1.0 / (k + rank)
            pool.setdefault(hit.path, hit)
    for path in sorted(scores, key=lambda item: -scores[item]):
        if len(out) >= limit:
            break
        if path in seen:
            continue
        seen.add(path)
        out.append(pool[path])
    return out


def merge_variants(primary, extra_lists, limit: int, mode: str | None = None):
    """Pick the merge strategy by configuration (``fill`` or ``top2``)."""
    if (mode or MERGE_MODE) == "top2":
        return merge_top2(primary, extra_lists, limit)
    return merge_hits(primary, extra_lists, limit)


def search_many(cfg, variants: Sequence[str], *, search, zone: str, limit, snippet_chars,
                fresh_seconds: float, deep: bool) -> list[list]:
    """Run one search per variant, concurrently, swallowing per-variant failures."""
    if not variants:
        return []

    def one(variant: str):
        return search(cfg, variant, zone=zone, limit=limit, snippet_chars=snippet_chars,
                      fresh_seconds=fresh_seconds, deep=deep).hits

    out: list[list] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(variants))) as pool:
            for hits in pool.map(one, variants):
                out.append(hits)
    except Exception:                                  # noqa: BLE001
        return []
    return out
