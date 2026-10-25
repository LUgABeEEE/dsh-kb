"""可选向量臂：把 bge-m3 嵌入 + ``vectors.sqlite`` 余弦接进生产检索。

设计取舍（2026-10-23，来自度量台的实证）：

* **默认关闭**（``rag.vector_enabled``）。关闭时不 import numpy、不发网络请求、
  行为与旧的纯 FTS5 完全一致 —— 接线的第一步不该改变任何人的搜索结果。
* **只在需要时补一路**：关键词命中数已达 ``limit`` 就不走向量。理由：实测关键词
  路径 12–14ms，而向量路径冷启动要 1.5s（ollama 5 分钟卸载），热态 ~40ms。
  自然语言改写型查询恰恰是关键词 0 命中的那类，所以这个触发条件正是它们的场景。
* **嵌入请求带 keep_alive**：否则每次空闲 5 分钟后首个查询都要付 1.5s。
* **任何异常都只降级、不报错**：向量臂挂掉时仍返回关键词结果，note 里写明原因。

融合用 RRF（倒数排名，k=60），与度量台 c3 同一口径：两路分数量纲不同（BM25 是负数、
余弦 0–1），直接相加没有意义。融合后的 score 是 RRF 值，不是 BM25 —— 这点写进 note。
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.request
from pathlib import Path

__all__ = ["VectorArm", "augment", "DEFAULT_ENDPOINT"]

DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/embed"
DEFAULT_KEEP_ALIVE = "30m"
RRF_K = 60

#: 进程内缓存：(库路径, 修改时间, 大小) -> (matrix, meta, dim, model)
_CACHE: dict[tuple, tuple] = {}

_D2Q_CACHE: dict[tuple, tuple] = {}

def _load_d2q(db: Path):
    """读 doc2query 生成问句库（index/d2q.sqlite）。不存在返回 None，不报错。"""
    if not db.is_file():
        return None
    st = db.stat()
    key = (str(db), int(st.st_mtime), int(st.st_size))
    cached = _D2Q_CACHE.get(key)
    if cached is None:
        np = _numpy()
        with sqlite3.connect("file:%s?mode=ro" % db, uri=True) as con:
            rows = con.execute("select path, window_start, vec from d2q").fetchall()
        if not rows:
            return None
        dim = len(rows[0][2]) // 4
        buffer = b"".join(r[2] for r in rows)
        matrix = np.frombuffer(buffer, dtype="<f4").reshape(len(rows), dim)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        matrix = matrix / norms[:, None]
        cached = (matrix, [(str(r[0]), int(r[1])) for r in rows])
        _D2Q_CACHE.clear(); _D2Q_CACHE[key] = cached
    return cached




def _endpoint() -> str:
    return os.environ.get("DSH_KB_EMBED_ENDPOINT", DEFAULT_ENDPOINT)


def _embed_provider() -> str:
    """嵌入后端：ollama（默认，bge-m3）/ onnx（Qwen3-Embedding-0.6B int8）。

    只给换模型 A/B 用：向量库必须与模型同源，切后端时也要切 rag.vectors_db。
    默认永远是 ollama —— 换模型是个未决实验，不该因为环境残留而悄悄生效。
    """
    return (os.environ.get("DSH_KB_EMBED_PROVIDER") or "ollama").strip().lower()


def _numpy():
    """延迟导入 numpy；可选地从 ``DSH_KB_NUMPY_PATH`` 借一份。

    向量臂是可选功能：没有 numpy 时这里抛 ImportError，调用方把它变成一条 note
    并退回纯词法检索 —— 不会崩，也不会静默改变排序。
    """
    try:
        import numpy  # noqa: F401
        return numpy
    except Exception:
        pass
    extra = os.environ.get("DSH_KB_NUMPY_PATH")
    if not extra:
        raise ImportError(
            "the vector arm needs numpy: install it, or point DSH_KB_NUMPY_PATH at a "
            "directory that contains it"
        )
    import sys

    if extra not in sys.path:
        sys.path.insert(0, extra)
    import numpy  # noqa: F401
    return numpy


GENERATE = "http://127.0.0.1:11434/api/generate"


def hyde_mimo(question: str, max_tokens: int = 160, timeout: float = 30.0) -> str | None:
    """让一个 OpenAI-compatible chat endpoint 写「假想答案」（HyDE）。

    需要 ``DSH_KB_HYDE_ENDPOINT`` / ``DSH_KB_HYDE_API_KEY`` / ``DSH_KB_HYDE_MODEL``；
    缺任何一个都返回 None，调用方退回纯查询向量（HyDE 本身默认关闭）。
    """
    base = (os.environ.get("DSH_KB_HYDE_ENDPOINT") or "").rstrip("/")
    key = os.environ.get("DSH_KB_HYDE_API_KEY") or ""
    model = os.environ.get("DSH_KB_HYDE_MODEL") or ""
    if not (base and key and model):
        return None
    try:
        prompt = ('用一段技术笔记的口吻，回答下面的问题。只写要点，60-120 字，不要解释、不要复述问题。问题：' + question)
        body = json.dumps({'model': model,
                           'messages': [{'role': 'user', 'content': prompt}],
                           'temperature': 0.2, 'max_tokens': max_tokens}).encode('utf-8')
        req = urllib.request.Request(base + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read().decode('utf-8'))
        text = ((d.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
    except Exception:
        return None
    text = text.strip()
    return text if len(text) >= 12 else None


def hyde_answer(question: str, model: str = "qwen2.5:7b", max_tokens: int = 100,
                timeout: float = 20.0) -> str | None:
    """HyDE：让本地模型先写一段「假想答案」，再拿它的向量去检索。

    动机（2026-10-23 实测）：口语问题与目标页没有共同词，但模型写的答案用词更像文档。
    失败一律返回 None（调用方退回纯查询向量），绝不因为生成失败影响检索。
    """
    prompt = "用一段技术笔记的口吻回答下面的问题。只写答案要点，60-120 字，不要解释、不要复述问题。问题：" + question
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.2, "num_predict": max_tokens}}).encode("utf-8")
    req = urllib.request.Request(GENERATE, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = json.loads(resp.read().decode("utf-8")).get("response") or ""
    except Exception:
        return None
    text = text.strip()
    return text if len(text) >= 12 else None


class VectorArm:
    """只读的向量库视图 + 查询嵌入。实例很轻，矩阵走模块缓存。"""

    def __init__(self, db: Path | str, keep_alive: str = DEFAULT_KEEP_ALIVE,
                 timeout: float = 8.0, use_d2q: bool = False):
        self.db = Path(db)
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.use_d2q = use_d2q
        self.np = None
        self.matrix = None
        self.meta: list[tuple[str, int]] = []
        self.dim = 0
        self.model = ""

    def load(self) -> "VectorArm":
        if self.matrix is not None:
            return self
        if not self.db.is_file():
            raise RuntimeError(f"向量库不存在：{self.db}")
        st = self.db.stat()
        key = (str(self.db), int(st.st_mtime), int(st.st_size))
        cached = _CACHE.get(key)
        if cached is None:
            np = _numpy()
            with sqlite3.connect(f"file:{self.db}?mode=ro", uri=True) as con:
                rows = con.execute(
                    "select path, startline, vec, model, length(vec) from vectors"
                ).fetchall()
            if not rows:
                raise RuntimeError("向量库为空")
            dims = {int(r[4]) // 4 for r in rows}
            if len(dims) != 1:
                raise RuntimeError(f"向量维度不一致：{sorted(dims)}")
            dim = dims.pop()
            buffer = b"".join(r[2] for r in rows)
            flat = np.frombuffer(buffer, dtype="<f4")
            matrix = flat.reshape(len(rows), dim)
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            matrix = matrix / norms[:, None]
            meta = [(str(r[0]), int(r[1])) for r in rows]
            cached = (matrix, meta, dim, str(rows[0][3] or ""))
            _CACHE.clear()
            _CACHE[key] = cached
        self.matrix, self.meta, self.dim, self.model = cached
        self.np = _numpy()
        return self

    def embed(self, text: str) -> "object":
        if _embed_provider() == "onnx":
            from .embed_onnx import embed_texts as _embed_texts

            vec = _embed_texts([text], is_query=True)[0]
            return self.np.asarray(vec, dtype="<f4")
        body = json.dumps({
            "model": self.model or os.environ.get("DSH_KB_EMBED_MODEL", "bge-m3"),
            "input": [text],
            "keep_alive": self.keep_alive,
        }).encode("utf-8")
        req = urllib.request.Request(
            _endpoint(), data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        vecs = payload.get("embeddings") or []
        if not vecs or not vecs[0]:
            raise RuntimeError("ollama 返回空嵌入")
        arr = self.np.asarray(vecs[0], dtype="<f4")
        norm = float(self.np.linalg.norm(arr)) or 1.0
        return arr / norm

    def query(self, text: str, prefixes: tuple[str, ...], depth: int = 32, hyde_text: str | None = None) -> list[tuple[str, int, float]]:
        self.load()
        q = self.embed(text)
        if hyde_text:
            # HyDE：把「假想答案」的向量与查询向量取平均再归一化。
            # 动机：口语问题与目标页没有共同词，但模型写的答案用词更像文档。
            hq = self.embed(hyde_text)
            q = q + hq
            n = float(self.np.linalg.norm(q)) or 1.0
            q = q / n
        scores = self.matrix @ q
        order = self.np.argsort(-scores)[: max(depth * 4, 64)]
        # 页级去重：同一页的多个块只保留**最高分**那一个。
        # 为什么要这样做：座席只有 limit 个（生产=5），块级候选很容易被同一页的
        # 三四个块占满，等于「给了 5 个答案位置却只答了 1 页」。实测页级去重把
        # 期望页名次拉前 10–20%（47→40 / 119→82 / 574→245）。
        out: list[tuple[str, int, float]] = []
        seen_path: dict[str, int] = {}
        for idx in order:
            path, line = self.meta[int(idx)]
            if prefixes and not str(path).startswith(prefixes):
                continue
            if path in seen_path:
                continue
            seen_path[path] = line
            out.append((path, line, float(scores[int(idx)])))
            if len(out) >= depth:
                break
        d2q = _load_d2q(self.db.parent / "d2q.sqlite") if self.use_d2q else None
        if d2q is not None:
            dmat, dmeta = d2q
            ds = dmat @ q
            best = {p: (ln, sc) for p, ln, sc in out}
            for k in range(len(dmeta)):
                dp, dln = dmeta[k]
                if prefixes and not dp.startswith(prefixes):
                    continue
                sc = float(ds[k])
                cur = best.get(dp)
                if cur is None or sc > cur[1]:
                    best[dp] = (dln, sc)
            out = sorted(((p, ln, sc) for p, (ln, sc) in best.items()), key=lambda t: -t[2])[:depth]
        return out




_TITLES: dict | None = None


def _page_titles(cfg) -> dict:
    """path -> 页面标题（来自 kb.sqlite.pages；取不到就空表，不影响检索）。"""
    global _TITLES
    if _TITLES is not None:
        return _TITLES
    table: dict = {}
    try:
        with sqlite3.connect("file:%s?mode=ro" % cfg.index_db, uri=True) as con:
            for path, title in con.execute("select path, title from pages"):
                if title:
                    table[str(path)] = str(title)[:60]
    except Exception:
        table = {}
    _TITLES = table
    return table

def _snippet(root: Path, rel: str, start: int, chars: int) -> str:
    try:
        with open(root / rel, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return ""
    if start < 1 or start > len(lines):
        return ""
    body = "".join(lines[start - 1 : start + 4]).strip()
    return body[:chars]


def augment(cfg, query: str, limit: int, existing: list, snippet_chars: int, zone: str = "z01", depth: int = 32, mode: str = "fill", hyde: str | None = None, rerank: bool = False) -> tuple[list, str]:
    """给一组关键词命中补上向量路，按 RRF 融合后返回 (新命中列表, 说明)。"""
    prefixes = tuple(cfg.query_prefixes(zone))
    arm = VectorArm(cfg.rag.vectors_db, use_d2q=bool(getattr(cfg.rag, "d2q_enabled", False)))
    depth = int(getattr(cfg.rag, "vector_depth", depth) or depth)
    vec = arm.query(query, prefixes, depth=depth, hyde_text=hyde)
    if rerank and len(vec) > 1 and getattr(cfg.rag, "rerank_enabled", False):
        try:
            from . import rerank as _rr
            topn = max(2, int(getattr(cfg.rag, "rerank_topn", 24) or 24))
            head = vec[:topn]
            _chars = int(getattr(cfg.rag, 'rerank_passage_chars', 480) or 480)
            _titles = (_page_titles(cfg)
                       if getattr(cfg.rag, 'rerank_title_boost', False) else {})
            passages = []
            for p, ln, _s in head:
                body = _snippet(cfg.mirror_dir, p, ln, _chars)
                title = _titles.get(str(p), '')
                passages.append(('【' + title + '】' + chr(10) + body) if title else body)
            scores = _rr.score_pairs(query, passages)
            if scores is not None and len(scores) == len(head):
                pairs = sorted(zip(scores, head), key=lambda t: -t[0])
                if getattr(cfg.rag, 'rerank_page_agg', False):
                    agg: dict = {}
                    for sc, h in pairs:
                        agg.setdefault(h[0], []).append((float(sc), h))
                    order = []
                    for _pth, items in agg.items():
                        items.sort(key=lambda t: -t[0])
                        ps = items[0][0] + (0.5 * items[1][0] if len(items) > 1 else 0.0)
                        order.append((ps, items))
                    order.sort(key=lambda t: -t[0])
                    ranked = [h for _ps, items in order for _sc, h in items]
                else:
                    ranked = [h for _sc, h in pairs]
                if getattr(cfg.rag, 'rerank_two_stage', False) and len(ranked) > 1:
                    import os as _os
                    _len2 = int(getattr(cfg.rag, 'rerank_stage2_maxlen', 384) or 384)
                    _ch2 = int(getattr(cfg.rag, 'rerank_stage2_chars', 800) or 800)
                    _os.environ['DSH_KB_RERANK_MAXLEN'] = str(_len2)
                    _rr._SESSION = None  # 让 rerank 用新截断重建 tokenizer/session
                    _head2 = ranked[:5]
                    _pass2 = [_snippet(cfg.mirror_dir, h[0], h[1], _ch2) for h in _head2]
                    _sc2 = _rr.score_pairs(query, _pass2)
                    if _sc2 is not None and len(_sc2) == len(_head2):
                        ranked = [h for _s2, h in sorted(zip(_sc2, _head2), key=lambda t: -t[0])] + ranked[5:]
                vec = ranked + vec[topn:]
        except Exception:
            pass
    if not vec:
        return existing, "vector arm: 无候选"
    if mode == "fill":
        # 只补齐：关键词顺序原样保留，向量独有候选接在后面。
        # 目的：不让向量路改写「精确查询的 top-1」（实测 `挂载` 会被换掉）。
        out = list(existing)
        seen_pairs = {(h.path, h.line) for h in out}
        root = cfg.mirror_dir
        from .index import Hit
        for rank, (path, line, score) in enumerate(vec, 1):
            if (path, line) in seen_pairs:
                continue
            if len(out) >= limit:
                break
            out.append(Hit(path=path, line=line,
                           score=round(-1.0 / (RRF_K + rank), 6),
                           snippet=_snippet(root, path, line, snippet_chars), zone=""))
            seen_pairs.add((path, line))
        note = (f"vector arm 补齐 {len(out) - len(existing)} 条（fill 模式，关键词顺序不变）；"
                f"库={Path(cfg.rag.vectors_db).name}")
        return out, note

    kw_w = float(getattr(cfg.rag, 'vector_rrf_kw_weight', 1.0) or 1.0)
    vec_w = float(getattr(cfg.rag, 'vector_rrf_vec_weight', 1.0) or 1.0)
    kw_rank = {
        (h.path, h.line): i + 1
        for i, h in enumerate(sorted(existing, key=lambda h: h.score))
    }
    fused: dict[tuple[str, int], float] = {}
    for (path, line), rank in kw_rank.items():
        fused[(path, line)] = fused.get((path, line), 0.0) + kw_w / (RRF_K + rank)
    for rank, (path, line, _score) in enumerate(vec, 1):
        fused[(path, line)] = fused.get((path, line), 0.0) + vec_w / (RRF_K + rank)
    have = {(h.path, h.line): h for h in existing}
    root = cfg.mirror_dir
    from .index import Hit

    merged: list = []
    for (path, line), score in sorted(fused.items(), key=lambda kv: -kv[1])[:limit]:
        hit = have.get((path, line))
        if hit is None:
            hit = Hit(path=path, line=line, score=score,
                      snippet=_snippet(root, path, line, snippet_chars), zone="")
        else:
            hit.score = score
        merged.append(hit)
    note = (f"vector arm 融合 {len(vec)} 候选（RRF k={RRF_K}，score 为 RRF 值）；"
            f"库={Path(cfg.rag.vectors_db).name} 模型={arm.model or 'bge-m3'}")
    return merged, note