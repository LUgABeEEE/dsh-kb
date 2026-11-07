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

图片第三路（P2 侧车，2026-11-06）沿同一口径接在最后：``vectors-image.sqlite``
里的图片向量由 :mod:`dsh_kb.image_embed` 离线构建，检索时用文本查询向量去比，
权重独立（``rag.image_arm_weight``，默认 0.6）。它默认关闭，关闭时连
``image_embed`` 都不 import；开启后任何失败（无 numpy / 库不存在 / 无 key /
HTTP 失败 / schema 不符）都只是 note 里的一句「图片路跳过」，文本结果照常返回。
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.request
from pathlib import Path

__all__ = ["VectorArm", "ImageArm", "augment", "image_arm_readiness", "DEFAULT_ENDPOINT",
          "RRF_K", "DEFAULT_IMAGE_WEIGHT"]

DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/embed"
DEFAULT_KEEP_ALIVE = "30m"
RRF_K = 60

#: 图片侧车的默认融合权重，**独立**于文本向量路的 1.0。
#: 实测图片路对文本查询是「锦上添花」，给 1.0 会让一张图压过一整页正文。
DEFAULT_IMAGE_WEIGHT = 0.6

#: 图片扩展名（与 ``image_embed.IMAGE_SUFFIXES`` 一致；这里单独写一份是为了
#: 让「关闭图片路」时连 image_embed 都不 import）。
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

#: 进程内缓存：(库路径, mtime, size, 同目录索引身份) -> (matrix, meta, dim, model,
#: 未能校验的原因, 丢弃的行数)。**键里必须带索引身份**：重分块只改 ``kb.sqlite``，
#: ``vectors.sqlite`` 的 (路径, mtime, size) 一点没变，只看它就会把上一代
#: （含孤儿行）的矩阵原样复用 —— 那正是这个缺陷能活过整次重嵌的原因。
_CACHE: dict[tuple, tuple] = {}

#: 图片库缓存：同结构，单独一张表，免得两个库互相把对方的缓存冲掉。
_IMAGE_CACHE: dict[tuple, tuple] = {}

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


def hyde_remote(question: str, max_tokens: int = 160, timeout: float = 30.0) -> str | None:
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


#: 向量库的同目录索引名。两者是同一个 home 的一对文件：
#: ``index/vectors.sqlite``（块向量）与 ``index/kb.sqlite``（块身份）。
_INDEX_BASENAME = "kb.sqlite"

#: 「拿不到索引」时的两条固定说明，写进 note 而不是抛错（见 _index_chunk_ids）。
_UNVERIFIED_MISSING = ("未找到同目录索引 {name}，未能校验索引一致性（按旧行为全部加载）")
_UNVERIFIED_LEGACY = "索引是旧格式（meta 无 chunk_id），未能校验索引一致性（按旧行为全部加载）"
_UNVERIFIED_UNREADABLE = ("索引 {name} 不可读（{kind}），未能校验索引一致性"
                          "（按旧行为全部加载）")
_UNVERIFIED_STORE = ("向量库是旧格式（无 chunk_id 列），未能校验索引一致性"
                     "（按旧行为全部加载）")


def _file_identity(path: Path) -> tuple:
    """``(路径, mtime_ns, size, inode)`` —— 缓存键里「这份文件现在是什么样」。

    用 ``st_mtime_ns`` 而不是秒：一次真实重建只要 16–21 s（远超 1 s 粒度），
    但**同一秒内**被重写是可能的（测试、脚本），而秒级 mtime 会让两次不同的
    索引算出同一个键 —— 那正是缓存要防的「上一代矩阵被复用」。inode 再兜一层：
    ``index.build`` 是「写 tmp 再 os.replace」，替换后 inode 必变。

    文件不在时返回全零元组：调用方据此知道「拿不到」，而不是拿一个会跟旧值
    撞上的键。
    """
    try:
        st = path.stat()
    except OSError:
        return (str(path), 0, 0, 0)
    return (str(path), int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))


def _index_chunk_ids(kb: Path) -> tuple[set[str] | None, str]:
    """当前索引里的 ``chunk_id`` 集合；**拿不到**时返回 ``(None, 原因)``。

    三种结果必须分开，否则会误删/误留：

    * ``set(...)``  —— 校验可用。**空集是合法的**：索引确实没有块，那么每一行
      向量都是孤儿，一条都不该加载。
    * ``None``      —— 校验不可用（索引缺失 / 旧格式没有 chunk_id / 打不开）。
      此时按改造前的行为全部加载，并把原因回报给 note —— 临时 home、``--home``
      指向别处、老的向量库副本都真实存在，突然「查询变纯词法」比「加载未校验的
      向量」更糟（captain 2026-11-06 的口径）。

    打开方式是 ``mode=ro``：校验绝不能创建或改写索引。
    """
    if not kb.is_file():
        return None, _UNVERIFIED_MISSING.format(name=kb.name)
    try:
        with sqlite3.connect(f"file:{kb}?mode=ro", uri=True) as con:
            columns = {row[1] for row in con.execute("PRAGMA table_info(meta)")}
            if "chunk_id" not in columns:
                return None, _UNVERIFIED_LEGACY
            ids = {str(row[0]) for row in con.execute("select chunk_id from meta")}
    except sqlite3.Error as exc:
        return None, _UNVERIFIED_UNREADABLE.format(
            name=kb.name, kind=exc.__class__.__name__)
    return ids, ""


def _vector_rows(con) -> tuple[list[tuple], bool]:
    """``(rows, has_chunk_id)`` —— rows 统一成 ``(chunk_id, path, startline, vec, model, nbytes)``。

    旧格式的 ``vectors``（没有 ``chunk_id`` 列，CAS 迁移前的库）仍然原样读出，
    ``chunk_id`` 位置补 ``None``：两条路径共用下面同一段矩阵构建代码，也保证
    老库不会因为「多了一列校验」而突然读不出来。
    """
    columns = {row[1] for row in con.execute("PRAGMA table_info(vectors)")}
    if "chunk_id" in columns:
        rows = con.execute(
            "select chunk_id, path, startline, vec, model, length(vec) from vectors"
        ).fetchall()
        return rows, True
    rows = con.execute(
        "select path, startline, vec, model, length(vec) from vectors").fetchall()
    return [(None, *row) for row in rows], False


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
        #: 加载时丢掉了多少条「chunk_id 不在当前索引里」的行（诊断用）。
        self.dropped = 0
        #: 非空 = 这次加载**没能**校验索引一致性，内容是给 note 的原因。
        self.unverified = ""

    def health_note(self) -> str:
        """向量库的健康提示，交给 ``augment`` 拼进 note；没事可说时返回空串。

        为什么要把「丢了多少行」说出来：这个缺陷原本的症状是**静默**错位。
        丢弃本身是对的（错位比缺失更糟），但如果一声不响，"向量库里有 7 万条
        旧块残留" 这件事就永远不会有人知道 —— 而它正是重嵌没做完的信号。
        """
        parts = []
        if self.dropped:
            parts.append(f"已忽略 {self.dropped} 条与当前索引不一致的向量"
                         f"（疑似重分块残留，重嵌后 rag-check --prune-chunks 清理）")
        if self.unverified:
            parts.append(self.unverified)
        return "；".join(parts)

    def load(self) -> "VectorArm":
        if self.matrix is not None:
            return self
        if not self.db.is_file():
            raise RuntimeError(f"向量库不存在：{self.db}")
        st = self.db.stat()
        kb = self.db.parent / _INDEX_BASENAME
        key = (str(self.db), int(st.st_mtime), int(st.st_size), _file_identity(kb))
        cached = _CACHE.get(key)
        if cached is None:
            np = _numpy()
            with sqlite3.connect(f"file:{self.db}?mode=ro", uri=True) as con:
                rows, has_chunk_id = _vector_rows(con)
            if has_chunk_id:
                live, why = _index_chunk_ids(kb)
            else:
                live, why = None, _UNVERIFIED_STORE
            dropped = 0
            if live is not None:
                kept = [row for row in rows if str(row[0]) in live]
                dropped = len(rows) - len(kept)
                rows = kept
            if not rows:
                if dropped:
                    raise RuntimeError(
                        f"向量库里 {dropped} 行都不在当前索引中（重新分块后没重嵌/"
                        f"没清残留），向量臂跳过：{self.db}")
                raise RuntimeError("向量库为空")
            dims = {int(r[5]) // 4 for r in rows}
            if len(dims) != 1:
                raise RuntimeError(f"向量维度不一致：{sorted(dims)}")
            dim = dims.pop()
            buffer = b"".join(r[3] for r in rows)
            flat = np.frombuffer(buffer, dtype="<f4")
            matrix = flat.reshape(len(rows), dim)
            norms = np.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1.0
            matrix = matrix / norms[:, None]
            meta = [(str(r[1]), int(r[2])) for r in rows]
            cached = (matrix, meta, dim, str(rows[0][4] or ""), why, dropped)
            _CACHE.clear()
            _CACHE[key] = cached
        (self.matrix, self.meta, self.dim, self.model,
         self.unverified, self.dropped) = cached
        self.np = _numpy()
        return self

    def embed(self, text: str) -> "object":
        if _embed_provider() == "onnx":
            from handrolled.embed_onnx import embed_texts as _embed_texts

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




class ImageArm:
    """只读的图片向量库视图（P2 侧车，默认关闭）。

    和 :class:`VectorArm` 同构：全库矩阵预加载 + 按 ``(路径, mtime, size)``
    缓存，归一化后用点积当余弦。区别是这里**没有**页级去重 —— 一张图就是一张图。

    本类只负责「图片空间里的排序」；库路径、模型、维度、凭据读取都在
    :mod:`dsh_kb.image_embed`，本模块不碰密钥。
    """

    def __init__(self, db: Path | str, timeout: float = 10.0):
        self.db = Path(db)
        self.timeout = timeout
        self.np = None
        self.matrix = None
        self.paths: list[str] = []
        self.total = 0
        self.dim = 0
        self.model = ""

    def load(self) -> "ImageArm":
        if self.matrix is not None:
            return self
        if not self.db.is_file():
            raise FileNotFoundError(f"图片向量库不存在：{self.db}")
        st = self.db.stat()
        key = (str(self.db), int(st.st_mtime), int(st.st_size))
        cached = _IMAGE_CACHE.get(key)
        if cached is None:
            from .image_embed import load_vectors

            paths, matrix, dim, model = load_vectors(self.db)
            cached = (paths, matrix, dim, model)
            _IMAGE_CACHE.clear()
            _IMAGE_CACHE[key] = cached
        self.paths, self.matrix, self.dim, self.model = cached
        self.total = len(self.paths)     # 诊断用：库里有几张，前缀命中几张
        self.np = _numpy()
        return self

    def embed_query(self, text: str) -> "object":
        """查询向量：文本侧（Query 指令），与图片语料同一个多模态空间。"""
        from .image_embed import embed_text

        np = self.np or _numpy()
        vec = embed_text(text, timeout=self.timeout)
        arr = np.asarray(vec, dtype="<f4")
        if self.dim and len(arr) != self.dim:
            raise RuntimeError(f"查询向量维度 {len(arr)} 与图片库 {self.dim} 不一致")
        norm = float(np.linalg.norm(arr)) or 1.0
        return arr / norm

    def query(self, text: str, prefixes: tuple[str, ...], depth: int = 32) -> list[tuple[str, int, float]]:
        """返回 ``[(相对路径, 1, 余弦)]``，已按分数降序、已做 zone 前缀过滤。

        行号恒为 1：图片是整文件命中的一个「块」，没有行概念。
        """
        self.load()
        if prefixes and not any(str(p).startswith(prefixes) for p in self.paths):
            # 库里有图，但一张都不在本次允许的前缀下（本库的真实状态：19 张全在
            # z3-skills/，而生产 z01 前缀是 z0-public/z1-restricted/...）。
            # 这时直接返回空，**不要**再发一次查询嵌入 —— 既省一次网络往返，
            # 也避免在 Ark 欠费（403）时白跑 3 次重试+退避。
            return []
        q = self.embed_query(text)
        scores = self.matrix @ q
        take = max(int(depth) * 2, 16)
        order = self.np.argsort(-scores)[:take]
        out: list[tuple[str, int, float]] = []
        for idx in order:
            path = self.paths[int(idx)]
            if prefixes and not str(path).startswith(prefixes):
                continue
            out.append((str(path), 1, float(scores[int(idx)])))
            if len(out) >= depth:
                break
        return out


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _image_settings(cfg) -> tuple[bool, float, Path]:
    """宽读图片侧车配置：缺字段就用默认值（不依赖配置层先长字段）。

    取值优先级与仓库其余 rag 项一致：``DSH_KB_RAG_<FIELD>`` 环境变量覆盖
    ``config.json``。当前 ``config.py`` 已有 ``image_arm_enabled/weight/db``
    三个字段；宽读保留是为了「旧配置 + 新代码」「新配置 + 旧测试替身」都能跑。
    """
    # 注意：能读到这里不代表检索一定会走进来 —— index.py 的 need_vector 门控
    # 还要求 vector_enabled 且（vector_always 或 FTS 未满额或弱命中留槽）。
    # 详见 augment() 顶部注释与 docs/semantic-arm.md「触发条件」。
    env_enabled = _env_flag("DSH_KB_RAG_IMAGE_ARM_ENABLED")
    enabled = bool(getattr(cfg.rag, "image_arm_enabled", False)) if env_enabled is None else env_enabled
    raw_weight = os.environ.get("DSH_KB_RAG_IMAGE_ARM_WEIGHT")
    if raw_weight is None:
        raw_weight = getattr(cfg.rag, "image_arm_weight", DEFAULT_IMAGE_WEIGHT)
    try:
        weight = float(raw_weight if raw_weight is not None else DEFAULT_IMAGE_WEIGHT)
    except (TypeError, ValueError):
        weight = DEFAULT_IMAGE_WEIGHT
    if weight < 0:
        weight = DEFAULT_IMAGE_WEIGHT
    db = (os.environ.get("DSH_KB_RAG_IMAGE_ARM_DB")
          or os.environ.get("DSH_KB_RAG_IMAGE_DB")  # 旧名，保留兼容
          or getattr(cfg.rag, "image_db", None)
          or _default_image_db(cfg))
    return enabled, weight, Path(db)


def _default_image_db(cfg) -> Path:
    """默认与文本向量库同目录的 ``vectors-image.sqlite``。"""
    vd = getattr(cfg.rag, "vectors_db", "") or ""
    if vd:
        return Path(vd).parent / "vectors-image.sqlite"
    home = getattr(cfg, "home", None)
    if not home:
        home = os.environ.get("DSH_KB_HOME") or "~/.dsh-kb"
    return Path(home).expanduser() / "index" / "vectors-image.sqlite"


def _image_prefixes(cfg, zone_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    """图片候选允许的路径前缀：默认与本次查询的 zone 前缀一致。

    **为什么要有这个开关（major-3）**：本库 z0-public / z1-restricted 里一张
    图都没有，19 张全在 ``z3-skills/``（该 zone ``index:false``，也不在
    ``query_prefixes("z01")`` 里）。也就是说默认配置下图片路候选恒为 0 ——
    这不是接线错误，是「本库 z01 没有图源」。

    ``rag.image_arm_prefixes``（或 ``DSH_KB_RAG_IMAGE_ARM_PREFIXES``，逗号分隔）
    可以显式放宽前缀，让 z01 查询也能召回 z3 的图片。**默认关着**：z3 是
    skills 素材区（配色图、示例截图），与知识查询语义无关，而且让 z01 的结果
    里出现 z3 路径会破坏 zone 约定。除非 z01 真的有了图、或调用方明确要跨区
    取图，否则不要把 z3 加进来。
    """
    raw = os.environ.get("DSH_KB_RAG_IMAGE_ARM_PREFIXES")
    if raw is None:
        raw = getattr(cfg.rag, "image_arm_prefixes", None)
    if not raw:
        return tuple(zone_prefixes)
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    else:
        try:
            parts = list(raw)
        except TypeError:
            return tuple(zone_prefixes)
    out: list[str] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        if not text.endswith("/"):
            text += "/"          # 统一带斜杠："z3-skills" 不该匹配 "z3-skills-old/"
        if text not in out:
            out.append(text)
    return tuple(out) or tuple(zone_prefixes)


def _image_candidates(cfg, query: str, prefixes: tuple[str, ...],
                      depth: int) -> tuple[list[tuple[str, int, float]], str, float]:
    """图片路候选 + 说明 + 生效权重。**绝不抛异常**，失败只写 note。

    覆盖的降级面：关闭 / 无 numpy / 库不存在 / 无 key / HTTP 失败 / schema 不符
    —— 任何一种都返回空候选，文本侧结果照常返回。

    2026-11-06 21:20 实测：火山方舟 PAYG 账户返回 ``403 AccountOverdueError``
    （欠费），此前 20:23 那批是最后一次成功。欠费/403 必须走这条通用降级路径，
    **不允许任何一条路径因为外部账户状态让 search 报错**；下面的 ``except
    Exception`` 就是它的兜底，note 里会写明 HTTP 403。
    """
    enabled, weight, db = _image_settings(cfg)
    if not enabled:
        return [], "", weight
    arm = ImageArm(db, timeout=float(getattr(cfg.rag, "image_timeout", 10.0) or 10.0))
    try:
        hits = arm.query(query, prefixes, depth=max(1, int(depth)))
    except FileNotFoundError:
        return [], "图片路跳过：库不存在", weight
    except ImportError as exc:
        return [], f"图片路跳过：缺 numpy（{exc}）", weight
    except Exception as exc:  # noqa: BLE001 - 可选路只降级
        return [], f"图片路跳过：{type(exc).__name__}: {exc}", weight
    if not hits and getattr(arm, "total", 0):
        # 库里有图但当前前缀一张都不匹配 —— 本库默认配置下的真实状态
        # （19 张全在 z3-skills/，而 z01 前缀是 z0-public/z1-restricted）。
        return [], (f"图片路 0 候选（库内 {arm.total} 张，当前前缀 "
                    f"{list(prefixes)} 无命中）"), weight
    return hits, f"图片路 {len(hits)} 候选（权重 {weight:g}，库={db.name}）", weight


def _hit_snippet(root: Path, path: str, line: int, chars: int) -> str:
    """图片命中给个人类可读片段；文本命中仍读镜像文件。

    图片是二进制，走 ``_snippet`` 会读出乱码，所以先按扩展名分流。
    """
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        return "[图片] " + Path(path).name
    return _snippet(root, path, line, chars)


def image_arm_readiness(cfg, *, db: Path | str | None = None, zone: str = "z01") -> dict:
    """回答「图片路到底有没有机会被用上」：库内数据分布 + 上游门控。

    只读、不联网、不 import numpy、不读密钥 —— 可以随时在生产上跑。返回的
    ``verdict`` 是一句人话结论；``usable`` 只有在「启用 + 库有数据 + 当前 zone
    前缀命中 + vector_always 打开」四条全满足时才是 True。

    用途：major-3/major-4 的验收数据都从这里出（CLI：
    ``python3 -m dsh_kb.image_embed --readiness``）。
    """
    enabled, weight, settings_db = _image_settings(cfg)
    db_path = Path(db) if db else settings_db
    try:
        zone_prefixes = tuple(cfg.query_prefixes(zone))
    except Exception:
        zone_prefixes = ()
    prefixes = _image_prefixes(cfg, zone_prefixes)
    info: dict = {
        "enabled": enabled,
        "weight": weight,
        "db": str(db_path),
        "db_exists": db_path.is_file(),
        "zone": zone,
        "zone_prefixes": list(zone_prefixes),
        "image_prefixes": list(prefixes),
        "rows": 0,
        "matched": 0,
        "rows_by_zone": {},
        "outside": [],
        "gate": {},
        "usable": False,
        "verdict": "",
    }
    paths: list[str] = []
    if info["db_exists"]:
        try:
            with sqlite3.connect("file:%s?mode=ro" % db_path, uri=True) as con:
                # 取全列而不是只取 path：schema 不对时要能被 sqlite 抓出来，
                # 不能因为「只读 path 也能读」就误报库是好的。
                rows = con.execute(
                    "select path, sha, dim, model, vec, updated from images"
                ).fetchall()
            paths = [str(r[0]) for r in rows]
        except sqlite3.Error as exc:
            info["verdict"] = f"图片库不可读或 schema 不符：{exc}"
            return info
        except Exception as exc:  # noqa: BLE001 - 诊断函数同样不抛
            info["verdict"] = f"图片库读取失败：{type(exc).__name__}: {exc}"
            return info
    info["rows"] = len(paths)
    by_zone: dict = {}
    for path in paths:
        head = path.split("/", 1)[0] + "/" if "/" in path else path
        by_zone[head] = by_zone.get(head, 0) + 1
    info["rows_by_zone"] = by_zone
    matched = [p for p in paths if p.startswith(prefixes)] if prefixes else list(paths)
    info["matched"] = len(matched)
    matched_set = set(matched)
    info["outside"] = sorted(p for p in paths if p not in matched_set)[:20]

    vector_enabled = bool(getattr(cfg.rag, "vector_enabled", False))
    vector_always = bool(getattr(cfg.rag, "vector_always", False))
    if not vector_enabled:
        gate_note = "文本向量臂未启用（rag.vector_enabled=false）⇒ index.search 不会调用 augment"
    elif not vector_always:
        gate_note = ("缺 rag.vector_always=true ⇒ FTS 命中满额时 index.search 不会调用 augment，"
                     "图片臂只能搭「命中不足/弱命中」的便车")
    else:
        gate_note = "vector_always=true ⇒ 即使 FTS 命中满额也会进 augment"
    info["gate"] = {
        "vector_enabled": vector_enabled,
        "vector_always": vector_always,
        "reachable_when_fts_full": bool(vector_enabled and vector_always),
        "note": gate_note,
    }

    if info["db_exists"] and info["rows"] and info["matched"] == 0:
        # 数据结论优先于开关状态：即使把 enabled 打开，本库 z01 也没有图源。
        verdict = (f"图片侧车在本库当前无可用图源：库内 {info['rows']} 张图片均不在 "
                   f"{zone} 前缀 {list(zone_prefixes)} 下（实际分布 {by_zone}）；"
                   + ("当前未启用（默认关）" if not enabled else "即使启用也不会产生候选"))
    elif not enabled:
        verdict = "未启用（rag.image_arm_enabled=false，默认值）"
    elif not info["db_exists"]:
        verdict = f"库不存在：{db_path}（先跑 image_embed --build）"
    elif info["rows"] == 0:
        verdict = "库为空"
    elif not vector_enabled:
        verdict = f"有 {info['matched']} 张前缀命中，但文本向量臂未启用 ⇒ 不会被调用"
    elif not vector_always:
        verdict = (f"有 {info['matched']} 张前缀命中，但缺 rag.vector_always=true："
                   f"FTS 命中满额的查询不会调用图片臂（只是偶尔生效）")
    else:
        verdict = f"可用：{info['matched']} 张候选前缀命中，且 vector_always 已开"
    info["verdict"] = verdict
    info["usable"] = bool(enabled and info["db_exists"] and info["rows"]
                          and info["matched"] and vector_enabled and vector_always)
    return info


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
    """给一组关键词命中补上向量路，按 RRF 融合后返回 (新命中列表, 说明)。

    图片臂的**上游触发条件**（major-4，不能在本模块里修）：index.py 只在
    ``vector_enabled and (vector_always or len(hits) < limit or weak-reserve)``
    时才调用本函数；生产 ``vector_trigger=auto`` 且没有 ``vector_always`` 时，
    FTS 命中满额就根本不会进来 —— 图片臂等于死代码。``vector_trigger=always``
    单独设也没用（它只影响弱命中的留槽，不改变 need_vector）。所以启用图片臂
    必须同时设 ``rag.vector_always=true``；这条前置条件写在
    docs/semantic-arm.md「触发条件」里，并有用例
    ``TestUpstreamGate.test_fts_full_query_reaches_image_arm_with_vector_always``
    用真实 index.search 证明「满额时仍被调用」。
    """
    prefixes = tuple(cfg.query_prefixes(zone))
    img_prefixes = _image_prefixes(cfg, prefixes)
    arm = VectorArm(cfg.rag.vectors_db, use_d2q=bool(getattr(cfg.rag, "d2q_enabled", False)))
    depth = int(getattr(cfg.rag, "vector_depth", depth) or depth)
    vec = arm.query(query, prefixes, depth=depth, hyde_text=hyde)
    # 加载时丢掉过旧块向量 / 没能校验索引一致性 —— 两者都要说出来，别静默：
    # 这个缺陷原本的症状就是「排序悄悄错位，没有任何提示」。
    health = arm.health_note()
    if rerank and len(vec) > 1 and getattr(cfg.rag, "rerank_enabled", False):
        try:
            from handrolled import rerank as _rr
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
    # 图片第三路（P2 侧车）：默认关闭；开启后与词法臂、文本向量臂一起做 RRF。
    # 位置在文本向量臂（含重排）之后 —— 关闭时这里一行都不执行，行为与旧版一致。
    img, img_note, img_w = _image_candidates(cfg, query, img_prefixes, depth)
    if not vec and not img:
        note = "vector arm: 无候选"
        if health:
            note += f"；{health}"
        if img_note:
            note += f"；{img_note}"
        return existing, note
    if mode == "fill":
        # 只补齐：关键词顺序原样保留，向量独有候选接在后面。
        # 目的：不让向量路改写「精确查询的 top-1」（实测 `挂载` 会被换掉）。
        out = list(existing)
        seen_pairs = {(h.path, h.line) for h in out}
        root = cfg.mirror_dir
        from .index import Hit
        # 图片候选接在文本向量候选之后：fill 模式下名次只决定补齐顺序，
        # 权重不参与（那条公式是 RRF 的，fill 仍沿用旧口径的 -1/(k+rank)）。
        for rank, (path, line, score) in enumerate(list(vec) + list(img), 1):
            if (path, line) in seen_pairs:
                continue
            if len(out) >= limit:
                break
            out.append(Hit(path=path, line=line,
                           score=round(-1.0 / (RRF_K + rank), 6),
                           snippet=_hit_snippet(root, path, line, snippet_chars), zone=""))
            seen_pairs.add((path, line))
        note = (f"vector arm 补齐 {len(out) - len(existing)} 条（fill 模式，关键词顺序不变）；"
                f"库={Path(cfg.rag.vectors_db).name}")
        if health:
            note += f"；{health}"
        if img_note:
            note += f"；{img_note}"
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
    # 图片路：权重独立（默认 0.6），名次从 1 起算，score 仍是 RRF 值。
    for rank, (path, line, _score) in enumerate(img, 1):
        fused[(path, line)] = fused.get((path, line), 0.0) + img_w / (RRF_K + rank)
    have = {(h.path, h.line): h for h in existing}
    root = cfg.mirror_dir
    from .index import Hit

    merged: list = []
    for (path, line), score in sorted(fused.items(), key=lambda kv: -kv[1])[:limit]:
        hit = have.get((path, line))
        if hit is None:
            hit = Hit(path=path, line=line, score=score,
                      snippet=_hit_snippet(root, path, line, snippet_chars), zone="")
        else:
            hit.score = score
        merged.append(hit)
    note = (f"vector arm 融合 {len(vec)} 候选（RRF k={RRF_K}，score 为 RRF 值）；"
            f"库={Path(cfg.rag.vectors_db).name} 模型={arm.model or 'bge-m3'}")
    if health:
        note += f"；{health}"
    if img_note:
        note += f"；{img_note}"
    return merged, note