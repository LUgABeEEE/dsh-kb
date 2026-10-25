"""ONNX 嵌入后端（Qwen3-Embedding-0.6B int8）—— 换模型 A/B 实验用。

为什么单独一个模块：
* 生产嵌入走 ollama（bge-m3）。要评估"换模型能不能把召回抬上去"，必须能在
  不改生产代码路径的前提下换掉**嵌入实现**，同时仍复用 VectorArm 的矩阵缓存、
  页级去重、重排与融合逻辑 —— 那些是逻辑资产，不该跟着模型一起换。
* 模型是 int8 ONNX（约 610 MB），本仓库不附带、不下载；路径由
  ``DSH_KB_EMBED_ONNX_ROOT`` 指定（其下 ``onnx/model_int8.onnx`` + ``tokenizer.json``），
  依赖库目录由 ``DSH_KB_ONNX_LIB`` 指定（可选，能 import 到即可）。

池化方式（Qwen3-Embedding 官方口径）：
* 取**最后一个 token** 的 hidden state（不是 CLS，也不是 mean）；
* L2 归一化；
* 查询侧加指令前缀 ``Instruct: {task}\nQuery: {query}``，文档侧不加 —— 这是它
  与非对称检索模型拉开差距的地方，和 bge-m3 的用法不同，别混。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

__all__ = ["available", "embed_texts", "embed_query", "dim", "MODEL_NAME"]

#: Both are unset by default: the ONNX backend is an experiment, not a shipped path.
_ROOT = os.environ.get("DSH_KB_EMBED_ONNX_ROOT")
_LIB = os.environ.get("DSH_KB_ONNX_LIB")
ROOT = Path(_ROOT) if _ROOT else None
LIB = Path(_LIB) if _LIB else None
MODEL_NAME = os.environ.get("DSH_KB_EMBED_ONNX_NAME", "qwen3-embedding-0.6b-int8")
INSTRUCTION = os.environ.get(
    "DSH_KB_EMBED_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)
DIM = 1024
MAX_LEN = int(os.environ.get("DSH_KB_EMBED_MAXLEN", "512"))
BATCH = int(os.environ.get("DSH_KB_EMBED_BATCH", "8"))

_LOCK = threading.Lock()
_SESSION = None
_TOKENIZER = None
_INPUT_NAMES = ()
_INPUT_META = ()


def _lib_path() -> None:
    if LIB is not None and LIB.is_dir() and str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))


def available() -> bool:
    """模型文件齐全且能 import 依赖时才为 True（不加载模型，不看网络）。"""
    if ROOT is None:
        return False
    need = [ROOT / "onnx" / "model_int8.onnx", ROOT / "tokenizer.json"]
    if not all(p.is_file() and p.stat().st_size > 1000 for p in need):
        return False
    try:
        _lib_path()
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except Exception:
        return False
    return True


def _load():
    global _SESSION, _TOKENIZER, _INPUT_NAMES
    if _SESSION is not None:
        return _SESSION, _TOKENIZER
    with _LOCK:
        if _SESSION is not None:
            return _SESSION, _TOKENIZER
        _lib_path()
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(os.environ.get("DSH_KB_EMBED_THREADS", "4"))
        opts.inter_op_num_threads = 1
        # 内存：推理 arena 会一直占着，和别的常驻服务共用一台小机器时很痛。默认
        # 中间张量池留到进程结束（实测 4 进程各占 3G），关掉后 RSS 降到 ~1.3G。
        opts.enable_cpu_mem_arena = os.environ.get("DSH_KB_EMBED_ARENA", "0") == "1"
        sess = ort.InferenceSession(
            str(ROOT / "onnx" / "model_int8.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        tok = Tokenizer.from_file(str(ROOT / "tokenizer.json"))
        tok.enable_truncation(max_length=MAX_LEN)
        tok.no_padding()
        _SESSION, _TOKENIZER = sess, tok
        global _INPUT_META
        _INPUT_NAMES = tuple(i.name for i in sess.get_inputs())
        _INPUT_META = tuple((i.name, list(i.shape or ())) for i in sess.get_inputs())
        return sess, tok


def dim() -> int:
    return DIM


def _postprocess(hidden, mask):
    """last-token pooling（按 attention_mask 取每条最后一个真实 token）+ L2。"""
    import numpy as np

    if isinstance(hidden, (list, tuple)):
        hidden = hidden[0]
    arr = np.asarray(hidden)
    if arr.ndim == 3:
        idx = np.asarray(mask).sum(axis=1) - 1
        idx = np.clip(idx, 0, arr.shape[1] - 1)
        arr = arr[np.arange(arr.shape[0]), idx]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_texts(texts, *, is_query: bool = False, batch_size: int | None = None):
    """批量嵌入，返回 list[list[float]]（float32 归一化向量）。"""
    if isinstance(texts, str):
        texts = [texts]
    sess, tok = _load()
    import numpy as np

    bs = batch_size or BATCH
    prepared = []
    for t in texts:
        body = str(t or "").strip()
        if is_query:
            body = "Instruct: " + INSTRUCTION + "\nQuery: " + body
        prepared.append(body or " ")
    out = []
    for i in range(0, len(prepared), bs):
        chunk = prepared[i : i + bs]
        encs = tok.encode_batch(chunk)
        maxlen = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), maxlen), dtype="int64")
        mask = np.zeros((len(encs), maxlen), dtype="int64")
        for r, e in enumerate(encs):
            ids[r, : len(e.ids)] = e.ids
            mask[r, : len(e.ids)] = 1
        feeds = {
            "input_ids": ids,
            "attention_mask": mask,
            "position_ids": np.tile(np.arange(maxlen, dtype="int64"), (len(encs), 1)),
        }
        # 这个 ONNX 导出是「带 KV cache」的解码器图：56 个 past 输入必须显式给空张量
        # （past_sequence_length=0）。不给就直接 ValueError；给了等价于无缓存的整段前向，
        # 我们要的正是 last_hidden_state 整段。
        for name, meta in _INPUT_META:
            if name in feeds:
                continue
            if name.startswith("past_key_values"):
                shape = [len(encs) if str(s).startswith("batch") else int(s)
                         if not str(s).startswith("past") else 0
                         for s in (meta or [])]
                feeds[name] = np.zeros(shape or (len(encs), 1, 0, 1), dtype="float32")
        hidden = sess.run(None, feeds)
        vecs = _postprocess(hidden, mask)
        out.extend([[float(x) for x in row] for row in vecs])
    return out


def embed_query(text: str):
    return embed_texts([text], is_query=True)[0]
