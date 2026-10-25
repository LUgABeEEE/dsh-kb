"""cross-encoder 重排（可选）：对弱命中查询的候选做 query-passage 交叉打分。

为什么单独一个模块：
  · 依赖 onnxruntime + tokenizers，装在一个独立目录（不进 user site），由
    ``DSH_KB_RERANK_ROOT`` 指向：该目录下要有 ``lib/``（能 import 的依赖）与
    ``models/model_quantized.onnx`` + ``models/tokenizer.json``；
  · 模型是 Xenova/bge-reranker-base 的 int8 ONNX（约 280 MB），本仓库不附带、不下载；
  · 全部延迟加载 —— 没装依赖、没下模型时，本模块 import 不报错、调用返回 None，检索退回原顺序。

设计约束（2026-10-23）：只在「弱命中」时调用（top-1 覆盖率 < vector_weak_ratio），
因为交叉编码器每对约 30–50ms（CPU），30 对 ≈ 1–2s，不能每次检索都付。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: Unset means the feature is off: a fresh clone has no model and no dependencies,
#: and nothing here may guess at a machine-specific location.
_ROOT = os.environ.get('DSH_KB_RERANK_ROOT')
ROOT = Path(_ROOT) if _ROOT else None
LIB = (ROOT / 'lib') if ROOT else None
MODEL_DIR = (ROOT / 'models') if ROOT else None
MODEL_FILE = (MODEL_DIR / 'model_quantized.onnx') if MODEL_DIR else None
TOKENIZER_FILE = (MODEL_DIR / 'tokenizer.json') if MODEL_DIR else None

_STATE: dict = {}


def available() -> bool:
    if ROOT is None:
        return False
    return MODEL_FILE.is_file() and TOKENIZER_FILE.is_file() and LIB.is_dir()


def _load():
    if 'session' in _STATE:
        return _STATE['session'], _STATE['tok']
    if not available():
        raise RuntimeError('重排依赖未安装或模型缺失')
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    import onnxruntime as ort
    from tokenizers import Tokenizer

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4)))
    sess = ort.InferenceSession(str(MODEL_FILE), so, providers=['CPUExecutionProvider'])
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    tok.enable_truncation(max_length=int(os.environ.get("DSH_KB_RERANK_MAXLEN", "192")))
    tok.enable_padding(length=None)
    _STATE['session'], _STATE['tok'] = sess, tok
    return sess, tok


def score_pairs(query: str, passages: list[str]) -> list[float] | None:
    """对 (query, passage) 逐对打分；任何失败都返回 None（调用方退回原顺序）。"""
    if not available() or not passages:
        return None
    try:
        sess, tok = _load()
        enc = tok.encode_batch([(query, p) for p in passages])
        import numpy as np

        ids = np.asarray([e.ids for e in enc], dtype='int64')
        mask = np.asarray([e.attention_mask for e in enc], dtype='int64')
        ttype = np.zeros_like(ids)
        feed = {i.name: v for i, v in zip(sess.get_inputs(), (ids, mask, ttype))}
        out = sess.run(None, feed)[0]
        out = np.asarray(out).reshape(len(passages), -1)
        return [float(row[0]) for row in out]
    except Exception:
        return None