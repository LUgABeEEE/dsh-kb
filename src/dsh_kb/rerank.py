"""cross-encoder rerank (optional): rescore candidates for weak-hit queries.

Why a separate module:

  · it needs ``onnxruntime`` + ``tokenizers`` in an isolated directory (not the
    user site), pointed at by ``DSH_KB_RERANK_ROOT``: that directory has to hold
    ``lib/`` (importable dependencies) and ``models/model_quantized.onnx`` +
    ``models/tokenizer.json``;
  · the model is ``Xenova/bge-reranker-base`` int8 ONNX (about 280 MB) — this
    repository ships no weights and downloads nothing;
  · everything loads lazily: with no dependencies and no model, importing this
    module succeeds, ``score_pairs`` returns ``None`` and search falls back to
    the original order.

Design constraint: it is only called for weak hits (top-1 coverage below
``vector_weak_ratio``), because a cross-encoder costs about 30–50 ms per pair on
CPU — 30 pairs is 1–2 s and must not be paid on every query.

Idle unload
-----------
Measured residency after the first search is roughly 780 MB (session ~395 MB,
the 17 MB ``tokenizer.json`` expanded to ~214 MB, plus a ~170 MB inference
arena) while a cold load takes only 0.92 s.  Giving it back is therefore cheap:

  · ``DSH_KB_RERANK_IDLE_S`` (default 600 s, ``0`` disables) — after that much
    quiet time the session and tokenizer are released and resident memory falls
    back to ~90 MB; the next call rebuilds them in ~0.92 s.
  · eviction runs on a daemon thread and shares one lock with inference, so a
    model is never dropped mid-call.
  · eviction does not change any score: same model, same input, bit-identical
    before and after a reload.
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path

#: Unset means the feature is off: a fresh clone has no model and no
#: dependencies, and nothing here may guess at a machine-specific location.
_ROOT = os.environ.get("DSH_KB_RERANK_ROOT")
ROOT = Path(_ROOT) if _ROOT else None
LIB = (ROOT / "lib") if ROOT else None
MODEL_DIR = (ROOT / "models") if ROOT else None
MODEL_FILE = (MODEL_DIR / "model_quantized.onnx") if MODEL_DIR else None
TOKENIZER_FILE = (MODEL_DIR / "tokenizer.json") if MODEL_DIR else None

#: Seconds of quiet time before the model is released (0 = keep it resident).
DEFAULT_IDLE_S = 600
#: Where to note load/eviction events.  Unset means "do not log": a library has
#: no business picking a log path on someone else's machine.
_LOG = os.environ.get("DSH_KB_RERANK_LOG")
LOG_PATH = Path(_LOG) if _LOG else None

_STATE: dict = {}
_LOCK = threading.Lock()          # inference and load/unload are mutually exclusive


def _idle_seconds() -> int:
    try:
        return max(0, int(os.environ.get("DSH_KB_RERANK_IDLE_S", str(DEFAULT_IDLE_S))))
    except (TypeError, ValueError):
        return DEFAULT_IDLE_S


def _log(msg: str) -> None:
    if LOG_PATH is None:
        return
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} rerank: {msg}\n")
    except OSError:
        pass


def available() -> bool:
    if ROOT is None:
        return False
    return MODEL_FILE.is_file() and TOKENIZER_FILE.is_file() and LIB.is_dir()


def status() -> dict:
    """Diagnostics: loaded or not, idle for how long, eviction threshold."""
    idle = _idle_seconds()
    loaded = "session" in _STATE
    age = (time.monotonic() - _STATE.get("last_used", 0.0)) if loaded else None
    return {
        "loaded": loaded,
        "idle_seconds": idle,
        "idle_for": round(age, 1) if age is not None else None,
        "model": str(MODEL_FILE) if MODEL_FILE else "",
    }


def _start_watchdog_locked() -> None:
    if _idle_seconds() <= 0 or _STATE.get("watchdog") is not None:
        return
    t = threading.Thread(target=_watchdog, name="kb-rerank-idle", daemon=True)
    _STATE["watchdog"] = t
    t.start()


def _watchdog() -> None:
    """Poll the idle age; unload when it exceeds the threshold."""
    empty_rounds = 0
    while True:
        idle = _idle_seconds()
        if idle <= 0:
            _STATE.pop("watchdog", None)
            return
        time.sleep(max(5.0, min(60.0, idle / 2.0)))
        with _LOCK:
            if "session" not in _STATE:
                empty_rounds += 1
                if empty_rounds >= 2:     # two quiet rounds ⇒ stop until next load
                    _STATE.pop("watchdog", None)
                    return
                continue
            empty_rounds = 0
            age = time.monotonic() - _STATE.get("last_used", 0.0)
            if age >= idle:
                _unload_locked(f"idle {age:.0f}s >= {idle}s")


def _unload_locked(reason: str = "manual") -> bool:
    """Release session and tokenizer (caller must hold ``_LOCK``)."""
    if "session" not in _STATE and "tok" not in _STATE:
        return False
    _STATE.pop("session", None)
    _STATE.pop("tok", None)
    gc.collect()                      # drop the Python objects first
    # The important step: onnxruntime's large allocations stay in the glibc
    # arena after free(), so RSS barely moves (measured: 436 MB left).  Only
    # malloc_trim gives the arena back to the kernel (436 MB → 63 MB).
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    _log(f"unloaded ({reason})")
    return True


def unload(reason: str = "manual") -> bool:
    """Release the model now (tests and operators)."""
    with _LOCK:
        return _unload_locked(reason)


def _load_locked():
    if "session" in _STATE:
        _STATE["last_used"] = time.monotonic()
        return _STATE["session"], _STATE["tok"]
    if not available():
        raise RuntimeError("rerank dependencies or model files are missing")
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    import onnxruntime as ort
    from tokenizers import Tokenizer

    t0 = time.monotonic()
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4)))
    sess = ort.InferenceSession(str(MODEL_FILE), so, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(TOKENIZER_FILE))
    tok.enable_truncation(max_length=int(os.environ.get("DSH_KB_RERANK_MAXLEN", "192")))
    tok.enable_padding(length=None)
    _STATE["session"], _STATE["tok"] = sess, tok
    _STATE["last_used"] = time.monotonic()
    _log(f"loaded in {time.monotonic() - t0:.2f}s (idle unload after {_idle_seconds()}s)")
    _start_watchdog_locked()
    return sess, tok


def score_pairs(query: str, passages: list[str]) -> list[float] | None:
    """Score (query, passage) pairs; any failure returns None (caller keeps order)."""
    if not available() or not passages:
        return None
    try:
        with _LOCK:                   # exclusive with eviction: never dropped mid-call
            sess, tok = _load_locked()
            enc = tok.encode_batch([(query, p) for p in passages])
            import numpy as np

            ids = np.asarray([e.ids for e in enc], dtype="int64")
            mask = np.asarray([e.attention_mask for e in enc], dtype="int64")
            ttype = np.zeros_like(ids)
            feed = {i.name: v for i, v in zip(sess.get_inputs(), (ids, mask, ttype))}
            out = sess.run(None, feed)[0]
            out = np.asarray(out).reshape(len(passages), -1)
            _STATE["last_used"] = time.monotonic()
            return [float(row[0]) for row in out]
    except Exception:
        return None
