"""深池 LLM 重排（`deep=true` 慢车道）。

为什么单独一层而不是改主链路：
* 主链路要快（p95 < 1s），LLM 一次 5~15s，只能按需触发；
* 需要「看得到更远的候选」——先把候选扩到 40 页，再让模型挑；
* 失败必须静默降级：API 挂了就退回普通排序，绝不影响检索可用性。

实测（2026-10-24，33 题目径）：普通 25/33 → 深池 + 挑 1 页 = 26/33，代价 p50 8.4s。
单次用量约 3.7k 输入 token / 120 输出 token。

注意一个实测过的失败模式：候选多时模型会报出**超出范围的编号**（例如只有 30 页却答 31）。
所以解析不到合法编号时，会用「前 10 页」重试一次，仍然失败才降级。
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Sequence

#: This is an opt-in slow lane: nothing below runs unless the deployment says
#: where to send the request and with which credentials.  No vendor endpoint,
#: no key file and no log path is assumed.
DEFAULT_MODEL = ""
DEFAULT_BASE = ""
CRED_PATH = os.environ.get("DSH_KB_DEEP_CREDENTIALS") or ""

LAST_RAW = ""
LAST_ERROR = ""
LOG_PATH = os.environ.get("DSH_KB_DEEP_LOG") or ""

PROMPT = """用户问题：%s

下面是知识库候选页面（编号 1-%d，含该页最相关片段）。请挑出最可能回答了该问题的 %d 个页面，
按可能性从高到低输出编号，逗号分隔，只输出编号（必须是 1 到 %d 之间的整数）。

%s"""


def _api_key() -> str | None:
    """API key: ``DSH_KB_DEEP_API_KEY`` first, then an optional credentials file.

    The file is whatever ``DSH_KB_DEEP_CREDENTIALS`` points at and only has to
    contain ``DEEP_API_KEY`` (top level or under ``refs``).  YAML is used when
    PyYAML happens to be importable, otherwise one line is matched with a regex —
    the MCP server interpreter may not have yaml installed.
    """
    global LAST_ERROR
    key = (os.environ.get("DSH_KB_DEEP_API_KEY") or "").strip()
    if key:
        return key
    if not CRED_PATH:
        LAST_ERROR = "no-key"
        return None
    try:
        raw = open(CRED_PATH, encoding="utf-8").read()
    except Exception as exc:
        LAST_ERROR = "cred-read: %s" % type(exc).__name__
        return None
    try:
        import yaml

        data = yaml.safe_load(raw) or {}
        refs = data.get("refs") or data
        value = refs.get("DEEP_API_KEY")
        if value:
            return str(value)
    except Exception:
        pass
    m = re.search(r"^\s*DEEP_API_KEY\s*:\s*(\S+)", raw, re.M)
    if m:
        return m.group(1).strip().strip(chr(34)).strip(chr(39))
    LAST_ERROR = "no-key"
    return None


def _endpoint() -> str:
    """OpenAI-compatible base URL, without the trailing slash."""
    return (os.environ.get("DSH_KB_DEEP_ENDPOINT") or DEFAULT_BASE).rstrip("/")


def _parse_order(text: str, n: int, pick: int) -> list[int]:
    """把「3,1,7」解析成 1 基下标；越界与重复丢掉。"""
    out: list[int] = []
    for raw in re.findall(r"\d+", text or ""):
        try:
            idx = int(raw)
        except ValueError:
            continue
        if 1 <= idx <= n and idx not in out:
            out.append(idx)
        if len(out) >= pick:
            break
    return out


def _request(query: str, pages: Sequence[tuple[str, str]], *, key: str, model: str,
             pick: int, timeout: float) -> tuple[list[int], str]:
    blocks = ["[%d] %s" % (i, path.split("/")[-1]) + chr(10) + (body or "")[:320]
              for i, (path, body) in enumerate(pages, 1)]
    prompt = PROMPT % (query, len(pages), pick, len(pages), chr(10) + chr(10).join(blocks))
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 400,
                       "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(_endpoint() + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    return _parse_order(text, len(pages), pick), text


def _log_failure(query: str, error: str, raw: str) -> None:
    """降级时留一行证据：MCP 进程看不到 stdout，只有文件能查。

    Nothing is written unless ``DSH_KB_DEEP_LOG`` is set — a library has no
    business picking a log location on someone else's machine.
    """
    if not LOG_PATH:
        return
    try:
        from datetime import datetime

        line = "%s\t%s\t%s\t%s" % (
            datetime.now().isoformat(timespec="seconds"),
            error,
            (query or "")[:40].replace("\t", " "),
            (raw or "").replace("\t", " ")[:40],
        )
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + chr(10))
    except Exception:
        pass


def pick_pages(
    query: str,
    pages: Sequence[tuple[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    pick: int = 1,
    timeout: float = 25.0,
) -> list[int] | None:
    """让模型从候选页面里挑 pick 个；失败返回 None。

    ``pages`` 是 [(path, passage)]，passage 由调用方切好（避免这里再读盘）。
    """
    global LAST_RAW, LAST_ERROR
    LAST_RAW, LAST_ERROR = "", ""
    if not pages:
        return []
    key = _api_key()
    if not key:
        LAST_ERROR = "no-key"
        return None
    if not _endpoint() or not model:
        LAST_ERROR = "no-endpoint-or-model"
        return None
    for attempt, subset in enumerate((list(pages), list(pages)[:10])):
        if attempt and len(subset) >= len(pages):
            break
        try:
            order, text = _request(query, subset, key=key, model=model, pick=pick,
                                   timeout=timeout if attempt == 0 else min(timeout, 15.0))
        except Exception as exc:  # noqa: BLE001 - 任何失败都降级
            LAST_ERROR = "%s: %s" % (type(exc).__name__, str(exc)[:120])
            continue
        LAST_RAW = (text or "").strip()[:80]
        if order:
            return order
        LAST_ERROR = "unparsable"
    _log_failure(query, LAST_ERROR, LAST_RAW)
    return None


def reorder(items: Sequence, order: Sequence[int]) -> list:
    """按 1 基下标把挑中的提到最前，其余保序。"""
    idx = [i - 1 for i in order if 0 < i <= len(items)]
    picked = [items[i] for i in idx]
    rest = [x for j, x in enumerate(items) if j not in set(idx)]
    return picked + rest