"""可选图片向量侧车：把镜像里的图片用火山方舟多模态模型向量化。

为什么要单独一个库：文本向量（``vectors.sqlite``）是 bge-m3 产的，维度 1024；
图片向量是 ``doubao-embedding-vision-251215`` 产的，维度 2048。两者不能塞进
同一张表，也不该互相影响 —— 图片侧车整库可以随时删掉重建。

实测得到的三条硬约束（照抄不要「优化」）：

1. **端点只能是** ``/api/v3/embeddings/multimodal``。``/api/v3/embeddings`` 会
   直接报 "does not support this api"。
2. **一次请求只放一条输入**。``input`` 数组传多条时接口**不报错**，只返回第一条
   的向量 —— 静默错位，最难查。（``tests/test_image_arm.py`` 对此有断言。）
3. **凭据键名由宿主指定**（``DSH_KB_IMAGE_KEY_NAME``，默认 ``IMAGE_EMBED_API_KEY``）。
   同一个服务通常既有订阅键又有按量键，拿错那个不会报「键错了」，而是报模型不支持——
   所以这里只认一个名字，绝不「找不到就换一个」。密钥只读进内存，绝不落盘、
   不写日志、不进异常消息。

默认关闭：``rag.image_arm_enabled``（或环境变量
``DSH_KB_RAG_IMAGE_ARM_ENABLED``）为真之前，检索路径连这个模块都不会 import。
构建是显式的离线动作（``--build``），不是检索副作用。

CLI::

    python3 -m dsh_kb.image_embed --build [--home /srv/kb] [--limit 20]
    python3 -m dsh_kb.image_embed --status

并发用 ``ThreadPoolExecutor``：一条一个 HTTP 请求，8 并发实测已经把 2048 维图片
（约 68 tokens/张）打到每秒几十张，再高只是给对端加压力。

外部账户状态（2026-11-06 21:20 实测）：火山方舟 PAYG 账户返回
``403 AccountOverdueError``（欠费），20:23 那批是最后一次成功嵌入。所以：

* 这条链路**必须默认关闭**，检索侧对任何失败（含 403 欠费）只能降级成一句
  note，绝不允许 search 因为外部账户状态报错；
* 欠费期间跑 ``--build`` 会把每张图记成 failed，但不写坏已有库（逐行 upsert +
  失败跳过），已有向量照常可读；
* 额度恢复后重跑 ``--build`` 即可（sha256 幂等，只会补没嵌过的）。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_ENDPOINT", "DEFAULT_MODEL", "DEFAULT_DIM", "IMAGE_SUFFIXES",
    "MIN_IMAGE_BYTES", "ImageFile", "ImageStoreError", "ImageAuthError", "ImageHTTPError",
    "discover_images", "sha256_file", "api_key", "embed_text", "embed_image",
    "build", "status", "load_vectors", "default_db", "main",
]

DEFAULT_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
DEFAULT_MODEL = "doubao-embedding-vision-251215"
#: 接口只接受 2048 / 1024（512 会 400）。默认 2048，且**不**对外暴露成可选项：
#: 换维度意味着整库重嵌，值得单独一次决定。
DEFAULT_DIM = 2048

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})

#: 小于这个字节数的「图片」不送。实测知识库里有 131 字节的 LFS 占位文件，
#: base64 发过去会报 "Invalid base64 image url"。
MIN_IMAGE_BYTES = 2048
#: 单张上限：整张图要 base64 进 JSON，太大既慢又容易撞请求体上限。
MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: 非对称指令（实测：Query 侧加指令后 MRR 0.9633 → 0.98）。
#: Corpus 侧（图片）**不带**结尾的 ``Query:``，Query 侧带。
_CORPUS_INSTRUCTIONS = "Target_modality: text.\nInstruction:根据这段文字描述，找到对应的图片"
_QUERY_INSTRUCTIONS = _CORPUS_INSTRUCTIONS + "\nQuery:"

#: 凭据文件位置。**故意没有默认值** —— 「凭据放哪」是部署事实，不是库该猜的东西。
#: 宿主用 ``DSH_KB_IMAGE_CREDENTIALS`` 指过来（或调用方显式传参）。
DEFAULT_CREDENTIALS = ""

#: 从凭据文件里读哪个键。默认中性名，宿主用 ``DSH_KB_IMAGE_KEY_NAME`` 改成自己
#: 实际用的名字（例如某个多模态服务的 PAYG 键）—— 厂商叫什么不该写死在库里。
#: 唯一的要求是**别混用同一个服务的订阅键**：那种键调 embedding 会因无权限而失败，
#: 而失败信息看起来像模型不存在，比直接说「键不对」难查得多。
DEFAULT_KEY_NAME = "IMAGE_EMBED_API_KEY"
CREDENTIAL_REF = os.environ.get("DSH_KB_IMAGE_KEY_NAME") or DEFAULT_KEY_NAME

#: 失败重试：3 次，指数退避（0.5s / 1s）。
RETRIES = 3
BACKOFF_S = 0.5
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_WORKERS = 8

_SCHEMA = """
create table if not exists images (
    path    text primary key,
    sha     text not null,
    dim     integer not null,
    model   text not null,
    vec     blob not null,
    updated integer not null
)
"""

_EXPECTED_COLUMNS = ("path", "sha", "dim", "model", "vec", "updated")

#: 凭据缓存：(文件路径, mtime, size) -> key。只在内存里，退出即消失。
_KEY_CACHE: dict[tuple, str | None] = {}


class ImageStoreError(RuntimeError):
    """图片向量库不可用（不存在 / 为空 / schema 不符 / 维度不一致）。"""


class ImageAuthError(RuntimeError):
    """读不到凭据（路径未配置，或文件里没有那个键）。消息里只有路径，绝不含密钥。"""


class ImageHTTPError(RuntimeError):
    """带状态码的 HTTP 失败：4xx（含欠费 403）不重试，5xx/429 才退避重试。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)

    @property
    def retryable(self) -> bool:
        return self.status == 429 or self.status >= 500


@dataclass(frozen=True)
class ImageFile:
    """磁盘上的一张图片（路径都相对镜像根，和文本索引同口径）。"""

    path: Path          # 绝对路径，仅用于读文件
    rel: str            # 相对镜像根的 POSIX 路径，写进库/命中
    sha: str
    size: int
    mtime: float


# ---------------------------------------------------------------- discovery --


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """整文件 sha256（十六进制）。幂等重建靠它，不靠 mtime。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _skip_dir(name: str) -> bool:
    return name in (".git", "node_modules", "__pycache__") or name.startswith(".")


def discover_images(mirror: Path | str, *, min_bytes: int = MIN_IMAGE_BYTES,
                    max_bytes: int = MAX_IMAGE_BYTES) -> tuple[list[ImageFile], dict]:
    """扫描镜像下的图片，返回 (可入索引的列表, 跳过统计)。

    跳过规则：扩展名不在 ``IMAGE_SUFFIXES``、小于 ``min_bytes``（LFS 占位图）、
    大于 ``max_bytes``、读不出来。返回的 ``rel`` 是 POSIX 风格，用于前缀（zone）
    过滤 —— 与文本命中完全一致。
    """
    root = Path(mirror)
    if not root.is_dir():
        raise FileNotFoundError(f"镜像目录不存在：{root}")
    found: list[ImageFile] = []
    skipped = {"small": 0, "large": 0, "unreadable": 0}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        for name in sorted(filenames):
            if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            path = Path(dirpath) / name
            try:
                st = path.stat()
            except OSError:
                skipped["unreadable"] += 1
                continue
            if st.st_size < min_bytes:
                skipped["small"] += 1
                continue
            if st.st_size > max_bytes:
                skipped["large"] += 1
                continue
            try:
                sha = sha256_file(path)
            except OSError:
                skipped["unreadable"] += 1
                continue
            rel = path.relative_to(root).as_posix()
            found.append(ImageFile(path=path, rel=rel, sha=sha,
                                   size=int(st.st_size), mtime=float(st.st_mtime)))
    found.sort(key=lambda it: it.rel)
    return found, skipped


# -------------------------------------------------------------- credentials --


def credentials_path() -> Path | None:
    """凭据文件位置；没配置时返回 ``None``（由调用方决定是报错还是静默跳过）。

    没有内置默认路径：库里写死一个部署路径，换个 home 就会读到别人的凭据，
    或者读到不存在的文件却报成「模型不可用」。
    """
    raw = os.environ.get("DSH_KB_IMAGE_CREDENTIALS") or DEFAULT_CREDENTIALS
    return Path(raw) if raw else None


def _parse_refs_loose(text: str) -> dict:
    """不依赖 PyYAML 的兜底解析：只认顶格 ``refs:`` 下两空格缩进的扁平键值。"""
    refs: dict = {}
    in_refs = False
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            in_refs = line.rstrip() == "refs:"
            continue
        if not in_refs:
            continue
        m = re.match(r"^ {2}([A-Za-z0-9_.\-]+):[ \t]*(.*)$", line)
        if not m:
            continue
        value = m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            refs[m.group(1)] = value
    return refs


def _read_refs(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        refs = (loaded or {}).get("refs")
        if isinstance(refs, dict):
            return {str(k): str(v) for k, v in refs.items() if v is not None}
    except Exception:
        pass
    return _parse_refs_loose(text)


def api_key(*, required: bool = False) -> str | None:
    """从凭据库取 ``CREDENTIAL_REF`` 指定的那个键；读不到返回 None。

    只认 ``CREDENTIAL_REF`` 指定的那一个键（宿主用 ``DSH_KB_IMAGE_KEY_NAME`` 命名）。
    **不**做任何「找不到就换一个」的猜测 —— 同一个服务常同时有订阅键和按量键，
    拿错那个不会报「键错了」，而是报模型不支持，静默换键只会把 401 变成更难查的 400。

    密钥只活在进程内存里（按文件 mtime 缓存）；不打印、不写文件、不进日志，
    抛出的异常消息里只有文件路径。
    """
    path = credentials_path()
    if path is None:
        if required:
            raise ImageAuthError(
                "未配置图片嵌入凭据路径：设 DSH_KB_IMAGE_CREDENTIALS 指向一个"
                f"含 refs.{CREDENTIAL_REF} 的文件")
        return None
    try:
        st = path.stat()
    except OSError:
        if required:
            raise ImageAuthError(
                f"读不到凭据文件 {path}（需要 refs.{CREDENTIAL_REF}）"
            ) from None
        return None
    cache_key = (str(path), int(st.st_mtime), int(st.st_size))
    if cache_key not in _KEY_CACHE:
        try:
            refs = _read_refs(path)
        except Exception:
            refs = {}
        value = refs.get(CREDENTIAL_REF) or ""
        _KEY_CACHE.clear()
        _KEY_CACHE[cache_key] = value or None
    key = _KEY_CACHE.get(cache_key)
    if not key and required:
        raise ImageAuthError(
            f"{path} 的 refs 里没有 {CREDENTIAL_REF}（可用 DSH_KB_IMAGE_KEY_NAME 改名；"
            f"注意别用订阅键，它调 embedding 会报模型不支持）"
        )
    return key


def _clear_key_cache() -> None:
    """测试用：丢掉缓存，让下一次 ``api_key`` 重新读文件。"""
    _KEY_CACHE.clear()


# ------------------------------------------------------------------- HTTP --


def _endpoint() -> str:
    return os.environ.get("DSH_KB_IMAGE_ENDPOINT") or DEFAULT_ENDPOINT


def _model() -> str:
    return os.environ.get("DSH_KB_IMAGE_MODEL") or DEFAULT_MODEL


def _redact(text: str, key: str | None) -> str:
    """异常/报错文本脱敏：绝不把密钥带进日志或 note。"""
    text = str(text)[:4000]
    if key:
        text = text.replace(key, "***")
    text = re.sub(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", "data:image/…;base64,***", text)
    return text[:400]


def _extract_embedding(payload: dict) -> list[float]:
    data = payload.get("data")
    # 实测：``data`` 是对象 {"embedding": [...], "object": "embedding"}，不是数组。
    if isinstance(data, dict):
        vec = data.get("embedding")
    elif isinstance(data, list) and data:
        vec = data[0].get("embedding")
    else:
        vec = None
    if not isinstance(vec, list) or not vec:
        raise ImageStoreError("多模态接口返回空向量")
    return [float(x) for x in vec]


def _post(payload: dict, key: str, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _endpoint(), data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 403 AccountOverdueError（欠费）也走这里：错误体里带 code，会被原样
        # 截断进 RuntimeError，再由 ragvec._image_candidates 变成一句 note。
        # 任何外部账户状态都不允许冒泡成一个会让 search 崩掉的异常。
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise ImageHTTPError(
            int(exc.code), f"HTTP {exc.code}: {_redact(detail or exc.reason, key)}"
        ) from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"HTTP 连接失败: {_redact(exc.reason, key)}") from None


def _with_retry(fn, key: str, attempts: int = RETRIES, backoff: float = BACKOFF_S):
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 单条失败不该炸掉整批
            last = exc
            # 403 AccountOverdueError / 401 / 400 这类 4xx 重试没有意义：账户
            # 状态与参数错误不会因为再等 1.5 秒变好，只会让每次 search 多付 3 次
            # 往返。只有 5xx / 429 / 网络错误才退避重试。
            if isinstance(exc, ImageHTTPError) and not exc.retryable:
                raise RuntimeError(_redact(str(exc), key)) from None
            if attempt < attempts:
                time.sleep(backoff * (2 ** (attempt - 1)))
    assert last is not None
    raise RuntimeError(_redact(str(last), key))


def _embed_payload(item: dict, key: str, timeout: float,
                   instructions: str = _CORPUS_INSTRUCTIONS) -> list[float]:
    """**一条一个请求**：``input`` 永远只有一个元素。

    接口对多元素 ``input`` 不报错、只返回第一条向量；这里把「一条」写死，
    免得调用方以为可以批量。Corpus 侧带指令但不带结尾的 ``Query:``。
    """
    payload = {
        "model": _model(),
        "input": [item],
        "instructions": instructions,
        "dimensions": DEFAULT_DIM,
    }
    return _extract_embedding(_post(payload, key, timeout))


def embed_text(text: str, *, key: str | None = None, timeout: float = DEFAULT_TIMEOUT_S,
               query: bool = True) -> list[float]:
    """把查询文本嵌入到图片空间（Query 侧带 ``Query:`` 指令）。"""
    api_key_value = key or api_key(required=True)
    item = {"type": "text", "text": text}
    payload = {
        "model": _model(),
        "input": [item],
        "instructions": _QUERY_INSTRUCTIONS if query else _CORPUS_INSTRUCTIONS,
        "dimensions": DEFAULT_DIM,
    }
    data = _with_retry(lambda: _post(payload, api_key_value, timeout), api_key_value)
    return _extract_embedding(data)


def _mime_for(suffix: str) -> str:
    suffix = suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(
        suffix.lstrip("."), "image/png"
    )


def embed_image(path: Path | str, *, key: str | None = None,
                timeout: float = DEFAULT_TIMEOUT_S) -> list[float]:
    """嵌入单张图片：读文件 → base64 data URL → **一条**请求。"""
    p = Path(path)
    api_key_value = key or api_key(required=True)
    raw = p.read_bytes()
    if len(raw) < MIN_IMAGE_BYTES:
        raise ImageStoreError(f"图片过小（{len(raw)}B）：{p.name}")
    b64 = base64.b64encode(raw).decode("ascii")
    url = f"data:{_mime_for(p.suffix)};base64,{b64}"
    item = {"type": "image_url", "image_url": {"url": url}}
    return _with_retry(lambda: _embed_payload(item, api_key_value, timeout), api_key_value)


def _pack(vec: list[float]) -> bytes:
    return struct.pack("<%df" % len(vec), *vec)


# -------------------------------------------------------------------- store --


def default_db(home: Path | str | None = None, vectors_db: Path | str | None = None) -> Path:
    """默认库位置：与文本向量库同目录的 ``vectors-image.sqlite``。"""
    if vectors_db:
        return Path(vectors_db).parent / "vectors-image.sqlite"
    base = Path(home) if home else Path(os.environ.get("DSH_KB_HOME") or "~/.dsh-kb").expanduser()
    return base / "index" / "vectors-image.sqlite"


def _configured_db(home: Path | str | None = None, home_arg: str | None = None) -> Path:
    """--db 没给时的库路径：先问配置，再回落到默认位置。

    为什么必须问配置：``ragvec`` 检索时读的是 ``rag.image_arm_db`` /
    ``DSH_KB_RAG_IMAGE_ARM_DB``。如果 build 只用 ``<home>/index`` 推导，用户一旦
    自定义了路径就会「embed 写 A、检索读 B」——两边都成功，只是互相看不见。
    """
    try:
        from .config import load_config

        overrides = {"home": str(home)} if home and not home_arg else (
            {"home": home_arg} if home_arg else {})
        configured = getattr(load_config(**overrides).rag, "image_db", "") or ""
        if configured:
            return Path(configured).expanduser()
    except Exception:
        pass  # 配置读不出来不是致命错误，回落到默认位置即可
    return default_db(home=home)


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _ensure_schema(con: sqlite3.Connection, db: Path) -> None:
    row = con.execute(
        "select name from sqlite_master where type='table' and name='images'"
    ).fetchone()
    if row is None:
        con.executescript(_SCHEMA)
        con.commit()
        return
    columns = tuple(r[1] for r in con.execute("PRAGMA table_info(images)"))
    if columns != _EXPECTED_COLUMNS:
        raise ImageStoreError(
            f"图片库 schema 不符：{db} 的 images 列是 {columns}，期望 {_EXPECTED_COLUMNS}；"
            f"删库重跑 --build 即可"
        )


def _open_store(db: Path) -> sqlite3.Connection:
    con = _connect(Path(db))
    try:
        _ensure_schema(con, Path(db))
    except Exception:
        con.close()
        raise
    return con


def load_vectors(db: Path | str):
    """读取整库并转成归一化矩阵：``(paths, matrix, dim, model)``。

    与文本臂同构：L2 归一化后余弦 = 点积。库不存在 / 表不对 / 维度不一致 /
    长度为 0 都抛 :class:`ImageStoreError`，调用方把它变成一条 note。
    """
    path = Path(db)
    if not path.is_file():
        raise ImageStoreError(f"图片向量库不存在：{path}")
    try:
        with sqlite3.connect("file:%s?mode=ro" % path, uri=True) as con:
            rows = con.execute("select path, vec, model, dim from images").fetchall()
    except sqlite3.Error as exc:
        raise ImageStoreError(f"图片库不可读或 schema 不符：{exc}") from None
    if not rows:
        raise ImageStoreError("图片库为空")
    # 先校验 schema/维度，再要 numpy：schema 问题不该被「缺 numpy」盖住。
    dims = {int(r[3]) for r in rows}
    if len(dims) != 1:
        raise ImageStoreError(f"图片向量维度不一致：{sorted(dims)}")
    dim = dims.pop()
    if dim <= 0:
        raise ImageStoreError(f"图片向量维度非法：{dim}")
    if any(len(r[1] or b"") != dim * 4 for r in rows):
        raise ImageStoreError("图片库里有向量的字节长度与 dim 不符")
    from .ragvec import _numpy  # 延迟导入：关闭图片路时不需要 numpy

    np = _numpy()
    buffer = b"".join(r[1] for r in rows)
    matrix = np.frombuffer(buffer, dtype="<f4").reshape(len(rows), dim)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    matrix = matrix / norms[:, None]
    return [str(r[0]) for r in rows], matrix, dim, str(rows[0][2] or "")


def status(db: Path | str, mirror: Path | str | None = None,
           prefixes: tuple[str, ...] | list[str] | None = None) -> dict:
    """只读状态：条数、维度、模型、更新时间、按 zone 分组，可选新鲜度。

    ``prefixes`` 给出时额外统计「这些路径前缀下有几张」——本库的 ``z01`` 前缀
    下是 0 张（19 张全在 ``z3-skills/``），这是判断图片路有没有图源的第一手数据。
    """
    path = Path(db)
    out: dict = {"db": str(path), "exists": path.is_file()}
    if not out["exists"]:
        return out
    try:
        with sqlite3.connect("file:%s?mode=ro" % path, uri=True) as con:
            rows = con.execute(
                "select path, sha, dim, model, updated, length(vec) from images"
            ).fetchall()
    except sqlite3.Error as exc:
        out.update({"error": f"{type(exc).__name__}: {exc}"})
        return out
    out["total"] = len(rows)
    out["dims"] = sorted({int(r[2]) for r in rows})
    out["models"] = sorted({str(r[3]) for r in rows})
    out["updated_min"] = min((int(r[4]) for r in rows), default=None)
    out["updated_max"] = max((int(r[4]) for r in rows), default=None)
    out["size_bytes"] = sum(int(r[5] or 0) for r in rows)
    zones: dict = {}
    for (rel, *_rest) in rows:
        head = str(rel).split("/", 1)[0] + "/"
        zones[head] = zones.get(head, 0) + 1
    out["zones"] = zones
    if prefixes is not None:
        pref = tuple(str(p) for p in prefixes)
        stored = [str(r[0]) for r in rows]
        matched = [p for p in stored if p.startswith(pref)] if pref else list(stored)
        out["prefixes"] = list(pref)
        out["matched"] = len(matched)
        matched_set = set(matched)
        out["outside"] = sorted(p for p in stored if p not in matched_set)[:20]
    if mirror is not None:
        items, skipped = discover_images(mirror)
        by_rel = {it.rel: it for it in items}
        stored = {str(r[0]): str(r[1]) for r in rows}
        out["orphan"] = sorted(set(stored) - set(by_rel))
        out["stale"] = sorted(p for p, sha in stored.items()
                              if p in by_rel and by_rel[p].sha != sha)
        out["untracked"] = sorted(set(by_rel) - set(stored))
        out["skipped"] = skipped
    return out


# -------------------------------------------------------------------- build --


def _batch_job(item: ImageFile, key: str, timeout: float) -> tuple[ImageFile, list[float]]:
    return item, embed_image(item.path, key=key, timeout=timeout)


def build(db: Path | str, mirror: Path | str, *, limit: int | None = None,
          workers: int = DEFAULT_WORKERS, timeout: float = DEFAULT_TIMEOUT_S,
          force: bool = False, prune: bool = False, min_bytes: int = MIN_IMAGE_BYTES,
          dim: int = DEFAULT_DIM, key: str | None = None,
          progress=None) -> dict:
    """增量构建图片向量库（sha256 幂等，只嵌变化的文件）。

    返回统计字典（可 JSON 序列化，不含密钥）。``limit`` 只嵌前 N 张，用于
    「先跑通 20 张」的验证；幂等意味着再跑一次会接着嵌剩下的。
    """
    started = time.time()
    db_path = Path(db)
    items, skipped = discover_images(mirror, min_bytes=min_bytes)
    stats: dict = {
        "db": str(db_path), "mirror": str(mirror), "model": _model(), "dim": dim,
        "discovered": len(items), "skipped_small": skipped["small"],
        "skipped_large": skipped["large"], "skipped_unreadable": skipped["unreadable"],
        "unchanged": 0, "todo": 0, "embedded": 0, "failed": 0,
        "pruned": 0, "errors": [], "elapsed_s": 0.0,
    }
    if limit is not None and limit >= 0:
        items = items[:limit]
        stats["limited_to"] = limit

    # 幂等的第二遍不需要凭据：全部未变时连密钥文件都不读。
    api_key_value = key or api_key(required=False)
    con = _open_store(db_path)
    try:
        existing = {
            str(row[0]): (str(row[1]), int(row[2]))
            for row in con.execute("select path, sha, dim from images")
        }
        todo = [it for it in items
                if force or existing.get(it.rel) != (it.sha, dim)]
        stats["unchanged"] = len(items) - len(todo)
        stats["todo"] = len(todo)
        if todo and not api_key_value:
            raise ImageAuthError(
                f"没有 {CREDENTIAL_REF}，无法嵌入 {len(todo)} 张图片"
            )
        done = 0
        if todo:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
                futures = {
                    pool.submit(_batch_job, it, api_key_value or "", timeout): it
                    for it in todo
                }
                for future in as_completed(futures):
                    item = futures[future]
                    done += 1
                    try:
                        _item, vec = future.result()
                    except Exception as exc:  # noqa: BLE001 - 单张失败继续
                        stats["failed"] += 1
                        if len(stats["errors"]) < 5:
                            stats["errors"].append(
                                {"path": item.rel, "error": _redact(str(exc), api_key_value)}
                            )
                        continue
                    if len(vec) != dim:
                        stats["failed"] += 1
                        if len(stats["errors"]) < 5:
                            stats["errors"].append(
                                {"path": item.rel, "error": f"维度 {len(vec)} != {dim}"}
                            )
                        continue
                    con.execute(
                        "insert into images(path, sha, dim, model, vec, updated) "
                        "values(?,?,?,?,?,?) "
                        "on conflict(path) do update set sha=excluded.sha, dim=excluded.dim, "
                        "model=excluded.model, vec=excluded.vec, updated=excluded.updated",
                        (item.rel, item.sha, dim, _model(), _pack(vec), int(time.time())),
                    )
                    con.commit()
                    stats["embedded"] += 1
                    if progress is not None:
                        try:
                            progress({"done": done, "total": len(todo),
                                      "embedded": stats["embedded"],
                                      "failed": stats["failed"], "path": item.rel})
                        except Exception:
                            pass
        if prune:
            live = {it.rel for it in items}
            stale_rows = [p for p in existing if p not in live]
            # 只在「这次扫描覆盖了整库」时剪枝：带 limit 的局部跑不该删别的行。
            if limit is None and stale_rows:
                con.executemany("delete from images where path=?", [(p,) for p in stale_rows])
                con.commit()
                stats["pruned"] = len(stale_rows)
    finally:
        con.close()
    stats["elapsed_s"] = round(time.time() - started, 3)
    return stats


# ---------------------------------------------------------------------- CLI --


def _default_mirror(explicit: str | None, home: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_mirror = os.environ.get("DSH_KB_MIRROR")
    if env_mirror:
        return Path(env_mirror).expanduser()
    try:
        from .config import load_config

        # 注意传 Path：config.load_config 的 overrides 是最后 replace 的，
        # 传字符串会让 cfg.home 变成 str（``str / "mirror"`` 直接 TypeError）。
        cfg = load_config(**({"home": Path(home).expanduser()} if home else {}))
        return cfg.mirror_dir
    except Exception:
        base = Path(home or os.environ.get("DSH_KB_HOME") or "~/.dsh-kb").expanduser()
        return base / "mirror"


def _default_home(home: str | None) -> Path:
    return Path(home or os.environ.get("DSH_KB_HOME") or "~/.dsh-kb").expanduser()


def _config_zone_prefixes(home: Path) -> tuple[str, ...] | None:
    """生产 zone（z01）的路径前缀；配置读不出来就返回 None。"""
    try:
        from .config import load_config

        return tuple(load_config(home=home).query_prefixes("z01"))
    except Exception:
        return None


def _parse_prefixes(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",")]
    out = tuple(p if p.endswith("/") else p + "/" for p in parts if p)
    return out or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m dsh_kb.image_embed",
        description="图片多模态侧车：向量化 / 查看状态（默认不参与检索）",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true", help="增量构建图片向量库")
    action.add_argument("--status", action="store_true", help="只读查看库状态")
    action.add_argument("--readiness", action="store_true",
                        help="评估图片路是否真的可能被用到（库内数据 + 上游门控）")
    parser.add_argument("--home", help="dsh-kb home（含 mirror/ 与 index/）")
    parser.add_argument("--mirror", help="镜像目录（默认 <home>/mirror）")
    parser.add_argument("--db", help="图片库路径（默认 <home>/index/vectors-image.sqlite）")
    parser.add_argument("--limit", type=int, default=None, help="只嵌前 N 张（验证用）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--force", action="store_true", help="忽略 sha，全部重嵌")
    parser.add_argument("--prune", action="store_true", help="删掉文件已消失的行")
    parser.add_argument("--prefixes", help="按这些路径前缀统计命中（逗号分隔，默认生产 z01 前缀）")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    home = _default_home(args.home)
    mirror = _default_mirror(args.mirror, args.home)
    db = Path(args.db).expanduser() if args.db else _configured_db(home, home_arg=args.home)

    if args.readiness:
        from .config import load_config
        from .ragvec import image_arm_readiness

        try:
            cfg = load_config(home=home)
        except Exception as exc:  # noqa: BLE001 - 诊断命令不该因配置崩掉
            print(f"读取配置失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        info = image_arm_readiness(
            cfg,
            db=Path(args.db).expanduser() if args.db else None,
            zone="z01",
        )
        if args.json:
            json.dump(info, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"图片路可用性：{'可用' if info['usable'] else '不可用'}")
            print(f"结论：{info['verdict']}")
            print(f"库：{info['db']}（{'存在' if info['db_exists'] else '不存在'}，"
                  f"{info['rows']} 行，按 zone {info['rows_by_zone']}）")
            print(f"{info['zone']} 前缀命中：{info['matched']} / {info['rows']}"
                  f"（前缀 {info['image_prefixes']}）")
            gate = info["gate"]
            print(f"门控：vector_enabled={gate['vector_enabled']} "
                  f"vector_always={gate['vector_always']} "
                  f"FTS满额可达={gate['reachable_when_fts_full']}")
            print(f"门控说明：{gate['note']}")
            if info["outside"]:
                print(f"不在前缀下的图片（前 {len(info['outside'])} 张）：{info['outside'][:5]} …")
        return 0

    if args.status:
        prefixes = _parse_prefixes(args.prefixes) or _config_zone_prefixes(home)
        info = status(db, mirror if mirror.is_dir() else None, prefixes=prefixes)
        if args.json:
            json.dump(info, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            print(f"库：{info['db']}")
            if not info.get("exists"):
                print("状态：不存在（还没跑过 --build）")
                return 0
            if info.get("error"):
                print(f"状态：不可读 —— {info['error']}")
                return 1
            print(f"条数：{info['total']}  维度：{info['dims']}  模型：{info['models']}")
            print(f"按 zone：{info['zones']}")
            print(f"更新时间：{info['updated_max']}（epoch 秒）")
            if "matched" in info:
                print(f"前缀命中：{info['matched']} / {info['total']}（前缀 {info['prefixes']}）")
                if info["outside"]:
                    print(f"不在前缀下的图片：{len(info['outside'])} 张（前 5：{info['outside'][:5]}）")
            for key in ("orphan", "stale", "untracked"):
                if key in info:
                    print(f"{key}：{len(info[key])}")
        return 0

    if not mirror.is_dir():
        print(f"镜像目录不存在：{mirror}（用 --home 或 --mirror 指定）", file=sys.stderr)
        return 2

    def _progress(p: dict) -> None:
        if p["done"] == p["total"] or p["done"] % 10 == 0:
            print(f"  [{p['done']}/{p['total']}] {p['path']}", file=sys.stderr)

    try:
        stats = build(db, mirror, limit=args.limit, workers=args.workers,
                      timeout=args.timeout, force=args.force, prune=args.prune,
                      progress=_progress)
    except ImageAuthError as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 2
    except ImageStoreError as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(stats, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"库：{stats['db']}")
        print(f"发现 {stats['discovered']} 张（跳过 <{MIN_IMAGE_BYTES}B："
              f"{stats['skipped_small']}，过大：{stats['skipped_large']}）")
        print(f"未变 {stats['unchanged']} / 待嵌 {stats['todo']} / 已嵌 "
              f"{stats['embedded']} / 失败 {stats['failed']} / 耗时 {stats['elapsed_s']}s")
        for err in stats["errors"]:
            print(f"  ! {err['path']}: {err['error']}", file=sys.stderr)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
