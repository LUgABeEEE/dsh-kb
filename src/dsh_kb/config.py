"""Configuration and zone policy.

Precedence (highest wins)
-------------------------
1. explicit keyword arguments (what the CLI parsed)
2. environment variables (``DSH_KB_*``)
3. ``<home>/config.json``
4. built-in defaults

``home`` defaults to ``~/.dsh-kb`` and holds **state only** (mirror, index,
manifest, logs, config).  It is deliberately separate from the code checkout:
a public clone therefore never carries personal paths or private data.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

#: Path fragments that must never enter the mirror or the index.  Matching is a
#: plain substring test against the bundle-relative path, so a directory named
#: ``credentials/`` (or ``密钥/``, "keys") is skipped along with everything
#: inside it.  Extend via ``never`` in the config file.
DEFAULT_NEVER: tuple[str, ...] = (
    "credentials",
    ".credentials",
    "secrets",
    "id_rsa",
    "密钥",
    "密钥库",
)

#: Path fragments excluded from the **index only** — still mirrored, still readable
#: with ``dsh-kb read``, just not searchable.  Deliberately separate from ``never``,
#: which also governs the mirror: reusing it for "process artifacts I do not want to
#: *search*" would make the next ``sync --prune`` delete the local copies, and that
#: is a different decision with a different blast radius.
#:
#: Empty by default: which directories are process artifacts is a property of a
#: particular bundle, not of the tool.  Matching is the same plain substring test
#: against the bundle-relative path.  Extend via ``index_never`` in the config file.
DEFAULT_INDEX_NEVER: tuple[str, ...] = ()

#: Zone policy.  A *zone* is a top-level directory of the bundle.
#:   index  — participate in the default full-text index
#:   inject — safe to inject into a model prompt without an explicit request
#:   mirror — copy to a local machine at all (``False`` keeps it remote-only)
DEFAULT_ZONES: tuple[dict, ...] = (
    {"name": "z0-public", "index": True, "inject": True, "mirror": True},
    {"name": "z1-restricted", "index": True, "inject": False, "mirror": True},
    {"name": "z2-sensitive", "index": False, "inject": False, "mirror": False},
)

#: Directories that are part of the layout rather than a privacy zone.
META_DIR = "_meta"
INCOMING_DIR = "_incoming"

#: RAG / vector-store settings.  The embeddings are produced outside this repo,
#: but dsh-kb has to know where the store lives, which model produced it and its
#: dimension — otherwise "the vector row no longer matches the page" is
#: undetectable.  Overridable per machine via ``DSH_KB_RAG_*``.
#: Empty means "under ``<home>/index/vectors.sqlite``", resolved at load time so
#: a public clone never carries a machine-specific path.  Point it elsewhere (or
#: set ``DSH_KB_RAG_VECTORS_DB``) when the store lives outside ``home``.
DEFAULT_RAG_VECTORS_DB = ""
DEFAULT_RAG_MODEL = "bge-m3"
DEFAULT_RAG_DIM = 1024


@dataclass(frozen=True)
class Zone:
    name: str
    index: bool = True
    inject: bool = False
    mirror: bool = True


@dataclass(frozen=True)
class RagConfig:
    """Where the embeddings live and how to interpret their rows.

    ``enabled`` is tri-state on purpose: ``None`` means "auto" — the feature
    counts as present exactly when the vector store file exists, so a machine
    that never built vectors reports the RAG side as *absent* instead of
    *broken*.  An explicit ``True``/``False`` (config file or
    ``DSH_KB_RAG_ENABLED``) overrides that probe.
    """

    vectors_db: str = DEFAULT_RAG_VECTORS_DB
    model: str = DEFAULT_RAG_MODEL
    dim: int = DEFAULT_RAG_DIM
    enabled: bool | None = None
    #: 放行「向量库在 home 之外」。默认 ``None`` = 没表态 ⇒ ``ragcheck`` 拒绝
    #: 带外库的 ``--prune``（踩过的形状：临时 home + 机器级 vectors 路径，
    #: 整库的向量看起来全是孤儿）。要在家的外面放库，就得显式
    #: 写 ``true``；即便放行，批量删除守卫仍然生效。
    allow_external_vectors_db: bool | None = None
    #: 可选向量臂（2026-10-23）：True 时「关键词命中不足 limit」才补一路向量检索。
    #: 默认 False —— 关闭时连 numpy 都不 import，行为与纯 FTS5 完全一致。
    vector_enabled: bool = False
    #: 融合方式：``fill`` 只补齐（不动关键词顺序，默认）/ ``rrf`` 全量倒数排名融合
    vector_mode: str = "fill"
    #: 语义路触发：auto = 关键词不足 或 勉强命中 都走；always / never
    vector_trigger: str = "auto"
    #: 勉强命中判据：top-1 页面真正命中的查询词数 < 该值 即视为勉强
    vector_weak_cov: int = 2
    #: 覆盖率比例阈值：top-1 块里命中的查询词占比低于此值即视为勉强命中
    vector_weak_ratio: float = 0.35
    #: HyDE：弱命中时让本地模型先写一段假想答案，再把那段嵌入去检索
    hyde_enabled: bool = False   # 2026-10-23 实测：+1 题但 p95 15.8s（本地 7B CPU），默认关
    #: cross-encoder 重排：只在弱命中时对候选交叉打分。
    #: 默认关闭 —— 它要 onnxruntime + tokenizers 和约 280 MB 模型（见 docs/semantic-arm.md），
    #: 打开前先设 DSH_KB_RERANK_ROOT。关闭时与纯 FTS5 逐字节一致。
    rerank_enabled: bool = False   # 一次 33 题评测：+1 题，top8+len192 时 p95 359ms
    rerank_topn: int = 8
    rerank_max_length: int = 192
    #: 重排时给 cross-encoder 的片段长度（默认 480 字 ≈ 5 行）
    rerank_passage_chars: int = 480
    #: 重排后是否按页聚合再排序（页分 = 最高分 + 0.5*次高分）。
    #: 动机：同一页的多个块会挤占 top-5，页级聚合能把「该页确实更相关」表达出来。
    rerank_page_agg: bool = False
    #: 两段重排：先粗排 topn，再对前 5 用更长上下文精排一次（只有 5 对，成本可控）。
    rerank_two_stage: bool = False
    #: 二段重排的截断长度与片段长度
    rerank_stage2_maxlen: int = 384
    rerank_stage2_chars: int = 800
    #: 重排时是否把页面标题拼在片段前（标题是强信号，几乎零成本）。
    rerank_title_boost: bool = False   # 截断长度：512→192 让 p95 从 908ms 降到 359ms，召回不变
    #: doc2query 问句向量是否参与语义路打分（2026-10-23 已生成 1,737 条 / 1,457 页）
    d2q_enabled: bool = False
    #: 弱命中时语义路的候选池深度（默认 32；实测 256 能把 8/11 改写题的答案页纳入池内）
    vector_depth: int = 32
    hyde_model: str = "qwen2.5:7b"
    hyde_prompt_tokens: int = 100
    #: 是否总是重排（默认只在弱命中时重排）。路线1 实测用：有些题关键词臂能召回
    #: 答案页（名次 6-17），但 top-5 被别的页占满 —— 这类题靠「弱命中才重排」永远救不回。
    #: 限幅重排：对最终顺序跑一次 cross-encoder，但每条位移不超过 ±N（0=关闭）。
    #: 2026-10-24 实测：±2 时 23→25/33（+2/−0），e2e 237→458ms。
    rerank_max_shift: int = 0
    #: 限幅重排额外多取的候选数（决定「能看到多远」的近邻）
    rerank_shift_pool: int = 5
    #: 深池 LLM 重排（慢车道）：deep=true 时把候选扩到 40 页、让模型挑 3 页提前。
    #: 2026-10-24 实测 25→26/33（0.788），代价 p50 7.5s；失败自动回退普通排序。
    deep_enabled: bool = False
    deep_candidates: int = 40
    deep_pick: int = 3
    #: 深排用的模型名：与 DSH_KB_DEEP_ENDPOINT 指向的 OpenAI-compatible 服务一致。
    deep_model: str = ""
    deep_timeout_s: float = 25.0
    deep_passage_chars: int = 340
    rerank_always: bool = False
    #: 是否**总是**跑向量臂（默认只在关键词不足/弱命中时跑）。路线1 实测用：
    #: 强命中题里也有答案页被更通用的页压住的情况，RRF 融合能纠正一部分。
    vector_always: bool = False
    #: RRF 融合时两路的权重（默认 1:1）。实测：关键词臂在 mined 桶更准，
    #: 给它更高权重能在不丢 paraphrase 收益的前提下把 mined 拉回来。
    vector_rrf_kw_weight: float = 1.0
    vector_rrf_vec_weight: float = 1.0
    #: 勉强命中时让给语义路的槽位数
    vector_weak_reserve: int = 2


@dataclass(frozen=True)
class Config:
    home: Path
    source: str = "auto"                 # auto | local | mount | smb | webdav
    local_root: str | None = None        # source=local
    mount: str | None = None             # source=mount (or auto probe)
    smb_share: str | None = None         # source=smb  (//host/share, no mount needed)
    webdav_root: str | None = None       # source=webdav
    webdav_repo: str | None = None       # checkout of the WebDAV client lib
    zones: tuple[Zone, ...] = field(default_factory=lambda: tuple(Zone(**z) for z in DEFAULT_ZONES))
    never: tuple[str, ...] = DEFAULT_NEVER
    index_never: tuple[str, ...] = DEFAULT_INDEX_NEVER
    stale_minutes: int = 15
    #: How much mtime difference still counts as "unchanged".  Network shares
    #: commonly truncate timestamps to one second (measured on a consumer NAS),
    #: so a value slightly under that keeps genuine edits detectable while
    #: absorbing rounding noise.  A same-second edit that also keeps the byte
    #: size identical is invisible by design — `sync --full` exists for that.
    mtime_tolerance: float = 1.0
    chunk_lines: int = 15
    snippet_chars: int = 240
    default_limit: int = 8
    max_read_lines: int = 200
    #: Vector store location / model / dimension (see :class:`RagConfig`).
    rag: RagConfig = field(default_factory=RagConfig)
    #: config.json 里存在、但本版本不认识的键。
    #: 配置文件多一个键不该让整个 CLI 崩掉（旧版本读新配置、用户加注释键都会遇到），
    #: 让 `doctor` 把它报出来，比静默忽略或直接 TypeError 都好。
    unknown_keys: tuple[str, ...] = ()

    # ---- derived paths -------------------------------------------------
    @property
    def mirror_dir(self) -> Path:
        return self.home / "mirror"

    @property
    def index_db(self) -> Path:
        return self.home / "index" / "kb.sqlite"

    @property
    def manifest_path(self) -> Path:
        return self.home / "state" / "manifest.json"

    @property
    def log_path(self) -> Path:
        return self.home / "logs" / "kb.log"

    @property
    def config_path(self) -> Path:
        return self.home / "config.json"

    # ---- policy helpers ------------------------------------------------
    def zone_for(self, rel: str) -> Zone | None:
        """Return the zone owning a bundle-relative POSIX path, if any."""
        head = rel.split("/", 1)[0]
        for zone in self.zones:
            if zone.name == head:
                return zone
        return None

    def is_never(self, rel: str) -> bool:
        """True when the path matches the never-index list."""
        lowered = rel.lower()
        return any(bad.lower() in lowered for bad in self.never)

    def mirror_allows(self, rel: str) -> bool:
        if self.is_never(rel):
            return False
        zone = self.zone_for(rel)
        return zone is None or zone.mirror

    def is_index_never(self, rel: str) -> bool:
        """True when the path is excluded from the index **but not** from the mirror."""
        lowered = rel.replace("\\", "/").lower()
        return any(bad.lower() in lowered for bad in self.index_never if bad)

    def index_allows(self, rel: str) -> bool:
        if not rel.endswith(".md"):
            return False
        if self.is_never(rel):
            return False
        if self.is_index_never(rel):
            return False
        zone = self.zone_for(rel)
        if zone is not None and not zone.index:
            return False
        if zone is None and rel.split("/", 1)[0] in (INCOMING_DIR,):
            return True          # proposals are searchable on purpose
        return True

    def zones_for_query(self, selector: str) -> tuple[str, ...]:
        """Expand a query zone selector into concrete directory names.

        ``all``  — every zone including sensitive ones (explicit opt-in)
        ``z01``  — the default: everything that is indexable
        ``z0``…  — one zone
        """
        if selector == "all":
            return tuple(z.name for z in self.zones)
        if selector in ("z01", "default"):
            return tuple(z.name for z in self.zones if z.index)
        return tuple(z.name for z in self.zones if z.name == selector or z.name.startswith(selector))

    def query_prefixes(self, selector: str) -> tuple[str, ...]:
        """Directory prefixes a query may return results from.

        Zone selection governs *privacy zones* only.  The layout directories
        (``_meta``, ``_incoming``) are structure, not private content, and stay
        searchable under every selector — otherwise the bundle's own
        conventions and pending proposals would silently vanish from search.
        """
        zones = self.zones_for_query(selector)
        return tuple(f"{z}/" for z in zones) + (f"{META_DIR}/", f"{INCOMING_DIR}/")


def _coerce_zones(raw) -> tuple[Zone, ...]:
    if not raw:
        return tuple(Zone(**z) for z in DEFAULT_ZONES)
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append(Zone(name=item))
        else:
            out.append(Zone(**item))
    return tuple(out)


#: rag 配置既接受嵌套写法 ``"rag": {"vectors_db": ...}``，也接受扁平写法
#: ``"rag_vectors_db": ...``（和环境变量 DSH_KB_RAG_* 同名，方便照抄）。
_RAG_FIELDS = ("vectors_db", "model", "dim", "enabled",
               "allow_external_vectors_db", "vector_enabled", "vector_mode", "vector_trigger", "vector_weak_cov", "vector_weak_reserve", "vector_weak_ratio", "hyde_enabled", "hyde_model", "hyde_prompt_tokens", "rerank_enabled", "rerank_topn", "rerank_max_length", "rerank_passage_chars", "rerank_page_agg", "rerank_title_boost", "rerank_two_stage", "rerank_stage2_maxlen", "rerank_stage2_chars", "rerank_max_shift", "rerank_shift_pool", "deep_enabled", "deep_candidates", "deep_pick", "deep_model", "deep_timeout_s", "deep_passage_chars", "d2q_enabled", "vector_depth", "rerank_always", "vector_always", "vector_rrf_kw_weight", "vector_rrf_vec_weight")
_RAG_ALIASES = {
    "rag_vectors_db": "vectors_db",
    "rag_model": "model",
    "rag_dim": "dim",
    "rag_enabled": "enabled",
    "rag_allow_external_vectors_db": "allow_external_vectors_db",
}


def _env_flag(raw: str) -> bool:
    """Parse a boolean env/config value; anything but the obvious falsy words is true."""
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _coerce_rag(data: dict) -> tuple[dict, list[str]]:
    """Pull rag settings out of ``config.json`` (nested ``rag`` or flat ``rag_*``).

    Unknown sub-keys surface as ``rag.<key>`` in ``unknown_keys`` instead of
    being silently dropped — a typo like ``dims`` must be visible in ``doctor``.
    A value of the wrong type is ignored and the default stands, which is how
    the environment-variable parsing above already treats bad numbers.
    """
    raw = data.pop("rag", None)
    values: dict = {}
    unknown: list[str] = []
    if isinstance(raw, dict):
        for key, val in raw.items():
            if key in _RAG_FIELDS:
                values[key] = val
            else:
                unknown.append(f"rag.{key}")
    elif raw is not None:
        unknown.append("rag")
    for flat, key in _RAG_ALIASES.items():
        if flat in data:
            values[key] = data.pop(flat)
    try:
        values["dim"] = int(values["dim"])
    except (KeyError, TypeError, ValueError):
        values.pop("dim", None)
    for flag in ("enabled", "allow_external_vectors_db"):
        if flag in values:
            values[flag] = (
                _env_flag(values[flag]) if isinstance(values[flag], str) else bool(values[flag])
            )
    return values, unknown


def load_config(**overrides) -> Config:
    """Build a :class:`Config` from env + config file + defaults.

    ``overrides`` are keyword replacements applied last (only non-``None``
    values), which is how the CLI layers explicit flags on top.
    """
    # `home` decides where config.json is read from, so it has to be resolved
    # *before* the file is loaded. It used to be resolved from the environment
    # only, with overrides applied at the end — so `--home X` relocated the
    # state directory but kept reading `~/.dsh-kb/config.json`, silently
    # falling back to the built-in zones and `never` list. Measured on a real
    # deployment: three `never` exclusions and every zone's `inject` flag were
    # ignored, and the exclusions are a privacy guard.
    home = Path(
        overrides.get("home") or os.environ.get("DSH_KB_HOME", "~/.dsh-kb")
    ).expanduser()
    data: dict = {}
    cfg_path = home / "config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}

    env_map = {
        "source": "DSH_KB_SOURCE",
        "local_root": "DSH_KB_LOCAL_ROOT",
        "mount": "DSH_KB_MOUNT",
        "smb_share": "DSH_KB_SMB_SHARE",
        "webdav_root": "DSH_KB_WEBDAV_ROOT",
        "webdav_repo": "DSH_KB_WEBDAV_REPO",
    }
    for key, env in env_map.items():
        if os.environ.get(env):
            data[key] = os.environ[env]
    for key, env, cast in (
        ("stale_minutes", "DSH_KB_STALE_MINUTES", int),
        ("snippet_chars", "DSH_KB_SNIPPET_CHARS", int),
        ("default_limit", "DSH_KB_DEFAULT_LIMIT", int),
        ("mtime_tolerance", "DSH_KB_MTIME_TOLERANCE", float),
    ):
        if os.environ.get(env):
            try:
                data[key] = cast(os.environ[env])
            except ValueError:
                pass

    # rag 配置：config.json（嵌套或扁平）在前，DSH_KB_RAG_* 环境变量覆盖在后，
    # 与上面「env 覆盖文件」的优先级保持一致。
    rag_values, rag_unknown = _coerce_rag(data)
    for key, env in (("vectors_db", "DSH_KB_RAG_VECTORS_DB"), ("model", "DSH_KB_RAG_MODEL")):
        if os.environ.get(env):
            rag_values[key] = os.environ[env]
    for key, env, cast in (("dim", "DSH_KB_RAG_DIM", int), ("enabled", "DSH_KB_RAG_ENABLED", _env_flag),
                           ("allow_external_vectors_db",
                            "DSH_KB_RAG_ALLOW_EXTERNAL_VECTORS_DB", _env_flag),
                           ("vector_enabled", "DSH_KB_RAG_VECTOR", _env_flag)):
        if os.environ.get(env):
            try:
                rag_values[key] = cast(os.environ[env])
            except ValueError:
                pass
    # 其余 rag 子键统一走 ``DSH_KB_RAG_<FIELD>``（与扁平写法同名，便于照抄）。
    # 早期只映射了上面几个；`rerank_*` / `deep_*` 只能写 config.json 是个陷阱 ——
    # 一台机器用环境变量开功能、另一台用文件，排查起来会以为是两套行为。
    _already = {"vectors_db", "model", "dim", "enabled", "allow_external_vectors_db"}
    _flags = {"vector_enabled", "hyde_enabled", "rerank_enabled", "rerank_page_agg",
              "rerank_title_boost", "rerank_two_stage", "rerank_always", "vector_always",
              "deep_enabled", "d2q_enabled"}
    _ints = {"vector_weak_cov", "vector_depth", "vector_weak_reserve", "rerank_topn",
             "rerank_max_length", "rerank_passage_chars", "rerank_stage2_maxlen",
             "rerank_stage2_chars", "rerank_max_shift", "rerank_shift_pool",
             "deep_candidates", "deep_pick", "deep_passage_chars", "hyde_prompt_tokens"}
    _floats = {"vector_weak_ratio", "vector_rrf_kw_weight", "vector_rrf_vec_weight",
               "deep_timeout_s"}
    for name in _RAG_FIELDS:
        if name in _already:
            continue
        raw = os.environ.get("DSH_KB_RAG_" + name.upper())
        if raw is None:
            continue
        try:
            if name in _flags:
                rag_values[name] = _env_flag(raw)
            elif name in _ints:
                rag_values[name] = int(raw)
            elif name in _floats:
                rag_values[name] = float(raw)
            else:
                rag_values[name] = raw
        except ValueError:
            pass
    if not rag_values.get("vectors_db"):
        rag_values["vectors_db"] = str(home / "index" / "vectors.sqlite")
    rag = RagConfig(**rag_values)

    zones = _coerce_zones(data.pop("zones", None))
    never = tuple(data.pop("never", None) or DEFAULT_NEVER)
    index_never = tuple(data.pop("index_never", None) or DEFAULT_INDEX_NEVER)
    data.pop("home", None)

    known = {f.name for f in fields(Config)}
    unknown = tuple(sorted(list(set(data) - known) + rag_unknown))
    data = {k: v for k, v in data.items() if k in known}
    cfg = Config(home=home, zones=zones, never=never, index_never=index_never,
                 rag=rag, unknown_keys=unknown, **data)
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean) if clean else cfg


def source_available(cfg: Config) -> tuple[str, str]:
    """Resolve ``source=auto`` into a concrete source and describe it.

    Returns ``(kind, location)``.  ``kind`` is one of ``local``, ``mount``,
    ``smb`` or ``webdav``; an unusable configuration resolves to
    ``("", reason)``.

    ``auto`` prefers, in order: a mounted share, a plain directory, an unmounted
    Samba share (only when ``smbclient`` is present), then WebDAV.  Mounts are
    first because walking a real filesystem is orders of magnitude cheaper than
    any per-directory network protocol.
    """
    if cfg.source == "local":
        return ("local", cfg.local_root or "") if cfg.local_root else ("", "local_root not set")
    if cfg.source == "mount":
        return ("mount", cfg.mount or "") if cfg.mount else ("", "mount not set")
    if cfg.source == "smb":
        return ("smb", cfg.smb_share or "") if cfg.smb_share else ("", "smb_share not set")
    if cfg.source == "webdav":
        return ("webdav", cfg.webdav_root or "") if cfg.webdav_root else ("", "webdav_root not set")

    for kind, path in (("mount", cfg.mount), ("local", cfg.local_root)):
        if path and _is_dir(path):
            return kind, path
    if cfg.smb_share and shutil.which("smbclient"):
        return "smb", cfg.smb_share
    if cfg.webdav_root:
        return "webdav", cfg.webdav_root
    return "", "no source configured"


def _is_dir(path: str) -> bool:
    """Directory test that tolerates Windows/UNC spellings on any platform."""
    text = str(path).strip().strip('"')
    if os.sep == "/" and text.startswith("\\\\"):
        text = text.replace("\\", "/")
    try:
        return os.path.isdir(os.path.expanduser(text))
    except OSError:
        return False
