"""Command line interface: ``dsh-kb <command>``.

Commands are grouped by whether they touch the network:

* **offline** — ``status``, ``index``, ``search``, ``read``, ``rag-check``,
  ``skills``: they only ever look at the local mirror, index and vector store,
  so they keep working with the share unreachable.  This is the property that
  makes the tool usable as a model tool.
* **network** — ``sync``, ``push``, ``doctor``: they talk to the share.
  ``doctor`` is diagnostic but **not strictly read-only**: it writes and deletes
  two probe files (one in ``home``, one under ``_incoming/`` on the source) to
  answer "is this writable?" — see :func:`cmd_doctor`.
* **read-only network** — ``diff``: it reads the source listing to report what
  changed, what is missing locally and what is drifting, but it never writes to
  the source, the mirror or the index.

Every command supports ``--json`` where a machine-readable result makes sense
(the MCP server and the agent both consume it).  ``dsh-kb <command> --help``
lists the flags of each command; the README carries the same table.

Command overview (the full flag list lives in ``--help``)::

    status
    doctor
    diff        [--incoming-days DAYS] [--json]
    rag-check   [--prune] [--force]
    supersede-check [--json] [--proposal FILE]
    sync        [--prune] [--full] [--dry-run]
    index
    refresh     [--prune] [--full]
    search      <query> [-n N] [--zone Z] [--snippet N] [--refresh] [--no-auto-refresh]
    read        <path> [--start N] [--lines N]
    skills      search|show|stats
    push        [files…] [--force] [--expect SHA256] [--log [N]] [--json]
    mcp
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from . import ragcheck, supersede
from .config import Config, load_config, source_available
from .index import build, index_age, read, search, stats
from .mirror import manifest_age, mirror_files, mirror_size, sync
from .mounts import describe_path
from .service import (
    DiffUnavailable,
    diff_payload,
    ensure_fresh,
    file_logger,
    open_configured_source,
    read_payload,
    search_with_refresh,
)
from .service import refresh as service_refresh
from .sources import (
    FilesystemSource,
    SmbSource,
    SourceError,
    SourceUnavailable,
    WebdavSource,
    open_source,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _emit(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(message, file=sys.stderr)


def _log_to(cfg: Config, quiet: bool):
    """Return an ``on_event`` callback that tees to stderr and the log file."""
    write_file = file_logger(cfg)

    def emit(message: str) -> None:
        write_file(message)
        _emit(message, quiet=quiet)

    return emit


def _config(args) -> Config:
    overrides = {
        "source": getattr(args, "source", None),
        "mount": getattr(args, "mount", None),
        "local_root": getattr(args, "local_root", None),
        "smb_share": getattr(args, "smb_share", None),
        "webdav_root": getattr(args, "webdav_root", None),
    }
    home = getattr(args, "home", None)
    if home:
        from pathlib import Path

        overrides["home"] = Path(home).expanduser()
    return load_config(**overrides)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_status(args) -> int:
    cfg = _config(args)
    kind, location = source_available(cfg)
    m_files = mirror_files(cfg)
    idx = stats(cfg)
    m_age, i_age = manifest_age(cfg), index_age(cfg)

    payload = {
        "version": __version__,
        "home": str(cfg.home),
        "source": {"kind": kind, "location": location, "note": "" if kind else location},
        "mirror": {"path": str(cfg.mirror_dir), "files": len(m_files), "bytes": mirror_size(cfg),
                   "age_seconds": m_age},
        "index": {"path": str(cfg.index_db), "pages": idx.pages, "chunks": idx.chunks,
                  "bytes": idx.bytes, "age_seconds": i_age},
        "zones": [{"name": z.name, "index": z.index, "inject": z.inject, "mirror": z.mirror}
                  for z in cfg.zones],
        "never": list(cfg.never),
        "stale_minutes": cfg.stale_minutes,
        "mtime_tolerance": cfg.mtime_tolerance,
    }
    # Where does the bundle actually live?  A Samba/CIFS mount behaves
    # differently from a local directory, so surface that explicitly.
    if kind in ("local", "mount"):
        payload["source"].update(describe_path(location))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    def age(value):
        return "never" if value is None else f"{value:.0f}s ago"

    print(f"dsh-kb {__version__}")
    print(f"  home    : {cfg.home}")
    print(f"  source  : {kind or '—'}  {location}")
    if payload["source"].get("mountpoint"):
        flags = []
        if payload["source"].get("network"):
            flags.append(payload["source"].get("fstype", "network"))
        if payload["source"].get("readonly"):
            flags.append("read-only")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  mount   : {payload['source'].get('device', '')} at "
              f"{payload['source']['mountpoint']}{suffix}")
    print(f"  mirror  : {len(m_files)} files, {mirror_size(cfg)} B, synced {age(m_age)}")
    print(f"  index   : {idx.pages} pages / {idx.chunks} chunks, {idx.bytes} B, built {age(i_age)}")
    print(f"  zones   : " + ", ".join(f"{z.name}{'' if z.mirror else ' (remote-only)'}" for z in cfg.zones))
    return 0


def cmd_sync(args) -> int:
    cfg = _config(args)
    source = open_configured_source(cfg)
    emit = _log_to(cfg, args.quiet)
    started_source = getattr(source, "label", "?")
    if args.dry_run:
        entries = source.walk()
        files = [e for e in entries if not e.is_dir]
        allowed = [e for e in files if cfg.mirror_allows(e.rel)]
        excluded = [e.rel for e in files if not cfg.mirror_allows(e.rel)]
        print(json.dumps({
            "source": started_source, "files": len(files), "would_mirror": len(allowed),
            "excluded": excluded[:50], "excluded_count": len(excluded),
        }, ensure_ascii=False, indent=2) if args.json else
            f"{started_source}: {len(files)} files, {len(allowed)} mirrorable, {len(excluded)} excluded")
        return 0

    result = sync(cfg, source, prune=args.prune, full=getattr(args, "full", False), on_event=emit)
    emit(f"sync: {result.summary}")
    if args.json:
        print(json.dumps({
            "source": result.source, "dirs": result.dirs, "files": result.files,
            "fetched": result.fetched, "unchanged": result.skipped,
            "excluded": result.excluded, "pruned": result.removed,
            "changed": result.changed, "seconds": round(result.duration, 3),
        }, ensure_ascii=False, indent=2))
    else:
        for rel in result.changed:
            print(f"  + {rel}")
        print(result.summary)
    return 0


def cmd_index(args) -> int:
    cfg = _config(args)
    emit = _log_to(cfg, args.quiet)
    result = build(cfg, on_event=emit)
    if args.json:
        print(json.dumps({"pages": result.pages, "chunks": result.chunks,
                          "bytes": result.bytes, "seconds": round(result.duration, 3)},
                         ensure_ascii=False, indent=2))
    return 0


def cmd_refresh(args) -> int:
    """sync + index in one shot — the unit a scheduler should call.

    刻意走用例层：那里有"内容没变就不重建索引"的判定。此前这里直接 `sync()+build()`，
    于是**每次定时刷新都要重写整个索引**（本库 3.4 MB），与"按需调用的轻量补充"
    这个定位相悖。同一件事只应有一处实现。
    """
    cfg = _config(args)
    emit = _log_to(cfg, args.quiet)
    result, built = service_refresh(cfg, prune=args.prune,
                                    full=getattr(args, "full", False), on_event=emit)
    if result.fetched or result.removed:
        emit(f"refresh: {result.summary} → {built.summary}")
    else:
        emit(f"refresh: {result.summary} → index unchanged (rebuild skipped)")
    if args.json:
        print(json.dumps({"sync": {"fetched": result.fetched, "unchanged": result.skipped,
                                   "excluded": len(result.excluded), "seconds": round(result.duration, 3)},
                          "index": {"pages": built.pages, "chunks": built.chunks,
                                    "bytes": built.bytes, "seconds": round(built.duration, 3)}},
                         ensure_ascii=False, indent=2))
    return 0


def cmd_search(args) -> int:
    cfg = _config(args)
    allow_refresh = not args.no_auto_refresh
    payload = search_with_refresh(
        cfg,
        args.query,
        zone=args.zone,
        limit=args.number,
        snippet_chars=args.snippet,
        allow_refresh=allow_refresh or args.refresh,
        on_event=_log_to(cfg, args.quiet),
    )
    if args.refresh and not payload["hits"]:
        pass  # force-refresh already happened; nothing more to try

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _emit(f"query: {payload['query']}  zones={','.join(payload['zones'])}  {payload['note']}",
          quiet=args.quiet)
    if not payload["hits"]:
        print("(no hits)")
        return 1
    for index, hit in enumerate(payload["hits"], 1):
        print(f"[{index}] {hit['path']}:{hit['line']}  score={hit['score']}")
        print(f"    {hit['snippet']}")
    return 0


def cmd_read(args) -> int:
    cfg = _config(args)
    try:
        payload = read_payload(cfg, args.path, start=args.start, lines=args.lines)
    except FileNotFoundError as exc:
        _emit(f"dsh-kb: {exc}")
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"--- {payload['path']}  L{payload['start']}-{payload['end']} / {payload['total']} ---")
    for entry in payload["lines"]:
        print(f"{entry['n']:5d}| {entry['text']}")
    return 0


#: 推送回执账。<home>/state/ 放的都是"可复核的状态"，跟 manifest 同一层。
PUSH_LOG_NAME = "push-log.jsonl"
#: 回执里 result 的全部取值；``refused`` 专指"我们主动不写"。
PUSH_RESULTS = ("ok", "forced", "unchanged", "refused", "failed")
#: 这两个结果说明本次推送没做成，命令因此返回 1。
PUSH_BAD_RESULTS = ("refused", "failed")


def _push_log_path(cfg: Config) -> Path:
    return cfg.home / "state" / PUSH_LOG_NAME


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_push_receipt(cfg: Config, record: dict) -> None:
    """一次尝试恰好追加一行 JSON。

    单次 ``write`` 到 ``O_APPEND`` 打开的文件：并发 push 最多让**行序**交错，
    不会把两行拼成一行。写失败只告警、不中断推送 —— 回执是审计副本，
    不该因为 home 满了就把"文件其实已经写上去"这件事报成失败。
    """
    path = _push_log_path(cfg)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError as exc:
        _emit(f"dsh-kb: cannot append push receipt to {path}: {exc} — "
              f"check that {path.parent} is writable")


def _normalise_expect(raw: str | None) -> tuple[str | None, str | None]:
    """校验 ``--expect``，返回 ``(小写哈希, 错误说明)``。

    接受大写与 ``sha256:`` 前缀：人从 ``sha256sum`` 或上一次 ``push --json``
    的 ``sha256`` 字段复制过来时，大小写/前缀不该成为拒绝的理由。
    """
    if raw is None:
        return None, None
    text = raw.strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        return None, (f"--expect needs a 64-hex-digit sha256 (got {raw!r}) — take it "
                      f"from `sha256sum <file>` or the `sha256` field of a previous `push --json`")
    return text, None


def _remote_files(source) -> dict:
    """远端文件清单 ``rel -> Entry``，一次 ``walk()`` 覆盖所有待推文件。

    为什么用 ``walk()`` 而不是 ``read()`` 判存在：``read()`` 在"文件不存在"和
    "文件存在但读不了"上抛的是同一个 ``SourceUnavailable``，拿它判存在会把权限
    问题误判成"远端没有 → 可以覆盖"，正是要堵的那个洞。代价是一次全量列举
    （本库几千个文件；push 是低频人工动作，且换来默认路径上不可能静默覆盖）。
    """
    return {entry.rel: entry for entry in source.walk() if not getattr(entry, "is_dir", False)}


def _plain_incoming_name(name: str) -> bool:
    """``name`` 必须是 ``_incoming/`` 下的**单个文件名**，而不是一条路径。

    这是写入目标的组成部分，所以它属于"能不能写"的一部分：``push ../index.md``
    会让 rel 变成 ``_incoming/../index.md``，落点直接跑到 bundle 顶层 —— 越过
    ``_incoming/`` 的写入不属于这个命令的职责。更糟的是存在性检查查的是**字面**
    路径，于是永远判"远端不存在"：既不比对内容、也不要 ``--force``，还回一句 ok，
    而回执里的 path 与实际落点并不一致（误报）。所以宁可拒绝，也不去"修正"名字。
    绝对路径与 Windows 分隔符一并挡掉：同一份配置在两种系统上都会用。
    """
    if not name or name in (".", ".."):
        return False
    if name.startswith(("/", "\\")):
        return False
    if "/" in name or "\\" in name:
        return False
    # POSIX 上 Path("C:x").is_absolute() 是 False，盘符要单独认。
    if Path(name).is_absolute() or (len(name) > 1 and name[1] == ":"):
        return False
    return True


def _push_one(source, incoming: Path, name: str, remote: dict, *,
              force: bool, expect: str | None) -> dict:
    """推一个文件，返回它的回执记录（不打印、不记账，便于单独测试）。

    判定顺序是有意的：先名字合法性（不许越出 ``_incoming/``），再本地完整性
    （能否读、``--expect`` 是否相符），再远端存在性，最后才谈写入。任何一步不
    成立都直接返回，绝不产生半次写入。
    """
    record: dict = {
        "ts": _timestamp(), "name": name, "path": None, "sha256": None,
        "bytes": None, "remoteExisted": False, "result": "failed",
    }
    if not _plain_incoming_name(name):
        # 越界写入不修不猜，直接拒；failed ⇒ 退出码 1（2 是"源不可用"，不是这个）。
        record["error"] = (f"name must be a plain file name inside _incoming/ "
                           f"(got {name!r}) — 提案名只能是文件名，不能带目录")
        return record
    rel = f"_incoming/{name}"
    record["path"] = rel
    local = incoming / name
    if not local.is_file():
        record["error"] = f"missing local file {local}"
        return record
    try:
        data = local.read_bytes()
    except OSError as exc:
        record["error"] = f"cannot read {local}: {exc}"
        return record

    digest = _sha256(data)
    record.update(sha256=digest, bytes=len(data))

    if expect is not None and digest != expect:
        # 不写、不算失败 —— 调用方声明的版本不对，最可能是推错文件了。
        record.update(result="refused", reason="expect-mismatch", expected=expect)
        return record

    entry = remote.get(rel)
    record["remoteExisted"] = entry is not None
    if entry is not None:
        try:
            existing = source.read(rel)
        except SourceError as exc:
            # 列得出来却读不出来：不猜、不覆盖。
            record.update(result="failed", error=f"cannot read remote copy: {exc}")
            return record
        record["remoteSha256"] = _sha256(existing)
        record["remoteBytes"] = len(existing)
        if record["remoteSha256"] == digest:
            record["result"] = "unchanged"
            return record
        if not force:
            record.update(result="refused", reason="exists")
            return record

    try:
        source.write(rel, data)
    except SourceError as exc:
        # Actionable: read-only mount, stale share, unreachable host.
        record.update(result="failed", error=str(exc))
        return record
    except Exception as exc:  # noqa: BLE001 - report and continue with the rest
        record.update(result="failed", error=f"{type(exc).__name__}: {exc}")
        return record

    # forced 时 remoteSha256/remoteBytes 记的是**被替换掉的那版**，回执因此能回答
    # "覆盖了什么"，而不只是"写了什么"。
    record["result"] = "forced" if entry is not None else "ok"
    return record


def _report_push_record(record: dict, emit) -> None:
    """把一条回执翻成人话；每条都带"下一步怎么办"。"""
    name = record["name"]
    short = (record.get("sha256") or "?")[:12]
    result = record["result"]
    if result == "ok":
        emit(f"  ↑ _incoming/{name}  {record['bytes']} B  sha256={short}")
    elif result == "forced":
        emit(f"  ↑ _incoming/{name} (forced)  {record['bytes']} B  sha256={short}  "
             f"replaced sha256={(record.get('remoteSha256') or '?')[:12]} "
             f"({record.get('remoteBytes')} B)")
    elif result == "unchanged":
        emit(f"  = _incoming/{name} (unchanged)  {record['bytes']} B  sha256={short}")
    elif record.get("reason") == "expect-mismatch":
        emit(f"  ! {name}: --expect mismatch — local sha256={short} ({record['bytes']} B) "
             f"vs expected {record['expected'][:12]}; nothing written")
    elif result == "refused":
        emit(f"  ! {name}: remote _incoming/{name} already exists with different content\n"
             f"      local  sha256={short} ({record['bytes']} B)\n"
             f"      remote sha256={(record.get('remoteSha256') or '?')[:12]} "
             f"({record.get('remoteBytes')} B)\n"
             f"      refused — pass --force to overwrite, or rename the proposal")
    else:
        emit(f"  ! {name}: {record.get('error') or 'failed'}")


def _push_payload(records: list[dict], cfg: Config) -> dict:
    """``--json`` 的结构化结果。

    每个文件一条放在 ``results``；只有一个文件时把这些字段再平铺到顶层 ——
    agent 侧最常见的是 `jq .result` / `jq .sha256`，不该逼它先取数组下标。
    """
    counts = {key: 0 for key in PUSH_RESULTS}
    for record in records:
        counts[record["result"]] = counts.get(record["result"], 0) + 1
    payload: dict = {
        "ok": not any(r["result"] in PUSH_BAD_RESULTS for r in records),
        "results": records,
        "counts": counts,
        "log": str(_push_log_path(cfg)),
    }
    if len(records) == 1:
        only = records[0]
        for key in ("name", "sha256", "bytes", "result", "remoteExisted",
                    "remoteSha256", "remoteBytes", "reason", "expected", "error"):
            if key in only:
                payload[key] = only[key]
    return payload


def _push_show_log(cfg: Config, args) -> int:
    """``push --log [N]``：打印最近 N 条回执。"""
    limit = args.log if args.log is not None else 10
    if limit < 0:
        _emit(f"dsh-kb: --log takes a non-negative count (got {limit})")
        return 1
    path = _push_log_path(cfg)
    if not path.is_file():
        if getattr(args, "json", False):
            print("[]")
        else:
            _emit(f"dsh-kb: no push receipts yet ({path})")
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        _emit(f"dsh-kb: cannot read {path}: {exc} — check permissions on {path.parent}")
        return 1

    records, unparsable = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            # 半行只可能来自被中断的写入；跳过并说出来，不能让 --log 自己崩掉。
            unparsable += 1
    if unparsable:
        _emit(f"dsh-kb: skipped {unparsable} unparsable line(s) in {path}")
    tail = records[-limit:] if limit else []
    if getattr(args, "json", False):
        print(json.dumps(tail, ensure_ascii=False, indent=2))
        return 0
    if not tail:
        if limit == 0 and records:
            # "no receipts" 会是假话 —— 账里有记录，只是被要求看 0 条。
            _emit(f"dsh-kb: --log 0 shows nothing ({len(records)} receipts on file)")
        else:
            _emit(f"dsh-kb: no push receipts yet ({path})")
        return 0
    print(f"--- last {len(tail)} of {len(records)} push receipts · {path} ---")
    for record in tail:
        size = record.get("bytes")
        print(f"{record.get('ts', '?')}  {str(record.get('result', '?')):<9}  "
              f"{str(record.get('name', '?')):<28} "
              f"{size if size is not None else '?'} B  "
              f"sha256={str(record.get('sha256') or '?')[:12]}  "
              f"remote={'yes' if record.get('remoteExisted') else 'no'}")
    return 0


def cmd_push(args) -> int:
    """Upload local proposal files to the bundle's ``_incoming/`` directory.

    这是本工具**唯一**的写动作，所以按"宁可拒绝，也不覆盖"设计：

    1. 推前先看远端有没有同名文件（一次 ``walk()``）；
    2. 有且字节相同 → 幂等跳过（``unchanged``），不发起写入；
    3. 有且字节不同 → **默认拒绝**，两端 sha256 前 12 位与字节数都打出来，
       要 ``--force`` 才覆盖；
    4. ``--expect`` 让调用方声明本地哈希，不符即不写（防"以为推的是这一版"）；
    5. 无论写入、跳过还是拒绝，都往 ``<home>/state/push-log.jsonl`` 记一行。

    另外在**任何写入之前**跑一次取代关系闸门（R1）：与某张现行页同 type、
    tags 有交集、标题 Dice ≥0.80 的提案，必须写 ``supersedes:`` 或
    ``not_superseding: "<理由>"``，否则整批拒绝。这是唯一写入口 ⇒ 拦在这里比
    事后从库里删一条谁也退不掉的渐进页便宜得多。``--allow-undeclared`` 是
    明确的绕过开关（用于"确实无关"而作者不想写理由的场景）。
    """
    cfg = _config(args)
    if args.log is not None:
        if args.files:
            _emit("dsh-kb: --log only reads receipts — it cannot push files in the "
                  "same call; run the two separately")
            return 1
        return _push_show_log(cfg, args)

    expect, expect_error = _normalise_expect(args.expect)
    if expect_error:
        _emit(f"dsh-kb: {expect_error}")
        return 1

    source = open_configured_source(cfg)
    incoming = cfg.mirror_dir / "_incoming"
    # 这两种"没东西可推"是正常状态（调度器会按天跑），提示语与退出码保持原样。
    if not incoming.is_dir():
        _emit("dsh-kb: mirror has no _incoming/ directory — nothing to push")
        return 0
    names = args.files or sorted(p.name for p in incoming.iterdir() if p.is_file())
    if not names:
        _emit("dsh-kb: _incoming/ is empty")
        return 0

    if expect is not None and len(names) > 1:
        # 一个哈希对多个文件没有意义；静默选取哪一个都等于没校验。
        shown = ", ".join(names[:3]) + ("…" if len(names) > 3 else "")
        _emit(f"dsh-kb: --expect only works with a single file (got {len(names)}: {shown}) — "
              f"verify them one by one, otherwise the check is meaningless")
        return 1

    if not hasattr(source, "write"):
        _emit(f"dsh-kb: source {getattr(source, 'label', '?')} is read-only")
        return 2
    # ---- R1 提案闸门：写之前拦，且一次报全（不要挤牙膏式一个个报） ----
    if not args.allow_undeclared:
        blocked: list[tuple[str, dict]] = []
        for name in names:
            local = incoming / name
            if local.suffix != ".md" or not local.is_file():
                continue
            try:
                text = local.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            verdict = supersede.gate(cfg, name, text)
            if not verdict["ok"]:
                blocked.append((name, verdict))
        if blocked:
            for name, verdict in blocked:
                _emit(f"dsh-kb: 提案闸门拦下 {name} —— {verdict['reason']}")
            _emit(f"dsh-kb: {len(blocked)} 份提案与现行页撞题却没声明 ⇒ 一份都没推。"
                  f"补 supersedes:/not_superseding: 后重试；确认无关可用 --allow-undeclared")
            return 1

    mount = getattr(source, "mount", None) or {}
    if mount.get("readonly"):
        # 提前一次性拒绝，比逐个文件在 write 里失败更好懂（doctor 也报同一条）。
        _emit(f"dsh-kb: source {getattr(source, 'label', '?')} is a read-only mount "
              f"({mount.get('mountpoint')}) — remount it rw, or push via another "
              f"source (e.g. --source webdav)")
        return 2

    remote = _remote_files(source)

    def emit(message: str) -> None:
        _emit(message, quiet=getattr(args, "quiet", False))

    records = []
    for name in names:
        record = _push_one(source, incoming, name, remote,
                          force=args.force, expect=expect)
        _append_push_receipt(cfg, record)
        _report_push_record(record, emit)
        records.append(record)

    if args.json:
        print(json.dumps(_push_payload(records, cfg), ensure_ascii=False, indent=2))
    return 1 if any(r["result"] in PUSH_BAD_RESULTS for r in records) else 0


def cmd_skills_search(args) -> int:
    """搜索 `z3-skills/` 资产区里的 skill 总库（转发库自带检索）。"""
    from .skills import SkillsUnavailable, search as skills_search

    try:
        rows = skills_search(" ".join(args.query), top=args.number, kind=args.kind,
                             verdict=args.verdict, risky_only=args.risky,
                             cfg=_config(args))
    except SkillsUnavailable as exc:
        print(f"dsh-kb: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    if not rows:
        print("  库里没有匹配的 skill")
        return 0
    print(f"  命中 {len(rows)} 条")
    for row in rows:
        print(f"  ▸ {row.get('name')}  [{row.get('verdict')}·{row.get('kind')}]  "
              f"{row.get('score')}分  {row.get('size_kb')}KB  {row.get('license') or '—'}")
        brief = row.get("brief") or row.get("zh") or ""
        if brief:
            print(f"      {str(brief)[:96]}")
        if row.get("deps"):
            print(f"      依赖: {str(row['deps'])[:70]}")
    return 0


def cmd_skills_show(args) -> int:
    """打印一条 skill 的开头若干行（正文在资产区的 `z3-skills/` 里）。"""
    from .skills import SkillsUnavailable, read_skill

    try:
        rel, text = read_skill(args.name, lines=args.lines, cfg=_config(args))
    except SkillsUnavailable as exc:
        print(f"dsh-kb: {exc}", file=sys.stderr)
        return 2
    print(f"# {rel}\n")
    print(text)
    return 0


def cmd_skills_stats(args) -> int:
    """库规模、verdict 分布与许可风险（无模型调用）。"""
    from .skills import SkillsUnavailable, stats as skills_stats

    try:
        info = skills_stats(cfg=_config(args))
    except SkillsUnavailable as exc:
        print(f"dsh-kb: {exc}", file=sys.stderr)
        return 2
    print(f"library : {info['library']}")
    print(f"count   : {info['count']} skills · {info['bytes'] / 1048576:.1f} MB")
    print(f"verdict : {info['verdicts']}")
    print(f"kinds   : {info['kinds']}")
    print(f"许可风险: {info['license_risk']} 条（manifest 标 license_risk；含「有声明缺许可文件」档，详见库内 LICENSE-RISK.md）")
    if info.get("updated"):
        print(f"updated : {info['updated']}")
    return 0


def cmd_mcp(args) -> int:
    """Run the MCP server on stdio.

    Intended for an agent host (the host speaks JSON-RPC on this process's stdin
    and stdout), not for a human.  Nothing may be printed to stdout here.
    """
    from .mcp_server import serve

    return serve(_config(args))


def cmd_doctor(args) -> int:
    """Probe the environment and print an actionable report.

    Severity is explicit: reads are the primary function, writes are optional.
    A share that is read-only is a **warning** (search works, ``push`` does
    not), not a failure — reporting it as a failure would train people to
    ignore the check.

    **Not strictly read-only** (this is deliberate, and documented so nobody
    uses it as one): two writability probes are performed on every run —

    * ``_writable(home)``: ``mkdir -p`` the home directory if missing, then
      create and delete ``<home>/.dsh-kb-write-probe``;
    * ``source.probe_writable("_incoming")``: when the source exposes it, write
      and delete a probe file under ``_incoming/`` on the **share** (needed to
      answer "can ``push`` work at all?").

    The probe files are removed again; nothing else is written and no document
    is touched.  Every other check here is a plain read.
    """
    cfg = _config(args)
    kind, location = source_available(cfg)
    checks: list[tuple[str, str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append((name, status, detail))

    add("home writable", "ok" if _writable(cfg.home) else "fail", str(cfg.home))
    add("source resolved", "ok" if kind else "fail", f"{kind or '—'} {location}")

    source, note = open_source(cfg)
    if source is None:
        add("source usable", "fail", note)
    else:
        info = source.describe() if hasattr(source, "describe") else {}
        where = info.get("device") or info.get("root") or location
        add("source usable", "ok", f"{kind} ({info.get('fstype', '?')}) {where}")
        if info.get("network"):
            target = info.get("mountpoint") or info.get("root") or where
            ro = " (read-only)" if info.get("readonly") else ""
            add("network share detected", "ok", f"{info.get('fstype')} at {target}{ro}")
        try:
            source.ensure_available()
            add("source reachable", "ok", "listing succeeded")
        except SourceError as exc:
            add("source reachable", "fail", str(exc))
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            add("source reachable", "fail", f"{type(exc).__name__}: {exc}")

        if hasattr(source, "probe_writable"):
            ok, why = source.probe_writable("_incoming")
            add("source writable (_incoming)", "ok" if ok else "warn", f"{why} — needed by `push`")
        elif kind == "webdav":
            probe = source.client.probe()
            add("webdav reachable", "ok" if probe.get("reachable") else "fail",
                probe.get("error") or f"HTTP {probe.get('status')}")

    if cfg.unknown_keys:
        add("config keys", "warn",
            "unknown in config.json: " + ", ".join(cfg.unknown_keys) + " (ignored)")
    add("mirror present", "ok" if cfg.mirror_dir.is_dir() else "warn", str(cfg.mirror_dir))
    add("index present", "ok" if cfg.index_db.exists() else "warn", str(cfg.index_db))

    # RAG 一致性：**计数级**检查（count/distinct，不遍历向量 blob），所以能放进
    # 每天被调用的 doctor 里而不会让耗时数量级恶化。
    # 向量库缺失、或 ollama 不可达（能查但暂时不能重建），都只报 warn：
    # 检索照常工作，报 fail 会训练人忽略这个检查（同 source writable 的处置）。
    rag_vectors = ragcheck.vectors_db(cfg)
    if not ragcheck.enabled(cfg):
        add("rag consistency", "warn",
            f"absent — no vector store at {rag_vectors} "
            f"(set DSH_KB_RAG_VECTORS_DB, or build the vectors)")
    else:
        try:
            rag = ragcheck.check(cfg)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            add("rag consistency", "warn", f"{type(exc).__name__}: {exc}")
        else:
            detail = (f"{rag['chunks']} chunks / {rag['vectors']} vectors"
                      f" · missing {rag['missing']} · orphan {rag['orphan']}"
                      f" · model≠{cfg.rag.model} {rag['model_mismatch']}"
                      f" · dim≠{cfg.rag.dim} {rag['dim_mismatch']}")
            status = "ok" if rag["ok"] else "warn"
            if not rag["ok"]:
                detail = f"{detail} — {rag['note']}"
            probe_ok, probe_why = _ollama_reachable()
            if not probe_ok:
                status = "warn"
                detail = f"{detail} — ollama 不可达（{probe_why}）：暂时无法重新嵌入"
            add("rag consistency", status, detail)

    # Supersede relations (取代关系).  Not built yet is a *warn* like an absent
    # vector store: the fields are optional, and a corpus that never used them
    # is not broken — it just has no答"which conclusion is current" yet.
    try:
        rel = supersede.check(cfg)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        add("supersede relations", "warn", f"{type(exc).__name__}: {exc}")
    else:
        if not rel["built"]:
            add("supersede relations", "warn", rel["note"])
        else:
            bad = {k: len(v) for k, v in rel["problems"].items() if v}
            add("supersede relations", "ok" if rel["ok"] else "fail",
                rel["note"] if not bad else
                rel["note"] + " — " + ", ".join(f"{k}={n}" for k, n in bad.items()))

    # the privacy invariant that matters most
    mirror_rels = [str(p.relative_to(cfg.mirror_dir)) for p in mirror_files(cfg)]
    leaked = [rel for rel in mirror_rels if not cfg.mirror_allows(rel)]
    add("no excluded paths in mirror", "ok" if not leaked else "fail", ", ".join(leaked[:5]))

    # 索引排除项也要可见：否则"某个目录突然搜不到了"只能靠猜（2026-10-19 加，
    # 与 _incoming / 暂存目录这几次决定直接相关）。
    excluded = [rel for rel in mirror_rels
                if rel.endswith(".md") and cfg.is_index_never(rel) and not cfg.is_never(rel)]
    if cfg.index_never:
        detail = (f"{len(cfg.index_never)} 条模式（{', '.join(cfg.index_never)}）→ "
                  f"{len(excluded)} 页不进索引，仍可 read")
        add("index exclusions", "ok", detail)
    else:
        add("index exclusions", "ok", "未配置（所有 .md 都进索引）")

    failed = any(status == "fail" for _, status, _ in checks)
    if args.json:
        print(json.dumps(
            {"ok": not failed,
             "checks": [{"name": n, "status": s, "detail": d} for n, s, d in checks]},
            ensure_ascii=False, indent=2))
        return 1 if failed else 0
    marks = {"ok": "ok", "warn": "~~", "fail": "!!"}
    for name, status, detail in checks:
        print(f"  [{marks[status]}] {name:28s} {detail}")
    return 1 if failed else 0


def _writable(path: Path) -> bool:
    import os

    probe = path / ".dsh-kb-write-probe"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# diff (read-only health report)
# --------------------------------------------------------------------------- #
#: attention 代码 → 中文短语。JSON 里保持稳定的英文代码，人读时翻译。
ATTENTION_LABELS = {
    "source-unavailable": "源不可用",
    "mirror-gap": "镜像缺口",
    "index-drift": "索引漂移",
    "incoming-backlog": "提案积压",
    "conflict-copy": "冲突副本",
    "mirror-escape": "镜像越界",
}


def _fmt_bytes(value) -> str:
    if not isinstance(value, (int, float)):
        return "大小未知"
    if value < 1024:
        return f"{int(value)} B"
    if value < 1048576:
        return f"{value / 1024:.1f} KB"
    return f"{value / 1048576:.1f} MB"


def _fmt_age(seconds) -> str:
    """把秒数说成"人能换算"的量级；未知就说未知，不猜。"""
    if not isinstance(seconds, (int, float)):
        return "未知"
    if seconds < 90:
        return f"{seconds:.0f} 秒"
    if seconds < 5400:
        return f"{seconds / 60:.0f} 分钟"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} 小时"
    return f"{seconds / 86400:.1f} 天"


def _signed_age(seconds) -> str:
    if not isinstance(seconds, (int, float)):
        return "mtime 未知"
    sign = "+" if seconds >= 0 else "-"
    return f"mtime {sign}{_fmt_age(abs(seconds))}"


def _describe_change(mark: str, item: dict) -> str:
    path = item["path"]
    if mark == "+":
        return f"{path}  {_fmt_bytes(item.get('size'))}"
    if mark == "-":
        return f"{path}  上次 {_fmt_bytes(item.get('previous_size'))}"
    delta = item.get("size_delta")
    delta_text = f"{delta:+d} B" if isinstance(delta, int) else "大小未知"
    return (f"{path}  {_fmt_bytes(item.get('previous_size'))} → "
            f"{_fmt_bytes(item.get('size'))}（{delta_text}），"
            f"{_signed_age(item.get('mtime_delta'))}")


def _describe_gap(mark: str, item: dict) -> str:
    """镜像侧的一条差异：``size`` 是本地大小，``previous_size`` 是快照记录的大小。

    与 :func:`_describe_change` 分开是因为 mtime 在这边不可比（镜像是本机写入
    时间），写"mtime 未知"只会是噪音。
    """
    path = item["path"]
    if mark == "+":
        return f"{path}  本地 {_fmt_bytes(item.get('size'))}"
    if mark == "-":
        return f"{path}  快照记录 {_fmt_bytes(item.get('previous_size'))}"
    return (f"{path}  快照 {_fmt_bytes(item.get('previous_size'))} → "
            f"本地 {_fmt_bytes(item.get('size'))}")


def _emit_diff(payload: dict) -> None:
    """人读版 diff：每组一小节，异常组后面直接跟"下一步做什么"。"""
    src = payload["source"]
    print(f"dsh-kb diff · {payload['home']}")
    print(f"  源     : {src['kind'] or '—'} {src['location']}  "
          f"[{'可用' if src['available'] else '不可用'}]")
    if src["note"]:
        print(f"           ! {src['note']}")

    content = payload["content"]
    last_sync = content.get("last_sync")
    if last_sync:
        print(f"  快照   : {content['manifest']}（上次同步 {last_sync}，"
              f"{_fmt_age(content.get('manifest_age_seconds'))}前）")
    else:
        print(f"  快照   : {content['manifest']}（还没有同步过）")

    # 1. 内容差异
    print("\n内容差异（远端 vs 上次快照）")
    counts = content["counts"]
    if content["note"]:
        print(f"  ! {content['note']}")
    elif not any(counts.values()):
        print("  ok 没有差异：远端与上次快照一致")
    else:
        # Markdown 与非 Markdown 分开列：日常的图片/JSON 变动不该把 .md 的
        # 差异挤出屏幕。
        for kind, label in (("markdown", "Markdown（.md，会进索引）"),
                            ("other", "其他文件（只镜像，不进索引）")):
            group = content.get(kind) or {}
            kind_counts = group.get("counts") or {}
            if not any(kind_counts.values()):
                continue
            print(f"  ── {label}: 新增 {kind_counts.get('added', 0)} / "
                  f"消失 {kind_counts.get('removed', 0)} / "
                  f"变更 {kind_counts.get('changed', 0)}")
            for mark, key, name in (("+", "added", "新增"), ("-", "removed", "消失"),
                                    ("~", "changed", "变更")):
                items = group.get(key) or []
                if not items:
                    continue
                total = kind_counts.get(key, len(items))
                tail = f"（样例 {len(items)}/{total}，--json 有全量列表）" if total > len(items) else ""
                print(f"    {mark} {name} {total} 个{tail}")
                for item in items:
                    print(f"        {_describe_change(mark, item)}")
        print("  提示：内容差异本身不算异常，下一次 `dsh-kb sync` 就会收敛")

    # 2. 镜像副本（本地 vs 快照）：t7 盲点 B —— 删掉已同步的文件必须看得见
    gaps = payload["mirror_gaps"]
    gap_counts = gaps["counts"]
    print(f"\n镜像副本（本地镜像 vs 上次快照，快照记录 {gaps['snapshot_files']} 个文件，"
          f"本地 {gaps['mirror_files']} 个）")
    if gaps.get("note"):
        print(f"  ! {gaps['note']}")
    if not any(gap_counts.values()):
        print("  ok 本地镜像没有缺口")
    else:
        def _gap_block(code: str, mark: str, heading: str, hint: str) -> None:
            total = gap_counts.get(code, 0)
            if not total:
                return
            md = (gaps.get("markdown") or {}).get(code) or []
            other = (gaps.get("other") or {}).get(code) or []
            print(f"  {'!!' if code != 'added' else '~~'} {heading} {total} 个"
                  f"（Markdown {len(md)} / 其他 {len(other)}）")
            for item in (md + other)[:8]:
                print(f"      {mark} {_describe_gap(mark, item)}")
            if total > 8:
                print(f"      … 其余 {total - 8} 个（--json 看样例列表）")
            if hint:
                print(f"     {hint}")

        _gap_block("removed", "-", "本地缺文件（快照里有、镜像里没有）",
                   "建议：`dsh-kb sync` 会把这些文件补回来")
        _gap_block("changed", "~", "本地文件大小与快照不符（可能被截断/改坏）",
                   "建议：`dsh-kb sync --full`（增量 sync 认为远端没变，不会重取）")
        _gap_block("added", "+", "本地多出文件（快照与源都不认识）",
                   "建议：确认无用后手工删除 —— `sync --prune` 只删「快照有、远端没有」"
                   "的文件，删不掉这些")

    # 3. 索引漂移
    idx = payload["index"]
    print("\n索引")
    if not idx["present"] and not idx["drift"]:
        # 全新 home：没索引是因为压根没有可索引的内容，别报成异常。
        print(f"  ok 还没有索引，镜像里也没有可索引的 .md（{idx['path']}）")
    elif not idx["present"]:
        print(f"  !! 索引不存在：{idx['path']}")
        print(f"     建议：`dsh-kb index` 建一次（镜像里现有 {idx['mirror_pages']} 个可索引 .md）")
    elif idx["drift"]:
        print(f"  !! 漂移：索引 meta 里 {idx['meta_paths']} 个路径，镜像里可索引 .md "
              f"{idx['mirror_pages']} 个（索引 {_fmt_age(idx['age_seconds'])}前）")
        print("     建议：`dsh-kb index` 重建（或 `dsh-kb refresh` 同步+重建一次）")
    else:
        print(f"  ok 一致：{idx['meta_paths']} 个路径，索引 {_fmt_age(idx['age_seconds'])}前")

    # 4. _incoming 积压
    inc = payload["incoming"]
    basis = "共享盘" if inc["basis"] == "source" else "本地镜像"
    print(f"\n_incoming 积压（来源 {basis}，阈值 {inc['threshold_seconds'] / 86400:.1f} 天）")
    if inc["note"]:
        print(f"  ! {inc['note']}")
    if not inc["files"]:
        print("  ok 没有待处理的提案")
    elif inc["stale"]:
        print(f"  !! {inc['files']} 份提案，最老的一份 {_fmt_age(inc['oldest_seconds'])}前"
              f"（{inc['oldest']}）")
        print(f"     建议：`dsh-kb read {inc['oldest']}` 看内容，转正到 z0/z1 或删掉")
    else:
        print(f"  ok {inc['files']} 份提案，最老的一份 {_fmt_age(inc['oldest_seconds'])}前（阈值内）")

    # 5. 冲突副本
    conf = payload["conflicts"]
    print(f"\n冲突副本（源上 {conf['pattern']}）")
    if conf["note"]:
        print(f"  ! {conf['note']}")
    elif not conf["count"]:
        print("  ok 没有冲突副本")
    else:
        print(f"  !! {conf['count']} 个：")
        for rel in conf["files"][:8]:
            print(f"      {rel}")
        if conf["count"] > 8:
            print(f"      … 其余 {conf['count'] - 8} 个")
        print("     建议：人工比对后只留一份，删掉带 .conflict- 的那份（别让它被当成正式文档）")

    # 6. 镜像越界
    esc = payload["escapes"]
    print("\n镜像越界（当前 zones/never 策略排除、却仍在镜像里的路径）")
    if not esc["count"]:
        print("  ok 没有越界文件")
    else:
        print(f"  !! {esc['count']} 个：")
        for item in esc["files"][:8]:
            why = ("策略收紧前同步进来的" if item["in_manifest"]
                   else "不是 sync 放进来的（手工拷贝？）")
            print(f"      {item['path']}  {_fmt_bytes(item['bytes'])}  {why}")
        print("     建议：确认无用后删掉；若是策略误伤，改 config.json 的 zones/never 再 "
              "`dsh-kb sync --prune`")

    # 结论
    attention = payload["attention"]
    if not src["available"] and "source-unavailable" not in attention:
        # 全新 home + 没有源：没有快照可对比，也就没有"差异"可报。配置问题由
        # doctor 负责喊（它返回 1），这里说清楚为什么不重复报警。
        print("\n注：源不可用，但本地还没有任何快照/镜像 —— 配置问题请跑 `dsh-kb doctor`"
              "（它返回 1），这里不重复报警")
    if attention:
        names = "、".join(ATTENTION_LABELS.get(code, code) for code in attention)
        print(f"\n结论：{len(attention)} 项需要注意（{names}）—— 退出码 10")
    else:
        print("\n结论：没有需要注意的项 —— 退出码 0")


def cmd_diff(args) -> int:
    """Read-only health diff: what changed, what is drifting, what is piling up.

    只读是硬约束：不写源、不写镜像、不触发 ``sync``、不重建索引 —— 这个命令会
    被巡检脚本按天调用，任何写动作都会把"看一眼"变成风险源。（它**读**源清单：
    源可用时会遍历远端列出差异，但只读。）

    两个方向、五组结果：

    * ``content``（远端 vs 上次快照）—— 源上变了什么。只报告，**不算异常**：
      下一次 ``sync`` 就会收敛。``markdown`` / ``other`` 把 Markdown 与非
      Markdown 的增/删/改分开计数。
    * ``mirror_gaps``（本地镜像 vs 快照/源清单）—— 本地缺了/多了/大小不符。
      ``removed`` / ``changed`` 算异常：索引里指向的页面可能已经读不到了。
      比对域覆盖**全部文件**，不只 ``.md``（2026-10-19 t7 审查：删掉
      ``mirror/z0-public/pic.png`` 后旧版 diff 报 0，而 sync 立刻 fetched=1）。
    * ``index`` / ``incoming`` / ``conflicts`` / ``escapes`` —— 与原来一致。

    退出码 0 / 10 / 11 三档，语义未变：0 干净；10 有需要人处理的项（源不可用、
    镜像缺口、索引漂移、积压、冲突副本、越界）；11 本地状态读不出来
    （home/mirror 形状不对、manifest 坏了）。10 与 11 分开，监控才能区分"有活
    要干"和"这台机器坏了"。
    """
    cfg = _config(args)
    days = float(getattr(args, "incoming_days", 7.0))
    threshold = max(0.0, days) * 86400
    try:
        payload = diff_payload(cfg, incoming_threshold=threshold)
    except DiffUnavailable as exc:
        print(f"dsh-kb: {exc}", file=sys.stderr)
        return 11
    except Exception as exc:  # noqa: BLE001 - 诊断命令不该以 traceback 收场
        print(f"dsh-kb: diff failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 11

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _emit_diff(payload)
    return 10 if payload["attention"] else 0


# --------------------------------------------------------------------------- #
# supersede-check (取代关系 + 提案闸门)
def cmd_supersede_check(args) -> int:
    """取代关系体检 / 提案闸门（纯只读）。

    两个用途共一个入口，因为它们读的是同一张 `pages` 表、回答的是同一个问题的
    两面：**"库里现在的取代关系自洽吗"**（A1–A5 + A7）与**"这份新提案要不要
    先声明一句"**（R1）。默认只报告；闸门用 ``--proposal``。
    """
    cfg = _config(args)
    if args.proposal:
        target = Path(args.proposal)
        if not target.is_file():
            _emit(f"dsh-kb: 找不到提案文件：{target}")
            return 2
        rel = target.name
        try:
            raw = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _emit(f"dsh-kb: 读不了 {target}: {exc}")
            return 2
        verdict = supersede.gate(cfg, rel, raw)
        if args.json:
            print(json.dumps(verdict, ensure_ascii=False, indent=1))
        else:
            mark = "ok" if verdict["ok"] else "必须声明"
            _emit(f"提案闸门 {target.name}：{mark} — {verdict['reason']}")
        return 0 if verdict["ok"] else 10

    report = supersede.check(cfg)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return 0 if report["ok"] else 10
    _emit(f"supersede: {report['note']}")
    for kind, rows in sorted(report.get("problems", {}).items()):
        for row in rows[:5]:
            _emit(f"  ✗ {kind}: {row}")
    return 0 if report["ok"] else 10


# rag-check (vector store <-> index consistency)
# --------------------------------------------------------------------------- #
#: 向量构建器用的 ollama 端点（kb-rag/kbindex.py 里硬编码同一地址）。doctor 只做
#: 一次极短探测：不可达意味着"能查、但暂时不能重建"，那是 warn 不是 fail。
#: 设 DSH_KB_RAG_OLLAMA=- 可跳过探测（没有 ollama 的机器/离线测试）。
RAG_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
RAG_OLLAMA_TIMEOUT = 0.5


def _ollama_reachable(endpoint: str | None = None) -> tuple[bool, str]:
    """探一次 ollama 的 ``/api/tags``（只读、短超时）。不可达是结论，不抛异常。"""
    import os
    import urllib.request

    url = (endpoint or os.environ.get("DSH_KB_RAG_OLLAMA") or RAG_OLLAMA_ENDPOINT).rstrip("/")
    if url in ("", "-"):
        return True, "skipped"
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=RAG_OLLAMA_TIMEOUT) as response:
            return True, f"HTTP {response.status}"
    except Exception as exc:  # noqa: BLE001 - 探不通就是结论
        return False, f"{type(exc).__name__}: {exc}"


def _emit_rag_check(payload: dict) -> None:
    """人读版：先给计数与差异，再给结论。dry-run 与真删必须一眼可分；
    拒绝时把原因、比例、两条路径与解锁方式都摆出来（不许静默 no-op）。"""
    print(f"向量库一致性 · {payload['vectors_db']}")
    print(f"  home      : {payload['home']}")
    print(f"  模型/维度 : {payload['model']} / {payload['dim']}"
          f"{'' if payload['enabled'] else '（配置里未启用）'}")
    if payload["store_missing"]:
        print("  !! 向量库不存在")
        print("     建议：用向量构建器生成 —— kb-rag/kbask-index（需要 ollama，见 "
              "kb-rag/README.md），或把 DSH_KB_RAG_VECTORS_DB 指到真实的 vectors.sqlite")
    else:
        print(f"  索引 chunks（meta 行数）: {payload['chunks']}")
        print(f"  向量行数               : {payload['vectors']}")
        print(f"  missing（有页无向量）  : {payload['missing']}")
        print(f"  orphan（有向量无页）   : {payload['orphan']}")
        if payload.get("chunk_level"):
            # 块级口径：页级"这页有向量"不等于"每一块都有向量"（一页 5 块坏 3 块
            # 在页级计数里是 0）。老库没有 startline 列时这里整个不打印，退化
            # 原因在结论 note 里。
            print(f"  块级 missing（有块无向量）: {payload['chunks_missing']}")
            print(f"  块级 orphan（有向量无块） : {payload['chunks_orphan']}")
            if payload.get("chunks_missing_sample"):
                print("  缺向量的块样本（最多 10 个，path:line）:")
                for rel in payload["chunks_missing_sample"]:
                    print(f"      {rel}")
        print(f"  模型不符 {payload['model']:<6}: {payload['model_mismatch']}")
        print(f"  维度不符 {payload['dim']:<6}: {payload['dim_mismatch']}")
        if payload.get("bad_path"):
            # 坏行（path 为 NULL/空/非文本）单独报一行：它们不参与 orphan 计数，
            # 只在 note 里出现太容易被跳过。
            print(f"  坏行（path 为 NULL/空）: {payload['bad_path']}")
        if payload["orphan_sample"]:
            print("  孤儿样本（最多 10 个）:")
            for rel in payload["orphan_sample"]:
                print(f"      {rel}")

    prune = payload["prune"]
    if prune["refused"]:
        _emit_prune_refusal(prune)
    elif not prune["removed"]:
        print("  → 没有孤儿行可删" + ("（dry-run）" if prune["dry_run"] else "（幂等：二次调用无变化）"))
    elif prune["dry_run"]:
        print(f"  → dry-run：待删 {prune['removed']} 行孤儿（数据库未改动），"
              f"保留 {prune['kept']} 行；确认后加 --prune 真删")
    else:
        print(f"  → 已删 {prune['removed']} 行孤儿，保留 {prune['kept']} 行")
    if prune["bypassed"]:
        print(f"  ! --force 越过了保护：{', '.join(prune['bypassed'])}（请确认这是同一个知识库）")
    print(f"  结论：{payload['note']}")
    if prune["refused"]:
        print("  退出码 11（拒绝执行：一行都没删）")
    else:
        print("  退出码 " + ("0（一致）" if payload["ok"] else "10（有需要注意项）"))


#: 拒绝原因 → 解锁方式。拒绝必须告诉人怎么继续，否则下一个人只会绕过整条命令。
PRUNE_UNLOCK = {
    ragcheck.REFUSED_EXTERNAL:
        "把 DSH_KB_RAG_VECTORS_DB 指到 home 里的向量库；确实要操作外部库就显式设 "
        "rag.allow_external_vectors_db=true（外部库仍受 50% 大批量保护）",
    ragcheck.REFUSED_BULK:
        "先确认 home 与 vectors_db 是同一个知识库，再带 --force 重跑",
    ragcheck.REFUSED_EMPTY_HOME:
        "先 `dsh-kb index` 建出索引；确认无误后也可以带 --force",
    ragcheck.REFUSED_INDEX:
        "先 `dsh-kb index` 重建索引（索引读不出来时没有任何可比对的对照面）",
}


def _emit_prune_refusal(prune: dict) -> None:
    """拒绝块：原因、比例、两条路径、解锁方式 —— 缺一项人就只能猜。"""
    print(f"  !! 拒绝执行（refused={prune['reason']}）"
          + ("：本次为 dry-run，未删除任何行" if prune["dry_run"] else "：一行都没有删"))
    print(f"     待删 {prune['orphan_rows']}/{prune['total']} 行"
          f"（{prune['ratio']:.0%}）· {prune['orphan_paths']} 个 path"
          f"（阈值 {ragcheck.BULK_DELETE_LIMIT:.0%}）")
    print("     疑似 home 与 vectors_db 不属于同一知识库：")
    print(f"       home       = {prune['home']}")
    print(f"       vectors_db = {prune['vectors_db']}"
          + ("  ← 在 home 之外" if prune["external"] else ""))
    print(f"     原因：{prune['note']}")
    print(f"     解锁：{PRUNE_UNLOCK.get(prune['reason'], '见上面的原因')}")


def cmd_rag_check(args) -> int:
    """向量库 ↔ 索引一致性：默认只报告（dry-run），``--prune`` 才真删孤儿行。

    为什么默认 dry-run：孤儿向量是**证据链**问题（``kbask`` 会引用已经删掉的页），
    删之前人该先看一眼 orphan_sample；而 ``--prune`` 只删「页面已不在 meta」的行，
    ``kb.sqlite`` 永远只读（见 ragcheck 的模块说明）。

    报告有两个粒度：页级 ``missing``/``orphan``（历史口径，CLI 与文档都在引用）
    与块级 ``chunks_missing``/``chunks_orphan``（按 ``(path, startline)`` 比对）。
    块级存在的原因很具体：一页 5 块只嵌了 2 块时，页级口径看到"这页有向量"，
    ``missing`` 是 0，而检索会静默漏掉 3 块。向量库/索引缺 ``startline`` 列时
    自动退化回页级，并在 ``note`` 里说明（不崩、也不假装比对过）。

    dry-run 下 ``prune()`` 返回的 removed/kept 是**预估值**：那正是 dry-run 的价值
    （"如果删会删几行"），所以文本里写"待删"、JSON 里用 ``prune.dry_run`` 与
    ``prune.label`` 标明，绝不把预估数字伪装成已删。

    ``--prune`` 只清理"整页都不在了"的行；页面还在、只是少了几块的**块级孤儿
    它删不掉**，那要靠重新嵌入（note 里给了建议）。

    ``--force`` 只用于越过"大批量删除"与"空索引"两道保护；**归属校验**（向量库
    必须在 home 之下）只能靠 ``rag.allow_external_vectors_db=true`` 显式放行 ——
    2026-10-19 的事故（临时 home + 默认生产向量库 → 删掉 5,521 行）就是这么发生的。

    退出码四档：

    * ``0``  一致；
    * ``1``  **用法错误**：``--force`` 没有搭配 ``--prune``（它单独存在什么也
      不强制，所以拒绝执行 —— 保持 1 而不是并进 10/11：那是"命令敲错了"，
      不是"库有问题"）；
    * ``10`` 有需要注意（缺失/损坏/页级或块级 missing/orphan/模型维度不符）；
    * ``11`` 拒绝执行或硬失败（``prune.refused`` / ``prune.ok=false`` / 本地状态
      读不出来）。
    """
    cfg = _config(args)
    if args.force and not args.prune:
        # --force 只影响"真删"；单独用它说明调用方以为自己能强制什么，其实什么都没发生。
        _emit("dsh-kb: --force 只在 --prune 时有意义（它解的是大批量/空索引两道保护）")
        return 1
    try:
        report = ragcheck.check(cfg)
        # 默认也走 prune(dry_run=True)：只读，但能给出"待删几行"的准确预估，
        # 并把三条防线（归属 / 空索引 / 大批量）的判定一并带出来。
        prune_result = ragcheck.prune(cfg, dry_run=not args.prune, force=args.force)
        if args.prune and prune_result["removed"]:
            # 真删过就再取一次状态：输出的是"现在的样子"，删了多少另由 prune 说明。
            report = ragcheck.check(cfg)
    except Exception as exc:  # noqa: BLE001 - 诊断命令不该以 traceback 收场
        print(f"dsh-kb: rag-check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 11

    vectors_path = ragcheck.vectors_db(cfg)
    store_missing = not vectors_path.is_file()
    refused = bool(prune_result["refused"])
    hard_failure = refused or not prune_result["ok"]
    payload = dict(report)
    payload.update({
        "ok": bool(report["ok"]) and not store_missing and not hard_failure,
        "vectors_db": str(vectors_path),
        "home": str(cfg.home),
        "model": cfg.rag.model,
        "dim": cfg.rag.dim,
        # 用 ragcheck 的口径（rag.enabled 为 None 时"文件存在即为真"），不再自己判一次。
        "enabled": ragcheck.enabled(cfg),
        "store_missing": store_missing,
        "prune": {
            **prune_result,
            "requested": bool(args.prune),
            # 脚本判态用 label/refused/reason；dry-run 与真删、拒绝与执行一眼可分。
            "label": ("refused" if refused
                      else "no-orphans" if not prune_result["removed"]
                      else "removed" if not prune_result["dry_run"]
                      else "would remove"),
        },
    })
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _emit_rag_check(payload)
    if hard_failure:
        return 11
    return 0 if payload["ok"] else 10


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def _add_common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Attach the flags shared by every command.

    They are declared on both the root parser and every sub-parser so that
    ``dsh-kb --json search x`` and ``dsh-kb search x --json`` both work.  On the
    sub-parsers the default is ``SUPPRESS`` so an omitted flag does not overwrite
    the value the root parser already stored.
    """
    default = argparse.SUPPRESS if suppress else False
    parser.add_argument("--home", default=argparse.SUPPRESS if suppress else None,
                        help="state directory (mirror, index, manifest, logs, config)")
    parser.add_argument("--json", action="store_true", default=default,
                        help="machine-readable output")
    parser.add_argument("--quiet", action="store_true", default=default,
                        help="suppress progress on stderr")
    parser.add_argument("--source", choices=["auto", "local", "mount", "smb", "webdav"],
                        default=argparse.SUPPRESS if suppress else None,
                        help="override the read source")
    for name, help_text in (
        ("--mount", "path of a mounted share (Samba/CIFS, NFS, or a local directory)"),
        ("--local-root", "path of a plain local bundle"),
        ("--smb-share", "Samba share reached with smbclient, e.g. //host/share (no mount needed)"),
        ("--webdav-root", "base URL of a WebDAV collection"),
    ):
        parser.add_argument(name, default=argparse.SUPPRESS if suppress else None, help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsh-kb",
        description="A personal Markdown knowledge base with a local mirror and index, "
                    "built to be queried cheaply by AI agents.",
    )
    parser.add_argument("--version", action="version", version=f"dsh-kb {__version__}")
    _add_common_flags(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text, parents=[])
        _add_common_flags(child, suppress=True)
        return child

    command("status", "show configuration, mirror and index state").set_defaults(func=cmd_status)
    command("doctor", "probe the environment and report problems").set_defaults(func=cmd_doctor)

    p = command("diff", "read-only: remote changes, local mirror gaps, drift and backlog")
    p.add_argument("--incoming-days", type=float, default=7.0, metavar="DAYS",
                   help="pending _incoming proposals older than this count as backlog (default 7)")
    p.set_defaults(func=cmd_diff)

    p = command("rag-check",
                "vector store <-> index consistency, page + chunk level "
                "(report; --prune deletes orphan pages)")
    p.add_argument("--prune", action="store_true",
                   help="really delete orphan vector rows (default: report only, dry-run)")
    p.add_argument("--force", action="store_true",
                   help="with --prune: override the bulk-delete and empty-index guards "
                        "(the ownership guard still needs rag.allow_external_vectors_db=true)")
    p.set_defaults(func=cmd_rag_check)

    p = command("supersede-check",
                "取代关系体检：互逆 / 无环 / 无悬挂 / 单后继（spec A1–A5）；"
                "--proposal 则对一份提案跑 R1 闸门")
    p.add_argument("--proposal", metavar="FILE", default=None,
                   help="对这份提案跑提案闸门（R1）：与现行页标题撞车却没声明就报 must_declare")
    p.set_defaults(func=cmd_supersede_check)

    p = command("sync", "refresh the mirror from the source")
    p.add_argument("--prune", action="store_true", help="delete local files that vanished remotely")
    p.add_argument("--full", action="store_true",
                   help="ignore the manifest and re-fetch everything (see the mtime caveat in the docs)")
    p.add_argument("--dry-run", action="store_true", help="list what would change")
    p.set_defaults(func=cmd_sync)

    command("index", "rebuild the local full-text index").set_defaults(func=cmd_index)

    command("mcp", "run the MCP server on stdio (for agent hosts)").set_defaults(func=cmd_mcp)

    p = command("refresh", "sync + index (what a scheduler should call)")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--full", action="store_true", help="re-fetch everything, ignoring the manifest")
    p.set_defaults(func=cmd_refresh)

    p = command("search", "ranked full-text search over the mirror")
    p.add_argument("query")
    p.add_argument("-n", "--number", type=int, default=None, help="max hits (default 8)")
    p.add_argument("--zone", default="z01", help="z01 (default) | z0 | z1 | z2 | all")
    p.add_argument("--snippet", type=int, default=None, help="snippet width in characters")
    p.add_argument("--refresh", action="store_true", help="force a refresh before searching")
    p.add_argument("--no-auto-refresh", action="store_true", help="never auto-refresh")
    p.set_defaults(func=cmd_search)

    p = command("read", "read a line window from the mirror")
    p.add_argument("path")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--lines", type=int, default=80)
    p.set_defaults(func=cmd_read)

    p = command("push", "upload local _incoming/ proposals to the bundle")
    p.add_argument("files", nargs="*",
                   help="file names inside the local mirror's _incoming/ (default: all of them)")
    p.add_argument("--force", action="store_true",
                   help="overwrite a remote _incoming file whose content differs")
    p.add_argument("--expect", metavar="SHA256", default=None,
                   help="refuse unless the local file's sha256 matches this digest")
    p.add_argument("--log", nargs="?", type=int, const=10, default=None, metavar="N",
                   help="print the most recent N push receipts (default 10) and exit")
    p.add_argument("--allow-undeclared", action="store_true",
                   help="skip the supersede gate: push a proposal that collides with a current "
                        "page's title without declaring supersedes:/not_superseding:")
    p.set_defaults(func=cmd_push)

    # 资产区入口：z3-skills 不进主子索引，所以给它一条自己的查询路径，
    # 实现上转发库自带的 bin/find.py（检索实现只有一份，见 skills.py 的说明）。
    p = command("skills", "query the skill library living in the z3-skills asset zone")
    ssub = p.add_subparsers(dest="skills_action", required=True)

    sp = ssub.add_parser("search", help="ranked search over the skill library")
    sp.add_argument("query", nargs="+")
    sp.add_argument("-n", "--number", type=int, default=10, help="max hits (default 10)")
    sp.add_argument("--kind", default=None, help="filter by use (ops/frontend/research/…)")
    sp.add_argument("--verdict", default=None, help="filter by verdict (recommended/candidate/…)")
    sp.add_argument("--risky", action="store_true", help="only skills without a license")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.set_defaults(func=cmd_skills_search)

    sp = ssub.add_parser("show", help="print the head of one skill")
    sp.add_argument("name")
    sp.add_argument("--lines", type=int, default=80)
    sp.set_defaults(func=cmd_skills_show)

    ssub.add_parser("stats", help="library size, verdicts and license risk").set_defaults(
        func=cmd_skills_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SourceError as exc:
        # Every source problem already reads as a sentence an operator can act on.
        print(f"dsh-kb: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("dsh-kb: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
