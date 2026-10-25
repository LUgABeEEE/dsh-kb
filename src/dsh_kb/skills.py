"""Skill 资产区（`z3-skills/`）的查询入口。

**设计取舍：检索实现只有一份。** 这里不重新实现分词与打分，而是转发给 skill 总库
自带的 `bin/find.py`（自研 SQLite 倒排索引：拉丁词 + 中文 bigram；索引落在库外的
本机缓存目录，不放网络盘）。两套实现必然漂移，所以 dsh-kb 只负责定位、转发与格式化。

**为什么 skill 不并进主子索引**：`z3-skills/` 在配置里是 `index: false`。资产区里
每个 skill 一份 `SKILL.md`，几千份文档进 FTS5 会把知识条目的检索淹没（见
`_meta/CONVENTIONS.md` 的「资产区」一节）。

**为什么这里没有硬编码路径**：库位置按「环境变量 → 真源根推导 → 单机默认」三级解析，
换机器只改配置，不改代码。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

#: 显式覆盖库位置（排障、多机迁移、库放在真源之外时用）
LIB_ENV = "DSH_KB_SKILLS_LIB"

#: 资产区在 bundle 内的约定位置（真源根之下的相对路径）
ASSET_SUBDIR = ("z3-skills", "skills-library")


class SkillsUnavailable(RuntimeError):
    """库或索引不可用（信息缺失/路径不对）——按"人话错误"上抛，不裸抛 OSError。"""


def _source_root(cfg) -> pathlib.Path | None:
    """知识库真源根目录（挂载点优先，其次本地目录），用于推导资产区位置。"""
    if cfg is None:
        return None
    for attr in ("mount", "local_root"):
        value = getattr(cfg, attr, None)
        if value:
            return pathlib.Path(str(value))
    return None


def library_root(cfg=None) -> pathlib.Path:
    """定位 skill 总库（认 `manifest.json` 存在）。

    解析顺序：

    1. `DSH_KB_SKILLS_LIB` —— 显式指定，优先级最高；
    2. `<真源根>/z3-skills/skills-library` —— 资产区的约定位置，跟着配置走；
    3. `~/kb/z3-skills/skills-library` —— 单机默认布局。
    """
    env = os.environ.get(LIB_ENV, "").strip()
    if env:
        path = pathlib.Path(env)
        if (path / "manifest.json").is_file():
            return path
        raise SkillsUnavailable(f"{LIB_ENV}={env} 下没有 manifest.json")
    candidates: list[pathlib.Path] = []
    root = _source_root(cfg)
    if root is not None:
        candidates.append(root.joinpath(*ASSET_SUBDIR))
    candidates.append(pathlib.Path.home().joinpath("kb", *ASSET_SUBDIR))
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    raise SkillsUnavailable(
        f"找不到 skill 总库（设 {LIB_ENV}，或把库放进真源根的 "
        f"{'/'.join(ASSET_SUBDIR)}）")


def manifest(cfg=None) -> dict:
    """读库清单（每条的元数据：名称/描述/许可/依赖/verdict/评分）。"""
    path = library_root(cfg) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SkillsUnavailable(f"读不了 {path}：{exc}") from exc


def stats(cfg=None) -> dict:
    """库规模快照：条数、体积、verdict 与用途分布、许可风险数。"""
    man = manifest(cfg)
    items = man.get("items") or []
    verdicts: dict[str, int] = {}
    kinds: dict[str, int] = {}
    risky = 0
    for item in items:
        verdict = str(item.get("verdict") or "?")
        kind = str(item.get("kind") or "?")
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        kinds[kind] = kinds.get(kind, 0) + 1
        # 口径（2026-10-15 起优先用显式标注）：manifest 若带 license_risk 字段
        # （z3-skills 库 2026-10-15 许可处置引入，含「有声明但缺 LICENSE 文件」的
        # declared-only 档），以它为准；否则回退到 SPDX 口径 —— 即 SPDX 为空或
        # NOASSERTION/NONE/UNKNOWN 视为未声明。
        if "license_risk" in item:
            if item.get("license_risk"):
                risky += 1
            continue
        license_id = str(item.get("license_spdx") or "").upper()
        if not license_id or license_id in ("NOASSERTION", "NONE", "UNKNOWN"):
            risky += 1
    return {
        "library": str(library_root(cfg)),
        "count": len(items),
        "bytes": sum(int(item.get("size_bytes") or 0) for item in items),
        "verdicts": dict(sorted(verdicts.items(), key=lambda kv: -kv[1])),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "license_risk": risky,
        "updated": man.get("updated") or man.get("audited") or "",
    }


def search(query: str, *, top: int = 10, kind: str | None = None,
           verdict: str | None = None, risky_only: bool = False,
           timeout: int = 60, cfg=None) -> list[dict]:
    """转发库自带的 `bin/find.py --json`，返回结构化命中列表。"""
    finder = library_root(cfg) / "bin" / "find.py"
    if not finder.is_file():
        raise SkillsUnavailable(f"库内缺少检索脚本：{finder}")
    cmd = [sys.executable, str(finder), *str(query).split(), "--json", "--top", str(top)]
    if kind:
        cmd += ["--kind", kind]
    if verdict:
        cmd += ["--verdict", verdict]
    if risky_only:
        cmd += ["--nolicense"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SkillsUnavailable(f"检索超时（>{timeout}s）") from exc
    if done.returncode != 0:
        why = (done.stderr or done.stdout).strip().splitlines()
        raise SkillsUnavailable(why[-1][:300] if why else "检索失败")
    payload = done.stdout.strip()
    if not payload:
        return []
    try:
        rows = json.loads(payload)
    except ValueError:
        return []
    return rows if isinstance(rows, list) else []


def skill_path(name: str, cfg=None) -> pathlib.Path:
    """定位一条 skill 的正文文件（`skills/<name>/SKILL.md`，名称大小写不敏感）。"""
    skills_dir = library_root(cfg) / "skills"
    direct = skills_dir / name
    for candidate in ("SKILL.md", "skill.md", "Skill.md"):
        if (direct / candidate).is_file():
            return direct / candidate
    lowered = name.lower()
    if skills_dir.is_dir():
        for child in sorted(skills_dir.iterdir()):
            if child.is_dir() and child.name.lower() == lowered:
                for candidate in ("SKILL.md", "skill.md", "Skill.md"):
                    if (child / candidate).is_file():
                        return child / candidate
    raise SkillsUnavailable(f"库里没有 skill：{name}")


def read_skill(name: str, *, lines: int = 80, cfg=None) -> tuple[str, str]:
    """取一条 skill 的开头若干行；返回（相对路径，正文窗口）。"""
    path = skill_path(name, cfg)
    text = path.read_text(encoding="utf-8", errors="replace")
    window = "\n".join(text.splitlines()[:lines])
    return path.relative_to(library_root(cfg)).as_posix(), window
