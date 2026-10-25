#!/usr/bin/env python3
"""Generate a synthetic bundle for benchmarking and for smoke tests.

The corpus mimics a real personal knowledge base: mostly short Chinese
paragraphs with latin identifiers, inline code and the occasional table.

Usage:
    python3 scripts/gen_corpus.py /tmp/kb/mirror 10000     # 10 000 pages
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

CN = ["环境", "配置", "备份", "恢复", "权限", "证书", "挂载", "镜像", "索引", "检索",
      "同步", "冲突", "区域", "策略", "凭据", "端口", "隧道", "容器", "服务", "日志",
      "快照", "校验", "阈值", "配额", "降级", "回滚", "审计", "脱敏", "脚本", "定时"]
EN = ["docker", "systemd", "fstab", "cifs", "webdav", "sqlite", "fts5", "rsync",
      "smbclient", "nginx", "landlock", "inode", "mutex", "sha256", "utf8",
      "token", "budget", "snippet", "manifest", "mirror"]


def make_page(index: int, rng: random.Random) -> str:
    lines = [
        "---",
        "type: lesson",
        f"title: 条目 {index:06d} 的标题",
        f"description: 第 {index} 页摘要，用于压测检索。",
        f"tags: [压测, 条目{index % 97}]",
        "---",
        "",
        f"# 条目 {index:06d}",
        "",
    ]
    for _ in range(14):
        lines.append(" ".join(rng.choice(CN) for _ in range(rng.randint(18, 34))) + "。")
        if rng.random() < 0.4:
            lines.append(f"- `{rng.choice(EN)}` 与 `{rng.choice(EN)}` 的交互："
                         f"{''.join(rng.choice(CN) for _ in range(rng.randint(8, 18)))}。")
        if rng.random() < 0.2:
            lines.append(f"| {rng.choice(CN)} | {rng.choice(EN)} | {rng.randint(1, 9999)} |")
    lines.append("")
    lines.append(f"唯一标识 UNIQ-{index:06d}-{rng.choice(EN).upper()}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="directory to fill (created if missing)")
    parser.add_argument("pages", type=int, help="number of pages to generate")
    parser.add_argument("--seed", type=int, default=42, help="deterministic output")
    parser.add_argument("--per-dir", type=int, default=500, help="pages per subdirectory")
    args = parser.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    for index in range(args.pages):
        folder = root / "z0-public" / f"{index // args.per_dir:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"page-{index:06d}.md").write_text(make_page(index, rng), encoding="utf-8")
    print(f"generated {args.pages} pages under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
