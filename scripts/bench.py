#!/usr/bin/env python3
"""Measure index build time, index size and query latency for a bundle.

Runs entirely offline against ``source=local`` and reports the numbers quoted in
the README, so anyone can reproduce (or contradict) them::

    python3 scripts/gen_corpus.py /tmp/kb/mirror 10000
    python3 -m dsh_kb index --quiet          # with DSH_KB_HOME / ... set
    python3 scripts/bench.py --home /tmp/kb --mirror /tmp/kb/mirror --pages 10000

Nothing here writes outside the given home directory.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

ZONE_CLAUSE = (
    "(m.path not like '%/%' or m.path like 'z0-public/%' or m.path like 'z1-restricted/%' "
    "or m.path like '_meta/%' or m.path like '_incoming/%')"
)


def corpus_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def timed_env(home: Path, mirror: Path) -> dict:
    return dict(
        os.environ,
        DSH_KB_HOME=str(home),
        DSH_KB_SOURCE="local",
        DSH_KB_LOCAL_ROOT=str(mirror),
        PYTHONPATH=str(SRC),
    )


def build_index(home: Path, mirror: Path) -> float:
    started = time.perf_counter()
    subprocess.run(
        [sys.executable, "-m", "dsh_kb", "index", "--quiet"],
        env=timed_env(home, mirror),
        capture_output=True,
        check=False,
    )
    return time.perf_counter() - started


def engine_latency(db: Path, expression: str, runs: int = 20) -> tuple[float, int]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        samples = []
        hits = 0
        for _ in range(runs):
            started = time.perf_counter()
            rows = con.execute(
                "select m.path, m.startline, m.text, bm25(chunks) as score "
                "from chunks join meta m on m.id = chunks.rowid "
                f"where chunks match ? and {ZONE_CLAUSE} order by score limit 16",
                (expression,),
            ).fetchall()
            samples.append((time.perf_counter() - started) * 1000)
            hits = len(rows)
    finally:
        con.close()
    return statistics.median(samples), hits


def cli_latency(home: Path, mirror: Path, query: str, runs: int = 3) -> float:
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        subprocess.run(
            [sys.executable, "-m", "dsh_kb", "search", query, "-n", "8",
             "--no-auto-refresh", "--quiet"],
            env=timed_env(home, mirror),
            capture_output=True,
            check=False,
        )
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, help="dsh-kb home (state directory)")
    parser.add_argument("--mirror", required=True, help="bundle directory to measure")
    parser.add_argument("--pages", type=int, default=None, help="page count, for the report")
    parser.add_argument("--query", default="配置", help="generic keyword query")
    parser.add_argument("--identifier", default="UNIQ-000123", help="exact identifier query")
    args = parser.parse_args()

    home, mirror = Path(args.home), Path(args.mirror)
    pages = args.pages or sum(1 for _ in mirror.rglob("*.md"))
    total = corpus_bytes(mirror)

    build_seconds = build_index(home, mirror)
    db = home / "index" / "kb.sqlite"
    size = db.stat().st_size if db.exists() else 0

    generic_ms, generic_hits = engine_latency(db, f'"{args.query}"')
    ident_ms, ident_hits = engine_latency(db, f'"{args.identifier}"')
    e2e_ms = cli_latency(home, mirror, args.query)

    print(f"pages                : {pages}")
    print(f"corpus               : {total / 1048576:.1f} MB")
    print(f"index build          : {build_seconds:.2f} s")
    print(f"index size           : {size / 1048576:.1f} MB  ({size / max(total, 1):.1f}x corpus)")
    print(f"engine, generic      : {generic_ms:.1f} ms  ({generic_hits} hits)")
    print(f"engine, identifier   : {ident_ms:.1f} ms  ({ident_hits} hits)")
    print(f"cli end-to-end       : {e2e_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
