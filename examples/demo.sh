#!/usr/bin/env bash
# Two-minute tour of dsh-kb: no share, no config file, nothing installed.
#
#   bash examples/demo.sh
#
# It builds a throwaway bundle in a temp directory, mirrors it, indexes it and
# runs one search — then deletes everything.  Read the output top to bottom:
# the search result is the whole product in one screen (path + line + snippet,
# roughly 2.4 KB regardless of how big the bundle gets).
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TMP=$(mktemp -d "${TMPDIR:-/tmp}/dsh-kb-demo-XXXXXX")
trap 'rm -rf "$TMP"' EXIT INT TERM

BUNDLE="$TMP/bundle"
KB_HOME="$TMP/home"
mkdir -p "$BUNDLE/z0-public" "$BUNDLE/z1-restricted" "$BUNDLE/z2-sensitive" "$BUNDLE/_incoming"

cat > "$BUNDLE/z0-public/mount.md" <<'PAGE'
# SMB mount going stale

Symptom: `search` returns nothing and the mount shows up as `autofs` in
`/proc/mounts`.

Cause: the autofs trigger hides the real filesystem, so the sync reads the
empty trigger directory and mirrors zero files.

Fix: resolve the real mount (or pass `--mount <real-path>`), then run
`dsh-kb refresh`.
PAGE

cat > "$BUNDLE/z0-public/retrieval.md" <<'PAGE'
# Why retrieval is cheap here

A result is a path, a line number and a 240-character snippet.  The size of a
result does not depend on the size of the corpus, so growing the knowledge base
does not make each lookup more expensive.
PAGE

cat > "$BUNDLE/z1-restricted/incident.md" <<'PAGE'
# Incident 2026-10-19

The index was rebuilt from a half-synced mirror.  Lesson: `refresh` is a pure
function of (remote listing, manifest) — make it cheap and run it often.
PAGE

cat > "$BUNDLE/z2-sensitive/personal.md" <<'PAGE'
# This page must never reach the local disk

If a search returns it, the zone policy is broken.
PAGE

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DSH_KB_HOME="$KB_HOME"
export DSH_KB_SOURCE="local"
export DSH_KB_LOCAL_ROOT="$BUNDLE"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/4  dsh-kb --version"
python3 -m dsh_kb --version

say "2/4  refresh: mirror the bundle, then index it"
python3 -m dsh_kb refresh

say "3/4  search \"mount going stale\"  (path + line + snippet, that is the whole result)"
python3 -m dsh_kb search "mount going stale"

say "4/4  the sensitive zone was never copied, and a default search cannot see it"
if [ -e "$KB_HOME/mirror/z2-sensitive" ]; then
  echo "!! z2-sensitive was mirrored — that is a bug, please report it"
  exit 1
fi
echo "z2-sensitive: not on disk  ✓"

echo
echo "That is the whole loop. Next: point DSH_KB_LOCAL_ROOT (or mount/webdav) at"
echo "your real bundle, copy examples/config.json next to it, and read README.md."
