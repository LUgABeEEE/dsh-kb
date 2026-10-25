# Security and privacy

## Reporting

Please do **not** open a public issue for a vulnerability. Use GitHub's private
[security advisory](https://github.com/LUgABeEEE/dsh-kb/security/advisories/new)
form; if that is unavailable, contact the maintainer directly and give us a
chance to fix it before disclosing.

Include the command or tool call, the version (`dsh-kb --version`), and what an
attacker gains. We will confirm receipt and keep you posted.

## Supported versions

`0.1.x` is alpha. Only the latest release gets fixes.

## What this tool is, security-wise

dsh-kb reads a directory or a share, keeps a **local mirror and index** of it,
and answers searches. It is honest about its boundaries:

* **It is not a sandbox.** It reads whatever you point it at, with your
  privileges. Do not point it at a share you would not let the process read in
  full — the `never` list and the zones are described below and are *not* a
  security boundary.
* **Credentials are never stored by dsh-kb.** They come from the environment or
  from files you own (`chmod 600`), and are passed to `smbclient`/HTTP at call
  time. Nothing writes them to the mirror, the index, the manifest or the log.
* **The `never` list is a safety net, not a boundary.** It keeps obvious
  credential paths (`.credentials`, `secrets`, `id_rsa`, …) out of the mirror
  and the index by *path* matching. It cannot understand intent; a page whose
  *name* merely contains such a word is excluded too. That trade is deliberate:
  over-excluding is the safe failure mode. Applications that must guarantee
  exclusion should keep those files outside the bundle entirely.
* **Zones are enforced at three layers** — the mirror, the index and the query —
  and the sensitive zone is never copied to disk, so a query cannot reach what
  was never mirrored. Tests cover each layer.
* **Writes are proposals.** An agent writes only into `_incoming/`; promoting a
  proposal into a zone is a human (or scheduled) action. There is no code path
  that lets a model edit a curated zone directly.
* **The MCP server has no write tools.** It exposes `search`, `read` and
  `status` only.

## What you should assume

* The mirror and the index are **plain files** in the state directory. Anyone
  who can read that directory can read the mirrored content. Put it somewhere
  with the same sensitivity as the source.
* Search results, snippets and logs can contain content from the knowledge base.
  Treat the log file as being as sensitive as the bundle.
* The WebDAV client verifies TLS by default and supports mutual TLS. Setting
  `DSH_KB_WEBDAV_CONNECT` separates the address it connects to from the name it
  verifies (`Host`/SNI), which is needed for appliances whose certificate CN is
  not an address — do not disable verification to work around that.

## Defence in depth in this repository

* `.github/workflows/ci.yml` fails the build on a credential-shaped assignment,
  a private IP address, or a tracked build artifact.
* The knowledge-base migration tooling used by the maintainer applies the same
  rules and additionally **scans its own output** before uploading, because
  trusting the input filter alone is not enough.
