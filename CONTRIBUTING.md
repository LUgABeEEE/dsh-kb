# Contributing

Thanks for looking. This project is small on purpose, and a few of its rules are
load-bearing — please read the short list below before sending a patch.

## Running the tests

No dependencies, no network, no fixtures to download:

```bash
python -m unittest discover -s tests -t . -v
```

There is nothing to install. If you find yourself needing `pip install` to run
the suite, that is a bug worth reporting.

## House rules

**1. No compiled runtime dependencies.** This is the headline feature, not an
accident. A pull request that adds a runtime dependency needs a very good
reason and a discussion first. Standard library only.

**2. The tests stay offline.** Every test must pass on a machine with no
network and no shares mounted. Transport tests use fakes and stubs (see
`tests/test_sources.py`, which drives the SMB parser through a stub
`smbclient`). If a feature cannot be tested offline, design it so it can.

**3. The MCP server is hand-rolled.** `src/dsh_kb/mcp_server.py` speaks
newline-delimited JSON-RPC on the standard library, and
`tests/test_mcp.py` proves it against the *official* MCP SDK. Do not replace it
with the SDK: dependency drift would become our users' problem.

**4. Docstrings explain *why*.** The code says what it does. Comments earn their
place by recording a reason, a constraint, or a scar. Where a bug was found the
hard way, the comment says how — several do, and they are the most useful lines
in the repository.

**5. Privacy invariants must not regress.** These are enforced by tests:

* the `z2-sensitive` zone is **never mirrored to disk**, not merely skipped at
  query time;
* SQLite never lives on a network share (its locking is unsound there);
* writes by an agent are proposals into `_incoming/`, never edits to a zone;
* the `never` list is applied at the mirror *and* the index layer.

## Adding a transport

Transports implement the small interface in `src/dsh_kb/sources.py`:

```python
def walk(self) -> list[Entry]: ...          # the whole tree, one call
def read(self, rel: str) -> bytes: ...
def write(self, rel: str, data: bytes) -> None   # optional
def describe(self) -> dict: ...             # for `doctor` and `status`
def ensure_available(self) -> None: ...     # raise SourceUnavailable with a sentence a human can act on
```

Two things matter more than the mechanics:

* **errors must read as sentences.** `ESTALE` becomes "the share looks stale:
  remount it". A raw traceback is a bug.
* **writes must create their own parents.** `MKCOL`/`mkdir` do not recurse;
  a bare `PUT` into a missing collection fails with an opaque 500.

## Commits

English, imperative, a short subject and a body that explains the *why*:

```
fix: create missing parent collections on WebDAV writes

A bare PUT into a missing collection is HTTP 500 errno 2, and MKCOL does not
recurse. A 217-file import failed 217 times before this existed.
```

## Reporting a problem

Open an issue with the command you ran, what you expected, and what happened —
`dsh-kb doctor` covers a lot of ground and its output is usually the fastest
path to a diagnosis. For anything security-related, see [SECURITY.md](SECURITY.md)
instead of opening a public issue.
