"""Test package.

The suite must behave the same on a laptop with a local model running and on a
CI runner with nothing installed.  Query rewriting and the cross-encoder are
both *environment-dependent* (they reach out to localhost), so they are pinned
off here; tests that exercise them opt in explicitly by overriding the module
constants.
"""

import os

#: Never let a test reach a local model by accident.
os.environ.setdefault("DSH_KB_REWRITE", "off")
os.environ.setdefault("DSH_KB_REWRITE_ENDPOINT", "http://127.0.0.1:1")
