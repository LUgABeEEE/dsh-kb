"""dsh-kb — a personal knowledge base that AI agents can query cheaply.

Design in one line
------------------
The **single source of truth** is a directory of Markdown files on shared
storage (NAS / network share / any folder).  Each machine keeps a **local
mirror** plus a **local full-text index**, so retrieval is millisecond-fast
and keeps working while the share is offline.

Nothing in this package assumes a particular vendor or path: everything
machine-specific lives in a config file outside the checkout.
"""

__version__ = "0.5.1"

__all__ = ["__version__"]
