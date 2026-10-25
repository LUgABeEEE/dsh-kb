"""A Model Context Protocol server over stdio — standard library only.

Why hand-rolled instead of depending on the official SDK:

* dsh-kb advertises **zero runtime dependencies**, which keeps installation and
  upgrades boring;
* the surface an agent actually needs is tiny — ``initialize``, ``tools/list``
  and ``tools/call`` over newline-delimited JSON-RPC 2.0;
* an SDK would make its release cadence our users' problem.

Interoperability is not taken on faith: ``tests/test_mcp.py`` drives this server
with the **official Python SDK client** and asserts the handshake, the tool list
and a real tool call.

Transport rules that matter
---------------------------
* stdout carries protocol frames and **nothing else** — every log line goes to
  the log file, never to stdout;
* the client may write a ``notifications/initialized`` frame with no ``id``;
  notifications must never be answered;
* tool failures are returned *inside* the result (``isError: true``) rather than
  as JSON-RPC errors, so the model can read and react to them.

Run it with ``dsh-kb mcp`` or ``python -m dsh_kb mcp``.
"""
from __future__ import annotations

import json
import sys

from .config import Config, load_config
from .service import file_logger, read_payload, search_with_refresh, status_payload
from .sources import SourceError

SERVER_NAME = "dsh-kb"
#: Newest first.  The handshake echoes the client's version when it is known.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

#: One line, because `instructions` lands in the model's context.
#: One line, because `instructions` lands in the model's context **every call**.
#: Wording matters as much as length: an earlier version said "search … before
#: reading files", which encouraged a lookup on every turn.  This knowledge base
#: is a *supplement* — it should be consulted when it is actually needed, so the
#: instruction says so explicitly.
INSTRUCTIONS = (
    "Optional local knowledge base: call `search` only when you need past decisions, "
    "environment facts or history — not routinely."
)


def tool_definitions() -> list[dict]:
    """The advertised tools.

    Kept deliberately small: every extra tool and every extra parameter is
    resident context the model pays for on each request.
    """
    return [
        {
            "name": "search",
            "description": "Search the knowledge base; returns path, line and a snippet. Expand a hit with `read`.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords or a phrase (CJK and latin may mix).",
                    },
                    "zone": {
                        "type": "string",
                        "enum": ["z01", "z0", "z1", "z2", "all"],
                        "description": "Zones to search. Default z01.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "description": "Max hits. Default 8.",
                    },
                    "snippet": {
                        "type": "integer",
                        "minimum": 60,
                        "maximum": 800,
                        "description": "Snippet width. Default 240.",
                    },
                    "deep": {
                        "type": "boolean",
                        "description": "Slow lane: LLM re-picks the best page (+5-10s). Default false.",
                    },
                },
            },
        },
        {
            "name": "read",
            "description": "Read a line window from a `path` returned by `search`.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path from `search`.",
                    },
                    "start": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "First line (1-based). Default 1.",
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Lines. Default 80.",
                    },
                },
            },
        },
        {
            "name": "status",
            "description": "Mirror/index age, source and counts. Use when results look stale.",
            "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
        },
        {
            "name": "skills",
            "description": "Search the curated skill library (724 third-party skills; kept out of the main index).",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Keywords."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "description": "Max hits. Default 8.",
                    },
                },
            },
        },
    ]


class ToolError(Exception):
    """A failure the model should see as a tool result, not a protocol error."""


def _as_int(value, name: str, *, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"{name} must be a number")
    number = int(value)
    if number < minimum or number > maximum:
        raise ToolError(f"{name} must be between {minimum} and {maximum}")
    return number


def call_tool(cfg: Config, name: str, arguments: dict | None) -> tuple[str, bool]:
    """Execute one tool.  Returns ``(text, is_error)``."""
    arguments = arguments or {}
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be an object")

    if name == "search":
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("query is required and must not be empty")
        zone = str(arguments.get("zone") or "z01")
        if zone not in ("z01", "z0", "z1", "z2", "all"):
            raise ToolError("zone must be one of z01, z0, z1, z2, all")
        payload = search_with_refresh(
            cfg,
            query,
            zone=zone,
            limit=_as_int(arguments.get("limit"), "limit", minimum=1, maximum=32, default=8),
            snippet_chars=_as_int(arguments.get("snippet"), "snippet",
                                  minimum=60, maximum=800, default=cfg.snippet_chars),
            on_event=file_logger(cfg),
            deep=bool(arguments.get("deep")),
        )
        return json.dumps(payload, ensure_ascii=False), False

    if name == "read":
        path = str(arguments.get("path") or "").strip()
        if not path:
            raise ToolError("path is required")
        payload = read_payload(
            cfg,
            path,
            start=_as_int(arguments.get("start"), "start", minimum=1, maximum=1_000_000, default=1),
            lines=_as_int(arguments.get("lines"), "lines", minimum=1,
                          maximum=cfg.max_read_lines, default=80),
        )
        return json.dumps(payload, ensure_ascii=False), False

    if name == "status":
        return json.dumps(status_payload(cfg), ensure_ascii=False), False

    if name == "skills":
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolError("query is required and must not be empty")
        from .skills import SkillsUnavailable, search as skills_search

        try:
            rows = skills_search(
                query,
                top=_as_int(arguments.get("limit"), "limit", minimum=1, maximum=32, default=8),
                cfg=cfg,
            )
        except SkillsUnavailable as exc:
            # 库不可用是"工具结果"，不是协议错误 —— 模型该看到人话原因（与 search 的约定一致）。
            raise ToolError(str(exc)) from exc
        return json.dumps({"count": len(rows), "hits": rows}, ensure_ascii=False), False

    raise ToolError(f"unknown tool: {name}")


def handle(cfg: Config, message: dict) -> dict | None:
    """Handle one JSON-RPC message.  Returns the response, or ``None`` for a
    notification (which must never be answered)."""
    request_id = message.get("id")
    if request_id is None:
        return None
    method = message.get("method")

    def ok(result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def fail(code: int, text: str) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": text}}

    if method == "initialize":
        params = message.get("params") or {}
        requested = str(params.get("protocolVersion") or "")
        version = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        return ok({
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": _version()},
            "instructions": INSTRUCTIONS,
        })

    if method == "ping":
        return ok({})

    if method == "tools/list":
        return ok({"tools": tool_definitions()})

    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        try:
            text, is_error = call_tool(cfg, name, params.get("arguments"))
        except ToolError as exc:
            return ok({"content": [{"type": "text", "text": str(exc)}], "isError": True})
        except SourceError as exc:
            return ok({"content": [{"type": "text", "text": str(exc)}], "isError": True})
        except FileNotFoundError as exc:
            return ok({"content": [{"type": "text", "text": str(exc)}], "isError": True})
        except Exception as exc:  # noqa: BLE001 - the model gets a readable message
            return ok({"content": [{"type": "text",
                                    "text": f"internal error: {type(exc).__name__}: {exc}"}],
                       "isError": True})
        return ok({"content": [{"type": "text", "text": text}], "isError": is_error})

    return fail(METHOD_NOT_FOUND, f"method not found: {method}")


def serve(cfg: Config | None = None, *, stdin=None, stdout=None) -> int:
    """Read newline-delimited JSON-RPC from stdin until EOF."""
    cfg = cfg or load_config()
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError as exc:
            _write(stdout, {"jsonrpc": "2.0", "id": None,
                            "error": {"code": PARSE_ERROR, "message": f"invalid JSON: {exc}"}})
            continue
        if not isinstance(message, dict):
            _write(stdout, {"jsonrpc": "2.0", "id": None,
                            "error": {"code": INVALID_REQUEST, "message": "message must be an object"}})
            continue
        try:
            response = handle(cfg, message)
        except Exception as exc:  # noqa: BLE001 - never die on one bad frame
            response = {"jsonrpc": "2.0", "id": message.get("id"),
                        "error": {"code": INTERNAL_ERROR,
                                  "message": f"{type(exc).__name__}: {exc}"}}
        if response is not None:
            _write(stdout, response)
    return 0


def _write(stdout, payload: dict) -> None:
    try:
        stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdout.flush()
    except (BrokenPipeError, ValueError):
        # The client went away; there is nowhere left to report that.
        pass


def _version() -> str:
    from . import __version__

    return __version__


def main(argv: list[str] | None = None) -> int:
    return serve()


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
