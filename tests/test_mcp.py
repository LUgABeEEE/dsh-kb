"""MCP server tests.

Three layers, deliberately:

1. **Protocol** — drive :func:`handle` directly and assert the JSON-RPC shape
   (handshake, notifications are never answered, tools, error surfaces).
2. **Process** — run ``python -m dsh_kb mcp`` as a real subprocess and speak
   newline-delimited JSON to it.  This is what catches "something printed to
   stdout and corrupted the stream".
3. **Interop** — drive the server with the **official MCP Python SDK**.  A
   hand-rolled protocol implementation is only worth keeping if the real client
   agrees with it; the test skips when the SDK is not installed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dsh_kb.config import Config  # noqa: E402
from dsh_kb.mcp_server import (  # noqa: E402
    INSTRUCTIONS,
    PROTOCOL_VERSIONS,
    handle,
    serve,
    tool_definitions,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"

FIXTURE = {
    "index.md": '---\nokf_version: "0.2"\n---\n\n# Index\n',
    "z0-public/mount.md": (
        "---\ntype: lesson\ntitle: 挂载失效的处置\n---\n\n"
        "先确认 `/proc/mounts`，再执行 `mount -a`。\n\n唯一标识 UNIQ-MCP-ALPHA-11111\n"
    ),
    "z2-sensitive/med.md": "---\ntype: fact\ntitle: 敏感\n---\n\nUNIQ-MCP-CANARY-99999\n",
}

try:  # optional: only used for the interop layer
    import mcp  # noqa: F401

    HAVE_SDK = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_SDK = False


def make_home(root: Path) -> tuple[Config, Path]:
    """Build a fixture bundle plus a synced mirror and index."""
    from dsh_kb.service import refresh

    bundle = root / "share"
    for rel, text in FIXTURE.items():
        path = bundle / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    cfg = Config(home=root / "home", source="local", local_root=str(bundle))
    refresh(cfg)
    return cfg, bundle


class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-mcp-"))
        cls.cfg, _ = make_home(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def rpc(self, method: str, params: dict | None = None, request_id: int = 1):
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        return handle(self.cfg, message)

    # ---- handshake -----------------------------------------------------
    def test_initialize_echoes_a_known_protocol_version(self):
        for version in PROTOCOL_VERSIONS:
            response = self.rpc("initialize", {"protocolVersion": version})
            self.assertEqual(response["result"]["protocolVersion"], version)

    def test_initialize_falls_back_for_an_unknown_version(self):
        response = self.rpc("initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(response["result"]["protocolVersion"], PROTOCOL_VERSIONS[0])

    def test_initialize_advertises_tools_and_server_info(self):
        result = self.rpc("initialize", {})["result"]
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "dsh-kb")
        self.assertTrue(result["instructions"])

    def test_notifications_are_never_answered(self):
        self.assertIsNone(handle(self.cfg, {"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_ping_is_answered(self):
        self.assertEqual(self.rpc("ping")["result"], {})

    def test_unknown_method_is_a_protocol_error(self):
        response = self.rpc("resources/list")
        self.assertEqual(response["error"]["code"], -32601)

    # ---- tools ---------------------------------------------------------
    def test_tool_list_shape(self):
        tools = self.rpc("tools/list")["result"]["tools"]
        names = [t["name"] for t in tools]
        self.assertEqual(names, ["search", "read", "status", "skills"])
        for tool in tools:
            self.assertIn("description", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])

    def test_tool_schemas_stay_small(self):
        """Every tool and parameter is resident context **on every call**.

        Measured: 1289 chars (~322 tokens) with search/read/status.  The `skills`
        tool (added 2026-10-14 — the curated skill library lives in the `z3-skills`
        asset zone and is deliberately kept out of the main index, so it needs its
        own entry point) brings the blob to **1674 chars** (~419 tokens).  Plus 125
        chars of instructions (~31 tokens) ≈ 450 tokens standing.  This knowledge
        base is a supplement consulted on demand, so the budget stays tight —
        adding a tool or a parameter is a real cost, not a freebie.
        """
        blob = json.dumps(tool_definitions(), ensure_ascii=False)
        self.assertLess(len(blob), 1800, "tool schemas grew past the documented budget")
        self.assertLess(len(INSTRUCTIONS), 200, "instructions grew past the budget")

    def test_instructions_do_not_encourage_routine_calls(self):
        """Wording is behaviour: "search before reading files" invited a lookup every turn."""
        self.assertIn("only when", INSTRUCTIONS)

    def test_search_returns_hits(self):
        response = self.rpc("tools/call", {"name": "search", "arguments": {"query": "挂载"}})
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["hits"][0]["path"], "z0-public/mount.md")
        self.assertGreaterEqual(payload["hits"][0]["line"], 1)

    def test_search_never_returns_the_sensitive_zone_by_default(self):
        response = self.rpc("tools/call",
                            {"name": "search", "arguments": {"query": "UNIQ-MCP-CANARY-99999"}})
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["count"], 0)

    def test_read_returns_a_window(self):
        response = self.rpc("tools/call",
                            {"name": "read", "arguments": {"path": "index.md", "lines": 2}})
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertEqual(payload["path"], "index.md")
        self.assertEqual(len(payload["lines"]), 2)
        self.assertEqual(payload["lines"][0]["n"], 1)

    def test_status_reports_freshness(self):
        response = self.rpc("tools/call", {"name": "status", "arguments": {}})
        payload = json.loads(response["result"]["content"][0]["text"])
        self.assertIn("mirror", payload)
        self.assertIn("index", payload)
        self.assertGreaterEqual(payload["mirror"]["files"], 2)

    def test_read_refuses_to_escape_the_mirror(self):
        response = self.rpc("tools/call",
                            {"name": "read", "arguments": {"path": "../../etc/passwd"}})
        self.assertTrue(response["result"]["isError"])

    # ---- tool failures are results, not protocol errors ----------------
    def test_unknown_tool_is_an_error_result(self):
        response = self.rpc("tools/call", {"name": "nope", "arguments": {}})
        self.assertNotIn("error", response)
        self.assertTrue(response["result"]["isError"])
        self.assertIn("unknown tool", response["result"]["content"][0]["text"])

    def test_missing_query_is_an_error_result(self):
        response = self.rpc("tools/call", {"name": "search", "arguments": {}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("query is required", response["result"]["content"][0]["text"])

    def test_out_of_range_limit_is_an_error_result(self):
        response = self.rpc("tools/call",
                            {"name": "search", "arguments": {"query": "x", "limit": 999}})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("limit", response["result"]["content"][0]["text"])

    def test_bad_zone_is_an_error_result(self):
        response = self.rpc("tools/call",
                            {"name": "search", "arguments": {"query": "x", "zone": "z9"}})
        self.assertTrue(response["result"]["isError"])


class TestProcess(unittest.TestCase):
    """Talk to a real subprocess: this is where stdout hygiene is proven."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-mcp-proc-"))
        self.cfg, _ = make_home(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _frames(self) -> list[dict]:
        requests = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": PROTOCOL_VERSIONS[0], "capabilities": {}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "search", "arguments": {"query": "挂载"}}},
        ]
        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests)
        env = dict(os.environ, DSH_KB_HOME=str(self.cfg.home),
                   DSH_KB_SOURCE="local", DSH_KB_LOCAL_ROOT=str(self.cfg.home.parent / "share"),
                   PYTHONPATH=str(SRC))
        done = subprocess.run([sys.executable, "-m", "dsh_kb", "mcp"], input=payload,
                              capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        out = [json.loads(line) for line in done.stdout.splitlines() if line.strip()]
        return out

    def test_stdio_exchange_is_clean_json_rpc(self):
        frames = self._frames()
        # three answers: initialize, tools/list, tools/call — the notification
        # must not be answered, and nothing else may appear on stdout
        self.assertEqual(len(frames), 3, [f.get("id") for f in frames])
        self.assertEqual([f["id"] for f in frames], [1, 2, 3])
        self.assertEqual(frames[2]["result"]["content"][0]["type"], "text")

    def test_search_over_the_wire_finds_the_document(self):
        frames = self._frames()
        payload = json.loads(frames[2]["result"]["content"][0]["text"])
        self.assertEqual(payload["hits"][0]["path"], "z0-public/mount.md")


@unittest.skipUnless(HAVE_SDK, "official MCP SDK not installed")
class TestOfficialSdkInterop(unittest.TestCase):
    """The hand-rolled server must satisfy the real client implementation."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="dsh-kb-mcp-sdk-"))
        self.cfg, self.bundle = make_home(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sdk_client_can_handshake_list_and_call(self):
        import asyncio

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = dict(os.environ, DSH_KB_HOME=str(self.cfg.home), DSH_KB_SOURCE="local",
                   DSH_KB_LOCAL_ROOT=str(self.bundle), PYTHONPATH=str(SRC))
        params = StdioServerParameters(command=sys.executable, args=["-m", "dsh_kb", "mcp"], env=env)

        async def scenario():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    result = await session.call_tool("search", {"query": "挂载"})
                    status = await session.call_tool("status", {})
                    return tools, result, status

        tools, result, status = asyncio.run(scenario())
        names = {t.name for t in tools.tools}
        self.assertEqual(names, {"search", "read", "status", "skills"})

        def is_error(call_result) -> bool:
            # The SDK has used both spellings across versions.
            return bool(getattr(call_result, "is_error", getattr(call_result, "isError", False)))

        payload = json.loads(result.content[0].text)
        self.assertEqual(payload["hits"][0]["path"], "z0-public/mount.md")
        self.assertFalse(is_error(result))

        described = json.loads(status.content[0].text)
        self.assertIn("mirror", described)
        self.assertFalse(is_error(status))


if __name__ == "__main__":
    unittest.main(verbosity=2)
