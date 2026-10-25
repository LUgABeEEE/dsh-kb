"""Tests for the `z3-skills` asset-zone entry point.

The skill library lives *inside* the knowledge bundle, so its location must be
resolved from configuration and never hardcoded — that keeps the project usable
on someone else's deployment without editing source.  These tests pin that
contract down, along with the "library missing" error path.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest

from dsh_kb import skills


class FakeConfig:
    """Only the two attributes the resolver looks at."""

    def __init__(self, mount: str | None = None, local_root: str | None = None):
        self.mount = mount
        self.local_root = local_root


def make_library(root: pathlib.Path) -> pathlib.Path:
    """A bundle-shaped library: manifest, one skill, and a stand-in finder."""
    lib = root.joinpath(*skills.ASSET_SUBDIR)
    (lib / "bin").mkdir(parents=True)
    (lib / "skills" / "demo").mkdir(parents=True)
    (lib / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\n# Demo\n", encoding="utf-8")
    (lib / "manifest.json").write_text(json.dumps({
        "items": [{"name": "demo", "verdict": "recommended", "kind": "ops",
                   "size_bytes": 10, "license_spdx": "MIT"}],
        "updated": "2026-01-01",
    }), encoding="utf-8")
    (lib / "bin" / "find.py").write_text(
        "import json\nprint(json.dumps([{'name': 'demo', 'score': 1}]))\n",
        encoding="utf-8")
    return lib


class EnvVar:
    """Set (or clear) the library override for the duration of a block."""

    def __init__(self, value: str | None):
        self.value = value

    def __enter__(self):
        self.previous = os.environ.get(skills.LIB_ENV)
        if self.value is None:
            os.environ.pop(skills.LIB_ENV, None)
        else:
            os.environ[skills.LIB_ENV] = self.value
        return self

    def __exit__(self, *exc):
        if self.previous is None:
            os.environ.pop(skills.LIB_ENV, None)
        else:
            os.environ[skills.LIB_ENV] = self.previous
        return False


class TestLibraryResolution(unittest.TestCase):
    def test_env_var_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = make_library(pathlib.Path(tmp))
            with EnvVar(str(lib)):
                self.assertEqual(skills.library_root(FakeConfig()), lib)

    def test_env_var_pointing_nowhere_is_an_error(self):
        with EnvVar("/nonexistent/skills-library"):
            with self.assertRaises(skills.SkillsUnavailable):
                skills.library_root(FakeConfig())

    def test_derived_from_source_root(self):
        """The library rides on the configured source root, mount or local."""
        with tempfile.TemporaryDirectory() as tmp:
            lib = make_library(pathlib.Path(tmp))
            with EnvVar(None):
                self.assertEqual(skills.library_root(FakeConfig(mount=tmp)), lib)
                self.assertEqual(skills.library_root(FakeConfig(local_root=tmp)), lib)

    def test_missing_library_says_what_to_do(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EnvVar(None):
                with self.assertRaises(skills.SkillsUnavailable) as ctx:
                    skills.library_root(FakeConfig(mount=tmp))
            self.assertIn(skills.LIB_ENV, str(ctx.exception))

    def test_stats_search_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_library(pathlib.Path(tmp))
            cfg = FakeConfig(mount=tmp)
            info = skills.stats(cfg)
            self.assertEqual(info["count"], 1)
            self.assertEqual(info["license_risk"], 0)
            self.assertEqual(info["verdicts"], {"recommended": 1})
            self.assertEqual(skills.search("demo", cfg=cfg)[0]["name"], "demo")
            rel, text = skills.read_skill("demo", cfg=cfg)
            self.assertTrue(rel.endswith("SKILL.md"))
            self.assertIn("A demo skill", text)

    def test_unknown_skill_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_library(pathlib.Path(tmp))
            with self.assertRaises(skills.SkillsUnavailable):
                skills.read_skill("nope", cfg=FakeConfig(mount=tmp))

    def test_source_carries_no_deployment_path(self):
        """A deployment path in the source would defeat the vendor-neutral goal."""
        source = pathlib.Path(skills.__file__).read_text(encoding="utf-8")
        for needle in ("/data/agent", "/mnt/nas", "/home/", "C:\\"):
            self.assertNotIn(needle, source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
