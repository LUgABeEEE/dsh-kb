# Release notes for maintainers

The repository is **private until it is ready to be useful to a stranger**. This
page is the definition of "ready", so the decision is a checklist rather than a
feeling.

## Definition of ready for public

**Correctness**

- [ ] CI green on 3.10–3.13 *and* the SDK-interop job green against the current
      official `mcp` release.
- [ ] No skipped test explains away a real gap (the interop test skipping
      locally is fine; it must run in CI).
- [ ] `dsh-kb doctor` and `dsh-kb mcp` verified by hand on a machine that has
      never run this code.

**Zero-friction promise**

- [ ] Quickstart in `README.md` works on a clean machine with **no**
      `pip install` — clone, run the tests, run the CLI.
- [ ] `pip install .` builds a wheel with no `Requires-Dist` entries.
- [ ] The console script works from the installed wheel, not just from a
      checkout.

**Privacy and secrets**

- [ ] The `hygiene` CI job passes: no credential-shaped assignment, no private
      IP, no tracked build artifact.
- [ ] `git log -p` reviewed once by eye for anything that ever leaked and was
      later deleted — history is published too.
- [ ] The maintainer's own bundle, credentials and machine paths appear nowhere,
      including in examples, screenshots and commit messages.

**Documentation**

- [ ] `README.md` first screen answers: what is this, who is it for, what does
      it cost to run, how do I try it in two minutes.
- [ ] `docs/deployment.md` covers the platforms actually claimed.
- [ ] Every documented config key exists, and every config key is documented.
- [ ] A demo exists that a reader can look at without running anything
      (a recorded terminal session or a short GIF).

**Repository hygiene**

- [ ] `CHANGELOG.md` has an entry for the release, and the version in
      `dsh_kb.__version__` matches the tag.
- [ ] Branch protection on `main` requires the CI jobs.
- [ ] Description, topics and licence are set on the host.
- [ ] Issue and pull-request templates exist; `SECURITY.md` explains private
      reporting.

**Known gaps that are acceptable to publish with — and are written down**

- Lexical by default. The optional semantic arm exists but is off, opt-in, and
  explicitly outside the zero-dependency promise (docs/semantic-arm.md).
- No Windows CI job, although Windows is a supported target (a Windows UNC
  share is the reason generated names are lowercased).
- The MCP server implements the three methods an agent needs and does not
  advertise prompts, resources, sampling or subscriptions.

## Release steps

```bash
# 1) version + changelog agree
$EDITOR src/dsh_kb/__init__.py CHANGELOG.md

# 2) the suite, from a clean checkout
python -m unittest discover -s tests -t . -v

# 3) a real wheel, installed into a throwaway venv
python -m pip wheel --no-deps . -w /tmp/whl
python -m venv /tmp/venv && /tmp/venv/bin/pip install /tmp/whl/*.whl
/tmp/venv/bin/dsh-kb --version

# 4) tag and push
git tag -a v0.1.0 -m "dsh-kb 0.1.0"
git push --follow-tags

# 5) publish the release, turning the changelog section into the notes
gh release create v0.1.0 --title "dsh-kb 0.1.0" --notes-file CHANGELOG.md
```

## Going public

**Publish by recreating the repository, not by flipping its visibility.**

```bash
# 1) publish from a fresh repository so nothing unreachable ships with it
gh repo create <owner>/dsh-kb --public --source=. --remote=origin --push \
  --description "A personal Markdown knowledge base with a local mirror and index, built to be queried cheaply by AI agents."

# 2) verify what the world can reach: the branch history AND its size
gh api repos/<owner>/dsh-kb/commits --jq '.[].sha' | wc -l
```

Why recreate instead of `--visibility public`: **a force-push does not delete
anything.** Rewritten commits stay retrievable by SHA on the host — measured on
this very repository, where three commits authored by a stale identity remained
reachable through the API after the branch had been rewritten and force-pushed.
If history ever contained something you would not publish, only a fresh
repository removes it. Do this while the repository is still private, one day
old, and has no forks or clones; it is far cheaper than asking the host to
garbage-collect later.

Then, and only then:

- [ ] Enable Issues/Discussions, set description and topics.
- [ ] Confirm the package metadata points at the new URL (`pyproject.toml`,
      `SECURITY.md`) — the `hygiene` CI job checks this against the repository
      being built, so a stale owner fails the build rather than shipping.

Do this *after* the checklist above, not before: making a repository private
again does not un-publish anything that was already fetched.
