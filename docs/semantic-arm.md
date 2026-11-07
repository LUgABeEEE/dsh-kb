# The optional semantic arm

dsh-kb 0.1.0 is a lexical search engine: SQLite FTS5, CJK bigrams, no
dependencies. This page documents the layer that sits **on top of** that and is
**off by default** — the part that can use embeddings, a cross-encoder, or a
remote model.

Read this before turning anything on: some of these layers send text (and, for
the image arm, pictures) off the machine, and none of them are covered by the
zero-dependency promise.

## What "off by default" guarantees

These hold with a stock configuration, and the test-suite asserts them:

- `import dsh_kb` never imports third-party code. Every optional dependency is
  imported inside a function.
- A search is pure FTS5. The optional paths are gated by config switches whose
  defaults are `false`, so results are identical to 0.1.0.
- Any failure in an optional layer — missing module, missing model, dead
  endpoint, unparsable answer — degrades to the lexical order. It never raises
  into a query and never silently changes ranking.

## The optional layers

| Layer | Module | Needs | Default |
|---|---|---|---|
| Vector arm (adds embedding hits to the keyword hits) | `ragvec.py` | `numpy` + a `vectors.sqlite` built outside this repo + an embedding endpoint (Ollama by default) | `rag.vector_enabled=false` |
| Image arm (adds pictures to the same RRF ranking — **no usable image source in this KB today**, see below) | `ragvec.py` + `image_embed.py` | `numpy` + a `vectors-image.sqlite` built by `python3 -m dsh_kb.image_embed --build` + a multimodal key in your own credential file (`DSH_KB_IMAGE_KEY_NAME` names it) | `rag.image_arm_enabled=false` |
| Cross-encoder rerank | `rerank.py` | `onnxruntime`, `tokenizers`, ~280 MB `bge-reranker-base` int8 ONNX | `rag.rerank_enabled=false`, and also gated by `rerank_max_shift>0` |
| ONNX embedding backend (A/B only) | `embed_onnx.py` | `onnxruntime`, `numpy`, ~610 MB `Qwen3-Embedding-0.6B` int8 ONNX | off; needs `DSH_KB_EMBED_PROVIDER=onnx` |
| Deep rerank (LLM picks pages) | `deeprank.py` | any OpenAI-compatible chat endpoint + key | `rag.deep_enabled=false`; MCP callers must also pass `deep=true` |

### Image arm: trigger conditions and what it can return here

Three gates stand between the image arm and a search result; all three have to be
open:

1. **Upstream gate (`index.py`)** — `search()` only calls `augment()` when
   `rag.vector_enabled` is on **and** one of: `rag.vector_always=true`, keyword
   hits fewer than `limit`, or a weak-match reserve. `rag.vector_trigger=always`
   alone does **not** open it (it only changes the weak-match reserve). With
   `vector_always` unset, a query whose FTS hits already fill `limit` never
   reaches the image arm, so enabling the image arm also requires
   `rag.vector_always=true`. (This is an `index.py` decision; the image arm
   cannot bypass it from `ragvec.py`.)
2. **Path prefixes** — the arm only returns rows under the current zone's
   `query_prefixes` (`z0-public/`, `z1-restricted/`, `_meta/`, `_incoming/` for
   `z01`). Rows outside are filtered out at query time.
   `rag.image_arm_prefixes` / `DSH_KB_RAG_IMAGE_ARM_PREFIXES` can widen this, but
   see the state note below.
3. **Account** — both `--build` and query-time embedding call Ark. Since
   **2026-11-06 21:20** the PAYG account returns `403 AccountOverdueError`; every
   failure (including that one) degrades to a `note`, never an exception.

`python3 -m dsh_kb.image_embed --readiness --home <home>` prints all three at
once; `usable` is true only when the arm is enabled, the store has rows, the
current zone's prefixes match some of them, and `vector_always` is on.

**State of this deployment (2026-11-06):** the mirror holds **0** images under
`z0-public/` and `z1-restricted/`, and **19** under `z3-skills/` (23 files, 4
below the 2 KB floor). `z3-skills` is `index:false` and outside
`query_prefixes("z01")`, and production has `vector_always` off. Therefore the
image arm returns **zero candidates** for every production `z01` search, and on a
full-hit query it is not even invoked. **Conclusion: the sidecar has no usable
image source in this KB today — it is wiring kept for a future source. Keep
`rag.image_arm_enabled=false`.**

Widening the prefixes to `z3-skills/` would make the 19 skill-asset pictures
(palette galleries, UI mockups) reachable from `z01` searches. Not recommended:
z3 is a skills-asset zone whose pages are deliberately not indexed, its images
are unrelated to knowledge queries, and returning z3 paths from a `z01` search
breaks the zone contract. The knob exists for the day images land in z0/z1, or
for an explicit cross-zone image search — it is not an activation path for this
KB.

`dsh-kb rag-check` works with or without any of them: it compares a vector store
against the index and reports absent stores as *absent*, not as broken.

## Model provenance

The repository ships **no weights and no download code**. If you want the
optional layers, fetch the models yourself and point the environment at them:

| Model | Upstream | Files the code expects |
|---|---|---|
| `bge-reranker-base` int8 ONNX | `Xenova/bge-reranker-base` on HuggingFace | `<DSH_KB_RERANK_ROOT>/models/model_quantized.onnx`, `.../tokenizer.json` |
| `Qwen3-Embedding-0.6B` int8 ONNX | `Qwen/Qwen3-Embedding-0.6B` (exported to ONNX yourself) | `<DSH_KB_EMBED_ONNX_ROOT>/onnx/model_int8.onnx`, `.../tokenizer.json` |
| `doubao-embedding-vision-251215` | Ark (remote, nothing shipped) | none — the key is read from the local credential file named by `DSH_KB_IMAGE_CREDENTIALS`, under the key named by `DSH_KB_IMAGE_KEY_NAME`; never from config, never logged |

`<DSH_KB_RERANK_ROOT>/lib/` is where the code looks for importable
`onnxruntime`/`tokenizers` when you keep them out of the system interpreter;
`DSH_KB_ONNX_LIB` overrides that directory.

Verify the files yourself — the project deliberately has no checksums to trust
and no opinion about where you keep them. With the variables unset, every
`available()` in these modules returns `False` and nothing is loaded.

## Turning it on

Environment (all equivalent to the `rag.*` keys in `config.json`):

```bash
# vector arm
export DSH_KB_RAG_VECTOR=1
export DSH_KB_RAG_VECTORS_DB=/srv/kb/index/vectors.sqlite   # default: <home>/index/vectors.sqlite
export DSH_KB_RAG_OLLAMA=http://127.0.0.1:11434             # embedding endpoint (bge-m3 by default)
export DSH_KB_NUMPY_PATH=/opt/kb-pyenv/lib                  # only if numpy is not importable

# cross-encoder rerank
export DSH_KB_RERANK_ROOT=/opt/kb-rerank                    # lib/ + models/
export DSH_KB_RAG_RERANK_MAX_SHIFT=2                        # same as rag.rerank_max_shift

# image arm (sends the query text AND the image bytes to Ark — see below)
export DSH_KB_RAG_IMAGE_ARM_ENABLED=1
export DSH_KB_RAG_VECTOR_ALWAYS=1                               # REQUIRED: else full-hit queries skip augment
export DSH_KB_RAG_IMAGE_ARM_DB=/srv/kb/index/vectors-image.sqlite   # default: next to vectors.sqlite
export DSH_KB_RAG_IMAGE_ARM_WEIGHT=0.6                         # independent of the text route
# export DSH_KB_RAG_IMAGE_ARM_PREFIXES=z0-public/,z1-restricted/ # only to deliberately widen zones

# deep rerank (sends page text to a remote model — see below)
export DSH_KB_RAG_DEEP_ENABLED=1
export DSH_KB_DEEP_ENDPOINT=https://api.example.com/v1
export DSH_KB_DEEP_MODEL=some-chat-model
export DSH_KB_DEEP_API_KEY=...                              # or DSH_KB_DEEP_CREDENTIALS=<file with DEEP_API_KEY>
```

`config.json` equivalents, if you prefer a file:

```json
{ "rag": { "vector_enabled": true, "vectors_db": "/srv/kb/index/vectors.sqlite",
           "rerank_enabled": true, "rerank_max_shift": 2,
           "image_arm_enabled": false, "image_arm_weight": 0.6,
           "deep_enabled": false, "deep_model": "" } }
```

Building the image store is an explicit offline step (it needs the Ark key, and
sending a whole mirror's pictures is not something a search should trigger):

```bash
python3 -m dsh_kb.image_embed --build              # incremental: sha256 decides
python3 -m dsh_kb.image_embed --build --limit 20   # smoke-test the first 20
python3 -m dsh_kb.image_embed --status             # count / dims / zones / orphans
                                                   # + production-zone prefix hits
python3 -m dsh_kb.image_embed --readiness          # can this arm ever return anything?
```

While Ark reports `403 AccountOverdueError`, `--build` records every image as
`failed` and exits non-zero, but leaves existing rows untouched (per-row upsert,
failures are skipped). Query-time embedding degrades the same way: one note, text
results unchanged. A 4xx such as 403 is **not retried** (an overdue account does
not recover by waiting); only 5xx/429/network errors use the 3-attempt backoff.

The `image_arm_*` keys are part of `_RAG_FIELDS` now, so the `config.json` form
above works directly; the environment variables override it as usual. The older
`DSH_KB_RAG_IMAGE_DB` spelling is still accepted for compatibility, but
`DSH_KB_RAG_IMAGE_ARM_DB` is the name to use. `image_arm_prefixes` is not a
`config.py` field yet: use `DSH_KB_RAG_IMAGE_ARM_PREFIXES` when you deliberately
widen the zone prefixes (see the state note above for why we do not).

`--build` resolves its target the same way the reader does: `--db` first, then
`rag.image_arm_db` from `config.json`, then `<home>/index/vectors-image.sqlite`.
Point both at the same file — a build that writes A while retrieval reads B
succeeds on both sides and simply never finds anything.

Files smaller than 2 KB are skipped on purpose: LFS pointer stubs are 131 bytes
and the API rejects them as `Invalid base64 image url`. One image = **one HTTP
request** — this endpoint silently returns only the first vector when `input`
carries several elements, so batching here is always a data-loss bug.

## Data egress — read this before enabling `deep`

| Path | What leaves the machine | Where |
|---|---|---|
| `deep=true` search | the query **and** up to `deep_candidates` (40) pages: file name + first `deep_passage_chars` (340) characters of each page body | `DSH_KB_DEEP_ENDPOINT` |
| HyDE (`rag.hyde_enabled`) | the query only | `DSH_KB_HYDE_ENDPOINT` — only when `DSH_KB_HYDE_PROVIDER=remote`; otherwise the local Ollama model (`rag.hyde_model`) writes it |
| Vector arm | the query and the page text being embedded | the embedding endpoint — `http://127.0.0.1:11434` (local Ollama) by default |
### One key name, never a fallback

`DSH_KB_IMAGE_KEY_NAME` defaults to `IMAGE_EMBED_API_KEY` and may hold any name
your credential file uses. The library deliberately reads **exactly one** name:

a provider that offers both a subscription key and a pay-as-you-go key returns
*"model not supported"* — not *"wrong key"* — when you use the subscription one
for embeddings. A library that silently tries the next key turns that into a 401
that looks like a model problem. One name, and the error says which name it
looked for.

`DSH_KB_IMAGE_CREDENTIALS` has **no default path**: where credentials live is a
deployment fact, and a built-in path would either read someone else's file on a
different machine or report a missing file as "the model is unavailable".

| Image arm (query) | the query text | `https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal` |
| Image arm (`--build`) | each indexed image, base64-encoded | same Ark endpoint |
| Everything else | nothing | — |

There is no telemetry and no automatic model download. If your bundle contains
pages you would not paste into a chat window, leave `deep` off — it is the one
path that sends page bodies to a third party.

Failures write one line to `DSH_KB_DEEP_LOG` only when that variable is set; a
library does not pick a log path on someone else's machine.

## How this was measured

Numbers in `CHANGELOG.md` come from **one 33-question set on one deployment**,
and are reported with their cost (e.g. "25/33 → 26/33, p50 8.4 s"). Treat them
as an existence proof that the layer can pay for itself, not as a benchmark you
can compare your own corpus against.

## Verifying a build

```bash
# the lexical suite never needs any of this
python3 -m unittest discover -s tests -t .

# the vector arm's own tests, with numpy borrowed from anywhere
DSH_KB_NUMPY_PATH=/opt/kb-pyenv/lib python3 -m unittest tests.test_ragvec -v

# the image sidecar (build/CLI/fusion/degradation; HTTP is stubbed, no key needed)
python3 -m unittest tests.test_image_arm -v
```

Without `numpy` those three tests **skip** rather than fail — the arm is
optional, and a clone with no third-party packages must still be green.

> **This library currently has no usable image source.** Measured 2026-11-06: `z0-public` 0 images, `z1-restricted` 0 images, `z3-skills` 23 (19 over 2 KB) — but `z3-skills` is `index: false` and outside `query_prefixes('z01')`, so production queries match 0 of 19. Keep the arm off until a real image source appears under z0/z1. `python3 -m dsh_kb.image_embed --readiness` reports the gate state without any network call.
