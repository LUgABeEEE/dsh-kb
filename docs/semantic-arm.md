# The optional semantic arm

dsh-kb 0.1.0 is a lexical search engine: SQLite FTS5, CJK bigrams, no
dependencies. This page documents the layer that sits **on top of** that and is
**off by default** — the part that can use embeddings, a cross-encoder, or a
remote model.

Read this before turning anything on: two of the four layers send text off the
machine, and none of them are covered by the zero-dependency promise.

## What "off by default" guarantees

These hold with a stock configuration, and the test-suite asserts them:

- `import dsh_kb` never imports third-party code. Every optional dependency is
  imported inside a function.
- A search is pure FTS5. The optional paths are gated by config switches whose
  defaults are `false`, so results are identical to 0.1.0.
- Any failure in an optional layer — missing module, missing model, dead
  endpoint, unparsable answer — degrades to the lexical order. It never raises
  into a query and never silently changes ranking.

## The four layers

| Layer | Module | Needs | Default |
|---|---|---|---|
| Vector arm (adds embedding hits to the keyword hits) | `ragvec.py` | `numpy` + a `vectors.sqlite` built outside this repo + an embedding endpoint (Ollama by default) | `rag.vector_enabled=false` |
| Cross-encoder rerank | `rerank.py` | `onnxruntime`, `tokenizers`, ~280 MB `bge-reranker-base` int8 ONNX | `rag.rerank_enabled=false`, and also gated by `rerank_max_shift>0` |
| ONNX embedding backend (A/B only) | `embed_onnx.py` | `onnxruntime`, `numpy`, ~610 MB `Qwen3-Embedding-0.6B` int8 ONNX | off; needs `DSH_KB_EMBED_PROVIDER=onnx` |
| Deep rerank (LLM picks pages) | `deeprank.py` | any OpenAI-compatible chat endpoint + key | `rag.deep_enabled=false`; MCP callers must also pass `deep=true` |

`dsh-kb rag-check` works with or without any of them: it compares a vector store
against the index and reports absent stores as *absent*, not as broken.

## Model provenance

The repository ships **no weights and no download code**. If you want the
optional layers, fetch the models yourself and point the environment at them:

| Model | Upstream | Files the code expects |
|---|---|---|
| `bge-reranker-base` int8 ONNX | `Xenova/bge-reranker-base` on HuggingFace | `<DSH_KB_RERANK_ROOT>/models/model_quantized.onnx`, `.../tokenizer.json` |
| `Qwen3-Embedding-0.6B` int8 ONNX | `Qwen/Qwen3-Embedding-0.6B` (exported to ONNX yourself) | `<DSH_KB_EMBED_ONNX_ROOT>/onnx/model_int8.onnx`, `.../tokenizer.json` |

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
           "deep_enabled": false, "deep_model": "" } }
```

## Data egress — read this before enabling `deep`

| Path | What leaves the machine | Where |
|---|---|---|
| `deep=true` search | the query **and** up to `deep_candidates` (40) pages: file name + first `deep_passage_chars` (340) characters of each page body | `DSH_KB_DEEP_ENDPOINT` |
| HyDE (`rag.hyde_enabled`) | the query only | `DSH_KB_HYDE_ENDPOINT` |
| Vector arm | the query and the page text being embedded | the embedding endpoint — `http://127.0.0.1:11434` (local Ollama) by default |
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
```

Without `numpy` those three tests **skip** rather than fail — the arm is
optional, and a clone with no third-party packages must still be green.
