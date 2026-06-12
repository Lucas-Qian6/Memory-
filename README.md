# Memory @ DeepResearch (single-task, MVP)

The **memory** pillar of the team's DeepResearch agent (the link that replaces
玻尔's "AI论文写作·帮我写综述"). This MVP is a small, **pluggable** memory
subsystem scoped to **one long-horizon task**, plus a runnable single-task demo.

- Capability write-up (Chinese): [`docs/memory-capability.md`](docs/memory-capability.md)
- Memory-design findings on the real search API (Chinese): [`docs/memory-schema-findings.md`](docs/memory-schema-findings.md)

## Idea in one minute

A single user query kicks off a **long-horizon task** (~15 minutes, many steps).
Within that one task the agent overflows its context window, repeats searches it
already ran, and loses the thread. This module gives it an **in-task memory** with
three facets — **working / episodic / semantic** — implemented as a `type` tag
over a single store, differing only by write/read policy.

Scope is deliberately narrow: **one task only.** No cross-session reuse, no
long-term knowledge base, no model-managed bridge. The store lives in process
memory and is thrown away when the task ends. Memory is **system-managed**: our
code extracts, stores, retrieves (RAG-style), and the pipeline (or the model, via
tool calls) only ever touches `MemoryManager`.

## Architecture

```
DeepResearch single-task pipeline (plan -> search -> read -> synthesize)
        |  only via the interface
        v
   MemoryManager   remember / recall / compress / update_state / get_state
        v
   MemoryStore     one store, in-memory + type tag (working|episodic|semantic)
        |                         |
        v                         v
   policies (write/read)     retrieval (lexical now, vector later)
```

The pipeline only ever touches `MemoryManager`. That is what makes a future
"split into per-facet backends (e.g. vector DB for semantic)" a backend swap
rather than a rewrite.

## Layout

| Path | What |
| --- | --- |
| `src/memory_dr/schema.py` | `MemoryItem` + `type` field (working/episodic/semantic), keyword extraction |
| `src/memory_dr/store.py` | single in-memory `MemoryStore` (optional JSON dump for inspection) |
| `src/memory_dr/retrieval.py` | lexical relevance + recency scoring |
| `src/memory_dr/policies.py` | per-facet write (dedup) + read (rank/assemble) |
| `src/memory_dr/manager.py` | `MemoryManager` — the public, pluggable API |
| `demo/search_client.py` | real Wenyon `/api/v1/search` client + normalizing adapters (mock fallback) |
| `demo/llm.py` | LLM plan / extract / reflect / synthesize over an Anthropic-compatible gateway |
| `demo/tools.py` | offline mock search, shaped like the real API response |
| `demo/pipeline.py` | LLM-driven, memory-augmented single-task research loop |
| `demo/run_demo.py` | one long-horizon task demo |

## Run

```bash
pip install -r requirements.txt              # anthropic, for the LLM loop
python demo/run_demo.py --mock --no-llm      # fully offline + deterministic
python demo/run_demo.py --mock               # offline search + real LLM loop
python demo/run_demo.py "your question"      # real search API + LLM loop
```

Copy `.env.example` to `.env`. The real-search path needs `SEARCH_API_BASE_URL`
(reachable from where you run it — the `svc-...` name only resolves inside the
cluster / dev box; tunnel it for a laptop). The LLM loop reuses
`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `MEMORY_DR_MODEL`. With
`--mock --no-llm` the core memory system needs no network and no API key.

### Using an Anthropic-compatible gateway (e.g. GPUGeek)

The demo honors `ANTHROPIC_BASE_URL`, so you can point it at a compatible
endpoint without code changes:

```bash
export ANTHROPIC_API_KEY="<your-gpugeek-key>"
export ANTHROPIC_BASE_URL="https://api.gpugeek.com"
export MEMORY_DR_MODEL="Vendor2/Claude-4.6-opus"
python demo/run_demo.py --llm
```

## Plugging into the real pipeline

```python
from memory_dr import MemoryManager

mem = MemoryManager(task_id="task-x")             # one long-horizon task

mem.update_state("goal: ...; plan: ...")          # working: keep the thread

if not mem.seen_action(f"search: {q}"):           # episodic: skip repeated work
    ...                                           # do the search ...
    mem.remember(f"search: {q}", type="episodic")

mem.remember(claim_text, type="semantic", source=paper_id,
             links={"evidence": evidence_span})   # findings survive overflow

context = mem.recall_context(question, type="semantic")  # inject top-k into prompt
state = mem.get_state()                           # re-read latest progress any step
```
