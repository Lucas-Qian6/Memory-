# Memory A/B eval harness

Proves the in-task memory system helps the deep-research agent by **ablation**:
the *same* `DeepResearchPipeline` runs twice per question, with only the injected
`MemoryManager` swapped.

- **arm ON**  - the real system (`MemoryManager` + `MemoryStore`): episodic dedup,
  relevance recall, bounded structured memory.
- **arm OFF** - `RawContextMemory` (in [arms.py](arms.py)): a finite recency
  context buffer with no dedup, no relevance ranking, no working-state thread -
  i.e. "a deep-research agent without external memory."

Everything else (planner, search API, extraction, synthesis, inject budget) is
identical, so any difference is attributable to memory.

## Run it

```bash
# deterministic, process metrics only (no LLM, offline mock search)
python eval/run_eval.py --mock --no-llm

# mock corpus + LLM loop + blind pairwise judge
python eval/run_eval.py --mock --rounds 5 --trials 2

# real search API (needs SEARCH_API_* in .env) + LLM + judge
python eval/run_eval.py --rounds 6 --trials 3
```

Useful flags: `--questions N` (cap to first N), `--rounds`, `--trials`,
`--page-size`, `--recall-top-k`, `--inject-char-budget` (tighten to force the OFF
arm's window to overflow sooner), `--no-judge`, `--seed`.

## What it measures (timeline P3 metrics)

Per run, from the pipeline counters + the `Tracer` event stream (no extra LLM
calls), see [metrics.py](metrics.py):

- **重复检索节省**: `searches_run`, `searches_skipped` - OFF re-runs duplicate
  searches (`skip=0`); ON skips them.
- **context 占用**: `context_chars` (chars injected into synthesis) + optional
  `llm_input_chars` (total prompt chars sent to the LLM).
- **证据覆盖**: `sources_cited` (distinct sources in the synthesized evidence),
  `claims_kept`.
- **cost**: `report_chars`, `llm_input_chars`.

Quality (only when an LLM is available) comes from a **blind pairwise**
LLM-as-judge ([judge.py](judge.py)) scoring faithfulness / coverage / coherence;
reports are shown in randomized order and mapped back to on/off afterwards.

## Reading the output

The runner prints a per-run line, then a `SUMMARY` table (ON, OFF, ON-OFF delta
averaged over questions x trials) and judge win-counts, and writes the full
result (config + raw rows + aggregates + judgments) to
`eval/results/eval_<timestamp>.json` (gitignored).

Expected shape: ON has higher `searches_skipped` and `sources_cited` and lower
`context_chars` for the same work; the judge prefers ON on coverage/coherence.

## Caveat: corpus size

The offline mock corpus is only 5 papers ([demo/tools.py](../demo/tools.py)), so
coverage/quality deltas under `--mock` are small - use it for the deterministic
process-metric signal. For a convincing **quality** comparison, run against the
real search API with `--rounds` high enough that the OFF arm's recency window
drops earlier-round findings.

## Process vs quality

Process metrics are deterministic under `--mock --no-llm` (reproducible). Quality
needs the LLM; `llm_input_chars` is `0` in `--no-llm` mode (no real calls) while
`context_chars` is still computed, so the context-usage comparison works offline.
