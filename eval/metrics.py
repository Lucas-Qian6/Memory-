"""Per-run metric extraction + aggregation for the memory A/B eval.

Metrics come from two cheap, already-present sources: the pipeline's own
counters and the ``Tracer`` event stream (demo/trace.py). Nothing here calls the
LLM, so these process metrics are deterministic given the same search results.

Mapping to the timeline P3 metrics:
- 重复检索节省 (redundant work avoided): ``searches_run`` / ``searches_skipped``
- context 占用 (context usage):           ``context_chars`` (+ ``llm_input_chars``)
- 证据覆盖 (evidence coverage):            ``sources_cited`` / ``claims_kept``
- cost:                                   ``report_chars`` / ``llm_input_chars``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Numeric fields we average across trials and diff between arms.
NUMERIC_FIELDS = [
    "steps",
    "searches_run",
    "searches_skipped",
    "claims_kept",
    "context_chars",
    "used_claims",
    "sources_cited",
    "report_chars",
    "llm_input_chars",
    # P2/P3 activity: near-duplicate merges, contradiction supersedes/conflicts.
    "merges",
    "supersedes",
    "conflicts",
]


def _last_synthesize(tracer) -> Optional[Dict[str, Any]]:
    """Return the payload of the last 'synthesize' llm trace event, if any."""
    if tracer is None:
        return None
    found = None
    for ev in tracer.all_events():
        p = ev.get("payload") or {}
        if ev.get("kind") == "llm" and p.get("call") == "synthesize":
            found = p
    return found


def collect(
    arm: str,
    question_id: str,
    trial: int,
    pipe,
    tracer,
    report: str,
    llm_input_chars: int = 0,
) -> Dict[str, Any]:
    """Build one metrics row for a finished (arm, question, trial) run."""
    syn = _last_synthesize(tracer) or {}
    stats = pipe.memory.stats()
    wc = getattr(pipe.memory, "write_counts", {}) or {}
    return {
        "arm": arm,
        "question_id": question_id,
        "trial": trial,
        # 重复检索节省
        "steps": pipe.steps,
        "searches_run": pipe.searches_run,
        "searches_skipped": pipe.searches_skipped,
        # 证据覆盖
        "claims_kept": stats.get("semantic", 0),
        "used_claims": syn.get("used_claims", 0),
        "sources_cited": len(syn.get("sources", []) or []),
        # context 占用 / cost
        "context_chars": syn.get("context_chars", 0),
        "report_chars": syn.get("report_chars", len(report or "")),
        "llm_input_chars": llm_input_chars,
        # P2 近重合并 / P3 一致性 (0 on the OFF arm, which does no merging)
        "merges": wc.get("merge", 0),
        "supersedes": wc.get("supersede", 0),
        "conflicts": wc.get("conflict", 0),
    }


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_by_arm(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Average each numeric field per arm."""
    out: Dict[str, Dict[str, float]] = {}
    for arm in ("on", "off"):
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        summary = {f: round(_mean([r.get(f, 0) for r in rows]), 1) for f in NUMERIC_FIELDS}
        summary["n_runs"] = len(rows)
        out[arm] = summary
    return out


def deltas(by_arm: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """ON minus OFF for each numeric field (positive => ON is higher)."""
    if "on" not in by_arm or "off" not in by_arm:
        return {}
    return {f: round(by_arm["on"][f] - by_arm["off"][f], 1) for f in NUMERIC_FIELDS}
