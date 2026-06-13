"""A/B eval harness: run the SAME DeepResearch pipeline with memory ON vs OFF.

For every curated question x trial we run both arms (only the injected
MemoryManager differs), collect process metrics from the pipeline + tracer, and
- when an LLM is available - score report quality with a blind pairwise judge.
Aggregates ON-vs-OFF deltas + judge win-rates, prints a summary, and dumps the
full result to ``eval/results/<timestamp>.json``.

Run:
    python eval/run_eval.py --rounds 6 --trials 3   # real search API + LLM (needs SEARCH_API_* + ANTHROPIC_* in .env)
    python eval/run_eval.py --no-judge              # skip LLM-as-judge quality scoring (process metrics only)

A real search API and an LLM are required; there is no offline/mock mode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "demo"))

import llm  # noqa: E402
from pipeline import DeepResearchPipeline  # noqa: E402
from search_client import SearchClient  # noqa: E402
from trace import Tracer  # noqa: E402
from run_demo import load_dotenv  # noqa: E402

import arms  # noqa: E402
import judge as judge_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import questions as questions_mod  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, "results")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="memory ON vs OFF A/B eval")
    p.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge quality scoring")
    p.add_argument("--rounds", type=int, default=3, help="max research rounds per run")
    p.add_argument("--trials", type=int, default=1, help="trials per question per arm")
    p.add_argument("--questions", type=int, default=0, help="cap to first N curated questions (0=all)")
    p.add_argument("--page-size", type=int, default=6, help="results per search")
    p.add_argument("--recall-top-k", type=int, default=10, help="claims recalled per query")
    p.add_argument("--inject-char-budget", type=int, default=4000, help="max chars injected into a prompt")
    p.add_argument("--seed", type=int, default=0, help="seed for judge order randomization")
    return p.parse_args()


def run_one(
    arm: str,
    question: str,
    args: argparse.Namespace,
    client: SearchClient,
) -> Tuple[Dict[str, Any], str]:
    """Run one arm on one question; return (raw metrics-ready objects, report)."""
    mem = arms.build_memory(arm, base_dir=None)
    tracer = Tracer(dump_path=None)
    llm.reset_usage()
    pipe = DeepResearchPipeline(
        mem,
        client,
        biz_types=["paper"],
        page_size=args.page_size,
        recall_top_k=args.recall_top_k,
        inject_char_budget=args.inject_char_budget,
        verbose=False,
        tracer=tracer,
    )
    report = pipe.run(question, rounds=args.rounds)
    return {"pipe": pipe, "tracer": tracer, "llm_input_chars": llm.get_usage()}, report


def main() -> None:
    args = parse_args()
    load_dotenv()

    qs = questions_mod.select(args.questions)
    client = SearchClient()

    judge_on = not args.no_judge

    print("=" * 70)
    print("MEMORY A/B EVAL  (arm A = memory ON, arm B = memory OFF ablation)")
    print(
        f"  search   : {os.environ.get('SEARCH_API_BASE_URL', 'real API')}\n"
        f"  reasoning: LLM {os.environ.get('MEMORY_DR_MODEL', '')}\n"
        f"  judge    : {'on' if judge_on else 'off'}\n"
        f"  questions: {len(qs)} | trials: {args.trials} | rounds: {args.rounds}"
    )
    print("=" * 70)

    config = {
        "judge": judge_on,
        "rounds": args.rounds,
        "trials": args.trials,
        "questions": [q["id"] for q in qs],
        "recall_top_k": args.recall_top_k,
        "inject_char_budget": args.inject_char_budget,
        "model": os.environ.get("MEMORY_DR_MODEL", ""),
    }

    # Stable, per-run output paths created up front. The summary JSON is rewritten
    # after EVERY run and the .jsonl gets one appended line per run/judgment, so a
    # slow or interrupted sweep still leaves a current, readable report on disk.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")
    rows_path = os.path.join(RESULTS_DIR, f"eval_{ts}.jsonl")
    print(f"  incremental output (updated per run):\n    rows   : {rows_path}\n    summary: {out_path}\n")

    rng = random.Random(args.seed)
    q_by_id = {q["id"]: q["question"] for q in qs}
    results: List[Dict[str, Any]] = []
    judgments: List[Dict[str, Any]] = []

    for q in qs:
        qid, question = q["id"], q["question"]
        for trial in range(1, args.trials + 1):
            pair: Dict[str, str] = {}
            for arm in arms.ARMS:
                handles, report = run_one(arm, question, args, client)
                row = metrics_mod.collect(
                    arm, qid, trial, handles["pipe"], handles["tracer"],
                    report, handles["llm_input_chars"],
                )
                results.append(row)
                pair[arm] = report
                _append_jsonl(rows_path, {"kind": "run", **row})
                print(
                    f"  [{arm:>3}] {qid:<20} t{trial}: "
                    f"run={row['searches_run']:>2} skip={row['searches_skipped']:>2} "
                    f"claims={row['claims_kept']:>3} ctx={row['context_chars']:>5} "
                    f"src={row['sources_cited']:>2}"
                )
                _write_json(out_path, _build_summary(config, results, judgments))

            # Judge this pair as soon as both arms are in, so quality verdicts are
            # persisted incrementally too (not deferred to a final batch pass).
            if judge_on and "on" in pair and "off" in pair:
                verdict = judge_mod.judge_pair(q_by_id[qid], pair["on"], pair["off"], rng=rng)
                if verdict:
                    jrow = {"question_id": qid, "trial": trial, **verdict}
                    judgments.append(jrow)
                    _append_jsonl(rows_path, {"kind": "judge", **jrow})
                    print(
                        f"    judged {qid} t{trial}: "
                        + " ".join(f"{d}={verdict[d]}" for d in judge_mod.DIMENSIONS)
                    )
                    _write_json(out_path, _build_summary(config, results, judgments))

    summary = _build_summary(config, results, judgments)
    _write_json(out_path, summary)
    _print_summary(summary["by_arm"], summary["deltas_on_minus_off"], summary["judge"], out_path)


def _build_summary(
    config: Dict[str, Any],
    results: List[Dict[str, Any]],
    judgments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the aggregate summary from whatever has been collected so far."""
    by_arm = metrics_mod.aggregate_by_arm(results)
    return {
        "config": config,
        "by_arm": by_arm,
        "deltas_on_minus_off": metrics_mod.deltas(by_arm),
        "judge": judge_mod.aggregate(judgments) if judgments else {},
        "judgments": judgments,
        "results": results,
    }


def _write_json(path: str, obj: Any) -> None:
    """Atomically (re)write a JSON file (tmp + replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    """Append one crash-safe line to the per-run .jsonl log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(by_arm, delta, judge_tally, out_path) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY  (averaged across questions x trials)")
    print("=" * 70)
    if "on" in by_arm and "off" in by_arm:
        print(f"  {'metric':<18}{'ON':>10}{'OFF':>10}{'ON-OFF':>10}")
        for f in metrics_mod.NUMERIC_FIELDS:
            print(f"  {f:<18}{by_arm['on'][f]:>10}{by_arm['off'][f]:>10}{delta.get(f, 0):>10}")
        print(
            "\n  reading it: OFF re-runs duplicate searches (skip=0) and its recency\n"
            "  window drops earlier-round findings -> fewer sources_cited; ON avoids\n"
            "  repeat work and recalls the most relevant claims within the same budget."
        )
    else:
        print("  (need both arms to compare)")

    if judge_tally:
        print("\n  LLM-judge win counts (blind pairwise):")
        for dim, tally in judge_tally.items():
            print(f"    {dim:<14} ON={tally['on']}  OFF={tally['off']}  tie={tally['tie']}")
    else:
        print("\n  LLM-judge: skipped (--no-judge).")

    print(f"\n  full results: {out_path}")


if __name__ == "__main__":
    main()
