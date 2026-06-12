"""A/B eval harness: run the SAME DeepResearch pipeline with memory ON vs OFF.

For every curated question x trial we run both arms (only the injected
MemoryManager differs), collect process metrics from the pipeline + tracer, and
- when an LLM is available - score report quality with a blind pairwise judge.
Aggregates ON-vs-OFF deltas + judge win-rates, prints a summary, and dumps the
full result to ``eval/results/<timestamp>.json``.

Run:
    python eval/run_eval.py --mock --no-llm                 # deterministic, process metrics only
    python eval/run_eval.py --mock --rounds 5 --trials 2    # mock corpus + LLM loop + judge
    python eval/run_eval.py --rounds 6 --trials 3           # real search API (needs SEARCH_API_* in .env)

Notes:
- Process metrics (searches saved, context chars, sources, claims) are
  deterministic under ``--mock --no-llm``. Quality (the judge) needs the LLM.
- The offline mock corpus is tiny (5 papers), so coverage/quality deltas are
  small; run against the real API for a convincing quality comparison.
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


def make_client(want_mock: bool) -> Tuple[SearchClient, bool]:
    """Build the search client, falling back to mock if the real API isn't set."""
    try:
        return SearchClient(mock=want_mock), want_mock
    except ValueError as e:
        print(f"[config] {e}\n[config] falling back to offline --mock search.")
        return SearchClient(mock=True), True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="memory ON vs OFF A/B eval")
    p.add_argument("--mock", action="store_true", help="use offline mock search")
    p.add_argument("--no-llm", action="store_true", help="deterministic loop (process metrics only)")
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
        use_llm=not args.no_llm,
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

    use_llm = not args.no_llm
    qs = questions_mod.select(args.questions)
    client, eff_mock = make_client(args.mock)

    llm_ready = use_llm and llm.llm_available()
    judge_on = llm_ready and not args.no_judge

    print("=" * 70)
    print("MEMORY A/B EVAL  (arm A = memory ON, arm B = memory OFF ablation)")
    print(
        f"  search   : {'mock (offline)' if eff_mock else os.environ.get('SEARCH_API_BASE_URL', 'real API')}\n"
        f"  reasoning: {'LLM ' + os.environ.get('MEMORY_DR_MODEL', '') if llm_ready else 'deterministic (no LLM)'}\n"
        f"  judge    : {'on' if judge_on else 'off'}\n"
        f"  questions: {len(qs)} | trials: {args.trials} | rounds: {args.rounds}"
    )
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    reports: Dict[Tuple[str, int], Dict[str, str]] = {}

    for q in qs:
        qid, question = q["id"], q["question"]
        for trial in range(1, args.trials + 1):
            reports.setdefault((qid, trial), {})
            for arm in arms.ARMS:
                handles, report = run_one(arm, question, args, client)
                row = metrics_mod.collect(
                    arm, qid, trial, handles["pipe"], handles["tracer"],
                    report, handles["llm_input_chars"],
                )
                results.append(row)
                reports[(qid, trial)][arm] = report
                print(
                    f"  [{arm:>3}] {qid:<20} t{trial}: "
                    f"run={row['searches_run']:>2} skip={row['searches_skipped']:>2} "
                    f"claims={row['claims_kept']:>3} ctx={row['context_chars']:>5} "
                    f"src={row['sources_cited']:>2}"
                )

    # --- quality: blind pairwise judge ---------------------------------
    judgments: List[Dict[str, Any]] = []
    if judge_on:
        rng = random.Random(args.seed)
        q_by_id = {q["id"]: q["question"] for q in qs}
        for (qid, trial), pair in reports.items():
            if "on" in pair and "off" in pair:
                verdict = judge_mod.judge_pair(q_by_id[qid], pair["on"], pair["off"], rng=rng)
                if verdict:
                    judgments.append({"question_id": qid, "trial": trial, **verdict})

    by_arm = metrics_mod.aggregate_by_arm(results)
    delta = metrics_mod.deltas(by_arm)
    judge_tally = judge_mod.aggregate(judgments) if judgments else {}

    summary = {
        "config": {
            "mock": eff_mock,
            "use_llm": use_llm,
            "llm_available": llm_ready,
            "judge": judge_on,
            "rounds": args.rounds,
            "trials": args.trials,
            "questions": [q["id"] for q in qs],
            "recall_top_k": args.recall_top_k,
            "inject_char_budget": args.inject_char_budget,
            "model": os.environ.get("MEMORY_DR_MODEL", "") if llm_ready else "",
        },
        "by_arm": by_arm,
        "deltas_on_minus_off": delta,
        "judge": judge_tally,
        "judgments": judgments,
        "results": results,
    }

    out_path = _write_results(summary)
    _print_summary(by_arm, delta, judge_tally, out_path)


def _write_results(summary: Dict[str, Any]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)
    return out_path


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
        print("\n  LLM-judge: skipped (no LLM / --no-judge / --no-llm).")

    print(f"\n  full results: {out_path}")


if __name__ == "__main__":
    main()
