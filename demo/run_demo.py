"""End-to-end demo: ONE long-horizon research task on the real search API.

A single user query kicks off a multi-step task. The agent plans sub-queries
with an LLM, hits the Wenyon search API, extracts claims, reflects, and writes a
cited synthesis - all while its in-task memory:

- skips queries it already ran this task (episodic dedup),
- keeps findings in memory so they survive context overflow (semantic),
- snapshots its goal/progress so it never loses the thread (working).

Run:
    python demo/run_demo.py                       # real API + LLM loop (needs config)
    python demo/run_demo.py "your question" --biz paper,scholar

A real search API and an LLM are required (no offline/mock mode).
Config (.env): SEARCH_API_BASE_URL/SEARCH_USER_ID/SEARCH_SCENE for the search
API; ANTHROPIC_API_KEY/ANTHROPIC_BASE_URL/MEMORY_DR_MODEL for the LLM loop.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import llm  # noqa: E402
from memory_dr import MemoryManager, MemoryStore  # noqa: E402
from pipeline import DeepResearchPipeline  # noqa: E402
from search_client import SearchClient  # noqa: E402

# Per-facet write-through directory so you can inspect the in-task store after a
# run: working/state.md, episodic.jsonl, semantic.json, and artifacts/.
STORE_DIR = os.path.join(os.path.dirname(__file__), ".demo_memory")

QUESTION = (
    "How does reaction context affect retrosynthesis prediction, and can "
    "LLM agents with memory help automate this research?"
)


def load_dotenv(path: str = None) -> None:
    """Minimal, dependency-free .env loader (existing env vars take precedence)."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=QUESTION, help="research question")
    parser.add_argument("--rounds", type=int, default=3, help="max research rounds")
    parser.add_argument("--biz", default="paper", help="comma-separated bizTypes (paper,scholar,patent)")
    parser.add_argument("--page-size", type=int, default=6, help="results per search")
    args = parser.parse_args()

    load_dotenv()  # pick up SEARCH_API_* and ANTHROPIC_* config

    # Fresh layered store for this single task; wipe the dir so each run starts
    # clean and reproducible (working/episodic/semantic + artifacts land here).
    if os.path.exists(STORE_DIR):
        shutil.rmtree(STORE_DIR)
    store = MemoryStore(base_dir=STORE_DIR)

    biz_types = [b.strip() for b in args.biz.split(",") if b.strip()]

    # Real search API + LLM are required (no offline/mock mode).
    client = SearchClient()

    banner("ONE LONG-HORIZON TASK  (single user query, many steps)")
    print(f"  question : {args.question}")
    print(f"  search   : real API ({os.environ.get('SEARCH_API_BASE_URL', '?')})")
    print(f"  reasoning: LLM loop ({os.environ.get('MEMORY_DR_MODEL', 'default model')})")
    print(f"  bizTypes : {biz_types}")

    mem = MemoryManager(store=store, task_id="task-alpha", judge=llm.judge_relation)
    pipe = DeepResearchPipeline(
        mem,
        client,
        biz_types=biz_types,
        page_size=args.page_size,
    )
    report = pipe.run(args.question, rounds=args.rounds)

    # --- in-task metrics -------------------------------------------------
    findings = mem.store.query(type="semantic")
    held_chars = sum(len(it.content) for it in findings)
    injected = mem.recall_context(args.question, type="semantic")
    state = mem.get_state()

    banner("WHAT MEMORY DID FOR THIS ONE TASK")
    print(
        f"  steps taken              : {pipe.steps}\n"
        f"  searches actually run    : {pipe.searches_run}\n"
        f"  searches skipped (dedup) : {pipe.searches_skipped}   <- repeat work avoided\n"
        f"  memory contents          : {mem.stats()}"
    )
    print(
        f"\n  findings kept in memory  : {len(findings)} claims ({held_chars} chars)\n"
        f"  injected into synthesis  : {len(injected)} chars (top-k)\n"
        f"    -> the rest stays in memory, off the live context window."
    )
    print("\n  latest working-memory snapshot (kept the thread across steps):")
    print("    " + (state.content.replace("\n", "\n    ") if state else "(none)"))

    print("\n--- Report ---")
    print(report)

    banner("TAKEAWAY")
    print(
        "Within ONE task the agent planned several rounds but ran each search only\n"
        "once, kept every finding in memory (so context can't overflow it away), and\n"
        "tracked its own goal/progress - all through the MemoryManager interface, with\n"
        "zero cross-session or long-term machinery."
    )
    print(f"\nInspect the raw in-task store under: {STORE_DIR}/")


if __name__ == "__main__":
    main()
