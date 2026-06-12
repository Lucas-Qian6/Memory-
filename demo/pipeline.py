"""A real, memory-augmented DeepResearch loop for ONE long-horizon task.

This replaces the earlier faked demo (staged `plan_round` + deterministic
synthesis). The loop is now LLM-driven over the real search API, but every
storage/recall decision still goes through the same `MemoryManager` surface:

    plan (recall) -> dedup (episodic) -> search -> remember episodic + semantic
                  -> snapshot (working) -> reflect -> synthesize (recall+compress)

The three facets each earn their keep on real data:
- working  : goal/plan + per-round progress snapshots (kept out of the prompt).
- episodic : one record per search query, so re-planning never repeats work.
- semantic : one record per extracted claim, with source + citation links, so
             findings survive context overflow and feed a cited synthesis.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import llm  # noqa: E402
from memory_dr import MemoryManager  # noqa: E402
from search_client import Paper, SearchClient  # noqa: E402


class DeepResearchPipeline:
    """One long task, many steps, one in-task memory."""

    def __init__(
        self,
        memory: MemoryManager,
        search_client: SearchClient,
        use_llm: bool = True,
        biz_types=("paper",),
        page_size: int = 6,
        max_subqueries: int = 3,
        max_claims_per_paper: int = 3,
        recall_top_k: int = 10,
        inject_char_budget: int = 4000,
        verbose: bool = True,
        tracer=None,
    ) -> None:
        self.memory = memory
        self.search = search_client
        self.use_llm = use_llm
        self.biz_types = list(biz_types)
        self.page_size = page_size
        self.max_subqueries = max_subqueries
        self.max_claims_per_paper = max_claims_per_paper
        # Recall/injection budget: how many claims we pull and how many chars of
        # them we are willing to inject into a prompt (the rest stays in memory).
        self.recall_top_k = recall_top_k
        self.inject_char_budget = inject_char_budget
        self.verbose = verbose
        self.searches_run = 0
        self.searches_skipped = 0
        self.steps = 0
        self.asked: List[str] = []
        self.tracer = tracer

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _trace(self, kind: str, payload: dict) -> None:
        if self.tracer is not None:
            self.tracer.record(kind, payload)

    # --- memory mapping: SearchResponse paper -> semantic MemoryItem ----
    def _paper_tags(self, paper: Paper, sub_query: str) -> List[str]:
        # Prefer the API's own subject labels; fall back to the query topic.
        if paper.subjects:
            return [s.lower() for s in paper.subjects[:4]]
        return llm._keywords(sub_query, limit=4)

    def _paper_links(self, paper: Paper, evidence: str) -> Dict[str, object]:
        """The citation payload a report needs without re-fetching the paper."""
        return {
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "venue": paper.venue,
            "doi": paper.doi,
            "url": paper.url,
            "evidence": evidence,
            "citationCount": paper.citation_count,
            "docId": paper.doc_id,
        }

    def _paper_raw(self, paper: Paper) -> str:
        """Raw source text archived under artifacts/ (off the live context)."""
        authors = ", ".join(paper.authors) if paper.authors else "(unknown)"
        meta = [
            f"# {paper.title}",
            "",
            f"- docId: {paper.doc_id}",
            f"- authors: {authors}",
            f"- year: {paper.year}",
            f"- venue: {paper.venue}",
            f"- doi: {paper.doi}",
            f"- url: {paper.url}",
            f"- citationCount: {paper.citation_count}",
            "",
            "## Abstract",
            "",
            paper.abstract or "(no abstract)",
            "",
        ]
        return "\n".join(meta)

    # --- one research action (a single search query) -------------------
    def research_step(self, question: str, sub_query: str) -> None:
        self.steps += 1
        # Episodic memory answers "have I already run this query this task?"
        if self.memory.seen_action(sub_query):
            self.searches_skipped += 1
            self._trace(
                "decision",
                {"what": "skip", "sub_query": sub_query, "reason": "already searched (episodic dedup)"},
            )
            self._log(f"  [skip] already searched: {sub_query}")
            return

        self.searches_run += 1
        self._log(f"  [search] {sub_query}")
        try:
            results = self.search.search(
                sub_query, biz_types=self.biz_types, page_size=self.page_size
            )
        except Exception as e:  # network/parse failure: log it, keep going
            self._trace("decision", {"what": "search_error", "sub_query": sub_query, "error": str(e)})
            self._log(f"  [error] search failed: {e}")
            self.memory.remember(
                sub_query,
                type="episodic",
                source="planner",
                links={"error": str(e)},
                step=self.steps,
            )
            return

        self.asked.append(sub_query)
        self._trace(
            "decision",
            {
                "what": "search",
                "sub_query": sub_query,
                "hits": len(results.papers),
                "papers": [
                    {"docId": p.doc_id, "title": p.title, "year": p.year} for p in results.papers
                ],
            },
        )
        # Record the action (episodic) with a light result fingerprint.
        self.memory.remember(
            sub_query,
            type="episodic",
            source="planner",
            links={"bizTypes": self.biz_types, "hits": len(results.papers)},
            step=self.steps,
        )

        # Extract findings (semantic) from each paper, with source + evidence.
        kept = 0
        for paper in results.papers:
            # Archive the raw source once (off the live context); claims point to it.
            uri = self.memory.archive(paper.doc_id, self._paper_raw(paper))
            claims = llm.extract_claims(
                question, paper, max_claims=self.max_claims_per_paper, use_llm=self.use_llm
            )
            self._trace(
                "llm",
                {
                    "call": "extract_claims",
                    "paper": paper.title,
                    "docId": paper.doc_id,
                    "n": len(claims),
                    "claims": [c.get("text", "") for c in claims],
                },
            )
            for claim in claims:
                evidence = claim.get("evidence", "")
                stored = self.memory.remember(
                    claim["text"],
                    type="semantic",
                    source=paper.doc_id,
                    tags=self._paper_tags(paper, sub_query),
                    links=self._paper_links(paper, evidence),
                    evidence=evidence,
                    uri=uri,
                    step=self.steps,
                )
                if stored is not None:
                    kept += 1
        self._log(f"    -> {len(results.papers)} papers, {kept} new claims kept")

    # --- the long-horizon loop -----------------------------------------
    def run(self, question: str, rounds: int = 3) -> str:
        self._log(f"\n=== Research task: {question} ===")

        # Working memory: write the goal/plan down so it survives a long task.
        self.memory.update_state(
            f"goal: {question}\nplan: decompose -> search -> extract -> synthesize",
            step=self.steps,
        )

        for r in range(1, rounds + 1):
            if self.tracer is not None:
                self.tracer.enter_round(r)
            self._log(f"\n-- round {r}/{rounds} (plan -> search -> extract) --")

            # Recall feeds planning: latest state + what we already know.
            state_text = self._state_text()
            known_text = self._known_text(question)
            if self.use_llm and r > 1:
                # Reflection: propose only NEW gap-filling queries, or stop.
                method = "decide_followups"
                sub_queries = llm.decide_followups(
                    question, known_text, self.asked, n=self.max_subqueries, use_llm=True
                )
            else:
                method = "plan_subqueries"
                sub_queries = llm.plan_subqueries(
                    question, state_text, known_text, n=self.max_subqueries, use_llm=self.use_llm
                )

            sub_queries = [q for q in sub_queries if q and q.strip()]
            if not sub_queries:
                self._trace(
                    "decision",
                    {"what": "stop", "method": method, "reason": "planner produced no further queries"},
                )
                self._log("  [done] planner produced no further queries; stopping early")
                break
            self._trace("decision", {"what": "plan", "method": method, "sub_queries": sub_queries})
            self._log("  planned: " + "; ".join(sub_queries))

            for idx, sub_query in enumerate(sub_queries, 1):
                if self.tracer is not None:
                    self.tracer.enter_step(idx, sub_query)
                self.research_step(question, sub_query)

            # Keep the thread: snapshot progress into working memory.
            if self.tracer is not None:
                self.tracer.set_phase("snapshot")
            self.memory.update_state(
                f"goal: {question}\n"
                f"progress: round {r}/{rounds} done; "
                f"{self.searches_run} searches run, {self.searches_skipped} skipped; "
                f"{self.memory.stats()['semantic']} findings kept",
                step=self.steps,
            )

        return self.synthesize(question)

    # --- recall helpers -------------------------------------------------
    def _state_text(self) -> str:
        state = self.memory.get_state()
        return state.content if state else ""

    def _known_text(self, question: str, top_k: Optional[int] = None) -> str:
        items = self.memory.recall(
            question, type="semantic", top_k=top_k or self.recall_top_k
        )
        return self._evidence_context(items, max_chars=self.inject_char_budget)

    def _evidence_context(self, items, max_chars: int = 4000) -> str:
        """Compact, citation-bearing context block for planning / synthesis."""
        if not items:
            return ""
        lines: List[str] = []
        used = 0
        for it in items:
            links = it.links or {}
            cite = links.get("title") or it.source
            if links.get("year"):
                cite = f"{cite}, {links['year']}"
            line = f"- {it.content} (source: {it.source}; {cite})"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    # --- synthesis: findings come from memory, not a bloated context ----
    def synthesize(self, question: str, top_k: Optional[int] = None) -> str:
        if self.tracer is not None:
            self.tracer.set_phase("synthesize")
        self._log("\n-- synthesize (recall + compress semantic memory) --")
        evidence = self.memory.recall(
            question, type="semantic", top_k=top_k or self.recall_top_k
        )
        context = self._evidence_context(evidence, max_chars=self.inject_char_budget)

        report = llm.synthesize(question, context, use_llm=self.use_llm)
        used_llm = report is not None
        if not report:
            # Deterministic fallback (offline / no LLM).
            lines = [f"# Mini literature review: {question}", ""]
            for it in evidence:
                links = it.links or {}
                title = links.get("title", "")
                ev = links.get("evidence", "")
                lines.append(f"- {it.content}  (source: {it.source}; {title}; {ev})")
            lines.append("")
            lines.append(
                f"Synthesized from {len(evidence)} recalled claims across "
                f"{len({it.source for it in evidence})} sources."
            )
            report = "\n".join(lines)

        self._trace(
            "llm",
            {
                "call": "synthesize",
                "used_llm": used_llm,
                "used_claims": len(evidence),
                "sources": sorted({it.source for it in evidence if it.source}),
                "report_chars": len(report),
                "report": report,
            },
        )
        return report
