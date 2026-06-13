"""Curated research questions for the memory A/B eval.

These are intentionally multi-part / broad so the loop runs several rounds and
accumulates enough findings that the memory-OFF arm's finite recency window
starts dropping earlier-round evidence (which is exactly what memory-ON fixes).

The topics span retrosynthesis, reaction context, Graph2Edits, LLM agents for
science, agent memory, and RAG - broad enough to produce real hits against the
live search API.
"""

from __future__ import annotations

from typing import Dict, List

QUESTIONS: List[Dict[str, str]] = [
    {
        "id": "retro_context",
        "question": (
            "How does reaction context affect retrosynthesis prediction, and can "
            "LLM agents with memory help automate this research?"
        ),
    },
    {
        "id": "retro_graph",
        "question": (
            "What graph-based methods exist for single-step retrosynthesis, and how "
            "do they compare to template-based baselines on benchmarks like USPTO-50K?"
        ),
    },
    {
        "id": "llm_agents_science",
        "question": (
            "How are LLM agents applied to scientific discovery, and what limitations "
            "or open challenges do surveys report about their effectiveness?"
        ),
    },
    {
        "id": "agent_memory",
        "question": (
            "What role does external memory play for long-horizon LLM agents, and how "
            "does structured memory reduce redundant tool calls and context overflow?"
        ),
    },
    {
        "id": "rag_quality",
        "question": (
            "How does retrieval-augmented generation reduce hallucination, and which "
            "factors (retrieval quality, chunking, re-ranking) most affect answer accuracy?"
        ),
    },
    {
        "id": "retro_tradeoffs",
        "question": (
            "Compare approaches to retrosynthesis prediction and the role of reaction "
            "context, citing reported accuracy improvements and their trade-offs."
        ),
    },
    {
        "id": "memory_for_research",
        "question": (
            "What are the trade-offs of memory-augmented agents for multi-step research "
            "tasks, in terms of evidence coverage, repeated work, and context usage?"
        ),
    },
    {
        # Contradiction probe (P3): explicitly invites competing/disagreeing claims
        # about the same metric across sources, so the consistency path
        # (merge / supersede / conflict) is exercised.
        "id": "retro_disputed",
        "question": (
            "Do graph-based retrosynthesis models actually outperform template-based "
            "methods on USPTO-50K top-1 accuracy? Surface any disagreement between "
            "sources on the reported numbers and on which approach wins."
        ),
    },
]


def select(limit: int = 0) -> List[Dict[str, str]]:
    """Return the question set, optionally capped to the first ``limit`` items."""
    if limit and limit > 0:
        return QUESTIONS[:limit]
    return QUESTIONS
