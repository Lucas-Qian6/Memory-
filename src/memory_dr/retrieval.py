"""Hybrid retrieval: keyword overlap + optional vector similarity + recency.

Lexical scoring is dependency-free so the demo runs anywhere. When embeddings
are available the caller passes a query vector + an ``{id: vector}`` map and the
relevance term blends cosine similarity with the lexical score; otherwise it is
pure lexical (the previous behavior, preserved).

Recency is **step-based** within a task: a single long-horizon task runs in
minutes, so wall-clock decay barely distinguishes items - the step ordinal does.
Items without a step fall back to wall-clock decay (kept for library/test use).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from .schema import MemoryItem, extract_keywords


def _recency_boost(created_at: float, half_life_hours: float = 72.0) -> float:
    """Exponential wall-clock decay in [0, 1]; fallback when no step is set."""
    age_hours = max(0.0, (time.time() - created_at) / 3600.0)
    return math.exp(-age_hours * math.log(2) / half_life_hours)


def _recency(item: MemoryItem, max_step: Optional[int]) -> float:
    """Step-based recency in (0, 1]; newer step -> higher. Wall-clock fallback."""
    if item.step is not None and max_step:
        return min(1.0, max(0.0, (item.step + 1) / (max_step + 1)))
    return _recency_boost(item.created_at)


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def score(
    query: str,
    item: MemoryItem,
    recency_weight: float = 0.25,
    query_vec: Optional[List[float]] = None,
    item_vec: Optional[List[float]] = None,
    vector_weight: float = 0.6,
    max_step: Optional[int] = None,
) -> float:
    q_terms = set(extract_keywords(query, limit=24))
    item_terms = set(item.keywords) | set(extract_keywords(item.content, limit=24))
    # Jaccard-ish relevance, normalized by query size so longer items aren't favored.
    lexical = len(q_terms & item_terms) / max(1, len(q_terms)) if q_terms else 0.0

    if query_vec is not None and item_vec is not None:
        relevance = vector_weight * _cosine(query_vec, item_vec) + (1 - vector_weight) * lexical
    else:
        relevance = lexical

    rec = _recency(item, max_step)
    # No usable query signal (no terms, no vector): rank by recency alone.
    if not q_terms and query_vec is None:
        return rec * recency_weight
    return (1 - recency_weight) * relevance + recency_weight * rec


def rank(
    query: str,
    items: List[MemoryItem],
    top_k: int = 5,
    min_score: float = 0.0,
    query_vec: Optional[List[float]] = None,
    vectors: Optional[Dict[str, List[float]]] = None,
) -> List[Tuple[MemoryItem, float]]:
    steps = [it.step for it in items if it.step is not None]
    max_step = max(steps) if steps else None
    scored: List[Tuple[MemoryItem, float]] = []
    for it in items:
        item_vec = vectors.get(it.id) if vectors else None
        scored.append(
            (it, score(query, it, query_vec=query_vec, item_vec=item_vec, max_step=max_step))
        )
    scored = [pair for pair in scored if pair[1] > min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
