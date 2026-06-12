"""Lightweight lexical retrieval: keyword overlap + recency.

Dependency-free on purpose so the demo runs anywhere. The scoring function is
the single seam where you'd later drop in embeddings / a vector DB.
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

from .schema import MemoryItem, extract_keywords


def _recency_boost(created_at: float, half_life_hours: float = 72.0) -> float:
    """Exponential recency decay in [0, 1]; newer memories score higher."""
    age_hours = max(0.0, (time.time() - created_at) / 3600.0)
    return math.exp(-age_hours * math.log(2) / half_life_hours)


def score(query: str, item: MemoryItem, recency_weight: float = 0.25) -> float:
    q_terms = set(extract_keywords(query, limit=24))
    if not q_terms:
        return _recency_boost(item.created_at) * recency_weight

    item_terms = set(item.keywords) | set(extract_keywords(item.content, limit=24))
    overlap = q_terms & item_terms
    # Jaccard-ish relevance, normalized by the query size so longer items
    # are not unfairly favored.
    relevance = len(overlap) / max(1, len(q_terms))

    return (1 - recency_weight) * relevance + recency_weight * _recency_boost(
        item.created_at
    )


def rank(
    query: str, items: List[MemoryItem], top_k: int = 5, min_score: float = 0.0
) -> List[Tuple[MemoryItem, float]]:
    scored = [(it, score(query, it)) for it in items]
    scored = [pair for pair in scored if pair[1] > min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
