"""Write and read policies.

This module is where the three in-task memory "facets" actually differ. The
store treats ``type`` as a plain field; the *behavior* (what is worth writing,
how to dedup, how to retrieve and assemble) lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import embeddings, retrieval
from .schema import MemoryItem, MemoryType
from .store import MemoryStore


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class WritePolicy:
    """Decides whether/how an item enters the store (selective write + dedup)."""

    dedup_threshold: float = 0.85

    def should_store(self, item: MemoryItem) -> bool:
        # Selectivity: drop trivially short content. Episodic actions are always
        # kept (they are the audit log that powers dedup of *work*).
        if item.type == "episodic":
            return True
        return len(item.content.strip()) >= 8

    def is_duplicate(self, item: MemoryItem, store: MemoryStore) -> Optional[MemoryItem]:
        """Return an existing near-duplicate of the same type, if any."""
        for existing in store.query(type=item.type):
            # Cheapest first: exact match via normalized content hash.
            if item.content_hash and existing.content_hash == item.content_hash:
                return existing
            if item.source and existing.source == item.source and (
                existing.content.strip() == item.content.strip()
            ):
                return existing
            if _jaccard(item.keywords, existing.keywords) >= self.dedup_threshold:
                return existing
        return None


@dataclass
class ReadPolicy:
    """Decides what gets recalled and how it is assembled into context."""

    top_k: int = 5
    min_score: float = 0.05

    def recall(
        self,
        query: str,
        store: MemoryStore,
        type: Optional[MemoryType] = None,
        task_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryItem]:
        candidates = store.query(type=type, task_id=task_id) if (
            type or task_id
        ) else store.all()
        # Semantic recall goes hybrid when embeddings are available: compute the
        # query vector once and hand the {id: vector} sidecar to the ranker.
        # Everything else (episodic/working) stays lexical.
        query_vec = None
        vectors = None
        if type == "semantic":
            query_vec = embeddings.encode(query)
            if query_vec is not None and hasattr(store, "semantic_vectors"):
                vectors = store.semantic_vectors()
        ranked = retrieval.rank(
            query,
            candidates,
            top_k=top_k or self.top_k,
            min_score=self.min_score,
            query_vec=query_vec,
            vectors=vectors,
        )
        return [item for item, _ in ranked]

    def assemble_context(self, items: List[MemoryItem], max_chars: int = 2000) -> str:
        """Turn recalled items into a compact, prompt-injectable block."""
        if not items:
            return "(no relevant memory)"
        lines: List[str] = []
        used = 0
        for it in items:
            src = f" [src: {it.source}]" if it.source else ""
            line = f"- ({it.type}) {it.content}{src}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)
