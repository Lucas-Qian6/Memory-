"""MemoryStore: a single in-task store with per-facet write-through sinks.

The store keeps an in-memory ``{id: MemoryItem}`` index (the thing ``query`` /
``all`` / ``seen_action`` read) and delegates *persistence* to one sink per
facet (see ``backends.py``): working -> ``state.md``, episodic ->
``episodic.jsonl``, semantic -> ``semantic.json`` + ``artifacts/`` + an
embedding sidecar. Pass ``base_dir`` to persist to disk for inspection; pass
``None`` to stay purely in-memory.

The public surface (``add/get/delete/all/query/clear/__len__``) is unchanged, so
``MemoryManager``, the policies, the tracer, and the demo keep working; swapping
storage is a backend change, not an API change.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from .backends import EpisodicSink, SemanticSink, WorkingSink
from .schema import MemoryItem, MemoryType, _hash_content


def _source_entry(item: MemoryItem) -> Dict[str, object]:
    """A compact per-source citation record for the merged ``links['sources']`` union."""
    links = item.links or {}
    return {
        "source": item.source,
        "title": links.get("title"),
        "year": links.get("year"),
        "doi": links.get("doi"),
        "url": links.get("url"),
        "docId": links.get("docId"),
        "evidence": item.evidence or links.get("evidence"),
    }


class MemoryStore:
    def __init__(self, base_dir: Optional[str] = None) -> None:
        # base_dir=None keeps the store purely in-memory (library / test default).
        # A directory enables per-facet write-through for inspection/debugging.
        self.base_dir = base_dir
        self._items: Dict[str, MemoryItem] = {}
        self._lock = threading.RLock()
        self.working_sink = WorkingSink(base_dir)
        self.episodic_sink = EpisodicSink(base_dir)
        self.semantic_sink = SemanticSink(base_dir)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

    # --- CRUD ----------------------------------------------------------
    def add(self, item: MemoryItem) -> MemoryItem:
        with self._lock:
            self._items[item.id] = item
            if item.type == "working":
                self.working_sink.write_latest(item)
            elif item.type == "episodic":
                self.episodic_sink.append(item)
            elif item.type == "semantic":
                self.semantic_sink.add(item)
        return item

    def update(self, item: MemoryItem) -> MemoryItem:
        """Persist an in-place mutation of an existing item (id unchanged).

        Used by the consistency/merge path after editing ``status`` / ``links`` /
        content. Semantic items are re-embedded + re-flushed; working rewrites its
        snapshot; episodic is an append-only log and is intentionally left as-is.
        """
        with self._lock:
            self._items[item.id] = item
            if item.type == "semantic":
                self.semantic_sink.add(item)
            elif item.type == "working":
                self.working_sink.write_latest(item)
        return item

    def merge_into(
        self,
        canonical: MemoryItem,
        dup: MemoryItem,
        canonical_text: Optional[str] = None,
    ) -> MemoryItem:
        """Fold near-duplicate ``dup`` into ``canonical`` (multi-source union).

        Keeps ``canonical.id`` stable so existing references survive. Unions source
        attribution into ``links['sources']`` so the merged claim can cite every
        contributing source, advances ``step`` to the newer of the two, and records
        a ``merged_from`` provenance edge. ``canonical_text`` (e.g. an LLM-merged
        sentence) replaces the content when given; otherwise the richer (longer) of
        the two texts is kept.
        """
        with self._lock:
            if canonical_text and canonical_text.strip():
                canonical.content = canonical_text.strip()
                canonical.content_hash = _hash_content(canonical.content)
            elif len(dup.content.strip()) > len(canonical.content.strip()):
                canonical.content = dup.content.strip()
                canonical.content_hash = _hash_content(canonical.content)

            # Union source attribution (seed from canonical's own citation payload).
            sources = canonical.links.get("sources")
            if not isinstance(sources, list):
                sources = [_source_entry(canonical)] if canonical.source else []
            seen = {s.get("source") for s in sources if isinstance(s, dict)}
            dup_entry = _source_entry(dup)
            if dup_entry.get("source") and dup_entry["source"] not in seen:
                sources.append(dup_entry)
            if sources:
                canonical.links["sources"] = sources

            # Recency: keep the newer step ordinal.
            if dup.step is not None:
                canonical.step = max(canonical.step or 0, dup.step)

            # Provenance edge to the folded-in duplicate.
            canonical.add_relation("merged_from", dup.id)

            self.update(canonical)
        return canonical

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            item = self._items.pop(item_id, None)
            if item is None:
                return False
            # Only semantic has mutable on-disk state; working/episodic are
            # append-only logs whose history we intentionally keep.
            if item.type == "semantic":
                self.semantic_sink.delete(item_id)
            return True

    def all(self) -> List[MemoryItem]:
        return list(self._items.values())

    def query(
        self,
        type: Optional[MemoryType] = None,
        task_id: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[MemoryItem]:
        """Filter by structured fields (no ranking; see retrieval.py for that)."""
        out = self._items.values()
        if type is not None:
            out = [it for it in out if it.type == type]
        if task_id is not None:
            out = [it for it in out if it.task_id == task_id]
        if tag is not None:
            out = [it for it in out if tag in it.tags]
        return list(out)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.working_sink.clear()
            self.episodic_sink.clear()
            self.semantic_sink.clear()

    # --- semantic recall helpers --------------------------------------
    def semantic_vectors(self) -> Dict[str, List[float]]:
        """The ``{id: embedding}`` sidecar used by hybrid vector recall."""
        return self.semantic_sink.vectors

    def write_artifact(self, source: Optional[str], content: str) -> Optional[str]:
        """Archive raw source text once under artifacts/; return a relative uri."""
        return self.semantic_sink.write_artifact(source, content)

    def __len__(self) -> int:
        return len(self._items)
