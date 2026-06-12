"""MemoryManager: the only surface a pipeline should touch.

Scoped to one long-horizon task. Keeping all access behind this class is what
makes the "single store now, per-type backends later" migration a localized
change. The pipeline never queries storage directly.
"""

from __future__ import annotations

from typing import List, Optional

from .policies import ReadPolicy, WritePolicy
from .schema import MemoryItem, MemoryType
from .store import MemoryStore


class MemoryManager:
    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        write_policy: Optional[WritePolicy] = None,
        read_policy: Optional[ReadPolicy] = None,
        task_id: Optional[str] = None,
    ) -> None:
        # NOTE: use explicit None checks. MemoryStore defines __len__, so an
        # empty (len 0) store is falsy; `store or MemoryStore()` would silently
        # discard a freshly-created empty store and fall back to the default path.
        self.store = store if store is not None else MemoryStore()
        self.write_policy = write_policy if write_policy is not None else WritePolicy()
        self.read_policy = read_policy if read_policy is not None else ReadPolicy()
        self.task_id = task_id

    # --- write ---------------------------------------------------------
    def remember(
        self,
        content: str,
        type: MemoryType = "semantic",
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        links: Optional[dict] = None,
        evidence: Optional[str] = None,
        uri: Optional[str] = None,
        step: Optional[int] = None,
    ) -> Optional[MemoryItem]:
        """Store an item, applying selectivity + dedup. Returns the stored
        (or pre-existing duplicate) item, or None if filtered out.

        ``evidence`` (supporting span), ``uri`` (pointer to archived raw source),
        and ``step`` (task step ordinal, drives recency) are optional first-class
        fields used by the semantic facet.
        """
        item = MemoryItem(
            content=content,
            type=type,
            source=source,
            tags=tags or [],
            links=links or {},
            evidence=evidence,
            uri=uri,
            step=step,
            task_id=self.task_id,
        )
        if not self.write_policy.should_store(item):
            return None
        dup = self.write_policy.is_duplicate(item, self.store)
        if dup is not None:
            return dup
        return self.store.add(item)

    # --- read ----------------------------------------------------------
    def recall(
        self,
        query: str,
        type: Optional[MemoryType] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryItem]:
        return self.read_policy.recall(
            query,
            self.store,
            type=type,
            top_k=top_k,
        )

    def recall_context(self, query: str, type: Optional[MemoryType] = None) -> str:
        return self.read_policy.assemble_context(self.recall(query, type=type))

    # --- dedup helper for "have I done this work?" ---------------------
    def seen_action(self, action: str, threshold: float = 0.6) -> bool:
        """True if a similar episodic action was already recorded.

        Lets the pipeline skip redundant searches/reads.
        """
        from . import retrieval

        for it in self.store.query(type="episodic"):
            if retrieval.score(action, it, recency_weight=0.0) >= threshold:
                return True
        return False

    # --- compression ---------------------------------------------------
    def compress(self, query: str, type: MemoryType = "semantic", top_k: int = 10) -> str:
        """Assemble a compact summary block of the most relevant memories.

        In the LLM-backed pipeline this feeds the synthesis step; here it is a
        deterministic concatenation so the demo runs offline.
        """
        items = self.recall(query, type=type, top_k=top_k)
        return self.read_policy.assemble_context(items, max_chars=4000)

    # --- task state: keep the thread across many steps -----------------
    def update_state(self, state: str, step: Optional[int] = None) -> MemoryItem:
        """Record the current task state as working memory.

        Lets the agent keep its goal / plan / outline / draft out of the live
        context and re-read the latest snapshot at any step of the long task.
        """
        return self.store.add(
            MemoryItem(
                content=state,
                type="working",
                source="state",
                step=step,
                task_id=self.task_id,
            )
        )

    # --- raw source archival: keep originals off the live context ------
    def archive(self, source: Optional[str], content: str) -> Optional[str]:
        """Persist raw source text under artifacts/ (once per source).

        Returns a relative uri to attach to the distilled claims, or None when
        the store is purely in-memory. The raw text never enters live context.
        """
        if hasattr(self.store, "write_artifact"):
            return self.store.write_artifact(source, content)
        return None

    def get_state(self) -> Optional[MemoryItem]:
        """Return the latest task-state snapshot, if any."""
        states = [
            it
            for it in self.store.query(type="working", task_id=self.task_id)
            if it.source == "state"
        ]
        if not states:
            return None
        return max(states, key=lambda it: it.created_at)

    # --- stats (handy for the demo) ------------------------------------
    def stats(self) -> dict:
        counts = {"working": 0, "episodic": 0, "semantic": 0}
        for it in self.store.all():
            counts[it.type] = counts.get(it.type, 0) + 1
        counts["total"] = len(self.store)
        return counts
