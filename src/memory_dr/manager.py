"""MemoryManager: the only surface a pipeline should touch.

Scoped to one long-horizon task. Keeping all access behind this class is what
makes the "single store now, per-type backends later" migration a localized
change. The pipeline never queries storage directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .policies import Judge, ReadPolicy, WritePolicy
from .schema import MemoryItem, MemoryType
from .store import MemoryStore


class MemoryManager:
    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        write_policy: Optional[WritePolicy] = None,
        read_policy: Optional[ReadPolicy] = None,
        task_id: Optional[str] = None,
        judge: Optional[Judge] = None,
    ) -> None:
        # NOTE: use explicit None checks. MemoryStore defines __len__, so an
        # empty (len 0) store is falsy; `store or MemoryStore()` would silently
        # discard a freshly-created empty store and fall back to the default path.
        self.store = store if store is not None else MemoryStore()
        self.write_policy = write_policy if write_policy is not None else WritePolicy()
        self.read_policy = read_policy if read_policy is not None else ReadPolicy()
        self.task_id = task_id
        # LLM relation-judge for near-duplicate / contradiction decisions. It is
        # required the moment the write policy finds a near-duplicate candidate
        # (it raises if missing); recall / working / episodic never need it.
        self.judge = judge
        # Introspection for tracing/metrics: the op of the most recent remember()
        # ("new" / "exact_dup" / "merge" / "supersede" / "conflict") + its target,
        # plus cumulative per-op counts (used by the eval to report P2/P3 activity).
        self.last_write: Optional[Dict[str, Any]] = None
        self.write_counts: Dict[str, int] = {}

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
        """Store an item, applying selectivity, dedup, near-duplicate merge and
        contradiction handling. Returns the canonical/stored item, or None when it
        is filtered out (too short / low value).

        For the semantic facet a near-duplicate of an existing claim is resolved by
        the LLM ``judge`` into one of: ``merge`` (fold in + union sources),
        ``supersede`` (retire the older, contradicting claim), or ``conflict``
        (keep both, cross-link as disputed). Nothing is deleted - retired items
        keep their evidence/artifacts and are simply excluded from recall.

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

        op, existing, verdict = self.write_policy.classify(item, self.store, judge=self.judge)
        target_id = existing.id if existing is not None else None

        if op == "exact_dup" and existing is not None:
            result: Optional[MemoryItem] = existing
        elif op == "merge" and existing is not None:
            result = self.store.merge_into(
                existing, item, canonical_text=verdict.get("canonical")
            )
            # Keep the folded-in duplicate on disk (status=merged) for provenance.
            item.mark_merged(result.id)
            self.store.add(item)
        elif op == "supersede" and existing is not None:
            if str(verdict.get("winner", "new")).lower() == "existing":
                # The incoming claim is the outdated one: store it, but retired.
                item.mark_superseded(existing.id)
                existing.add_relation("supersedes", item.id)
            else:
                # Default: the new claim wins and retires the existing one.
                item.add_relation("supersedes", existing.id)
                existing.mark_superseded(item.id)
            self.store.update(existing)
            result = self.store.add(item)
        elif op == "conflict" and existing is not None:
            item.add_relation("conflicts_with", existing.id)
            existing.add_relation("conflicts_with", item.id)
            self.store.update(existing)
            result = self.store.add(item)
        else:
            op = "new"
            result = self.store.add(item)

        self.last_write = {
            "op": op,
            "target_id": target_id,
            "id": result.id if result is not None else None,
        }
        self.write_counts[op] = self.write_counts.get(op, 0) + 1
        return result

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
