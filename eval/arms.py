"""The two eval arms: memory ON (the real system) vs memory OFF (an ablation).

The pipeline only ever touches memory through the ``MemoryManager`` surface, so
an A/B comparison is just "inject a different manager into the same pipeline."

``RawContextMemory`` simulates a deep-research agent with NO external memory -
only a finite context window. It keeps findings in a flat buffer and, on recall,
returns the most RECENT ones that fit (no dedup, no relevance ranking, no
vectors, no working-state thread). That reproduces the honest failure modes the
real memory fixes:

- no episodic dedup  -> it re-runs searches it already ran (wasted work),
- recency-only recall -> on a long task, earlier-round findings fall out of the
  window and never reach synthesis (lost evidence),
- no working snapshot -> no persistent goal/progress thread.

Everything else (planner, search, extraction, synthesis, inject budget) is
identical to the ON arm, so the delta isolates the memory system's value.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "demo"))

import llm  # noqa: E402
from memory_dr import MemoryManager, MemoryStore  # noqa: E402
from memory_dr.schema import MemoryItem, MemoryType  # noqa: E402


class RawContextMemory(MemoryManager):
    """Memory-OFF ablation: a finite recency context buffer, no memory machinery."""

    def __init__(self, task_id: Optional[str] = None) -> None:
        # Keep a real (empty, unused) store so inherited attributes stay valid;
        # actual "storage" is the flat buffer below.
        super().__init__(store=MemoryStore(), task_id=task_id)
        self._buffer: List[MemoryItem] = []

    # --- write: append-only, no dedup, no embedding --------------------
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
        self._buffer.append(item)
        return item

    # --- read: recency window only (no relevance, no vectors) ----------
    def recall(
        self,
        query: str,
        type: Optional[MemoryType] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryItem]:
        items = [it for it in self._buffer if type is None or it.type == type]
        # Newest first (by step, then insertion time): a finite context window.
        items.sort(
            key=lambda it: (it.step if it.step is not None else -1, it.created_at),
            reverse=True,
        )
        k = top_k if top_k is not None else self.read_policy.top_k
        return items[:k]

    # --- no dedup of work: every planned query gets run ----------------
    def seen_action(self, action: str, threshold: float = 0.6) -> bool:
        return False

    # --- no working-state thread ---------------------------------------
    def update_state(self, state: str, step: Optional[int] = None) -> MemoryItem:
        # Return a transient snapshot but do NOT persist it -> get_state() is empty.
        return MemoryItem(
            content=state, type="working", source="state", step=step, task_id=self.task_id
        )

    def get_state(self) -> Optional[MemoryItem]:
        return None

    # --- no raw-source archival ----------------------------------------
    def archive(self, source: Optional[str], content: str) -> Optional[str]:
        return None

    # --- stats over the buffer (so metrics still work) -----------------
    def stats(self) -> dict:
        counts = {"working": 0, "episodic": 0, "semantic": 0}
        for it in self._buffer:
            counts[it.type] = counts.get(it.type, 0) + 1
        counts["total"] = len(self._buffer)
        return counts


def build_memory(
    arm: str, base_dir: Optional[str] = None, task_id: str = "task-alpha"
) -> MemoryManager:
    """Factory: ``arm='on'`` -> real MemoryManager; ``arm='off'`` -> ablation."""
    if arm == "on":
        return MemoryManager(
            store=MemoryStore(base_dir=base_dir), task_id=task_id, judge=llm.judge_relation
        )
    if arm == "off":
        return RawContextMemory(task_id=task_id)
    raise ValueError(f"unknown arm {arm!r}; expected 'on' or 'off'")


ARMS = ("on", "off")
