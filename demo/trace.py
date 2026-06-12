"""Tracing layer for the memory inspector (zero changes to `src/memory_dr`).

Two pieces:
- ``Tracer`` - a thread-safe, ordered event buffer. The pipeline sets the current
  round/step context; reads/writes/decisions are appended as events the web UI
  polls. Also dumps ``demo/.demo_trace.json`` for offline inspection.
- ``TracingMemoryManager`` - a thin subclass of ``MemoryManager`` that emits an
  event on every read/write, then delegates to ``super()``. Because the whole
  codebase only touches memory through ``MemoryManager``, subclassing captures
  every operation faithfully without editing the core.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory_dr import MemoryItem, MemoryManager  # noqa: E402
from memory_dr.schema import MemoryType  # noqa: E402


def _preview(text: Optional[str], n: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _item_brief(item: Optional[MemoryItem]) -> Optional[Dict[str, Any]]:
    if item is None:
        return None
    return {
        "id": item.id,
        "facet": item.type,
        "source": item.source,
        "preview": _preview(item.content),
        "tags": list(item.tags or []),
        "links_keys": list((item.links or {}).keys()),
    }


class Tracer:
    """Ordered, thread-safe event log with a notion of "current step"."""

    def __init__(self, dump_path: Optional[str] = None) -> None:
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._seq = 0
        self.round = 0
        self.step = 0
        self.sub_query: Optional[str] = None
        self.phase = "init"
        self.dump_path = dump_path

    # --- context the pipeline sets so events are attributable -----------
    def enter_round(self, r: int) -> None:
        with self._lock:
            self.round = r
            self.step = 0
            self.sub_query = None
            self.phase = "plan"
        self.record("info", {"event": "round_start", "round": r})

    def enter_step(self, step_idx: int, sub_query: str, phase: str = "search") -> None:
        with self._lock:
            self.step = step_idx
            self.sub_query = sub_query
            self.phase = phase

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    # --- the single append point ----------------------------------------
    def record(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "ts": time.time(),
                "round": self.round,
                "step": self.step,
                "sub_query": self.sub_query,
                "phase": self.phase,
                "kind": kind,
                "payload": payload,
            }
            self._events.append(event)
            self._dump_locked()
            return event

    # --- read side (for the HTTP polling endpoint) ----------------------
    def events_since(self, seq: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    def all_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def _dump_locked(self) -> None:
        if not self.dump_path:
            return
        try:
            tmp = self.dump_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"events": self._events}, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.dump_path)
        except OSError:
            pass


class TracingMemoryManager(MemoryManager):
    """``MemoryManager`` that emits a trace event on each read/write."""

    def __init__(self, *args: Any, tracer: Optional[Tracer] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tracer = tracer

    def _emit(self, kind: str, payload: Dict[str, Any]) -> None:
        if self.tracer is not None:
            self.tracer.record(kind, payload)

    # --- writes ---------------------------------------------------------
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
        before = len(self.store)
        item = super().remember(
            content,
            type=type,
            source=source,
            tags=tags,
            links=links,
            evidence=evidence,
            uri=uri,
            step=step,
        )
        stored_new = len(self.store) > before
        if item is None:
            self._emit(
                "write",
                {
                    "op": "remember",
                    "facet": type,
                    "source": source,
                    "preview": _preview(content),
                    "stored": False,
                    "reason": "filtered (too short / low value)",
                },
            )
        else:
            self._emit(
                "write",
                {
                    "op": "remember",
                    "facet": item.type,
                    "source": item.source,
                    "id": item.id,
                    "preview": _preview(item.content),
                    "tags": list(item.tags or []),
                    "links_keys": list((item.links or {}).keys()),
                    "stored": stored_new,
                    "dedup_hit": not stored_new,
                },
            )
        return item

    def update_state(self, state: str, step: Optional[int] = None) -> MemoryItem:
        item = super().update_state(state, step=step)
        self._emit(
            "write",
            {"op": "update_state", "facet": "working", "id": item.id, "preview": _preview(state), "stored": True},
        )
        return item

    # --- reads ----------------------------------------------------------
    def recall(
        self,
        query: str,
        type: Optional[MemoryType] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryItem]:
        items = super().recall(query, type=type, top_k=top_k)
        self._emit(
            "recall",
            {
                "op": "recall",
                "query": _preview(query, 120),
                "facet": type,
                "top_k": top_k,
                "count": len(items),
                "returned": [_item_brief(it) for it in items],
            },
        )
        return items

    def seen_action(self, action: str, threshold: float = 0.6) -> bool:
        hit = super().seen_action(action, threshold=threshold)
        self._emit(
            "recall",
            {"op": "seen_action", "query": _preview(action, 120), "threshold": threshold, "hit": hit},
        )
        return hit

    def get_state(self) -> Optional[MemoryItem]:
        item = super().get_state()
        self._emit(
            "recall",
            {"op": "get_state", "hit": item is not None, "returned": [_item_brief(item)] if item else []},
        )
        return item


def snapshot(memory: MemoryManager) -> Dict[str, List[Dict[str, Any]]]:
    """Current memory contents grouped by facet, in insertion (created) order."""
    out: Dict[str, List[Dict[str, Any]]] = {"working": [], "episodic": [], "semantic": []}
    for it in memory.store.all():
        out.setdefault(it.type, []).append(
            {
                "id": it.id,
                "content": it.content,
                "source": it.source,
                "tags": list(it.tags or []),
                "links": it.links or {},
                "uri": it.uri,
                "created_at": it.created_at,
            }
        )
    for facet in out:
        out[facet].sort(key=lambda x: x["created_at"])
    return out
