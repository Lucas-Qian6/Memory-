"""Per-facet persistence sinks - the disk side of the single-task store.

Each facet is persisted the way it is *retrieved*, not in one generic blob:

- ``WorkingSink``   -> a single ``working/state.md`` (latest snapshot only).
- ``EpisodicSink``  -> an append-only ``episodic.jsonl`` action log.
- ``SemanticSink``  -> structured ``semantic.json`` records + raw source text
                       under ``artifacts/`` + an in-memory ``{id: embedding}``
                       sidecar that powers vector recall.

Writes are **inline** (synchronous write-through on each ``add``/``delete``),
matching the single-task scope: there is no serving latency to defer writes
for, and inline writes keep the web-UI read->decide->write trace coherent. When
``base_dir`` is ``None`` the sinks skip all disk I/O (pure in-memory), but the
semantic sink still computes embeddings so vector recall works in-memory too.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from . import embeddings
from .schema import MemoryItem


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _safe_name(name: str) -> str:
    """Filesystem-safe basename for an artifact (docIds are numeric, but be safe)."""
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(name)]
    out = "".join(keep).strip("._") or "source"
    return out[:120]


class WorkingSink:
    """Latest task-state snapshot, rendered as a human-readable markdown file."""

    def __init__(self, base_dir: Optional[str]) -> None:
        self.path = os.path.join(base_dir, "working", "state.md") if base_dir else None

    def write_latest(self, item: MemoryItem) -> None:
        if not self.path:
            return
        step = item.step if item.step is not None else "-"
        body = (
            "# Working memory - current task state\n\n"
            f"- id: `{item.id}`\n"
            f"- step: {step}\n"
            f"- updated_at: {item.created_at}\n\n"
            f"{item.content}\n"
        )
        _atomic_write(self.path, body)

    def clear(self) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


class EpisodicSink:
    """Append-only JSONL log of actions taken (the audit trail behind dedup)."""

    def __init__(self, base_dir: Optional[str]) -> None:
        self.path = os.path.join(base_dir, "episodic.jsonl") if base_dir else None

    def append(self, item: MemoryItem) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def clear(self) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


class SemanticSink:
    """Structured claim records + raw source artifacts + an embedding sidecar."""

    def __init__(self, base_dir: Optional[str]) -> None:
        self.json_path = os.path.join(base_dir, "semantic.json") if base_dir else None
        self.artifacts_dir = os.path.join(base_dir, "artifacts") if base_dir else None
        self.vectors: Dict[str, List[float]] = {}
        self._records: Dict[str, dict] = {}

    def add(self, item: MemoryItem) -> None:
        vec = embeddings.encode(item.content)
        if vec is not None:
            self.vectors[item.id] = vec  # type: ignore[assignment]
        self._records[item.id] = item.to_dict()
        self._flush_json()

    def delete(self, item_id: str) -> None:
        self.vectors.pop(item_id, None)
        if self._records.pop(item_id, None) is not None:
            self._flush_json()

    def write_artifact(self, source: Optional[str], content: str) -> Optional[str]:
        """Persist raw source text once (off the live context); return a relative uri."""
        if not self.artifacts_dir or not source:
            return None
        fname = _safe_name(source) + ".md"
        path = os.path.join(self.artifacts_dir, fname)
        if not os.path.exists(path):
            _atomic_write(path, content or "")
        return os.path.join("artifacts", fname)

    def clear(self) -> None:
        self.vectors.clear()
        self._records.clear()
        if self.json_path and os.path.exists(self.json_path):
            os.remove(self.json_path)

    def _flush_json(self) -> None:
        if not self.json_path:
            return
        _atomic_write(
            self.json_path,
            json.dumps({"items": list(self._records.values())}, ensure_ascii=False, indent=2),
        )
