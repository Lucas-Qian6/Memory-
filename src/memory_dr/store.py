"""MemoryStore: a single in-memory store, scoped to one task.

Within one long-horizon task the store lives in process memory (no cross-session
persistence). Pass a ``path`` only if you want an optional write-through JSON
dump for inspection/debugging. The ``type`` field is just a filterable
namespace, so a future "split into per-type backends" change is a backend swap.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from .schema import MemoryItem, MemoryType


class MemoryStore:
    def __init__(self, path: Optional[str] = None) -> None:
        # path=None keeps the store purely in-memory (the single-task default).
        # A path enables an optional write-through JSON dump for inspection.
        self.path = path
        self._items: Dict[str, MemoryItem] = {}
        self._lock = threading.RLock()
        self._load()

    # --- optional inspection persistence -------------------------------
    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        for d in raw.get("items", []):
            item = MemoryItem.from_dict(d)
            self._items[item.id] = item

    def _flush(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(
                {"items": [it.to_dict() for it in self._items.values()]},
                fh,
                ensure_ascii=False,
                indent=2,
            )
        os.replace(tmp, self.path)

    # --- CRUD ----------------------------------------------------------
    def add(self, item: MemoryItem) -> MemoryItem:
        with self._lock:
            self._items[item.id] = item
            self._flush()
        return item

    def get(self, item_id: str) -> Optional[MemoryItem]:
        return self._items.get(item_id)

    def delete(self, item_id: str) -> bool:
        with self._lock:
            existed = self._items.pop(item_id, None) is not None
            if existed:
                self._flush()
        return existed

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
            self._flush()

    def __len__(self) -> int:
        return len(self._items)
