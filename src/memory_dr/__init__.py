"""memory_dr: a small, pluggable in-task memory subsystem for a DeepResearch agent.

Scoped to one long-horizon task (one user query, many steps). The three memory
facets (working / episodic / semantic) are a single ``type`` tag over one
``MemoryStore``; they differ only by write/read policy.

Public surface is intentionally tiny: build a ``MemoryManager`` and call
``remember`` / ``recall`` / ``compress`` / ``update_state`` / ``get_state``.
"""

from .schema import MemoryItem, MemoryType
from .store import MemoryStore
from .manager import MemoryManager
from .policies import ReadPolicy, WritePolicy

__all__ = [
    "MemoryItem",
    "MemoryType",
    "MemoryStore",
    "MemoryManager",
    "ReadPolicy",
    "WritePolicy",
]
