"""Memory item schema.

One item type for all three in-task memory facets. The ``type`` field is what
makes the split a tag over a single store rather than three separate systems.
All memory here is scoped to a single long-horizon task (one user query):

- ``working``  - current task state (goal, sub-questions, plan/outline, draft)
- ``episodic`` - actions already taken (searches/reads), to avoid repeat work
- ``semantic`` - findings (claims + source/evidence), to survive context overflow
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional

MemoryType = Literal["working", "episodic", "semantic"]

VALID_TYPES = ("working", "episodic", "semantic")


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class MemoryItem:
    """A single unit of memory within one task.

    Granularity depends on ``type``: a ``semantic`` item is typically one claim,
    an ``episodic`` item is one action (a search, a paper read), and a
    ``working`` item is a piece of current task state.
    """

    content: str
    type: MemoryType = "semantic"
    source: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    links: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.type not in VALID_TYPES:
            raise ValueError(
                f"invalid memory type {self.type!r}; expected one of {VALID_TYPES}"
            )
        if not self.keywords:
            self.keywords = extract_keywords(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryItem":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "by", "as", "at", "that", "this", "it", "from", "we", "our",
    "using", "use", "used", "can", "has", "have", "was", "were", "than",
}


def extract_keywords(text: str, limit: int = 12) -> List[str]:
    """Cheap, dependency-free keyword extraction (lowercased token set).

    Good enough for the MVP's lexical retrieval; swappable for embeddings later.
    """
    tokens: List[str] = []
    seen = set()
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) <= 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
        if len(tokens) >= limit:
            break
    return tokens
