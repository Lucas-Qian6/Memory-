"""Memory item schema.

One item type for all three in-task memory facets. The ``type`` field is what
makes the split a tag over a single store rather than three separate systems.
All memory here is scoped to a single long-horizon task (one user query):

- ``working``  - current task state (goal, sub-questions, plan/outline, draft)
- ``episodic`` - actions already taken (searches/reads), to avoid repeat work
- ``semantic`` - findings (claims + source/evidence), to survive context overflow
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional

MemoryType = Literal["working", "episodic", "semantic"]

VALID_TYPES = ("working", "episodic", "semantic")

# Per-item lifecycle (set by the consistency + merge logic in the write policy):
# - ``active``     : current, recallable.
# - ``superseded`` : replaced by a newer, contradicting claim (kept for provenance).
# - ``merged``     : folded into a canonical near-duplicate claim (kept for provenance).
# The read path recalls only ``active`` items; the others stay on disk so the
# trail behind a claim is never destroyed.
VALID_STATUSES = ("active", "superseded", "merged")


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _hash_content(text: str) -> str:
    """Stable md5 of normalized content, for cheap exact-duplicate detection.

    Normalization (strip + lowercase + whitespace-collapse) so that trivially
    reformatted duplicates of the same claim/action hash to the same value.
    """
    norm = " ".join((text or "").strip().lower().split())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


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
    # ``links`` carries both citation payload (title/authors/doi/url/...) and
    # relation edges to other items by id: ``supersedes`` / ``conflicts_with`` /
    # ``merged_from`` (lists of ids), ``superseded_by`` / ``merged_into`` (scalar
    # ids), plus ``sources`` (list of citation dicts, unioned when near-duplicate
    # claims are merged - the basis for multi-source attribution).
    links: Dict[str, Any] = field(default_factory=dict)
    # P1 storage-layering fields:
    evidence: Optional[str] = None  # exact supporting span behind a claim
    uri: Optional[str] = None  # pointer to raw source archived under artifacts/
    step: Optional[int] = None  # task step ordinal (drives recency, not wall-clock)
    status: str = "active"  # one of VALID_STATUSES; read path recalls only "active"
    content_hash: str = ""  # md5 of normalized content, for exact dedup
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
        if not self.content_hash:
            self.content_hash = _hash_content(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryItem":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    # --- relation / lifecycle bookkeeping (consistency + merge) ---------
    def add_relation(self, kind: str, other_id: str) -> None:
        """Append a directed edge to another item's id under ``links[kind]``.

        For the list-valued id relations (``supersedes`` / ``conflicts_with`` /
        ``merged_from``). De-dupes and tolerates a pre-existing scalar value.
        """
        bucket = self.links.setdefault(kind, [])
        if not isinstance(bucket, list):
            bucket = [bucket]
            self.links[kind] = bucket
        if other_id not in bucket:
            bucket.append(other_id)

    def mark_superseded(self, by_id: str) -> None:
        """Retire this item: ``status`` -> ``superseded`` + back-pointer to the winner."""
        self.status = "superseded"
        self.links["superseded_by"] = by_id

    def mark_merged(self, into_id: str) -> None:
        """Retire this item into a canonical: ``status`` -> ``merged`` + back-pointer."""
        self.status = "merged"
        self.links["merged_into"] = into_id


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
