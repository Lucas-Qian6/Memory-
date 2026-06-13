"""Client for the Wenyon search API (`POST /api/v1/search`).

The pipeline only depends on the normalized ``Paper`` / ``Scholar`` / ``Patent``
dataclasses below. Every quirk of the real response (`" | "`-joined authors,
``" _ "`` junk in ``venue``, nullable ``year``, the broken ``subjects`` array)
is isolated in the ``_parse_*`` adapters - the single place to fix field names
when scholar/patent shapes show up live.

Config comes from the environment so "where the API lives" is just a setting:
- ``SEARCH_API_BASE_URL``  e.g. ``http://svc-...:8080`` or a tunneled host
- ``SEARCH_USER_ID``       defaults to ``user_123``
- ``SEARCH_SCENE``         optional business-isolation tag
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

DEFAULT_USER_ID = "user_123"
SEARCH_PATH = "/api/v1/search"


# --------------------------------------------------------------------------
# Normalized result types (what the pipeline + memory mapping consume)
# --------------------------------------------------------------------------
@dataclass
class Paper:
    doc_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    is_open_access: Optional[bool] = None
    score: Optional[float] = None
    subjects: List[str] = field(default_factory=list)
    title_zh: Optional[str] = None
    biz_type: str = "paper"


@dataclass
class Scholar:
    doc_id: str
    name: str
    raw: Dict = field(default_factory=dict)
    biz_type: str = "scholar"


@dataclass
class Patent:
    doc_id: str
    title: str
    raw: Dict = field(default_factory=dict)
    biz_type: str = "patent"


@dataclass
class SearchResults:
    query: str
    papers: List[Paper] = field(default_factory=list)
    scholars: List[Scholar] = field(default_factory=list)
    patents: List[Patent] = field(default_factory=list)
    total: int = 0
    trace_id: Optional[str] = None
    raw: Dict = field(default_factory=dict)

    def all_items(self) -> List[object]:
        return [*self.papers, *self.scholars, *self.patents]


# --------------------------------------------------------------------------
# Field-mapping adapters (isolate all real-API quirks here)
# --------------------------------------------------------------------------
def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _flatten_authors(authors) -> List[str]:
    """Real API sometimes packs multiple names into one `" | "`-joined string."""
    out: List[str] = []
    if not authors:
        return out
    if isinstance(authors, str):
        authors = [authors]
    for entry in authors:
        if not isinstance(entry, str):
            continue
        for name in entry.split("|"):
            name = name.strip()
            if name:
                out.append(name)
    return out


def _clean_venue(venue) -> Optional[str]:
    """Strip trailing/duplicated `" _ "` junk: "Computer _ Computer" -> "Computer"."""
    if not venue or not isinstance(venue, str):
        return None
    parts = [p.strip() for p in venue.split("_") if p.strip()]
    deduped: List[str] = []
    for p in parts:
        if p not in deduped:
            deduped.append(p)
    if not deduped:
        return None
    return " - ".join(deduped)


def _parse_subjects(subjects) -> List[str]:
    """`subjects` is null or a comma-split fragment of a JSON array; recover names."""
    if not subjects:
        return []
    if isinstance(subjects, list):
        blob = ",".join(str(s) for s in subjects)
    else:
        blob = str(subjects)
    return re.findall(r'"name"\s*:\s*"([^"]+)"', blob)


def _derive_url(doi: Optional[str], arxiv_id: Optional[str]) -> Optional[str]:
    if doi:
        return f"https://doi.org/{doi}"
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    return None


def _parse_paper(d: Dict) -> Paper:
    doi = d.get("doi")
    arxiv_id = d.get("arxivId")
    title = (d.get("titleEn") or d.get("titleZh") or "").strip()
    return Paper(
        doc_id=str(d.get("docId") or doi or title),
        title=title,
        title_zh=d.get("titleZh"),
        abstract=(d.get("snippet") or "").strip(),
        authors=_flatten_authors(d.get("authors")),
        venue=_clean_venue(d.get("venue")),
        year=_as_int(d.get("year")),
        doi=doi,
        arxiv_id=arxiv_id,
        url=_derive_url(doi, arxiv_id),
        citation_count=_as_int(d.get("citationCount")),
        is_open_access=d.get("isOpenAccess"),
        score=d.get("score"),
        subjects=_parse_subjects(d.get("subjects")),
    )


def _parse_scholar(d: Dict) -> Scholar:
    # Best-effort until a non-empty scholar list is seen live.
    name = d.get("name") or d.get("nameZh") or d.get("nameEn") or ""
    return Scholar(doc_id=str(d.get("docId") or d.get("id") or name), name=str(name).strip(), raw=d)


def _parse_patent(d: Dict) -> Patent:
    # Best-effort until a non-empty patent list is seen live.
    title = d.get("titleEn") or d.get("titleZh") or d.get("title") or ""
    return Patent(doc_id=str(d.get("docId") or d.get("id") or title), title=str(title).strip(), raw=d)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
class SearchClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        user_id: Optional[str] = None,
        scene: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SEARCH_API_BASE_URL", "")).rstrip("/")
        self.user_id = user_id or os.environ.get("SEARCH_USER_ID", DEFAULT_USER_ID)
        self.scene = scene if scene is not None else os.environ.get("SEARCH_SCENE")
        self.timeout = timeout
        if not self.base_url:
            raise ValueError(
                "SEARCH_API_BASE_URL is not set. Set it (e.g. http://svc-...:8080) "
                "in your environment/.env; a real search API is required."
            )

    def search(
        self,
        query: str,
        biz_types: Sequence[str] = ("paper",),
        page: int = 1,
        page_size: int = 10,
    ) -> SearchResults:
        biz_types = list(biz_types)
        raw = self._post(query, biz_types, page, page_size)
        return self._parse(query, raw)

    def _post(self, query: str, biz_types: List[str], page: int, page_size: int) -> Dict:
        body: Dict[str, object] = {
            "userId": self.user_id,
            "query": query,
            "page": page,
            "pageSize": page_size,
            "bizTypes": biz_types,
        }
        if self.scene:
            body["scene"] = self.scene
        req = urllib.request.Request(
            self.base_url + SEARCH_PATH,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            raise RuntimeError(f"search API call failed ({self.base_url}{SEARCH_PATH}): {e}") from e

    def _parse(self, query: str, raw: Dict) -> SearchResults:
        return SearchResults(
            query=query,
            papers=[_parse_paper(d) for d in (raw.get("papers") or [])],
            scholars=[_parse_scholar(d) for d in (raw.get("scholars") or [])],
            patents=[_parse_patent(d) for d in (raw.get("patents") or [])],
            total=_as_int(raw.get("total")) or 0,
            trace_id=raw.get("traceId"),
            raw=raw,
        )
