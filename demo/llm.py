"""LLM helpers for the deep-research loop, over an Anthropic-compatible gateway.

Reuses ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` / ``MEMORY_DR_MODEL`` from
the environment (the GPUGeek/Claude gateway already wired in ``.env``).

Every function degrades to a deterministic fallback when the LLM is unavailable
(``anthropic`` not installed, no key, network/JSON error) or when ``use_llm`` is
False - so ``--mock`` and ``--no-llm`` runs stay fully offline and reproducible.
The four functions map onto the loop: plan -> (per result) extract -> reflect
(follow-ups) -> synthesize.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "by", "as", "at", "that", "this", "it", "from", "we", "our",
    "using", "use", "used", "can", "has", "have", "was", "were", "than", "how",
    "does", "do", "what", "which", "and", "can",
}

_client_cache = None
_client_tried = False


# --------------------------------------------------------------------------
# Low-level gateway access
# --------------------------------------------------------------------------
def _get_client():
    global _client_cache, _client_tried
    if _client_tried:
        return _client_cache
    _client_tried = True
    try:
        import anthropic
    except Exception:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    try:
        _client_cache = (
            anthropic.Anthropic(base_url=base_url) if base_url else anthropic.Anthropic()
        )
    except Exception:
        _client_cache = None
    return _client_cache


def llm_available() -> bool:
    return _get_client() is not None


def _complete(prompt: str, max_tokens: int = 1024, use_llm: bool = True) -> Optional[str]:
    if not use_llm:
        return None
    client = _get_client()
    if client is None:
        return None
    model = os.environ.get("MEMORY_DR_MODEL", "claude-sonnet-4-6")
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception:
        return None


def _extract_json(text: Optional[str]):
    """Pull the first JSON array/object out of a possibly chatty completion."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except Exception:
        return None


def _keywords(text: str, limit: int = 24) -> List[str]:
    out, seen = [], set()
    for raw in (text or "").lower().replace("/", " ").replace("-", " ").split():
        tok = "".join(ch for ch in raw if ch.isalnum())
        if len(tok) <= 2 or tok in _STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= limit:
            break
    return out


def _dedup_keep_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out


# --------------------------------------------------------------------------
# Loop steps
# --------------------------------------------------------------------------
def plan_subqueries(
    question: str,
    state: str = "",
    known: str = "",
    n: int = 3,
    use_llm: bool = True,
) -> List[str]:
    """Decompose the research question into focused search queries."""
    prompt = (
        "You are planning literature searches for a deep-research task.\n"
        f"Research question:\n{question}\n\n"
        f"Current task state:\n{state or '(none)'}\n\n"
        f"Findings gathered so far:\n{known or '(none yet)'}\n\n"
        f"Propose {n} focused search queries (short keyword phrases, NOT questions) "
        "that gather evidence to answer the question. Cover distinct sub-topics.\n"
        "Return ONLY a JSON array of strings."
    )
    data = _extract_json(_complete(prompt, max_tokens=400, use_llm=use_llm))
    if isinstance(data, list):
        queries = [str(q) for q in data if isinstance(q, (str, int, float)) and str(q).strip()]
        if queries:
            return _dedup_keep_order(queries)[:n]
    return _fallback_subqueries(question, n)


def decide_followups(
    question: str,
    known: str,
    asked: List[str],
    n: int = 2,
    use_llm: bool = True,
) -> List[str]:
    """Reflect on findings and propose NEW gap-filling queries, or stop ([])."""
    prompt = (
        "You are running a deep-research loop and deciding whether to search more.\n"
        f"Research question:\n{question}\n\n"
        f"Findings gathered so far:\n{known or '(none yet)'}\n\n"
        f"Queries already run (do NOT repeat these):\n{chr(10).join('- ' + q for q in asked) or '(none)'}\n\n"
        "If the findings already cover the question well, return an empty JSON array [].\n"
        f"Otherwise return up to {n} NEW search queries (short keyword phrases) that "
        "target the most important remaining gaps.\n"
        "Return ONLY a JSON array of strings."
    )
    data = _extract_json(_complete(prompt, max_tokens=400, use_llm=use_llm))
    if isinstance(data, list):
        asked_lower = {a.strip().lower() for a in asked}
        fresh = [
            str(q).strip()
            for q in data
            if str(q).strip() and str(q).strip().lower() not in asked_lower
        ]
        return _dedup_keep_order(fresh)[:n]
    # Deterministic fallback: no reflection model -> stop after the first round.
    return []


def extract_claims(
    question: str,
    paper,
    max_claims: int = 3,
    use_llm: bool = True,
) -> List[Dict[str, str]]:
    """Extract atomic, question-relevant claims (+ supporting evidence span)."""
    abstract = getattr(paper, "abstract", "") or ""
    title = getattr(paper, "title", "") or ""
    if not abstract:
        return []
    prompt = (
        "Extract atomic factual claims from a paper abstract that are RELEVANT to a "
        "research question.\n"
        f"Research question:\n{question}\n\n"
        f"Paper title: {title}\n"
        f"Abstract:\n{abstract}\n\n"
        f"Return up to {max_claims} claims. Each claim must be one self-contained "
        "sentence, and 'evidence' must be an exact supporting span copied from the "
        "abstract. If nothing is relevant, return [].\n"
        "Return ONLY a JSON array of objects with keys 'claim' and 'evidence'."
    )
    data = _extract_json(_complete(prompt, max_tokens=700, use_llm=use_llm))
    claims: List[Dict[str, str]] = []
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                text = str(obj.get("claim") or obj.get("text") or "").strip()
                evidence = str(obj.get("evidence") or text).strip()
                if len(text) >= 8:
                    claims.append({"text": text, "evidence": evidence})
        if claims:
            return claims[:max_claims]
    return _fallback_claims(question, abstract, max_claims)


def synthesize(question: str, context: str, use_llm: bool = True) -> Optional[str]:
    """Write a cited literature-review answer from recalled claims (or None)."""
    if not context.strip():
        return None
    prompt = (
        "Write a concise literature-review answer to the research question using ONLY "
        "the recalled memory claims below. Cite sources inline like [source-id]. Do "
        "not invent facts beyond the claims.\n\n"
        f"Research question:\n{question}\n\n"
        f"Recalled claims:\n{context}\n\n"
        "Write 1-2 short paragraphs followed by a 'Sources' list."
    )
    return _complete(prompt, max_tokens=1024, use_llm=use_llm)


# --------------------------------------------------------------------------
# Deterministic fallbacks
# --------------------------------------------------------------------------
def _fallback_subqueries(question: str, n: int) -> List[str]:
    kws = _keywords(question, limit=3 * n)
    chunks = [kws[i : i + 3] for i in range(0, len(kws), 3)]
    queries = [" ".join(c) for c in chunks if c][:n]
    return queries or [question.strip()]


def _fallback_claims(question: str, abstract: str, max_claims: int) -> List[Dict[str, str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) >= 30]
    q_terms = set(_keywords(question))
    scored = sorted(
        ((len(q_terms & set(_keywords(s))), s) for s in sentences),
        key=lambda x: x[0],
        reverse=True,
    )
    picked = [s for ov, s in scored if ov > 0][:max_claims]
    if not picked:
        picked = [s for _, s in scored[:1]]
    return [{"text": s, "evidence": s} for s in picked]
