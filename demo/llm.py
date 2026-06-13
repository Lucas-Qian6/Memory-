"""LLM helpers for the deep-research loop, over an Anthropic-compatible gateway.

Reuses ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` / ``MEMORY_DR_MODEL`` from
the environment (the GPUGeek/Claude gateway already wired in ``.env``).

An LLM is **required**. Every call raises ``RuntimeError`` when the gateway is
unavailable (``anthropic`` not installed, no key) or when the model output can't
be parsed - there is no deterministic/offline fallback. The functions map onto
the loop: plan -> (per result) extract -> reflect (follow-ups) -> synthesize,
plus ``judge_relation`` for in-task memory write decisions.
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

# Lightweight per-run accounting of prompt chars actually sent to the LLM. The
# eval harness resets this before a run and reads it after, as a token proxy for
# "how much context did this arm feed the model" (real tokens would need the API
# usage field). Only real ``_complete`` calls increment it.
_usage_chars = 0


def reset_usage() -> None:
    global _usage_chars
    _usage_chars = 0


def get_usage() -> int:
    """Total prompt chars sent to the LLM since the last ``reset_usage()``."""
    return _usage_chars


# --------------------------------------------------------------------------
# Low-level gateway access (LLM is mandatory: fail fast, never degrade)
# --------------------------------------------------------------------------
def _get_client():
    global _client_cache
    if _client_cache is not None:
        return _client_cache
    try:
        import anthropic
    except Exception as e:  # package missing
        raise RuntimeError(
            "anthropic is required (an LLM is mandatory): pip install anthropic"
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; an LLM is required to run this loop."
        )
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    _client_cache = (
        anthropic.Anthropic(base_url=base_url) if base_url else anthropic.Anthropic()
    )
    return _client_cache


def _complete(prompt: str, max_tokens: int = 1024) -> str:
    client = _get_client()
    global _usage_chars
    _usage_chars += len(prompt)
    model = os.environ.get("MEMORY_DR_MODEL", "claude-sonnet-4-6")
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


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
    data = _extract_json(_complete(prompt, max_tokens=400))
    if isinstance(data, list):
        queries = [str(q) for q in data if isinstance(q, (str, int, float)) and str(q).strip()]
        if queries:
            return _dedup_keep_order(queries)[:n]
    raise RuntimeError("plan_subqueries: LLM did not return a usable JSON array of queries")


def decide_followups(
    question: str,
    known: str,
    asked: List[str],
    n: int = 2,
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
    data = _extract_json(_complete(prompt, max_tokens=400))
    if isinstance(data, list):
        asked_lower = {a.strip().lower() for a in asked}
        fresh = [
            str(q).strip()
            for q in data
            if str(q).strip() and str(q).strip().lower() not in asked_lower
        ]
        return _dedup_keep_order(fresh)[:n]
    raise RuntimeError("decide_followups: LLM did not return a JSON array")


def extract_claims(
    question: str,
    paper,
    max_claims: int = 3,
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
    data = _extract_json(_complete(prompt, max_tokens=700))
    if isinstance(data, list):
        claims: List[Dict[str, str]] = []
        for obj in data:
            if isinstance(obj, dict):
                text = str(obj.get("claim") or obj.get("text") or "").strip()
                evidence = str(obj.get("evidence") or text).strip()
                if len(text) >= 8:
                    claims.append({"text": text, "evidence": evidence})
        return claims[:max_claims]
    raise RuntimeError("extract_claims: LLM did not return a JSON array")


def synthesize(question: str, context: str) -> Optional[str]:
    """Write a cited literature-review answer from recalled claims."""
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
    return _complete(prompt, max_tokens=1024)


# --------------------------------------------------------------------------
# In-task memory: relation judge (near-duplicate merge + contradiction handling)
# --------------------------------------------------------------------------
def judge_relation(new, cand) -> Dict[str, object]:
    """Decide how a NEW claim relates to a similar EXISTING stored claim.

    Called by ``WritePolicy.classify`` once embedding similarity flags a near
    duplicate. Returns a JSON-shaped dict::

        {"relation": "duplicate"|"merge"|"supersede"|"conflict"|"distinct",
         "canonical": "<combined sentence>",   # for merge / duplicate
         "winner": "new"|"existing",            # for supersede (which is current)
         "reason": "<short>"}

    Raises ``RuntimeError`` if the LLM output can't be parsed (no fallback).
    """
    new_text = getattr(new, "content", str(new))
    cand_text = getattr(cand, "content", str(cand))
    new_src, cand_src = getattr(new, "source", None), getattr(cand, "source", None)
    new_step, cand_step = getattr(new, "step", None), getattr(cand, "step", None)
    prompt = (
        "You maintain an AI agent's in-task research memory. A NEW claim was just "
        "extracted and is highly similar to an EXISTING stored claim. Decide their "
        "relation so the memory stays consistent and non-redundant.\n\n"
        f"NEW claim (source={new_src}, step={new_step}):\n{new_text}\n\n"
        f"EXISTING claim (source={cand_src}, step={cand_step}):\n{cand_text}\n\n"
        "Choose exactly ONE relation:\n"
        '- "duplicate": same assertion, no new information.\n'
        '- "merge": same assertion but complementary detail or a different source '
        "worth keeping -> combine into one claim.\n"
        '- "supersede": they CONTRADICT and one is a corrected/updated version -> '
        "the newer/more authoritative one replaces the other.\n"
        '- "conflict": they CONTRADICT but you cannot tell which is correct -> keep '
        "both, flag as disputed.\n"
        '- "distinct": actually different claims -> keep both.\n\n'
        'For "merge"/"duplicate" also return "canonical": one self-contained '
        "sentence capturing the combined claim.\n"
        'For "supersede" also return "winner": "new" or "existing" (which claim is '
        "current/true).\n"
        'Return ONLY a JSON object with key "relation" (plus optional "canonical", '
        '"winner", "reason").'
    )
    data = _extract_json(_complete(prompt, max_tokens=400))
    if isinstance(data, dict) and data.get("relation"):
        return data
    raise RuntimeError("judge_relation: LLM did not return a usable JSON object")
