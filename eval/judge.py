"""Blind pairwise LLM-as-judge for report quality (memory ON vs OFF).

To avoid position/identity bias the two reports are shown in randomized order as
"Report 1"/"Report 2"; the model picks a winner per dimension and we map the
labels back to on/off afterwards. Reuses the same gateway as the loop
(``llm._complete`` + ``llm._extract_json``). When a report is missing or the
model output can't be parsed the judge returns ``None`` and the harness skips
that pair (an LLM itself is mandatory: ``llm._complete`` raises if unavailable).

Dimensions:
- faithfulness: claims supported by the cited sources, nothing invented.
- coverage:     how completely the report answers all parts of the question.
- coherence:    stays on the original goal; organized and consistent.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "demo"))

import llm  # noqa: E402

DIMENSIONS = ("faithfulness", "coverage", "coherence")


def _prompt(question: str, report1: str, report2: str) -> str:
    return (
        "You are judging two answers to a research question. Compare them on three "
        "dimensions and pick the better report for each (or 'tie').\n\n"
        f"Research question:\n{question}\n\n"
        f"--- Report 1 ---\n{report1}\n\n"
        f"--- Report 2 ---\n{report2}\n\n"
        "Dimensions:\n"
        "- faithfulness: claims are supported by the cited sources; nothing invented.\n"
        "- coverage: how completely it answers ALL parts of the question.\n"
        "- coherence: stays on the original goal; organized and internally consistent.\n\n"
        "Return ONLY a JSON object with keys 'faithfulness', 'coverage', 'coherence' "
        "(each exactly '1', '2', or 'tie') and a short 'reason'."
    )


def _winner_to_arm(value: str, flipped: bool) -> Optional[str]:
    """Map a '1'/'2'/'tie' verdict back to 'on'/'off'/'tie'.

    When ``flipped`` is True, Report 1 was OFF and Report 2 was ON.
    """
    v = str(value).strip().lower()
    if v in ("1", "report 1", "report1"):
        label = 1
    elif v in ("2", "report 2", "report2"):
        label = 2
    else:
        return "tie"
    if label == 1:
        return "off" if flipped else "on"
    return "on" if flipped else "off"


def judge_pair(
    question: str,
    report_on: str,
    report_off: str,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, str]]:
    """Blind pairwise verdict. Returns {dim: 'on'|'off'|'tie', 'reason': str} or None."""
    if not report_on or not report_off:
        return None
    r = rng or random
    flipped = r.random() < 0.5  # flipped => Report 1 is OFF
    report1, report2 = (report_off, report_on) if flipped else (report_on, report_off)

    data = llm._extract_json(llm._complete(_prompt(question, report1, report2), max_tokens=600))
    if not isinstance(data, dict):
        return None

    verdict: Dict[str, str] = {}
    for dim in DIMENSIONS:
        verdict[dim] = _winner_to_arm(data.get(dim, "tie"), flipped) or "tie"
    verdict["reason"] = str(data.get("reason", ""))[:300]
    return verdict


def aggregate(judgments: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    """Tally wins/ties per dimension across all pairwise judgments."""
    out: Dict[str, Dict[str, int]] = {
        dim: {"on": 0, "off": 0, "tie": 0} for dim in DIMENSIONS
    }
    for j in judgments:
        if not j:
            continue
        for dim in DIMENSIONS:
            v = j.get(dim, "tie")
            if v in out[dim]:
                out[dim][v] += 1
    return out
