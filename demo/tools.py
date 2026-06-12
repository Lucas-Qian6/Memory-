"""Offline mock of the Wenyon search API, shaped like the real response.

This is the ``--mock`` fallback so the deep-research loop runs without cluster
access. Each entry mirrors a real ``/api/v1/search`` ``papers[]`` item (same
field names + the same quirks: ``" | "``-joined authors, ``" _ "`` junk in
``venue``, nullable ``year``), so the SAME ``_parse_paper`` path in
``search_client.py`` is exercised in mock mode as in a live call.
"""

from __future__ import annotations

from typing import Dict, List

# Raw, API-shaped paper records (see the confirmed live schema in the plan).
MOCK_PAPERS: List[Dict] = [
    {
        "docId": "900000000000000001",
        "titleZh": "基于序列图编辑的逆合成预测",
        "titleEn": "Graph2Edits: Retrosynthesis via Sequential Graph Edits",
        "snippet": (
            "Graph2Edits formulates single-step retrosynthesis as a sequence of "
            "graph edit operations on the molecular graph. The model autoregressively "
            "predicts edits, generating synthons and then candidate reactants. On the "
            "USPTO-50K benchmark it reaches 55.1% top-1 accuracy, improving over "
            "template-based baselines. The edit formulation also yields more "
            "chemically valid intermediate structures."
        ),
        "highlights": None,
        "score": 211.4,
        "authors": ["Weihe Zhong | Ziduo Yang | Calvin Yu-Chian Chen"],
        "venue": "Nature Communications _ ",
        "year": 2023,
        "doi": "10.1038/s41467-023-38851-5",
        "arxivId": None,
        "subjects": None,
        "citationCount": 88,
        "isOpenAccess": True,
        "scoreDetails": {"recallScore": 19.8, "coarseScore": 255.7, "fineScore": 211.4, "blendScore": 0.0, "finalScore": 211.4},
    },
    {
        "docId": "900000000000000002",
        "titleZh": "反应上下文感知的逆合成预测",
        "titleEn": "Reaction-Context-Aware Retrosynthesis Prediction",
        "snippet": (
            "Incorporating reaction context, such as reagents and conditions, improves "
            "retrosynthesis prediction accuracy. Conditioning on reaction context gives "
            "a +3.2% top-1 improvement over a context-free baseline. An ablation shows "
            "that reaction context reduces implausible precursor predictions and fewer "
            "chemically invalid suggestions are produced."
        ),
        "highlights": None,
        "score": 209.5,
        "authors": ["A. Researcher | B. Coauthor"],
        "venue": "Journal of Chemical Information and Modeling _ JCIM",
        "year": None,
        "doi": "10.1021/acs.jcim.3c00001",
        "arxivId": None,
        "subjects": [
            "[{\"id\": \"80031\"",
            " \"count\": 1",
            " \"name\": \"Chemical Sciences\"",
            " \"research_categories_id\": \"34\"}]",
        ],
        "citationCount": 12,
        "isOpenAccess": False,
        "scoreDetails": {"recallScore": 20.4, "coarseScore": 264.1, "fineScore": 209.5, "blendScore": 0.0, "finalScore": 209.5},
    },
    {
        "docId": "900000000000000003",
        "titleZh": "面向科学发现的大语言模型智能体综述",
        "titleEn": "A Survey of LLM Agents for Scientific Discovery",
        "snippet": (
            "Large language model (LLM) agents have been applied to literature review, "
            "tool use, and hypothesis generation across scientific domains. This survey "
            "categorizes agent applications and the orchestration patterns behind them. "
            "However, broad superiority of LLM agents over traditional workflows is not "
            "yet established, and reported results remain largely task-specific."
        ),
        "highlights": None,
        "score": 205.2,
        "authors": ["C. Lead", "D. Second", "E. Third"],
        "venue": "Computer _ Computer",
        "year": 2024,
        "doi": "10.48550/arXiv.2401.00001",
        "arxivId": "2401.00001",
        "subjects": None,
        "citationCount": 34,
        "isOpenAccess": False,
        "scoreDetails": {"recallScore": 19.6, "coarseScore": 254.7, "fineScore": 205.2, "blendScore": 0.0, "finalScore": 205.2},
    },
    {
        "docId": "900000000000000004",
        "titleZh": "面向长程任务大语言模型智能体的记忆机制",
        "titleEn": "Memory Mechanisms for Long-Horizon LLM Agents",
        "snippet": (
            "External memory lets agents persist findings beyond the context window, "
            "paging structured notes in and out to sustain long-horizon tasks. "
            "Structured memory reduces redundant tool calls in multi-step tasks: an "
            "episodic action log yields 38% fewer repeated searches. The study also "
            "finds that selective writing matters, since storing everything turns memory "
            "into a second polluted context."
        ),
        "highlights": None,
        "score": 203.9,
        "authors": ["F. Memory | G. Agent | H. Horizon"],
        "venue": "Transactions on Machine Learning Research _ TMLR",
        "year": 2025,
        "doi": "10.48550/arXiv.2502.00002",
        "arxivId": "2502.00002",
        "subjects": [
            "[{\"id\": \"80017\"",
            " \"count\": 1",
            " \"name\": \"Information and Computing Sciences\"",
            " \"research_categories_id\": \"46\"}",
            " {\"id\": \"80181\"",
            " \"count\": 1",
            " \"name\": \"Artificial Intelligence\"",
            " \"research_categories_id\": \"4602\"}]",
        ],
        "citationCount": 5,
        "isOpenAccess": True,
        "scoreDetails": {"recallScore": 20.4, "coarseScore": 248.9, "fineScore": 203.9, "blendScore": 0.0, "finalScore": 203.9},
    },
    {
        "docId": "900000000000000005",
        "titleZh": "检索增强生成综述",
        "titleEn": "Retrieval-Augmented Generation: A Survey",
        "snippet": (
            "Retrieval-augmented generation (RAG) grounds language model outputs in "
            "retrieved documents, reducing hallucination on knowledge-intensive tasks. "
            "Retrieval quality is the dominant factor in end-to-end answer accuracy. "
            "Chunking strategy and embedding choice materially affect what the retriever "
            "surfaces, and re-ranking retrieved passages further improves faithfulness."
        ),
        "highlights": None,
        "score": 199.1,
        "authors": ["I. Retrieval", "J. Generation"],
        "venue": "Foundations and Trends in IR _ FnTIR",
        "year": 2023,
        "doi": "10.48550/arXiv.2312.00003",
        "arxivId": "2312.00003",
        "subjects": None,
        "citationCount": 120,
        "isOpenAccess": False,
        "scoreDetails": {"recallScore": 19.2, "coarseScore": 250.3, "fineScore": 199.1, "blendScore": 0.0, "finalScore": 199.1},
    },
]


def _overlap(query: str, paper: Dict) -> int:
    """Crude lexical overlap of query tokens against title + snippet."""
    q = {t for t in query.lower().replace("-", " ").split() if len(t) > 2}
    haystack = " ".join(
        str(paper.get(k, "")) for k in ("titleEn", "titleZh", "snippet")
    ).lower()
    return sum(1 for t in q if t in haystack)


def mock_search(
    query: str,
    biz_types: List[str],
    page: int = 1,
    page_size: int = 10,
) -> Dict:
    """Return a SearchResponse-shaped dict from the offline corpus.

    Only ``paper`` is populated; ``scholar`` / ``patent`` come back empty (as in
    the live sample we captured), so the loop's paper path is what gets exercised.
    """
    papers: List[Dict] = []
    if "paper" in biz_types:
        scored = sorted(
            ((_overlap(query, p), p) for p in MOCK_PAPERS),
            key=lambda x: x[0],
            reverse=True,
        )
        hits = [p for ov, p in scored if ov > 0] or [p for _, p in scored[:2]]
        start = (page - 1) * page_size
        papers = hits[start : start + page_size]
    return {
        "traceId": "mock-trace",
        "total": len(papers),
        "page": page,
        "pageSize": page_size,
        "papers": papers,
        "scholars": [],
        "patents": [],
        "facets": None,
        "workerTrace": None,
        "fallback": False,
    }
