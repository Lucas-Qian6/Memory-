"""Optional local embeddings for semantic recall.

Uses ``sentence-transformers`` if it is installed; otherwise every call returns
``None`` so retrieval falls back to the pure-stdlib lexical scorer. This keeps
the MVP runnable with zero extra dependencies (the same degrade-to-None pattern
as ``demo/llm.py``: no key / no package / any error -> ``None``).

Model is chosen via ``MEMORY_DR_EMBED_MODEL`` (default ``all-MiniLM-L6-v2``)
and loaded lazily + cached on first use.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

_model_cache = None
_model_tried = False


def _get_model():
    global _model_cache, _model_tried
    if _model_tried:
        return _model_cache
    _model_tried = True
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    name = os.environ.get("MEMORY_DR_EMBED_MODEL", "all-MiniLM-L6-v2")
    try:
        _model_cache = SentenceTransformer(name)
    except Exception:
        _model_cache = None
    return _model_cache


def embeddings_available() -> bool:
    return _get_model() is not None


def encode(
    texts: Union[str, Sequence[str]],
) -> Optional[Union[List[float], List[List[float]]]]:
    """Encode text(s) into normalized vectors.

    - ``str`` in  -> single ``List[float]`` (or ``None`` if unavailable).
    - sequence in -> ``List[List[float]]`` (or ``None`` if unavailable).

    Returns ``None`` (not a zero vector) whenever embeddings can't be produced,
    which the retrieval layer treats as "use lexical scoring".
    """
    model = _get_model()
    if model is None:
        return None
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    if not batch:
        return None if single else []
    try:
        vecs = model.encode(batch, normalize_embeddings=True)
    except Exception:
        return None
    out = [[float(x) for x in v] for v in vecs]
    return out[0] if single else out
