"""
Sparse retrieval: BM25 over chunk text. -> component docs/retrieval.md, section 3
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from src.indexing.build_index import load_metadata_store

TOP_K = 20  # retrieval.md, section 3

# Lowercase, strip punctuation, but keep dotted code identifiers (`nn.Linear`) intact
# rather than aggressively stemming -- retrieval.md, section 3.
_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _bm25_index() -> tuple[BM25Okapi, tuple[dict, ...]]:
    """Built once from the persisted metadata store (indexing.md, section 4) -- in-process
    per retrieval.md's open question ("in-process is simpler for a project of this size")."""
    chunks = load_metadata_store()
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus), chunks


def bm25_retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """-> retrieval.md, section 3. Returns `[(chunk_id, bm25_score), ...]` ranked by BM25
    score, most relevant first. Chunks with zero term overlap are dropped rather than
    padding out to `k` with noise that would still earn RRF rank credit downstream."""
    bm25, chunks = _bm25_index()
    scores = bm25.get_scores(_tokenize(query))
    ranked_positions = np.argsort(scores)[::-1]

    results = []
    for pos in ranked_positions[:k]:
        if scores[pos] <= 0:
            break  # descending order -- everything after this is <= 0 too
        results.append((chunks[pos]["chunk_id"], float(scores[pos])))
    return results
