"""
Dense retrieval: FAISS cosine similarity search. -> component docs/retrieval.md, section 2
"""

from __future__ import annotations

import numpy as np

from src.indexing.build_index import load_index, load_metadata_store
from src.indexing.embedder import embed_query

TOP_K = 20  # retrieval.md, section 2


def dense_retrieve_by_vector(query_vector: np.ndarray, k: int = TOP_K) -> list[tuple[str, float]]:
    """Core FAISS search, given an already-computed query embedding -- the reusable half
    of `dense_retrieve`, split out so hyde_fallback.py can search with the hypothetical
    document's embedding instead of embedding a query string itself
    (component docs/hyde-fallback.md, pipeline step 3)."""
    index = load_index()
    chunks_by_position = load_metadata_store()

    scores, positions = index.search(query_vector.reshape(1, -1), min(k, index.ntotal))

    results = []
    for score, pos in zip(scores[0], positions[0]):
        if pos == -1:
            continue  # FAISS pads with -1 when the index has fewer than k vectors
        results.append((chunks_by_position[pos]["chunk_id"], float(score)))
    return results


def dense_retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """-> retrieval.md, section 2. Returns `[(chunk_id, dense_score), ...]` ranked by
    cosine similarity, most similar first."""
    return dense_retrieve_by_vector(embed_query(query), k=k)
