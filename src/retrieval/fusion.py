"""
Reciprocal Rank Fusion. -> component docs/retrieval.md, section 4
"""

from __future__ import annotations

from src.indexing.build_index import load_metadata_by_chunk_id

RRF_K = 60  # retrieval.md, section 4: "standard default from the original RRF paper"
TOP_N = 5   # retrieval.md, section 4-5: chunks passed to the correction stage


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    *,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """-> retrieval.md, section 4-5. Fuses by RANK, not raw score (dense cosine and BM25
    scores are on incomparable scales -- section 4), then hydrates the top-N fused
    chunk_ids with full metadata for the correction stage. Returns
    `[{"chunk_id", "rrf_score", "text", "heading_path", "content_type", "source_path"}, ...]`
    ranked by `rrf_score` descending."""
    rrf_scores: dict[str, float] = {}
    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)

    ranked_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_n]
    metadata_by_id = load_metadata_by_chunk_id()

    return [
        {
            "chunk_id": chunk_id,
            "rrf_score": rrf_scores[chunk_id],
            "text": metadata_by_id[chunk_id]["text"],
            "heading_path": metadata_by_id[chunk_id]["heading_path"],
            "content_type": metadata_by_id[chunk_id]["content_type"],
            "source_path": metadata_by_id[chunk_id]["source_path"],
        }
        for chunk_id in ranked_ids
        if chunk_id in metadata_by_id  # defensive: skip if metadata store is out of sync
    ]
