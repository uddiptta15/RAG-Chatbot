"""
HyDE fallback: single-retry corrective retrieval. -> component docs/hyde-fallback.md

Triggered by src/graph.py when the relevance filter finds too few relevant chunks
(`len(relevant_chunks) < MIN_RELEVANT`, src/relevance_filter.py). Retries dense retrieval
with a hypothetical answer's embedding instead of the query's, re-fuses with the ORIGINAL
query's BM25 leg (pipeline step 4 -- BM25 already matches exact query terms, a fabricated
passage wouldn't help it), and re-runs the relevance filter. Runs at most once per turn --
src/graph.py has no edge looping back into this node.
"""

from __future__ import annotations

import warnings

from google.genai import types

from src.indexing.embedder import embed_query
from src.llm_client import call_with_retry, get_client
from src.relevance_filter import MIN_RELEVANT, filter_chunks
from src.retrieval.dense import TOP_K, dense_retrieve_by_vector
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.sparse import bm25_retrieve

HYDE_MODEL = "gemini-flash-latest"

_MAX_OUTPUT_TOKENS = 512  # a short passage, not a full answer

_SYSTEM_PROMPT = """Output ONLY the passage itself -- no preamble, no meta-commentary
("Here is a passage that..."), no restating the question. Just the passage, as if it were
lifted directly from a textbook section."""

_PROMPT_TEMPLATE = (
    "Write a short passage that would answer this question, in the style of a technical "
    "course textbook: {query}"
)


def _generate_hypothetical_doc(query: str) -> str:
    """-> pipeline step 1."""
    response = call_with_retry(
        get_client().models.generate_content,
        model=HYDE_MODEL,
        contents=_PROMPT_TEMPLATE.format(query=query),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=_MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text.strip()


def _exhausted(fused: list[dict]) -> dict:
    """-> pipeline step 7, exhausted branch: pass whatever chunks scored highest (the
    fused top-N, already rrf_score-ranked) to generation with a hedge flag, rather than
    confidently hallucinating from nothing."""
    return {"relevant_chunks": fused, "low_confidence_context": True}


def hyde_fallback(query: str) -> dict:
    """-> component docs/hyde-fallback.md. `query` is the router's standalone_query (the
    same query normal retrieval used), NOT the original user message -- generation.py
    still only ever sees the true original message (CLAUDE.md, "history ownership rule").
    Returns `{"relevant_chunks": [...], "low_confidence_context": bool}`."""
    try:
        hypothetical_doc = _generate_hypothetical_doc(query)
        # -> step 2: same embedding model/provider as indexing (src/indexing/embedder.py).
        # Embedded with RETRIEVAL_QUERY task type (via embed_query) despite the text being
        # answer-shaped -- the task type reflects this vector's ROLE (searching the index),
        # not the surface form of the text it was derived from.
        hyde_embedding = embed_query(hypothetical_doc)

        dense_results = dense_retrieve_by_vector(hyde_embedding, k=TOP_K)      # step 3
        bm25_results = bm25_retrieve(query, k=TOP_K)                          # step 4
        fused = reciprocal_rank_fusion(dense_results, bm25_results)           # step 5
    except Exception as exc:  # noqa: BLE001 -- deliberate: this is already the last-resort
        # retry; if it breaks too, there's nothing left to fall back to except the hedge.
        warnings.warn(f"HyDE fallback failed, returning no context with low_confidence_context: {exc}")
        return {"relevant_chunks": [], "low_confidence_context": True}

    relevant_chunks = filter_chunks(query, fused)  # step 6

    if len(relevant_chunks) >= MIN_RELEVANT:  # step 7, success branch
        return {"relevant_chunks": relevant_chunks, "low_confidence_context": False}
    return _exhausted(fused)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "what is the vanishing gradient problem"
    result = hyde_fallback(query)
    print(f"low_confidence_context={result['low_confidence_context']}")
    for c in result["relevant_chunks"]:
        print(f"  {c['chunk_id']}: {c['text'][:80]}")
