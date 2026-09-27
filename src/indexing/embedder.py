"""
Embedding step. -> component docs/indexing.md, section 3

Uses a local Hugging Face embedding model via `fastembed` (ONNX Runtime, no torch) rather
than Gemini or a Sentence Transformer model -- avoids two real constraints hit during
development:
  1. `sentence-transformers`'s `torch` dependency couldn't install (Windows Long Path
     support disabled; torch ships files whose paths exceed 260 chars).
  2. Gemini's free-tier `embed_content` quota (~1000/day) can't cover this corpus
     (2660+ chunks) in one day, and a network-dependent embedding call also means every
     retrieval-time query pays a round-trip + is subject to that same quota.
Running fully locally sidesteps both: no daily cap ever, no network latency at query time,
no GPU required. LLM calls (router/relevance_filter/hyde_fallback/generation) still go
through Gemini via src/llm_client.py -- only embeddings moved.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

# BAAI/bge-small-en-v1.5: 384-dim, ~67MB, retrieval-tuned (asymmetric query/passage
# encoding built in), no GPU needed. Swap to BAAI/bge-base-en-v1.5 (768-dim, ~210MB) for a
# quality/speed tradeoff -- same framing indexing.md originally gave for Sentence
# Transformer models ("speed" vs. "quality matters more than latency").
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Lazy singleton. The ONNX model weights download once from Hugging Face Hub on
    first use and are cached locally afterward (no repeated downloads, no network needed
    once cached)."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Batch-embed arbitrary texts. `task_type` distinguishes indexing-time documents
    (`RETRIEVAL_DOCUMENT`, the default, via fastembed's `passage_embed`) from query-time
    embeddings (`RETRIEVAL_QUERY`, via `query_embed` -- see `embed_query`) -- BGE models are
    asymmetric-retrieval-aware (a different instruction prefix is baked in per side) just
    like the Gemini embeddings this replaced, so the task_type split carries over
    unchanged. Returns a float32 `(n, dim)` array; BGE's output is already L2-normalized
    (confirmed empirically -- norm 1.0), so no extra normalization step is needed before
    FAISS inner-product search (contrast with indexing.md, section 4's original note, which
    assumed a model that might not normalize on its own)."""
    model = _get_model()
    if task_type == "RETRIEVAL_QUERY":
        vectors = list(model.query_embed(texts))
    else:
        vectors = list(model.passage_embed(texts))
    return np.array(vectors, dtype="float32")


def embed_chunks(chunks) -> np.ndarray:
    """Embed a list of `Chunk` (or dict) objects' `text` field only -- metadata is never
    embedded, just carried alongside. -> section 3."""
    texts = [c.text if hasattr(c, "text") else c["text"] for c in chunks]
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string with `RETRIEVAL_QUERY` -- the query-side half of the
    asymmetric retrieval embeddings, paired with `embed_chunks`'s `RETRIEVAL_DOCUMENT` side.
    -> retrieval.md, section 2 ("same embedding model as indexing")."""
    return embed_texts([query], task_type="RETRIEVAL_QUERY")[0]
