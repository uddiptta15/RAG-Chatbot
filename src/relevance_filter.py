"""
Relevance filter: statement-level contextual relevancy scoring + keep/discard.
-> component docs/relevance-filter.md

Runs after retrieval's RRF fusion (component docs/retrieval.md, sections 4-5) and before
generation (or hyde_fallback.py, on failure). Per-chunk scoring (section 2), batched
across all N retrieved chunks into a single LLM call (section 5) rather than one call per
chunk.
"""

from __future__ import annotations

import warnings

from google.genai import types
from pydantic import BaseModel

from src.llm_client import call_with_retry, get_client

RELEVANCE_FILTER_MODEL = "gemini-flash-latest"

THRESHOLD = 0.5   # relevance-filter.md, section 3: "majority of statements relevant"
MIN_RELEVANT = 1  # relevance-filter.md, section 4: at least one relevant chunk found

_MAX_OUTPUT_TOKENS = 4096  # statement-level breakdown across up to N chunks in one call


class _Statement(BaseModel):
    text: str
    relevant: bool


class _ChunkJudgment(BaseModel):
    chunk_id: str
    statements: list[_Statement]


class _RelevanceJudgments(BaseModel):
    judgments: list[_ChunkJudgment]


_SYSTEM_PROMPT = """You are the relevance filter in a Corrective-RAG pipeline. For each
passage, decide whether it actually helps answer the question -- this is a PRECISION
check on already-retrieved passages, not a second retrieval step.

A passage is relevant if it contains information that could plausibly be used to answer
the question -- not just topical/keyword overlap.
- FALSE POSITIVE to avoid: a passage mentions the same keywords/topic but doesn't actually
  address what the question is asking (e.g. question asks "how do you compute gradients
  for softmax", passage defines softmax but never discusses gradients) -- statements like
  this should be marked NOT relevant.
- FALSE NEGATIVE to avoid: a passage uses different terminology than the question but does
  answer it (e.g. question says "why is my loss NaN", passage discusses "numerical
  instability from exploding gradients") -- do not mark a statement irrelevant just because
  the wording differs from the question.

For each passage:
1. Break it into individual factual statements/claims.
2. For each statement, decide if it is relevant to answering the question.

Return one judgment entry per passage, using the exact chunk_id given for each passage."""


def _build_prompt(query: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[chunk_id: {c['chunk_id']}]\n{c['text']}" for c in chunks)
    return f"Question: {query}\n\nPassages:\n{passages}"


def _relevancy_score(judgment: _ChunkJudgment) -> float:
    """-> section 3: `relevancy_score = (# statements judged relevant) / (total # statements)`."""
    if not judgment.statements:
        return 0.0
    relevant_count = sum(1 for s in judgment.statements if s.relevant)
    return relevant_count / len(judgment.statements)


def filter_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """-> sections 3-4. Returns the subset of `chunks` whose `relevancy_score >= THRESHOLD`,
    each with a `relevancy_score` field attached (this is also DeepEval's/RAGAS's
    Contextual Relevancy metric, per section 3 -- attaching the score lets eval.md reuse
    the same production judgments rather than recomputing). Chunks with no text are
    skipped defensively (nothing to judge).

    On total failure (API error, missing/malformed structured output) this fails CLOSED --
    returns an empty list rather than passing chunks through unfiltered. An empty result
    routes to the HyDE fallback via the graph's MIN_RELEVANT check (src/graph.py), which is
    the correction pipeline's existing "not enough relevant content" path -- silently
    disabling the precision check would be worse than that."""
    scorable = [c for c in chunks if c.get("text", "").strip()]
    if not scorable:
        return []

    try:
        response = call_with_retry(
            get_client().models.generate_content,
            model=RELEVANCE_FILTER_MODEL,
            contents=_build_prompt(query, scorable),
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=_RelevanceJudgments,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
        )
        parsed = response.parsed
        if parsed is None:
            raise ValueError("Gemini returned no parsed structured output")
    except Exception as exc:  # noqa: BLE001 -- deliberate fail-closed, see docstring above
        warnings.warn(f"relevance filter failed, discarding all chunks (fail closed -> HyDE fallback): {exc}")
        return []

    scores_by_id = {judgment.chunk_id: _relevancy_score(judgment) for judgment in parsed.judgments}

    kept = []
    for chunk in scorable:
        score = scores_by_id.get(chunk["chunk_id"])
        if score is None:
            # Model omitted a judgment for this chunk_id -- treat as discarded rather than
            # guessing; this is a precision filter, so erring strict is the safer failure mode.
            continue
        if score >= THRESHOLD:
            kept.append({**chunk, "relevancy_score": score})
    return kept


if __name__ == "__main__":
    _demo_chunks = [
        {
            "chunk_id": "demo-1",
            "text": "Softmax converts logits into a probability distribution. The gradient of the softmax with respect to its inputs is computed via the Jacobian of the softmax function.",
            "heading_path": ["Softmax Regression"],
            "content_type": "prose",
            "source_path": "demo",
        },
        {
            "chunk_id": "demo-2",
            "text": "The frozen lake environment has a 4x4 grid of states. It is used to demonstrate value iteration.",
            "heading_path": ["Value Iteration"],
            "content_type": "prose",
            "source_path": "demo",
        },
    ]
    result = filter_chunks("how do you compute gradients for softmax", _demo_chunks)
    for c in result:
        print(f"{c['chunk_id']} (score={c['relevancy_score']:.2f}): {c['text'][:80]}")
