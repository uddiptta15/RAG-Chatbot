"""
LangGraph wiring for the Corrective-RAG course chatbot.

See CLAUDE.md's "Architecture summary" for the diagram this file implements. This module only
wires nodes together — all actual logic lives in the per-component modules it imports (each
component's own doc is linked from CLAUDE.md and from that module's docstring).

    route
      -> RETRIEVE            -> retrieve -> relevance_filter -> (enough)     -> generate
                                                               -> (not enough) -> hyde_fallback -> generate
      -> ANSWER_FROM_HISTORY -> generate
      -> CHITCHAT            -> generate

`history` ownership: this graph NEVER writes to `history` — main.py owns it and appends the raw
user/assistant turns itself, once per call to `graph.invoke(...)`. `standalone_query` produced by
the router is transient graph state only; it is never appended to `history` and is never passed
to `generate_node` (see CLAUDE.md, "`history` ownership rule", and generation.md, section 1).
"""

from __future__ import annotations

import warnings
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.router import classify_and_rewrite
from src.retrieval.dense import dense_retrieve
from src.retrieval.sparse import bm25_retrieve
from src.retrieval.fusion import reciprocal_rank_fusion
from src.relevance_filter import filter_chunks, MIN_RELEVANT
from src.hyde_fallback import hyde_fallback
from src.generation import generate


class GraphState(TypedDict, total=False):
    """
    Per-turn state. Owned inputs (never mutated here) vs. graph-local outputs:

    Inputs, set once by main.py's graph.invoke(...) call for this turn:
        query: str          # original user message, raw, exactly as typed
        history: list[dict] # original conversation history, raw (role/content dicts)

    Set by route_node (router.md):
        category: str            # "RETRIEVE" | "ANSWER_FROM_HISTORY" | "CHITCHAT"
        standalone_query: str     # RETRIEVE only. Transient retrieval-only rewrite -- must never
                                  # be written to `history` and must never reach generate_node.

    Set by retrieve_node (retrieval.md, section 5):
        retrieved_chunks: list[dict]   # top-N fused chunks, pre-filter, full metadata attached

    Set by relevance_filter_node / hyde_fallback_node (relevance-filter.md, hyde-fallback.md):
        relevant_chunks: list[dict]    # chunks that passed the relevancy threshold
        low_confidence_context: bool   # True only if HyDE fallback was exhausted without
                                       # reaching MIN_RELEVANT (hyde-fallback.md)

    Set by generate_node (generation.md):
        answer: str   # final response text; this is the only key main.py reads back out
    """

    query: str
    history: list[dict]

    category: str
    standalone_query: str

    retrieved_chunks: list[dict]
    relevant_chunks: list[dict]
    low_confidence_context: bool

    answer: str


def route_node(state: GraphState) -> dict:
    """Classify the turn and, on RETRIEVE, produce the transient standalone query. -> router.md"""
    result = classify_and_rewrite(query=state["query"], history=state["history"])
    return {
        "category": result["category"],
        "standalone_query": result.get("standalone_query", ""),
    }


def _route_condition(state: GraphState) -> str:
    return state["category"]


def retrieve_node(state: GraphState) -> dict:
    """Hybrid dense + BM25 retrieval, fused via RRF, over the standalone query. -> retrieval.md

    Unlike every other node here, `dense_retrieve`/`bm25_retrieve`/`reciprocal_rank_fusion`
    don't fail safely on their own (no built index, embedding API error, etc. all raise) --
    so this node catches that itself, failing to zero retrieved chunks rather than crashing
    the turn. That composes correctly with the rest of the pipeline: relevance_filter_node
    short-circuits to `[]` on empty input, which routes to hyde_fallback_node (which fails
    to a hedge on its own), which reaches generate_node (which fails to an apology on its
    own) -- so the graph still always returns an `answer`."""
    query = state["standalone_query"]
    try:
        dense_results = dense_retrieve(query)
        bm25_results = bm25_retrieve(query)
        fused_chunks = reciprocal_rank_fusion(dense_results, bm25_results)
    except Exception as exc:  # noqa: BLE001 -- deliberate, see docstring above
        warnings.warn(f"retrieval failed, returning no chunks (routes to HyDE fallback): {exc}")
        fused_chunks = []
    return {"retrieved_chunks": fused_chunks}


def relevance_filter_node(state: GraphState) -> dict:
    """Statement-level contextual relevancy filter; discards chunks below threshold.
    -> relevance-filter.md, sections 3-4"""
    relevant_chunks = filter_chunks(query=state["standalone_query"], chunks=state["retrieved_chunks"])
    return {"relevant_chunks": relevant_chunks, "low_confidence_context": False}


def _relevance_condition(state: GraphState) -> str:
    """Enough relevant chunks -> generation. Too few -> single HyDE retry.
    -> relevance-filter.md, section 4"""
    if len(state.get("relevant_chunks", [])) >= MIN_RELEVANT:
        return "enough"
    return "not_enough"


def hyde_fallback_node(state: GraphState) -> dict:
    """Single-retry HyDE reformulation -> re-fuse -> re-filter. Runs at most once per query.
    -> hyde-fallback.md"""
    result = hyde_fallback(query=state["standalone_query"])
    return {
        "relevant_chunks": result["relevant_chunks"],
        "low_confidence_context": result["low_confidence_context"],
    }


def generate_node(state: GraphState) -> dict:
    """Answer synthesis. Always uses the ORIGINAL history + ORIGINAL query, never
    `standalone_query`. -> generation.md, section 1"""
    answer = generate(
        category=state["category"],
        history=state["history"],
        query=state["query"],
        relevant_chunks=state.get("relevant_chunks", []),
        low_confidence_context=state.get("low_confidence_context", False),
    )
    return {"answer": answer}


def build_graph():
    """Compiles the graph described in CLAUDE.md's Architecture summary. Called once by main.py."""
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("relevance_filter", relevance_filter_node)
    graph.add_node("hyde_fallback", hyde_fallback_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        _route_condition,
        {
            "RETRIEVE": "retrieve",
            "ANSWER_FROM_HISTORY": "generate",
            "CHITCHAT": "generate",
        },
    )

    graph.add_edge("retrieve", "relevance_filter")

    graph.add_conditional_edges(
        "relevance_filter",
        _relevance_condition,
        {
            "enough": "generate",
            "not_enough": "hyde_fallback",
        },
    )

    graph.add_edge("hyde_fallback", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
