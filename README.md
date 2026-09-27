# Corrective-RAG Course Chatbot

A Retrieval-Augmented Generation chatbot that answers questions over [Dive into Deep
Learning](https://d2l.ai) (d2l.ai), built around the **Corrective RAG** pattern: retrieved
context is checked for relevance before it ever reaches the answer, and a single corrective
retry (HyDE reformulation) kicks in automatically when the first retrieval attempt comes up
short — rather than generating from whatever was retrieved, right or wrong.

## Why Corrective RAG

Plain RAG trusts retrieval unconditionally: whatever comes back from the vector store gets
handed straight to the LLM, even if it's off-topic or only tangentially related. This
project treats retrieval as *fallible* and adds a correction step on top of it:

```
user turn + conversation history
   -> [router] RETRIEVE (rewritten standalone query) | ANSWER_FROM_HISTORY | CHITCHAT

   -> [retrieval] hybrid dense (FAISS) + sparse (BM25) search -> Reciprocal Rank Fusion
   -> [relevance filter] statement-level LLM relevancy scoring — discards chunks that
      merely mention the right topic without actually answering the question
   -> too few relevant chunks? -> [HyDE fallback] reformulate as a hypothetical answer,
      re-embed, re-retrieve, re-filter (single retry, no infinite loops)
   -> [generation] answer synthesis, grounded ONLY in the chunks that survived filtering,
      citing which course section each point came from
```

Every stage is designed to degrade gracefully rather than crash: a failed classification,
an empty retrieval, an LLM outage at any step — the pipeline always still returns an
answer (hedged, if the underlying retrieval genuinely came up empty) instead of raising.

## Architecture

LangGraph wires the pipeline above into a stateful graph (`src/graph.py`); each node
delegates to its own module:

| Stage | Module | Notes |
|---|---|---|
| Conversation router | `src/router.py` | Classifies the turn + rewrites it into a standalone query for retrieval (the rewrite never touches conversation history or reaches generation — only the raw user message does) |
| Indexing | `src/indexing/` | Structural chunking by heading hierarchy, local embeddings, FAISS |
| Retrieval | `src/retrieval/` | Dense (FAISS cosine) + sparse (BM25) search, fused via Reciprocal Rank Fusion |
| Relevance filter | `src/relevance_filter.py` | Statement-level contextual relevancy scoring (the same metric definition DeepEval's Contextual Relevancy metric uses) |
| HyDE fallback | `src/hyde_fallback.py` | One corrective retry when retrieval comes up short |
| Generation | `src/generation.py` | Grounded answer synthesis with section citations |

## Tech stack

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **LLM**: Google Gemini (`google-genai`) — classification, relevance judgments, and
  generation all use structured/JSON-schema output rather than prompt-parsed JSON
- **Embeddings**: local `BAAI/bge-small-en-v1.5` via [`fastembed`](https://github.com/qdrant/fastembed)
  (ONNX Runtime — no PyTorch, no API calls, no daily quota)
- **Retrieval**: FAISS (`faiss-cpu`) for dense search, `rank_bm25` for sparse search
- **Validation**: Pydantic, for every structured LLM output

## Getting started

```bash
git clone <this-repo-url>
cd "RAG Chatbot"
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (https://ai.google.dev/gemini-api/docs/api-key)
```

Build the corpus and index (see [`data/README.md`](data/README.md) for what this does and
why it's a separate step — the corpus itself isn't checked into this repo):

```bash
git clone --depth 1 --branch master https://github.com/d2l-ai/d2l-en data/d2l-en
python -m src.indexing.build_index
```

Then run the chatbot:

```bash
python main.py
```

## Known limitations

- **Free-tier LLM quota**: Gemini's free tier caps `generate_content` at ~20 requests/day,
  and every conversational turn costs 2-4 of them (router, always; relevance filter +
  generation on retrieval turns; +1 more if the HyDE fallback triggers) — so the free tier
  supports roughly 5-10 real turns/day. A paid key removes this ceiling.
- **`## Exercises` sections are excluded from the index** — they're questions, not
  declarative statements, and don't fit the relevance filter's statement-level scoring well.
- **No evaluation harness yet** (`eval/`) — DeepEval-based metrics (context precision/recall,
  answer relevancy, faithfulness) aren't implemented yet.
