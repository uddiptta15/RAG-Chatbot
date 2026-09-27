"""
Conversation router: classify + rewrite, in one LLM call. -> component docs/router.md

Decides, per turn, whether the query needs the full retrieval pipeline (RETRIEVE, with a
transient rewritten standalone_query), is answerable from conversation history alone
(ANSWER_FROM_HISTORY), or is chitchat/meta (CHITCHAT). -> router.md, section 2.

`standalone_query` is retrieval-only and transient: src/graph.py never writes it to
`history` and src/generation.py never sees it -- see CLAUDE.md, "history ownership rule".
"""

from __future__ import annotations

import warnings
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel

from src.llm_client import call_with_retry, get_client

# router.md, open questions: "whether the router needs to be near-instant ... since it
# runs on every single turn, unlike the relevance filter" -- resolved here in favor of a
# fast/cheap tier model rather than the app's main generation model.
ROUTER_MODEL = "gemini-flash-latest"

# router.md, section 4 + open questions: "how many history turns (k) to feed the router
# -- start with last 3-5 turns"
HISTORY_TURNS = 5

_MAX_OUTPUT_TOKENS = 512  # classification + a short rewrite; output is small JSON


class RouterDecision(BaseModel):
    category: Literal["RETRIEVE", "ANSWER_FROM_HISTORY", "CHITCHAT"]
    standalone_query: Optional[str] = None


_SYSTEM_PROMPT = """You are routing a message in a course-assistant chatbot conversation.

Decide which category the CURRENT user message belongs to:
- RETRIEVE: a question about course content (may depend on prior turns for context)
- ANSWER_FROM_HISTORY: answerable from the conversation itself, not course-content-specific
- CHITCHAT: greeting, small talk, or a meta question about the assistant

If RETRIEVE, also rewrite the message into a standalone question that makes sense without
needing the conversation history (resolve pronouns/references using the history).

Rewrite rules:
- Resolve pronouns/references (it, that, the previous one) using the history -- nothing else.
- Do NOT add any information, assumption, or detail not present in the history or the
  current message.
- Do NOT paraphrase or substitute synonyms for technical terms -- keep exact terminology as
  the user wrote it (this matters for downstream keyword-based retrieval).
- If the message is already standalone, return it UNCHANGED -- do not rewrite for its own sake.
- Only set standalone_query when category is RETRIEVE; omit it otherwise.

If the categories genuinely seem to overlap or you are unsure, default to RETRIEVE --
under-retrieving is worse than an unnecessary retrieval."""


def _format_history(history: list[dict], turns: int = HISTORY_TURNS) -> str:
    if not history:
        return "(no prior turns)"
    windowed = history[-turns * 2 :]
    return "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in windowed)


def _safe_default(query: str) -> dict:
    """router.md, section 6: on low-confidence/ambiguous classification, default to
    RETRIEVE. Extended here to cover outright failures too (API error, malformed/missing
    structured output) -- for the same reason: under-retrieving is worse than an
    unnecessary retrieval, and this keeps the graph moving instead of crashing the turn."""
    return {"category": "RETRIEVE", "standalone_query": query}


def classify_and_rewrite(query: str, history: list[dict]) -> dict:
    """-> router.md, section 3. Returns `{"category": ...}`, or on RETRIEVE,
    `{"category": "RETRIEVE", "standalone_query": ...}`. Never mutates `history` --
    matches CLAUDE.md's ownership rule (only main.py appends to it)."""
    user_prompt = (
        f"Conversation history (last {HISTORY_TURNS} turns):\n{_format_history(history)}\n\n"
        f"Current user message: {query}"
    )

    try:
        response = call_with_retry(
            get_client().models.generate_content,
            model=ROUTER_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=RouterDecision,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
        )
        decision = response.parsed
        if decision is None:
            raise ValueError("Gemini returned no parsed structured output")
    except Exception as exc:  # noqa: BLE001 -- deliberate: any failure falls back safely
        warnings.warn(f"router classification failed, defaulting to RETRIEVE: {exc}")
        return _safe_default(query)

    if decision.category == "RETRIEVE":
        return {"category": "RETRIEVE", "standalone_query": decision.standalone_query or query}
    return {"category": decision.category}


if __name__ == "__main__":
    import sys

    _history: list[dict] = []
    print("Router smoke test. Type 'exit' to quit.")
    while True:
        text = input("\nYou: ")
        if text.lower() == "exit":
            break
        result = classify_and_rewrite(text, _history)
        print(result)
        _history.append({"role": "user", "content": text})
        _history.append({"role": "assistant", "content": f"[{result['category']}]"})
