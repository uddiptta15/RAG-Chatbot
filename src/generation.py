"""
Generation: prompt assembly (raw history/query) + LLM call. -> component docs/generation.md

Always receives the ORIGINAL `history` and ORIGINAL user `query` -- never the router's
standalone_query. That rewrite is retrieval-only and transient; showing it here would risk
answering a slightly different question than the user actually asked, in a tone that drifts
from the conversation (section 1). See also CLAUDE.md, "history ownership rule".
"""

from __future__ import annotations

import warnings

from google.genai import types

from src.llm_client import call_with_retry, get_client

# Same lightweight tier as every other component in this app (router.py, relevance_filter.py,
# hyde_fallback.py) -- keeps quota usage uniform on the free Gemini API tier rather than
# reaching for a heavier model only here.
GENERATION_MODEL = "gemini-flash-latest"

_MAX_OUTPUT_TOKENS = 2048

_RETRIEVE_SYSTEM = """You are a course assistant. Answer using ONLY the provided context below.
If the context doesn't fully answer the question, say so rather than filling gaps from
general knowledge.

Reference which section an answer point came from where natural, using its heading path
(e.g. "As covered in Linear Regression > Vectorization for Speed, ...")."""

# -> section 5: appended when hyde_fallback.py was exhausted (low_confidence_context: true).
_HEDGE_NOTE = """
Note: the retrieved context may not fully cover this question. If it doesn't, say so
clearly rather than presenting a partial or uncertain answer as complete."""

_HISTORY_SYSTEM = """You are a course assistant. Answer using the conversation history.
Do not fabricate course content that wasn't already discussed."""

_CHITCHAT_SYSTEM = "You are a friendly course assistant. Respond briefly and naturally."


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{turn['role'].capitalize()}: {turn['content']}" for turn in history)


def _format_context(relevant_chunks: list[dict]) -> str:
    """-> section 4, citing sources: each chunk's heading_path rides alongside its text so
    the model can name the section an answer point came from."""
    return "\n".join(f"[{' > '.join(c['heading_path'])}] {c['text']}" for c in relevant_chunks)


def _build_retrieve_prompt(
    history: list[dict], query: str, relevant_chunks: list[dict], low_confidence_context: bool
) -> tuple[str, str]:
    system = _RETRIEVE_SYSTEM + (_HEDGE_NOTE if low_confidence_context else "")
    user = (
        f"Conversation history:\n{_format_history(history)}\n\n"
        f"Retrieved context:\n{_format_context(relevant_chunks)}\n\n"
        f"Current user message: {query}"
    )
    return system, user


def _build_history_prompt(history: list[dict], query: str) -> tuple[str, str]:
    user = f"Conversation history:\n{_format_history(history)}\n\nCurrent user message: {query}"
    return _HISTORY_SYSTEM, user


def _build_chitchat_prompt(query: str) -> tuple[str, str]:
    # No history block -- router.md, section 5: "little or no history needed here".
    return _CHITCHAT_SYSTEM, f"Current user message: {query}"


def _fallback_message(category: str, exc: Exception) -> str:
    """This is the graph's terminal node -- there's nowhere further to fail toward, so on
    total failure it must still return SOME user-facing string rather than raising and
    crashing main.py's CLI loop."""
    warnings.warn(f"generation failed for category={category}: {exc}")
    return "Sorry, I ran into a problem generating a response. Please try again."


def generate(
    category: str,
    history: list[dict],
    query: str,
    relevant_chunks: list[dict] | None = None,
    low_confidence_context: bool = False,
) -> str:
    """-> generation.md, section 2. `query`/`history` are always the ORIGINAL user message
    and conversation history -- see module docstring. `relevant_chunks`/
    `low_confidence_context` are only meaningful on the `RETRIEVE` path."""
    relevant_chunks = relevant_chunks or []

    if category == "RETRIEVE":
        system, user = _build_retrieve_prompt(history, query, relevant_chunks, low_confidence_context)
    elif category == "ANSWER_FROM_HISTORY":
        system, user = _build_history_prompt(history, query)
    elif category == "CHITCHAT":
        system, user = _build_chitchat_prompt(query)
    else:
        raise ValueError(f"unknown category: {category!r}")

    try:
        response = call_with_retry(
            get_client().models.generate_content,
            model=GENERATION_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=_MAX_OUTPUT_TOKENS,
            ),
        )
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001 -- deliberate, see _fallback_message
        return _fallback_message(category, exc)


if __name__ == "__main__":
    history: list[dict] = []
    print("Generation smoke test (CHITCHAT/ANSWER_FROM_HISTORY only -- RETRIEVE needs a built index). Type 'exit' to quit.")
    while True:
        text = input("\nYou: ")
        if text.lower() == "exit":
            break
        category = "CHITCHAT" if len(history) == 0 else "ANSWER_FROM_HISTORY"
        answer = generate(category, history, text)
        print(f"[{category}] Assistant: {answer}")
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer})
