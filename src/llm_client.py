"""
Shared Gemini client for every LLM-calling component (src/router.py, relevance_filter.py,
hyde_fallback.py, generation.py). Centralizes .env loading and client construction so each
component doesn't repeat the same boilerplate.

Per-component model choice, prompts, and structured-output schemas stay local to each
module (each has its own doc under component docs/) -- this file only owns the one thing
they all share: how to get a configured `genai.Client`.

**Free-tier quota, confirmed live against this project's key (not documentation --
observed directly in production error responses)**:
  - `embed_content`: ~1000/day (moot now -- src/indexing/embedder.py moved to a local
    model; see indexing.md, section 3).
  - `generate_content` (every LLM call in this app): **only 20/day**, on whatever model
    `gemini-flash-latest` currently resolves to. Every user turn costs 2-4 of these
    (router always; +relevance_filter +generation on RETRIEVE; +1 more if HyDE triggers)
    -- so the free tier supports roughly 5-10 real conversation turns per day, total,
    across all testing and use. This is a known, accepted limitation for now (not a bug to
    fix here) -- see router.md's open question about a rule-based CHITCHAT shortcut for a
    partial mitigation that hasn't been implemented.
"""

from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Lazy singleton so importing a component module doesn't require GEMINI_API_KEY to
    already be set (e.g. before the user has filled in .env)."""
    global _client
    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY from the environment
    return _client


_T = TypeVar("_T")

# Observed in practice (see indexing.md's open questions):
# - gemini-flash-latest intermittently returns 503 UNAVAILABLE ("This model is currently
#   experiencing high demand") -- Google's own overload signal, not a bug in this app, and
#   it often clears within a few seconds.
# - The free tier's embed_content quota is 100 requests/minute; exceeding it returns 429
#   RESOURCE_EXHAUSTED with an explicit "retry in Ns" -- tens of seconds, not a few, so
#   this needs a real wait, not a short exponential backoff.
# Retrying (with the right wait for each case) meaningfully reduces how often every
# component's own safe-default/fail-closed fallback fires for a purely transient reason.
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 2.0       # 503 / other transient errors: short exponential backoff
DEFAULT_RATE_LIMIT_DELAY = 60.0  # 429 fallback if the server didn't say how long to wait

_RETRY_DELAY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*s")


def _rate_limit_wait_seconds(exc: errors.ClientError) -> float:
    """Parse Google's own suggested wait out of a 429's `RetryInfo` detail (e.g.
    `{"retryDelay": "52s"}`) rather than guessing -- falls back to `DEFAULT_RATE_LIMIT_DELAY`
    if the response shape doesn't have it. A couple of seconds of slack is added since the
    quota window resets on the server's clock, not ours."""
    details = getattr(exc, "details", None) or {}
    error_details = details.get("error", details).get("details", []) if isinstance(details, dict) else []
    for entry in error_details:
        retry_delay = entry.get("retryDelay") if isinstance(entry, dict) else None
        if retry_delay:
            match = _RETRY_DELAY_RE.match(str(retry_delay))
            if match:
                return float(match.group(1)) + 2.0
    return DEFAULT_RATE_LIMIT_DELAY


def call_with_retry(fn: Callable[..., _T], *args, **kwargs) -> _T:
    """Retries `fn(*args, **kwargs)` on transient errors -- 5xx (`ServerError`) with short
    exponential backoff, or 429 rate-limit (`ClientError`) waiting however long the server
    actually asked for. Re-raises immediately on any other error (bad request, auth, etc.),
    since retrying wouldn't help those. Each component's own try/except around its
    `call_with_retry(...)` call is still the last line of defense if every retry is
    exhausted."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except errors.ServerError as exc:
            last_exc = exc
            wait = BASE_DELAY_SECONDS * (2**attempt)
        except errors.ClientError as exc:
            if exc.code != 429:
                raise
            last_exc = exc
            wait = _rate_limit_wait_seconds(exc)
        if attempt < MAX_RETRIES - 1:
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc
