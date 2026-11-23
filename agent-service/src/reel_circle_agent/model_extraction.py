"""Model-based preference extraction (port of `extractWithModel` and `mergeModelExtraction` in api/agent.ts).

Which model, by environment variable:
- AGENT_MODEL: `<provider>:<model>` for LangChain's `init_chat_model`, for example
  `openai:gpt-4.1-mini`, `anthropic:claude-haiku-5-5` or `google_genai:gemini-2.5-flash`.
- Without AGENT_MODEL, an OPENAI_API_KEY alone selects OpenAI with OPENAI_MODEL (default
  gpt-4.1-mini) and the optional OPENAI_BASE_URL, exactly like the TypeScript agent.
- Neither: no model; extraction is deterministic.

The model only interprets the message. Its output is bounded to the allowed genres, moods and
value ranges before it can touch the conversation's preferences, and any failure (after a bounded
number of retries for transient errors) falls back to the deterministic extractor.
"""

import asyncio
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from .js_compat import strip_one_trailing_slash, unique
from .preferences import Preferences

ALLOWED_GENRES = [
    "Action",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
]
ALLOWED_MOODS = ["Gentle", "Playful", "Electric", "Reflective", "Twisty"]
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
MAX_ATTEMPTS = 3
REQUEST_TIMEOUT_SECONDS = 15

SYSTEM_PROMPT = (
    "Return one JSON object containing only movie-viewing preferences from the latest turn. Preserve relevant "
    "current preferences. Keys: genres, excludedGenres, platforms, moods, maxRuntime, yearRange, languages, "
    "peopleCount, viewingContext, clarificationQuestion. Use canonical English TMDB genre names and ISO 639-1 "
    "language codes. Use empty arrays or null for unknown values."
)

Genre = Literal[
    "Action",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
]
Mood = Literal["Gentle", "Playful", "Electric", "Reflective", "Twisty"]


class MoviePreferences(BaseModel):
    """The structured output the model must return (same fields and enums as the TypeScript JSON schema)."""

    genres: list[Genre]
    excludedGenres: list[Genre]  # noqa: N815 - wire name
    platforms: list[str]
    moods: list[Mood]
    maxRuntime: float | None  # noqa: N815
    yearRange: list[float] | None  # noqa: N815
    languages: list[str]
    peopleCount: float | None  # noqa: N815
    viewingContext: str | None  # noqa: N815
    clarificationQuestion: str | None  # noqa: N815


@dataclass(frozen=True)
class ModelExtraction:
    """Outcome of asking the model: bounded preferences, or why it failed."""

    status: Literal["not-configured", "failed", "succeeded"]
    extracted: dict[str, Any] | None = None
    error: str | None = None


class InvalidModelOutput(ValueError):
    """The model answered with something that isn't a preferences object."""


@lru_cache(maxsize=1)
def configured_chat_model() -> BaseChatModel | None:
    """The chat model the environment selects, built once (clients are expensive to create)."""
    spec = os.environ.get("AGENT_MODEL")
    options: dict[str, Any] = {"timeout": REQUEST_TIMEOUT_SECONDS, "max_retries": 0}
    if not spec:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        spec = f"openai:{os.environ.get('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL}"
    provider, _, model = spec.partition(":") if ":" in spec else ("", "", spec)
    if provider in ("", "openai"):
        # Only OpenAI gets temperature 0: current Claude models reject custom sampling parameters.
        options["temperature"] = 0
        if os.environ.get("OPENAI_BASE_URL"):
            options["base_url"] = strip_one_trailing_slash(os.environ["OPENAI_BASE_URL"])
    return init_chat_model(model, model_provider=provider or "openai", **options)


def _string_list(value: Any, allowed: list[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    return unique(item for item in value if isinstance(item, str) and (allowed is None or item in allowed))


def _number(value: Any) -> float | None:
    return (
        float(value)
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def bound_extraction(value: Any) -> dict[str, Any]:
    """Keeps only allowed values and sensible ranges (port of `sanitizeExtraction`)."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise InvalidModelOutput(f"expected an object, got {type(value).__name__}")
    max_runtime = _number(value.get("maxRuntime"))
    people = _number(value.get("peopleCount"))
    year_range = value.get("yearRange")
    years = [_number(year) for year in year_range] if isinstance(year_range, list) and len(year_range) == 2 else None
    extracted = {
        "genres": _string_list(value.get("genres"), ALLOWED_GENRES),
        "excludedGenres": _string_list(value.get("excludedGenres"), ALLOWED_GENRES),
        "platforms": _string_list(value.get("platforms")),
        "moods": _string_list(value.get("moods"), ALLOWED_MOODS),
        "maxRuntime": min(600, max_runtime) if max_runtime and max_runtime > 0 else None,
        "yearRange": years if years and all(year is not None for year in years) else None,
        "languages": _string_list(value.get("languages")),
        "peopleCount": max(1, min(20, math.floor(people))) if people is not None else None,
        "viewingContext": value["viewingContext"][:80] if isinstance(value.get("viewingContext"), str) else None,
        "clarificationQuestion": (
            value["clarificationQuestion"][:240] if isinstance(value.get("clarificationQuestion"), str) else None
        ),
    }
    for key in ("maxRuntime", "peopleCount"):
        if isinstance(extracted[key], float) and extracted[key].is_integer():
            extracted[key] = int(extracted[key])
    if extracted["yearRange"]:
        extracted["yearRange"] = [int(year) if year.is_integer() else year for year in extracted["yearRange"]]
    return extracted


def merge_model_extraction(current: Preferences, extracted: dict[str, Any], now: int) -> Preferences:
    """Empty lists and missing values keep the current preferences (port of `mergeModelExtraction`)."""
    merged = {**current}
    for key in ("genres", "excludedGenres", "platforms", "moods", "languages"):
        merged[key] = extracted[key] if extracted[key] else current.get(key)
    for key in ("maxRuntime", "yearRange", "peopleCount", "viewingContext"):
        merged[key] = extracted[key] if extracted[key] is not None else current.get(key)
    merged["lastUpdatedAt"] = now
    return {key: value for key, value in merged.items() if value is not None}


def _status_code(error: BaseException) -> int | None:
    for candidate in (error, getattr(error, "response", None)):
        status = getattr(candidate, "status_code", None) or getattr(candidate, "status", None)
        if isinstance(status, int):
            return status
    return None


def _failure_reason(error: BaseException) -> str:
    status = _status_code(error)
    if status is not None:
        return f"http-{status}"
    if isinstance(error, TimeoutError) or "Timeout" in type(error).__name__:
        return "timeout"
    if isinstance(error, ValueError):  # pydantic validation, output parsing, InvalidModelOutput
        return "invalid-output"
    return "request-error"


def _is_retryable(error: BaseException) -> bool:
    status = _status_code(error)
    if status is not None:
        return status in (408, 409, 429) or status >= 500
    return _failure_reason(error) in ("timeout", "invalid-output") or "Connection" in type(error).__name__


async def extract_with_model(model: BaseChatModel | None, message: str, current: Preferences) -> ModelExtraction:
    """Asks the model to interpret the turn, retrying transient failures up to MAX_ATTEMPTS times."""
    if model is None:
        return ModelExtraction("not-configured")
    structured = model.with_structured_output(MoviePreferences, method="json_schema")
    prompt = [("system", SYSTEM_PROMPT), ("user", json.dumps({"current": current, "latestTurn": message}))]
    delay = float(os.environ.get("AGENT_MODEL_RETRY_DELAY_SECONDS", "0.5"))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return ModelExtraction("succeeded", bound_extraction(await structured.ainvoke(prompt)))
        except Exception as error:  # noqa: BLE001 - every failure is classified, then retried or reported
            if attempt == MAX_ATTEMPTS or not _is_retryable(error):
                return ModelExtraction("failed", error=_failure_reason(error))
            await asyncio.sleep(delay * 2 ** (attempt - 1))
    raise AssertionError("unreachable")
