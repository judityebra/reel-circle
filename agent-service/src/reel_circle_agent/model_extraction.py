"""Model-based preference extraction (port of `extractWithModel`, `sanitizeExtraction` and
`mergeModelExtraction` in api/agent.ts).

Which model, by environment variable:
- AGENT_MODEL: `<provider>:<model>` for LangChain's `init_chat_model`, for example
  `openai:gpt-4.1-mini`, `anthropic:claude-haiku-5-5` or `google_genai:gemini-2.5-flash`.
- Without AGENT_MODEL, an OPENAI_API_KEY alone selects OpenAI with OPENAI_MODEL (default
  gpt-4.1-mini) and the optional OPENAI_BASE_URL, exactly like the TypeScript agent.
- Neither: no model; extraction is deterministic.

The model only interprets the message. Its output is bounded to the allowed genres, moods and
value ranges before it can touch the conversation's preferences. Transient failures are retried
within a time budget; any failure then falls back to the deterministic extractor.
"""

import asyncio
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from .js_compat import js_number_value, js_slice, strip_one_trailing_slash, unique
from .preferences import Preferences

ALLOWED_GENRES = [
    "Action", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller",
]  # fmt: skip
ALLOWED_MOODS = ["Gentle", "Playful", "Electric", "Reflective", "Twisty"]
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = (
    "Return one JSON object containing only movie-viewing preferences from the latest turn. Preserve relevant "
    "current preferences. Keys: genres, excludedGenres, platforms, moods, maxRuntime, yearRange, languages, "
    "peopleCount, viewingContext, clarificationQuestion. Use canonical English TMDB genre names and ISO 639-1 "
    "language codes. Use empty arrays or null for unknown values."
)


def _one_of(allowed: list[str]) -> Any:
    """A string list whose JSON schema lists the allowed values. Validation accepts any string, so an
    unexpected value is filtered out by `bound_extraction`, as in TypeScript, instead of failing the turn."""
    return Field(json_schema_extra={"items": {"type": "string", "enum": allowed}})


class MoviePreferences(BaseModel):
    """The structured output the model must return (the TypeScript JSON schema's fields)."""

    genres: Annotated[list[str], _one_of(ALLOWED_GENRES)]
    excludedGenres: Annotated[list[str], _one_of(ALLOWED_GENRES)]  # noqa: N815 - wire name
    platforms: list[str]
    moods: Annotated[list[str], _one_of(ALLOWED_MOODS)]
    maxRuntime: float | None  # noqa: N815
    yearRange: Annotated[list[float] | None, Field(json_schema_extra={"minItems": 2, "maxItems": 2})]  # noqa: N815
    languages: list[str]
    peopleCount: float | None  # noqa: N815
    viewingContext: str | None  # noqa: N815
    clarificationQuestion: str | None  # noqa: N815


@dataclass(frozen=True)
class RetryPolicy:
    """How long extraction may keep a turn waiting: attempts, backoff, and one overall time budget.

    The budget must stay below the TypeScript proxy's timeout (AGENT_SERVICE_TIMEOUT_MS, 4 s), so
    a slow model falls back here instead of the whole turn falling back to TypeScript.
    """

    attempts: int = 3
    base_delay_seconds: float = 0.25
    budget_seconds: float = 3.0


DEFAULT_RETRY_POLICY = RetryPolicy()


@dataclass(frozen=True)
class ExtractionOutcome:
    """Result of asking the model: bounded preferences, or why it could not be used."""

    status: Literal["not-configured", "failed", "succeeded"]
    extracted: dict[str, Any] | None = None
    error: str | None = None


class InvalidModelOutput(ValueError):
    """The model answered with something that isn't a preferences object."""


class ModelConfigurationError(RuntimeError):
    """The configured model could not be built (missing key, unknown provider, ...)."""


class _UnavailableModel:
    """Stands in for a model whose configuration is broken, so turns fall back instead of failing."""

    def __init__(self, reason: Exception):
        self.reason = reason

    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> "_UnavailableModel":
        return self

    async def ainvoke(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ModelConfigurationError(str(self.reason)) from self.reason


@lru_cache(maxsize=1)
def configured_chat_model() -> BaseChatModel | _UnavailableModel | None:
    """The chat model the environment selects, built once per process (clients are expensive to create)."""
    spec = os.environ.get("AGENT_MODEL")
    if not spec:
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        spec = f"openai:{os.environ.get('OPENAI_MODEL') or DEFAULT_OPENAI_MODEL}"
    provider, _, model = spec.partition(":") if ":" in spec else ("openai", "", spec)
    options: dict[str, Any] = {"max_retries": 0, "timeout": DEFAULT_RETRY_POLICY.budget_seconds}
    if provider == "openai":
        # Only OpenAI gets temperature 0: current Claude models reject custom sampling parameters.
        options["temperature"] = 0
        if os.environ.get("OPENAI_BASE_URL"):
            options["base_url"] = strip_one_trailing_slash(os.environ["OPENAI_BASE_URL"])
    try:
        return init_chat_model(model, model_provider=provider, **options)
    except Exception as error:  # noqa: BLE001 - a broken configuration must not fail every turn
        return _UnavailableModel(error)


def _string_list(value: Any, allowed: list[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    return unique(item for item in value if isinstance(item, str) and (allowed is None or item in allowed))


def _number(value: Any) -> float | None:
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    return float(value) if is_number and math.isfinite(value) else None


def bound_extraction(value: Any) -> dict[str, Any]:
    """Keeps only allowed values and sensible ranges (port of `sanitizeExtraction`)."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise InvalidModelOutput(f"expected an object, got {type(value).__name__}")
    max_runtime = _number(value.get("maxRuntime"))
    people = _number(value.get("peopleCount"))
    year_range = value.get("yearRange")
    years = [_number(year) for year in year_range] if isinstance(year_range, list) and len(year_range) == 2 else []
    context, question = value.get("viewingContext"), value.get("clarificationQuestion")
    return {
        "genres": _string_list(value.get("genres"), ALLOWED_GENRES),
        "excludedGenres": _string_list(value.get("excludedGenres"), ALLOWED_GENRES),
        "platforms": _string_list(value.get("platforms")),
        "moods": _string_list(value.get("moods"), ALLOWED_MOODS),
        "maxRuntime": js_number_value(min(600, max_runtime)) if max_runtime and max_runtime > 0 else None,
        "yearRange": [js_number_value(year) for year in years] if years and None not in years else None,
        "languages": _string_list(value.get("languages")),
        "peopleCount": max(1, min(20, math.floor(people))) if people is not None else None,
        "viewingContext": js_slice(context, 80) if isinstance(context, str) else None,
        "clarificationQuestion": js_slice(question, 240) if isinstance(question, str) else None,
    }


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
    """The HTTP status behind a provider error, looking through `response` and the `__cause__` chain."""
    seen: BaseException | None = error
    while seen is not None:
        for candidate in (seen, getattr(seen, "response", None)):
            for attribute in ("status_code", "status", "code"):
                status = getattr(candidate, attribute, None)
                if isinstance(status, int) and 100 <= status < 600:
                    return status
        if "RateLimit" in type(seen).__name__:
            return 429
        seen = seen.__cause__
    return None


def _classify(error: BaseException) -> tuple[str, bool]:
    """Why a model call failed, as a `modelError` value, and whether trying again could help."""
    if isinstance(error, ModelConfigurationError):
        return "configuration-error", False
    status = _status_code(error)
    if status is not None:
        return f"http-{status}", status in (408, 409, 429) or status >= 500
    if isinstance(error, TimeoutError) or "Timeout" in type(error).__name__:
        return "timeout", True
    if isinstance(error, ValueError):  # pydantic validation, output parsing, InvalidModelOutput
        return "invalid-json", True
    return "request-error", "Connection" in type(error).__name__


def _supports_json_mode(model: Any) -> bool:
    return getattr(model, "_llm_type", "") == "openai-chat"


async def _attempts(model: Any, prompt: list[tuple[str, str]], policy: RetryPolicy) -> ExtractionOutcome:
    structured = model.with_structured_output(MoviePreferences, method="json_schema")
    for attempt in range(1, policy.attempts + 1):
        try:
            return ExtractionOutcome("succeeded", bound_extraction(await structured.ainvoke(prompt)))
        except Exception as error:  # noqa: BLE001 - every failure is classified, then retried or reported
            reason, retryable = _classify(error)
            if reason in ("http-400", "http-422") and _supports_json_mode(model):
                # Like the TypeScript agent: an OpenAI-compatible server without JSON-schema support gets JSON mode.
                structured, retryable = model.with_structured_output(MoviePreferences, method="json_mode"), True
            if attempt == policy.attempts or not retryable:
                return ExtractionOutcome("failed", error=reason)
            await asyncio.sleep(policy.base_delay_seconds * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


async def extract_with_model(
    model: Any, message: str, current: Preferences, policy: RetryPolicy = DEFAULT_RETRY_POLICY
) -> ExtractionOutcome:
    """Asks the model to interpret the turn, within the policy's attempts and time budget."""
    if model is None:
        return ExtractionOutcome("not-configured")
    prompt = [("system", SYSTEM_PROMPT), ("user", json.dumps({"current": current, "latestTurn": message}))]
    try:
        async with asyncio.timeout(policy.budget_seconds):
            return await _attempts(model, prompt, policy)
    except TimeoutError:
        return ExtractionOutcome("failed", error="timeout")
