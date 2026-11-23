"""Evaluators for one turn of a scenario: plain functions from (expectation, response) to a score.

They know nothing about LangSmith or Langfuse; runners for those tools wrap these same functions,
so every tool scores exactly the same checks. The hard-constraint checks are written here again,
independently of the agent's own filter, so a bug in the agent can't grade itself.

A successful turn (GLOSSARY.md) is one where every evaluator passes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Expectation = dict[str, Any]
Response = dict[str, Any]

_LANGUAGE_NAMES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "ko": "korean",
    "ja": "japanese",
    "de": "german",
    "it": "italian",
}


@dataclass(frozen=True)
class Score:
    name: str
    passed: bool
    reason: str = ""


def _language(value: str) -> str:
    return _LANGUAGE_NAMES.get(value.lower(), value.lower())


def _broken_constraints(movie: dict[str, Any], preferences: dict[str, Any]) -> list[str]:
    """Which hard constraints a recommended movie breaks."""
    broken = []
    if preferences.get("maxRuntime") and movie["runtime"] > preferences["maxRuntime"]:
        broken.append("runtime")
    if set(movie["genres"]) & set(preferences.get("excludedGenres") or []):
        broken.append("excluded genre")
    year_range = preferences.get("yearRange")
    if year_range and not year_range[0] <= movie["year"] <= year_range[1]:
        broken.append("year range")
    languages = preferences.get("languages") or []
    if languages and _language(movie.get("language") or "") not in {_language(code) for code in languages}:
        broken.append("language")
    genres = preferences.get("genres") or []
    if genres and not set(movie["genres"]) & set(genres):
        broken.append("requested genres")
    platforms = preferences.get("platforms") or []
    if platforms and not set(movie["platforms"]) & set(platforms):
        broken.append("platforms")
    return broken


def evaluate_path(expect: Expectation, response: Response) -> Score:
    """The route the agent took: the expected nodes in order, the expected pause (or none), no step limit."""
    tools = [step["tool"] for step in response["meta"]["trace"]]
    if "step_limit" in tools:
        return Score("path", False, "the turn hit the step limit")
    expected_pause = expect.get("pause")
    actual_pause = (response.get("pause") or {}).get("type")
    if actual_pause != expected_pause:
        return Score("path", False, f"expected pause {expected_pause!r}, got {actual_pause!r}")
    if "path" in expect and tools != expect["path"]:
        return Score("path", False, f"expected {' > '.join(expect['path'])}, got {' > '.join(tools) or 'nothing'}")
    return Score("path", True)


def evaluate_final_answer(expect: Expectation, response: Response) -> Score:
    """The answer itself: no broken hard constraint, earlier exclusions kept, the expected results."""
    preferences = response["preferences"]
    recommendations = response["recommendations"]
    for item in recommendations:
        broken = _broken_constraints(item["movie"], preferences)
        if broken:
            return Score("final_answer", False, f"{item['movie']['id']} breaks {', '.join(broken)}")
    wrong = [key for key, value in expect.get("preferences", {}).items() if preferences.get(key) != value]
    if wrong:
        return Score("final_answer", False, f"unexpected preferences: {', '.join(wrong)}")
    missing = [genre for genre in expect.get("excludes", []) if genre not in preferences.get("excludedGenres", [])]
    if missing:
        return Score("final_answer", False, f"lost exclusions: {', '.join(missing)}")
    results = expect.get("results", {})
    ids = [item["movie"]["id"] for item in recommendations]
    if "count" in results and len(ids) != results["count"]:
        return Score("final_answer", False, f"expected {results['count']} results, got {len(ids)}")
    if "min" in results and len(ids) < results["min"]:
        return Score("final_answer", False, f"expected at least {results['min']} results, got {len(ids)}")
    if "top" in results and ids[: len(results["top"])] != results["top"]:
        return Score("final_answer", False, f"expected {results['top']} first, got {ids}")
    if "nothingFits" in expect:
        tightest = (response.get("nothingFits") or {}).get("tightestConstraint")
        if tightest != expect["nothingFits"]:
            return Score("final_answer", False, f"expected nothing fits ({expect['nothingFits']}), got {tightest!r}")
    return Score("final_answer", True)


EVALUATORS: list[Callable[[Expectation, Response], Score]] = [evaluate_path, evaluate_final_answer]


def evaluate_turn(expect: Expectation, response: Response) -> list[Score]:
    return [evaluator(expect, response) for evaluator in EVALUATORS]
