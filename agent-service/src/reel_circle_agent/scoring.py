"""Hard constraints, group scoring and explanations (port of src/lib/recommendation.ts).

Movies and persons stay as the dictionaries the browser sent, so every recommended movie is echoed
back exactly as it arrived.
"""

import math
import re
from collections.abc import Callable
from typing import Any, Literal

from .js_compat import js_round, js_str, js_sum, js_to_fixed, strip_accents_lower
from .preferences import Preferences

Movie = dict[str, Any]
Person = dict[str, Any]
Recommendation = dict[str, Any]
RankingMode = Literal["balanced", "average", "least-misery", "nash"]

_LANGUAGE_ALIASES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "it": "italian",
}


def _language_name(value: str) -> str:
    """An ISO 639-1 code or a language name, as a lower-case language name."""
    return _LANGUAGE_ALIASES.get(value.lower(), value.lower())


def _meets_max_runtime(movie: Movie, preferences: Preferences) -> bool:
    limit = preferences.get("maxRuntime")
    return not limit or movie["runtime"] <= limit


def _meets_excluded_genres(movie: Movie, preferences: Preferences) -> bool:
    return not any(genre in movie["genres"] for genre in preferences.get("excludedGenres") or [])


def _meets_year_range(movie: Movie, preferences: Preferences) -> bool:
    year_range = preferences.get("yearRange")
    return not year_range or year_range[0] <= movie["year"] <= year_range[1]


def _meets_languages(movie: Movie, preferences: Preferences) -> bool:
    languages = preferences.get("languages") or []
    language = movie.get("language")
    return not languages or bool(
        language and any(_language_name(wanted) == _language_name(language) for wanted in languages)
    )


def _meets_genres(movie: Movie, preferences: Preferences) -> bool:
    genres = preferences.get("genres") or []
    return not genres or any(genre in movie["genres"] for genre in genres)


def _meets_platforms(movie: Movie, preferences: Preferences) -> bool:
    platforms = preferences.get("platforms") or []
    return not platforms or any(platform in movie["platforms"] for platform in platforms)


#: The hard constraints, in the order the filter applies them (see GLOSSARY.md).
HARD_CONSTRAINTS: dict[str, Callable[[Movie, Preferences], bool]] = {
    "maxRuntime": _meets_max_runtime,
    "excludedGenres": _meets_excluded_genres,
    "yearRange": _meets_year_range,
    "languages": _meets_languages,
    "genres": _meets_genres,
    "platforms": _meets_platforms,
}


def broken_constraints(movie: Movie, preferences: Preferences) -> list[str]:
    """The hard constraints a movie fails, by name."""
    return [name for name, meets in HARD_CONSTRAINTS.items() if not meets(movie, preferences)]


def _clamp(value: float) -> int:
    return max(0, min(100, js_round(value)))


def _film_key(film: dict[str, Any]) -> str:
    if film.get("tmdbId"):
        return f"tmdb:{js_str(film['tmdbId'])}"
    title = re.sub(r"[^a-z0-9]+", " ", strip_accents_lower(film["title"])).strip()
    year = film.get("year")
    return f"{title}::{'' if year is None else js_str(year)}"


def movie_features(movie: Movie) -> list[str]:
    return [
        *([f"director:{movie['director']}"] if movie.get("director") else []),
        *([f"language:{movie['language']}"] if movie.get("language") else []),
        *(f"mood:{mood}" for mood in movie.get("moods") or []),
        f"decade:{math.floor(movie['year'] / 10) * 10}s",
    ]


def aggregate_group_score(scores: list[int], mode: RankingMode) -> int:
    if not scores:
        return 0
    if mode == "least-misery":
        return _clamp(min(scores))
    if mode == "nash":
        product = 1.0
        for score in scores:
            product *= max(score / 100, 0.01)
        return _clamp(math.pow(product, 1 / len(scores)) * 100)
    average = js_sum(scores) / len(scores)
    if mode == "average":
        return _clamp(average)
    disagreement = max(scores) - min(scores) if len(scores) > 1 else 0
    return _clamp(average - disagreement * 0.2)


def _person_score(person: Person, movie: Movie, mood_matches: int, preferences: Preferences) -> dict[str, Any]:
    genre_matches = sum(1 for genre in movie["genres"] if genre in person.get("preferredGenres", []))
    taste = person.get("tasteWeights") or {}
    features = person.get("featureWeights") or {}
    learned = js_sum(taste.get(genre, 0) for genre in movie["genres"])
    feature_preference = js_sum(features.get(feature, 0) for feature in movie_features(movie))
    explainable = _clamp(
        movie["rating"] * 14
        + genre_matches * 14
        + mood_matches * 20
        + learned * 16
        + feature_preference * 6
        + (8 if movie["runtime"] <= 100 else 0)
    )
    neural = (preferences.get("neuralScores") or {}).get(person["id"], {}).get(movie["id"])
    blend = max(0, min(1, preferences.get("neuralBlend") or 0))
    score = explainable if neural is None else _clamp(explainable * (1 - blend) + neural * blend)
    return {"profileId": person["id"], "score": score}


def score_candidates(candidates: list[Movie], persons: list[Person], preferences: Preferences) -> list[Recommendation]:
    """Movies nobody has watched that meet every hard constraint, ranked best first for the group."""
    watched = {_film_key(film) for person in persons for film in person.get("watched") or []}
    requested_moods = preferences.get("moods") or []
    recommendations = []
    for movie in candidates:
        if _film_key(movie) in watched or broken_constraints(movie, preferences):
            continue
        moods = [mood for mood in requested_moods if mood in (movie.get("moods") or [])]
        match_by_profile = [_person_score(person, movie, len(moods), preferences) for person in persons]
        score = aggregate_group_score(
            [match["score"] for match in match_by_profile], preferences.get("rankingMode") or "balanced"
        )
        shared = [
            genre for genre in movie["genres"] if all(genre in person.get("preferredGenres", []) for person in persons)
        ]
        reasons = [
            *([f"Mood match: {', '.join(moods)}"] if moods else []),
            f"Shared taste: {', '.join(shared)}" if shared else f"{movie['genres'][0]} bridges this group",
            f"Quick {js_str(movie['runtime'])}-minute watch"
            if movie["runtime"] <= 100
            else f"{js_str(movie['runtime'])} minutes",
            f"{js_to_fixed(movie['rating'], 1)} community rating",
            *(
                ["Neural taste model included"]
                if preferences.get("neuralScores") and preferences.get("neuralBlend")
                else []
            ),
        ]
        recommendations.append({"movie": movie, "score": score, "matchByProfile": match_by_profile, "reasons": reasons})
    # `sorted` is stable, like Array.prototype.sort, so ties keep the candidate pool's order.
    return sorted(recommendations, key=lambda recommendation: -recommendation["score"])


def explain_preference_match(movie: Movie, preferences: Preferences) -> str:
    genres = [genre for genre in preferences.get("genres") or [] if genre in movie["genres"]]
    moods = [mood for mood in preferences.get("moods") or [] if mood in (movie.get("moods") or [])]
    year_range = preferences.get("yearRange")
    details = [
        *([f"your {' and '.join(genres).lower()} preference"] if genres else []),
        *([f"the {' and '.join(moods).lower()} mood"] if moods else []),
        *([f"your under-{js_str(preferences['maxRuntime'])}-minute limit"] if preferences.get("maxRuntime") else []),
        *([f"your {js_str(year_range[0])}-{js_str(year_range[1])} release window"] if year_range else []),
        *(
            [f"your {movie['language'].upper()} language choice"]
            if preferences.get("languages") and movie.get("language")
            else []
        ),
    ]
    if not details:
        return f"{movie['title']} rises to the top from the group's ratings and learned taste signals."
    excluded = preferences.get("excludedGenres") or []
    exclusions = f" It also avoids {' and '.join(excluded).lower()}." if excluded else ""
    return f"{movie['title']} matches {', '.join(details)}.{exclusions}"
