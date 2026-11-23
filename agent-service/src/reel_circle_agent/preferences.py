"""Deterministic extraction of conversation preferences from one turn (port of src/lib/constraints.ts).

Preferences travel as plain dictionaries with the browser's camelCase keys, so a turn's input and
output keep the exact wire shape the React app and the parity fixtures use.
"""

import re
from typing import Any

from .js_compat import js_round, strip_accents_lower, unique

Preferences = dict[str, Any]

# Every pattern is ASCII-only, like JavaScript's `\b`, and runs on accent-stripped lower-case text.
_A = re.ASCII

GENRE_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("Comedy", re.compile(r"\b(comedy|comic|funny|comedia|divertid[oa]|gracios[oa]|romcom)\b", _A)),
    ("Drama", re.compile(r"\b(drama|dramatic[oa])\b", _A)),
    ("Mystery", re.compile(r"\b(mystery|misterio|intriga)\b", _A)),
    ("Romance", re.compile(r"\b(romance|romantic|romantico|romantica|romcom)\b", _A)),
    ("Sci-Fi", re.compile(r"\b(sci[ -]?fi|science fiction|ciencia ficcion)\b", _A)),
    ("Horror", re.compile(r"\b(horror|scary|terror|miedo|spooky)\b", _A)),
    ("Thriller", re.compile(r"\b(thriller|suspense|tension)\b", _A)),
    ("Action", re.compile(r"\b(action|accion|adrenaline)\b", _A)),
    ("Fantasy", re.compile(r"\b(fantasy|fantasia|magical|magica)\b", _A)),
    ("Animation", re.compile(r"\b(animation|animated|animacion|anime)\b", _A)),
    ("Documentary", re.compile(r"\b(documentary|documental)\b", _A)),
    ("Crime", re.compile(r"\b(crime|criminal|crimen)\b", _A)),
]

MOOD_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("Gentle", re.compile(r"\b(gentle|cozy|calm|suave|tranquil[oa]|acogedor[a]?)\b", _A)),
    (
        "Playful",
        re.compile(r"\b(playful|witty|jugueton[a]?|ingenios[oa]|camp|campy|cunt|cunty|slay|girly|iconic)\b", _A),
    ),
    ("Electric", re.compile(r"\b(electric|intense|intens[oa]|energetic[oa])\b", _A)),
    ("Reflective", re.compile(r"\b(reflective|thoughtful|reflexiv[oa]|contemplativ[oa])\b", _A)),
    ("Twisty", re.compile(r"\b(twisty|mind.?bending|retorcid[oa]|sorprendente)\b", _A)),
]

PLATFORM_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("Netflix", re.compile(r"\bnetflix\b", _A)),
    ("Prime Video", re.compile(r"\b(prime video|amazon prime)\b", _A)),
    ("Movistar Plus+", re.compile(r"\bmovistar(?: plus|\+)?\b", _A)),
    ("Filmin", re.compile(r"\bfilmin\b", _A)),
    ("MUBI", re.compile(r"\bmubi\b", _A)),
    ("Max", re.compile(r"\b(?:hbo )?max\b", _A)),
    ("Disney+", re.compile(r"\bdisney(?: plus|\+)?\b", _A)),
    ("SkyShowtime", re.compile(r"\bskyshowtime\b", _A)),
]

LANGUAGE_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("en", re.compile(r"\b(?:english|ingles)\b", _A)),
    ("es", re.compile(r"\b(?:spanish|espanol)\b", _A)),
    ("fr", re.compile(r"\b(?:french|frances)\b", _A)),
    ("ko", re.compile(r"\b(?:korean|coreano)\b", _A)),
    ("ja", re.compile(r"\b(?:japanese|japones)\b", _A)),
]

_NEGATION = re.compile(
    r"(?:\bno\b|\bnot\b|without|avoid|exclude|dont want|do not want|nada de|sin)\s*(?:anything\s+)?$", _A
)
_NOT_FUTURISTIC = re.compile(
    r"\b(?:not futuristic|nothing futuristic|dont want anything futuristic|no futuristic)\b", _A
)
_RESET = re.compile(
    r"\b(start over|reset|clear (?:my )?preferences|forget that|empezar de nuevo|borrar preferencias)\b", _A
)
_LESS_INTENSE = re.compile(r"\b(less intense|not (?:too )?intense|menos intens[oa])\b", _A)
_PEOPLE = re.compile(r"\b(?:with|for|somos|con)\s+(\d{1,2})\s+(?:friends?|people|personas?|amigos?)\b", _A)
_MOVIE_NIGHT_WITH = re.compile(r"\bmovie night with (\d{1,2})\b", _A)
_SINCE_YEAR = re.compile(r"\b(?:from|after|since|desde|posterior a)\s+(19\d{2}|20\d{2})\b", _A)
_DECADE = re.compile(r"\b((?:19|20)\d0)s\b", _A)
_DARK = re.compile(r"\b(dark|darker|mysterious|oscura?|misterios[oa])\b", _A)
_DATE_NIGHT = re.compile(r"\b(date night|with my (?:boyfriend|girlfriend|partner)|cita)\b", _A)
_MOVIE_NIGHT = re.compile(r"\b(movie night|noche de cine)\b", _A)
_HOURS = re.compile(r"(?:under|less than|menos de|maximo)\s*(one|una?|two|dos|\d+(?:\.5)?)\s*(?:hours?|horas?)", _A)
_HOUR_AND_HALF = re.compile(r"\b(hora y media|an hour and a half)\b", _A)
_MINUTES = re.compile(r"(?:under|less than|menos de|maximo)\s*(\d{2,3})", _A)
_SHORT = re.compile(r"\b(short|quick|corta|rapida)\b", _A)
_FAIR = re.compile(r"\b(fair|everyone|nadie odie|para todos)\b", _A)
_BALANCED = re.compile(r"\b(consensus|balanced|equilibrad[oa])\b", _A)
_HOUR_WORDS = {"one": 1, "un": 1, "una": 1, "two": 2, "dos": 2}


def create_conversation_preferences(now: int) -> Preferences:
    return {"genres": [], "excludedGenres": [], "platforms": [], "moods": [], "lastUpdatedAt": now}


def _is_negated(text: str, match_index: int) -> bool:
    return bool(_NEGATION.search(text[max(0, match_index - 28) : match_index]))


def _genre_signals(text: str) -> tuple[list[str], list[str]]:
    included: list[str] = []
    excluded: list[str] = []
    for genre, pattern in GENRE_TERMS:
        match = pattern.search(text)
        if match:
            (excluded if _is_negated(text, match.start()) else included).append(genre)
    if _NOT_FUTURISTIC.search(text):
        excluded.append("Sci-Fi")
    return unique(included), unique(excluded)


def parse_constraint_prompt(prompt: str) -> Preferences:
    """Constraints stated in one message, without any conversation context."""
    text = strip_accents_lower(prompt)
    result: Preferences = {
        "genres": _genre_signals(text)[0],
        "platforms": [platform for platform, pattern in PLATFORM_TERMS if pattern.search(text)],
        "moods": [mood for mood, pattern in MOOD_TERMS if pattern.search(text)],
    }
    hours = _HOURS.search(text)
    minutes = _MINUTES.search(text)
    if hours:
        amount = _HOUR_WORDS.get(hours.group(1))
        result["maxRuntime"] = js_round((amount if amount is not None else float(hours.group(1))) * 60)
    elif _HOUR_AND_HALF.search(text):
        result["maxRuntime"] = 90
    elif minutes:
        result["maxRuntime"] = int(minutes.group(1))
    elif _SHORT.search(text):
        result["maxRuntime"] = 100

    if _FAIR.search(text):
        result["rankingMode"] = "least-misery"
    elif _BALANCED.search(text):
        result["rankingMode"] = "balanced"
    return result


def _first(*values: Any) -> Any:
    """JavaScript `a ?? b`: the first value that is not missing."""
    return next((value for value in values if value is not None), None)


def update_conversation_preferences(current: Preferences, prompt: str, now: int) -> Preferences:
    """Merges one turn into the conversation's preferences, keeping earlier exclusions."""
    text = strip_accents_lower(prompt).replace("’", "").replace("'", "")
    base = create_conversation_preferences(now) if _RESET.search(text) else current
    parsed = parse_constraint_prompt(prompt)
    included, excluded = _genre_signals(text)
    remove_intense = bool(_LESS_INTENSE.search(text))
    people = _PEOPLE.search(text) or _MOVIE_NIGHT_WITH.search(text)
    since_year = _SINCE_YEAR.search(text)
    decade = _DECADE.search(text)
    languages = [language for language, pattern in LANGUAGE_TERMS if pattern.search(text)]

    genres = [genre for genre in unique([*(base.get("genres") or []), *included]) if genre not in excluded]
    excluded_genres = [genre for genre in unique([*base["excludedGenres"], *excluded]) if genre not in included]
    moods = [
        mood
        for mood in unique([*(base.get("moods") or []), *parsed["moods"], *(["Twisty"] if _DARK.search(text) else [])])
        if not (remove_intense and mood == "Electric")
    ]
    if decade:
        year_range = [int(decade.group(1)), int(decade.group(1)) + 9]
    elif since_year:
        year_range = [int(since_year.group(1)), 2100]
    else:
        year_range = base.get("yearRange")
    if _DATE_NIGHT.search(text):
        viewing_context = "date-night"
    elif _MOVIE_NIGHT.search(text):
        viewing_context = "movie-night"
    else:
        viewing_context = base.get("viewingContext")

    merged: Preferences = {
        **base,
        "genres": genres,
        "excludedGenres": excluded_genres,
        "platforms": parsed["platforms"] if parsed["platforms"] else base.get("platforms"),
        "moods": moods,
        "maxRuntime": _first(parsed.get("maxRuntime"), base.get("maxRuntime")),
        "rankingMode": _first(parsed.get("rankingMode"), base.get("rankingMode")),
        "yearRange": year_range,
        "languages": languages if languages else base.get("languages"),
        "peopleCount": int(people.group(1)) if people else base.get("peopleCount"),
        "viewingContext": viewing_context,
        "lastUpdatedAt": now,
    }
    # JSON drops `undefined`; dropping None keeps the same wire shape.
    return {key: value for key, value in merged.items() if value is not None}
