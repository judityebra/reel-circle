"""Local semantic vectors, lexical evidence and the bounded rerank (port of src/lib/semantic-retrieval.ts).

Without an embedding provider, the query and each movie become 384-dimension hashed bag-of-words
vectors. Semantic similarity is worth a bounded 28% of the final score, so vibe phrases can reorder
movies that meet the hard constraints but never bring back one that doesn't. Embedding providers
and Qdrant arrive in a later ticket.
"""

import math
import re

from .js_compat import js_int32, js_round, js_sum, strip_accents_lower, unique
from .scoring import Movie, Recommendation

DIMENSIONS = 384
LOCAL_MODEL = "local-hash-v1"

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "but",
    "for",
    "i",
    "in",
    "is",
    "it",
    "me",
    "movie",
    "of",
    "on",
    "or",
    "something",
    "the",
    "to",
    "want",
    "with",
}
_CONCEPTS = {
    "autumn": ["autumn", "fall", "leaves", "harvest", "cozy", "nostalgic", "warm"],
    "beautiful": ["beautiful", "visual", "cinematography", "lush", "elegant", "stunning"],
    "comfort": ["comfort", "cozy", "gentle", "warm", "hopeful", "uplifting"],
    "feminine": ["feminine", "female", "woman", "women", "girl", "stylish"],
    "paris": ["paris", "french", "france", "romance", "city"],
    "rainy": ["rainy", "rain", "melancholy", "quiet", "cozy", "atmospheric"],
    "mysterious": ["mysterious", "mystery", "enigmatic", "suspense", "twisty"],
    "elegant": ["elegant", "refined", "stylish", "beautiful", "lush"],
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(value: str) -> list[str]:
    return [
        token for token in _TOKEN.findall(strip_accents_lower(value)) if len(token) > 1 and token not in _STOP_WORDS
    ]


def _expand(tokens: list[str]) -> list[str]:
    return unique(term for token in tokens for term in [token, *_CONCEPTS.get(token, [])])


def _hash(token: str) -> int:
    """32-bit FNV-1a, as the TypeScript code computes it with `Math.imul`."""
    result = 2166136261
    for character in token:
        result = js_int32(js_int32(result ^ ord(character)) * 16777619)
    return abs(result) % DIMENSIONS


def semantic_vector(value: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for token in _expand(_tokenize(value)):
        vector[_hash(token)] += 1
    magnitude = math.sqrt(js_sum(item * item for item in vector)) or 1
    return [item / magnitude for item in vector]


def cosine(left: list[float], right: list[float]) -> float:
    """Similarity of two unit vectors, floored at zero."""
    return max(0.0, js_sum(a * b for a, b in zip(left, right, strict=True)))


def movie_document(movie: Movie) -> str:
    """The text a movie is embedded and matched by."""
    parts = [
        movie.get("title"),
        movie.get("overview"),
        " ".join(movie.get("genres") or []),
        " ".join(movie["keywords"]) if movie.get("keywords") is not None else None,
        " ".join(movie["cast"]) if movie.get("cast") is not None else None,
        movie.get("director"),
        movie.get("tone"),
        " ".join(movie["moods"]) if movie.get("moods") is not None else None,
        movie.get("language"),
    ]
    return " ".join(part for part in parts if part)


def matched_terms(query: str, movie: Movie) -> list[str]:
    """The query's (concept-expanded) terms that also appear in the movie's document."""
    document_terms = set(_expand(_tokenize(movie_document(movie))))
    return [term for term in _expand(_tokenize(query)) if term in document_terms]


def rerank(
    recommendations: list[Recommendation], scores: dict[str, float], terms: dict[str, list[str]]
) -> list[Recommendation]:
    """Blends each movie's semantic score (0-1) into its group score at a bounded 28%, best first.

    A movie missing from `scores` counts as no semantic match and shows no matched terms.
    """
    reranked = []
    for recommendation in recommendations:
        movie_id = recommendation["movie"]["id"]
        semantic_score = js_round(scores.get(movie_id, 0) * 100)
        reranked.append(
            {
                **recommendation,
                "score": js_round(recommendation["score"] * 0.72 + semantic_score * 0.28),
                "semanticScore": semantic_score,
                "semanticTerms": terms.get(movie_id, [])[:5] if movie_id in scores else [],
            }
        )
    return sorted(reranked, key=lambda recommendation: -recommendation["score"])
