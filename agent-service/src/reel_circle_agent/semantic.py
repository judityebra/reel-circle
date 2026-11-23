"""Local semantic retrieval (port of src/lib/semantic-retrieval.ts and the local path of api/semantic-service.ts).

Without an embedding provider, the query and each movie become 384-dimension hashed bag-of-words
vectors. Semantic similarity is worth a bounded 28% of the final score, so vibe phrases can reorder
movies that meet the hard constraints but never bring back one that doesn't. Embedding providers
and Qdrant arrive in a later ticket.
"""

import math
import re
from typing import Any

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


def _cosine(left: list[float], right: list[float]) -> float:
    return max(0.0, js_sum(a * b for a, b in zip(left, right, strict=True)))


def movie_document(movie: Movie) -> str:
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


def retrieve_semantically(query: str, recommendations: list[Recommendation]) -> dict[str, Any]:
    """Reranks recommendations by similarity to the user's message, with local vectors."""
    query_vector = semantic_vector(query)
    query_terms = _expand(_tokenize(query))
    reranked = []
    for recommendation in recommendations:
        document = movie_document(recommendation["movie"])
        document_terms = set(_expand(_tokenize(document)))
        semantic_score = js_round(_cosine(query_vector, semantic_vector(document)) * 100)
        reranked.append(
            {
                **recommendation,
                "score": js_round(recommendation["score"] * 0.72 + semantic_score * 0.28),
                "semanticScore": semantic_score,
                "semanticTerms": [term for term in query_terms if term in document_terms][:5],
            }
        )
    return {
        "recommendations": sorted(reranked, key=lambda recommendation: -recommendation["score"]),
        "retrievalMode": "local",
        "embeddingMode": "local",
        "embeddingModel": LOCAL_MODEL,
        "dimensions": DIMENSIONS,
    }
