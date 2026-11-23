"""Semantic retrieval with independent fallbacks (port of api/semantic-service.ts).

    provider embeddings --fails--> local hashed vectors       (fallbackReason "embedding-failed")
    Qdrant search       --fails--> in-memory cosine search    (fallbackReason "qdrant-failed")

Each external service can fail on its own without failing the turn.
"""

from typing import Any

import httpx

from .embeddings import EmbeddingError, embed_texts
from .qdrant import QdrantError, qdrant_configured, search_qdrant
from .scoring import Recommendation
from .semantic import DIMENSIONS, LOCAL_MODEL, cosine, matched_terms, movie_document, rerank, semantic_vector


def _result(
    recommendations: list[Recommendation],
    retrieval: str,
    embedding: str,
    model: str,
    dimensions: int,
    fallback: str | None = None,
) -> dict[str, Any]:
    result = {
        "recommendations": recommendations,
        "retrievalMode": retrieval,
        "embeddingMode": embedding,
        "embeddingModel": model,
        "dimensions": dimensions,
    }
    return {**result, "fallbackReason": fallback} if fallback else result


def _local(query: str, recommendations: list[Recommendation], fallback: str | None = None) -> dict[str, Any]:
    query_vector = semantic_vector(query)
    scores = {
        item["movie"]["id"]: cosine(query_vector, semantic_vector(movie_document(item["movie"])))
        for item in recommendations
    }
    terms = {item["movie"]["id"]: matched_terms(query, item["movie"]) for item in recommendations}
    return _result(rerank(recommendations, scores, terms), "local", "local", LOCAL_MODEL, DIMENSIONS, fallback)


async def retrieve_semantically(
    query: str, recommendations: list[Recommendation], client: httpx.AsyncClient
) -> dict[str, Any]:
    """Reranks the movies that meet the hard constraints by similarity to the user's message."""
    if not recommendations:
        return _local(query, recommendations)
    movies = [item["movie"] for item in recommendations]
    try:
        batch = await embed_texts([query, *(movie_document(movie) for movie in movies)], client)
    except EmbeddingError:
        return _local(query, recommendations, "embedding-failed")

    query_vector, movie_vectors = batch.vectors[0], batch.vectors[1:]
    terms = {movie["id"]: matched_terms(query, movie) for movie in movies}
    memory_scores = {
        movie["id"]: cosine(query_vector, vector) for movie, vector in zip(movies, movie_vectors, strict=True)
    }
    memory_mode = "embedding-memory" if batch.mode == "provider" else "local"

    if qdrant_configured():
        try:
            hits = await search_qdrant(
                query_vector,
                [(movie["id"], vector) for movie, vector in zip(movies, movie_vectors, strict=True)],
                len(movies),
                client,
            )
        except QdrantError:
            return _result(
                rerank(recommendations, memory_scores, terms),
                memory_mode,
                batch.mode,
                batch.model,
                batch.dimensions,
                "qdrant-failed",
            )
        if hits is not None:
            scores = {hit.movie_id: hit.score for hit in hits}
            return _result(rerank(recommendations, scores, terms), "qdrant", batch.mode, batch.model, batch.dimensions)

    return _result(
        rerank(recommendations, memory_scores, terms), memory_mode, batch.mode, batch.model, batch.dimensions
    )
