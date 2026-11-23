"""Semantic retrieval with independent fallbacks (port of api/semantic-service.ts).

    provider embeddings --fails--> local hashed vectors       (fallbackReason "embedding-failed")
    Qdrant search       --fails--> in-memory cosine search    (fallbackReason "qdrant-failed")

Each external service can fail on its own, in any way, without failing the turn: like the
TypeScript service's bare `catch`, every error from a service means "use the fallback".
"""

from typing import Any, Literal

import httpx

from .embeddings import EmbeddingBatch, embed_texts
from .qdrant import qdrant_configured, search_qdrant
from .scoring import Recommendation
from .semantic import DIMENSIONS, LOCAL_MODEL, cosine, matched_terms, movie_document, rerank, semantic_vector

RetrievalMode = Literal["qdrant", "embedding-memory", "local"]
FallbackReason = Literal["embedding-failed", "qdrant-failed"]

_LOCAL_BATCH_DESCRIPTION = EmbeddingBatch(vectors=[], mode="local", model=LOCAL_MODEL, dimensions=DIMENSIONS)


def _retrieval_result(
    reranked: list[Recommendation], mode: RetrievalMode, batch: EmbeddingBatch, fallback: FallbackReason | None = None
) -> dict[str, Any]:
    result = {
        "recommendations": reranked,
        "retrievalMode": mode,
        "embeddingMode": batch.mode,
        "embeddingModel": batch.model,
        "dimensions": batch.dimensions,
    }
    return {**result, "fallbackReason": fallback} if fallback else result


def _local_retrieval(query: str, recommendations: list[Recommendation], fallback: FallbackReason | None = None):
    """Local hashed vectors and in-memory cosine search: no external service involved."""
    query_vector = semantic_vector(query)
    movies = [item["movie"] for item in recommendations]
    scores = {movie["id"]: cosine(query_vector, semantic_vector(movie_document(movie))) for movie in movies}
    terms = {movie["id"]: matched_terms(query, movie) for movie in movies}
    return _retrieval_result(rerank(recommendations, scores, terms), "local", _LOCAL_BATCH_DESCRIPTION, fallback)


async def retrieve_semantically(
    query: str, recommendations: list[Recommendation], client: httpx.AsyncClient
) -> dict[str, Any]:
    """Reranks the movies that meet the hard constraints by similarity to the user's message."""
    if not recommendations:
        return _local_retrieval(query, recommendations)
    movies = [item["movie"] for item in recommendations]
    try:
        batch = await embed_texts([query, *(movie_document(movie) for movie in movies)], client)
    except Exception:  # noqa: BLE001 - any provider failure means local vectors, as in TypeScript
        return _local_retrieval(query, recommendations, "embedding-failed")

    query_vector = batch.vectors[0]
    pool_vectors = [(movie["id"], vector) for movie, vector in zip(movies, batch.vectors[1:], strict=True)]
    terms = {movie["id"]: matched_terms(query, movie) for movie in movies}
    in_memory = rerank(
        recommendations, {movie_id: cosine(query_vector, vector) for movie_id, vector in pool_vectors}, terms
    )
    in_memory_mode: RetrievalMode = "embedding-memory" if batch.mode == "provider" else "local"

    if qdrant_configured():
        try:
            hits = await search_qdrant(query_vector, pool_vectors, len(movies), client)
        except Exception:  # noqa: BLE001 - any Qdrant failure means in-memory search, as in TypeScript
            return _retrieval_result(in_memory, in_memory_mode, batch, "qdrant-failed")
        if hits is not None:
            scores = {hit.movie_id: hit.score for hit in hits}
            return _retrieval_result(rerank(recommendations, scores, terms), "qdrant", batch)

    return _retrieval_result(in_memory, in_memory_mode, batch)
