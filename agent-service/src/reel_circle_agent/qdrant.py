"""Candidate-scoped vector search in Qdrant (port of api/qdrant-client.ts).

Configured by QDRANT_URL, QDRANT_API_KEY and QDRANT_COLLECTION (default `reel-circle-<dimensions>`).
Each turn upserts the candidate pool's vectors and searches only among those points, so Qdrant
can reorder the movies that meet the hard constraints but never add one.
"""

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx

from .js_compat import encode_uri_component, js_number, strip_one_trailing_slash


class QdrantError(Exception):
    """Qdrant answered with an error at some step."""


@dataclass(frozen=True)
class SemanticMatch:
    movie_id: str
    score: float


def point_id(movie_id: str) -> str:
    """A stable UUID-shaped point id, identical to the TypeScript client's."""
    digest = hashlib.sha256(movie_id.encode("utf-8")).hexdigest()[:32]
    return f"{digest[0:8]}-{digest[8:12]}-4{digest[13:16]}-a{digest[17:20]}-{digest[20:]}"


def qdrant_configured() -> bool:
    return bool(os.environ.get("QDRANT_URL"))


async def search_qdrant(
    query_vector: list[float], candidate_pool: list[tuple[str, list[float]]], limit: int, client: httpx.AsyncClient
) -> list[SemanticMatch] | None:
    """Matches for the query within the candidate pool (movie id, vector), or None when there's nothing to search."""
    base_url = strip_one_trailing_slash(os.environ.get("QDRANT_URL", ""))
    if not base_url or not candidate_pool:
        return None
    configured_collection = os.environ.get("QDRANT_COLLECTION")  # `??`: an empty name is kept, as in TypeScript
    collection = f"reel-circle-{len(query_vector)}" if configured_collection is None else configured_collection
    api_key = os.environ.get("QDRANT_API_KEY")
    headers = {"api-key": api_key} if api_key else {}
    collection_url = f"{base_url}/collections/{encode_uri_component(collection)}"

    async def send(method: str, url: str, body: dict[str, Any] | None = None) -> httpx.Response:
        try:
            return await client.request(method, url, json=body, headers=headers)
        except httpx.HTTPError as error:
            raise QdrantError(f"Qdrant request failed: {error!r}") from error

    existing = await send("GET", collection_url)
    if existing.status_code == 404:
        created = await send("PUT", collection_url, {"vectors": {"size": len(query_vector), "distance": "Cosine"}})
        if created.status_code >= 400:
            raise QdrantError(f"Qdrant collection creation failed: {created.status_code}")
    elif existing.status_code >= 400:
        raise QdrantError(f"Qdrant collection lookup failed: {existing.status_code}")

    points = [
        {"id": point_id(movie_id), "vector": vector, "payload": {"movieId": movie_id}}
        for movie_id, vector in candidate_pool
    ]
    upserted = await send("PUT", f"{collection_url}/points?wait=true", {"points": points})
    if upserted.status_code >= 400:
        raise QdrantError(f"Qdrant upsert failed: {upserted.status_code}")

    query = {
        "query": query_vector,
        "filter": {"must": [{"has_id": [point_id(movie_id) for movie_id, _ in candidate_pool]}]},
        "limit": max(1, min(limit, len(candidate_pool))),
        "with_payload": True,
        "with_vector": False,
    }
    searched = await send("POST", f"{collection_url}/points/query", query)
    legacy = searched.status_code == 404
    if legacy:
        legacy_query = {key: value for key, value in query.items() if key != "query"}
        searched = await send("POST", f"{collection_url}/points/search", {**legacy_query, "vector": query_vector})
    if searched.status_code >= 400:
        raise QdrantError(f"Qdrant search failed: {searched.status_code}")
    try:
        body = searched.json()
    except ValueError as error:
        raise QdrantError("Qdrant returned invalid JSON") from error
    # Mirrors the TypeScript reading of the answer: anything unexpected means no hits, not an error.
    result = body.get("result") if isinstance(body, dict) else None
    hits = result if legacy else (result.get("points") if isinstance(result, dict) else [])
    matches = []
    for hit in hits if isinstance(hits, list) else []:
        payload = hit.get("payload") if isinstance(hit, dict) else None
        movie_id = payload.get("movieId") if isinstance(payload, dict) else None
        if not movie_id:
            continue
        score = js_number(hit.get("score") or 0)
        if math.isnan(score):
            raise QdrantError(f"Qdrant returned a non-numeric score for {movie_id}")
        matches.append(SemanticMatch(movie_id, max(0.0, min(1.0, score))))
    return matches
