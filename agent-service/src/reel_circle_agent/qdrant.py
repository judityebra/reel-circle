"""Candidate-scoped vector search in Qdrant (port of api/qdrant-client.ts).

Configured by QDRANT_URL, QDRANT_API_KEY and QDRANT_COLLECTION (default `reel-circle-<dimensions>`).
Each turn upserts the candidate pool's vectors and searches only among those points, so Qdrant
can reorder the movies that meet the hard constraints but never add one.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx


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
    query_vector: list[float], candidates: list[tuple[str, list[float]]], limit: int, client: httpx.AsyncClient
) -> list[SemanticMatch] | None:
    """Matches for the query among the candidates, or None when Qdrant isn't configured or there's nothing to search."""
    base_url = os.environ.get("QDRANT_URL", "").rstrip("/")
    if not base_url or not candidates:
        return None
    collection = os.environ.get("QDRANT_COLLECTION") or f"reel-circle-{len(query_vector)}"
    api_key = os.environ.get("QDRANT_API_KEY")
    headers = {"api-key": api_key} if api_key else {}
    collection_url = f"{base_url}/collections/{quote(collection, safe='')}"

    async def call(method: str, url: str, body: dict[str, Any] | None = None) -> httpx.Response:
        try:
            return await client.request(method, url, json=body, headers=headers)
        except httpx.HTTPError as error:
            raise QdrantError(f"Qdrant request failed: {error!r}") from error

    existing = await call("GET", collection_url)
    if existing.status_code == 404:
        created = await call("PUT", collection_url, {"vectors": {"size": len(query_vector), "distance": "Cosine"}})
        if created.status_code >= 400:
            raise QdrantError(f"Qdrant collection creation failed: {created.status_code}")
    elif existing.status_code >= 400:
        raise QdrantError(f"Qdrant collection lookup failed: {existing.status_code}")

    points = [
        {"id": point_id(movie_id), "vector": vector, "payload": {"movieId": movie_id}}
        for movie_id, vector in candidates
    ]
    upserted = await call("PUT", f"{collection_url}/points?wait=true", {"points": points})
    if upserted.status_code >= 400:
        raise QdrantError(f"Qdrant upsert failed: {upserted.status_code}")

    query = {
        "query": query_vector,
        "filter": {"must": [{"has_id": [point_id(movie_id) for movie_id, _ in candidates]}]},
        "limit": max(1, min(limit, len(candidates))),
        "with_payload": True,
        "with_vector": False,
    }
    searched = await call("POST", f"{collection_url}/points/query", query)
    legacy = searched.status_code == 404
    if legacy:
        legacy_query = {key: value for key, value in query.items() if key != "query"}
        searched = await call("POST", f"{collection_url}/points/search", {**legacy_query, "vector": query_vector})
    if searched.status_code >= 400:
        raise QdrantError(f"Qdrant search failed: {searched.status_code}")
    try:
        result = searched.json().get("result")
    except ValueError as error:
        raise QdrantError("Qdrant returned invalid JSON") from error

    hits = result if legacy else (result.get("points") if isinstance(result, dict) else [])
    return [
        SemanticMatch(hit["payload"]["movieId"], max(0.0, min(1.0, hit.get("score") or 0)))
        for hit in (hits if isinstance(hits, list) else [])
        if isinstance(hit, dict) and isinstance(hit.get("payload"), dict) and hit["payload"].get("movieId")
    ]
