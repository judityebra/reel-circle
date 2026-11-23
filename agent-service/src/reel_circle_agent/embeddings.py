"""Embedding providers (port of api/embedding-client.ts).

Configured by environment variables, read on every turn:
- EMBEDDING_PROVIDER: "gemini" for Google's native API; anything else means OpenAI-compatible.
- EMBEDDING_BASE_URL: the OpenAI-compatible base URL (not needed for Gemini).
- EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_DIMENSIONS (Gemini only, 128-3072, default 768).

Without a model (or, for OpenAI-compatible providers, without a base URL), texts get local hashed
vectors and nothing leaves the service.
"""

import math
import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote

import httpx

from .semantic import DIMENSIONS, LOCAL_MODEL, semantic_vector


class EmbeddingError(Exception):
    """The provider failed or returned vectors that can't be used."""


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    mode: Literal["provider", "local"]
    model: str
    dimensions: int


def _gemini_dimensions() -> int:
    try:
        configured = float(os.environ.get("EMBEDDING_DIMENSIONS", ""))
    except ValueError:
        configured = 0
    return max(128, min(3072, int(configured) if math.isfinite(configured) and configured else 768))


def _is_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, int | float) and not isinstance(item, bool) and math.isfinite(item) for item in value)
    )


async def embed_texts(texts: list[str], client: httpx.AsyncClient) -> EmbeddingBatch:
    """Embeds the query (first) and the movie documents, with the configured provider or local vectors."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
    base_url = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
    model = os.environ.get("EMBEDDING_MODEL", "")
    api_key = os.environ.get("EMBEDDING_API_KEY")
    if not model or (provider != "gemini" and not base_url):
        return EmbeddingBatch([semantic_vector(text) for text in texts], "local", LOCAL_MODEL, DIMENSIONS)

    try:
        if provider == "gemini":
            dimensions = _gemini_dimensions()
            requests = [
                {
                    "model": f"models/{model}",
                    "content": {
                        "parts": [
                            {
                                "text": f"task: search result | query: {text}"
                                if index == 0
                                else f"title: none | text: {text}"
                            }
                        ]
                    },
                    "embedContentConfig": {"outputDimensionality": dimensions, "autoTruncate": True},
                }
                for index, text in enumerate(texts)
            ]
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='')}:batchEmbedContents",
                json={"requests": requests},
                headers={"x-goog-api-key": api_key} if api_key else {},
            )
        else:
            response = await client.post(
                f"{base_url}/embeddings",
                json={"model": model, "input": texts, "encoding_format": "float"},
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
    except httpx.HTTPError as error:
        raise EmbeddingError(f"Embedding request failed: {error!r}") from error
    if response.status_code >= 400:
        raise EmbeddingError(f"Embedding request failed: {response.status_code}")
    try:
        payload = response.json()
    except ValueError as error:
        raise EmbeddingError("Embedding provider returned invalid JSON") from error

    if provider == "gemini":
        vectors = [item.get("values") for item in payload.get("embeddings") or []]
    else:
        items = sorted(payload.get("data") or [], key=lambda item: item.get("index") or 0)
        vectors = [item.get("embedding") for item in items]
    if len(vectors) != len(texts) or not all(_is_vector(vector) for vector in vectors):
        raise EmbeddingError("Embedding provider returned invalid vectors")
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise EmbeddingError("Embedding provider returned inconsistent dimensions")
    return EmbeddingBatch([[float(item) for item in vector] for vector in vectors], "provider", model, dimensions)
