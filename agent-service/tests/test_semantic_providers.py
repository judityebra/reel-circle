"""Semantic retrieval with embedding providers and Qdrant, faked at the HTTP transport.

Each fake stands in for one external service: an OpenAI-compatible or Gemini embedding provider, and
a Qdrant cluster. Every failure must fall back on its own, as the TypeScript service does: a broken
provider means local vectors, a broken Qdrant means in-memory search over the provider's vectors.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from reel_circle_agent.api import app, outbound_transport

MOVIES = [
    {
        "id": "space",
        "title": "Solar War",
        "year": 2021,
        "runtime": 95,
        "genres": ["Sci-Fi"],
        "platforms": [],
        "rating": 4.9,
    },
    {
        "id": "paris",
        "title": "Paris in October",
        "year": 2020,
        "runtime": 95,
        "genres": ["Romance"],
        "platforms": [],
        "rating": 4.0,
        "overview": "Rainy cafes and autumn leaves.",
    },
]
TURN = {"message": "something atmospheric", "candidates": MOVIES, "profiles": [], "limit": 5}


def _vector_for(text: str) -> list[float]:
    """The fake provider puts Paris (and the query) on one axis and everything else on the other."""
    return [1.0, 0.0] if "Paris" in text or "atmospheric" in text else [0.0, 1.0]


class FakeServices:
    """A fake embedding provider and Qdrant cluster behind one transport, recording every request."""

    def __init__(self, embedding_status: int = 200, embedding_body: dict | None = None, qdrant: dict | None = None):
        self.embedding_status = embedding_status
        self.embedding_body = embedding_body
        self.qdrant = qdrant or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "embeddings.test":
            texts = json.loads(request.content)["input"]
            body = self.embedding_body or {
                "data": [{"index": i, "embedding": _vector_for(t)} for i, t in enumerate(texts)]
            }
            return httpx.Response(self.embedding_status, json=body)
        if request.url.host == "generativelanguage.googleapis.com":
            texts = [item["content"]["parts"][0]["text"] for item in json.loads(request.content)["requests"]]
            return httpx.Response(200, json={"embeddings": [{"values": _vector_for(text)} for text in texts]})
        if request.url.host == "qdrant.test":
            key = f"{request.method} {request.url.path.rsplit('/', 1)[-1]}"
            status, body = self.qdrant.get(key, (200, {}))
            return httpx.Response(status, json=body)
        raise AssertionError(f"unexpected request to {request.url}")


@pytest.fixture
def services(monkeypatch):
    for name in [
        "EMBEDDING_PROVIDER",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_DIMENSIONS",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_COLLECTION",
    ]:
        monkeypatch.delenv(name, raising=False)

    def install(fake: FakeServices) -> FakeServices:
        app.dependency_overrides[outbound_transport] = lambda: httpx.MockTransport(fake)
        return fake

    yield install
    app.dependency_overrides.clear()


def _turn() -> dict:
    response = TestClient(app).post("/agent", json=TURN)
    assert response.status_code == 200, response.text
    return response.json()


def _openai_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embeddings.test/v1/")
    monkeypatch.setenv("EMBEDDING_MODEL", "semantic-model")
    monkeypatch.setenv("EMBEDDING_API_KEY", "embedding-secret")


QDRANT_HITS = {
    "result": {
        "points": [
            {"score": 0.95, "payload": {"movieId": "paris"}},
            {"score": 0.05, "payload": {"movieId": "space"}},
        ]
    }
}


def test_provider_embeddings_rank_movies_in_memory_without_qdrant(services, monkeypatch):
    _openai_provider(monkeypatch)
    fake = services(FakeServices())
    body = _turn()
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["paris", "space"]
    assert body["meta"]["semantic"] == {
        "retrievalMode": "embedding-memory",
        "embeddingMode": "provider",
        "embeddingModel": "semantic-model",
        "dimensions": 2,
    }
    [request] = fake.requests
    assert str(request.url) == "https://embeddings.test/v1/embeddings"
    assert request.headers["authorization"] == "Bearer embedding-secret"
    sent = json.loads(request.content)
    assert sent["model"] == "semantic-model" and sent["encoding_format"] == "float"
    assert sent["input"][0] == "something atmospheric" and len(sent["input"]) == 3


def test_gemini_embeddings_use_task_prefixes_and_bounded_dimensions(services, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_API_KEY", "gemini-secret")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "99999")
    fake = services(FakeServices())
    body = _turn()
    assert body["recommendations"][0]["movie"]["id"] == "paris"
    assert body["meta"]["semantic"]["embeddingModel"] == "gemini-embedding-2"
    [request] = fake.requests
    assert request.url.path == "/v1beta/models/gemini-embedding-2:batchEmbedContents"
    assert request.headers["x-goog-api-key"] == "gemini-secret"
    sent = json.loads(request.content)["requests"]
    assert sent[0]["content"]["parts"][0]["text"] == "task: search result | query: something atmospheric"
    assert sent[1]["content"]["parts"][0]["text"].startswith("title: none | text: Solar War")
    assert sent[0]["embedContentConfig"] == {"outputDimensionality": 3072, "autoTruncate": True}
    assert sent[0]["model"] == "models/gemini-embedding-2"


@pytest.mark.parametrize(
    "fake",
    [
        FakeServices(embedding_status=503),
        FakeServices(embedding_body={"data": [{"index": 0, "embedding": [1.0, "x"]}]}),
        FakeServices(
            embedding_body={
                "data": [
                    {"index": 0, "embedding": [1.0]},
                    {"index": 1, "embedding": [1.0, 0.0]},
                    {"index": 2, "embedding": [0.0, 1.0]},
                ]
            }
        ),
    ],
    ids=["provider-error", "invalid-vectors", "inconsistent-dimensions"],
)
def test_a_failing_provider_falls_back_to_local_vectors_and_says_why(services, monkeypatch, fake):
    _openai_provider(monkeypatch)
    services(fake)
    body = _turn()
    assert body["meta"]["semantic"] == {
        "retrievalMode": "local",
        "embeddingMode": "local",
        "embeddingModel": "local-hash-v1",
        "dimensions": 384,
        "fallbackReason": "embedding-failed",
    }
    assert body["recommendations"][0]["movie"]["id"] == "paris"


def test_qdrant_creates_the_collection_upserts_the_pool_and_ranks_by_its_scores(services, monkeypatch):
    _openai_provider(monkeypatch)
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.test/")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    fake = services(FakeServices(qdrant={"GET reel-circle-2": (404, {}), "POST query": (200, QDRANT_HITS)}))
    body = _turn()
    assert body["meta"]["semantic"]["retrievalMode"] == "qdrant"
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["paris", "space"]
    assert body["recommendations"][0]["semanticScore"] == 95
    qdrant = [request for request in fake.requests if request.url.host == "qdrant.test"]
    assert [(request.method, request.url.path) for request in qdrant] == [
        ("GET", "/collections/reel-circle-2"),
        ("PUT", "/collections/reel-circle-2"),
        ("PUT", "/collections/reel-circle-2/points"),
        ("POST", "/collections/reel-circle-2/points/query"),
    ]
    assert all(request.headers["api-key"] == "qdrant-secret" for request in qdrant)
    assert json.loads(qdrant[1].content) == {"vectors": {"size": 2, "distance": "Cosine"}}
    points = json.loads(qdrant[2].content)["points"]
    assert [point["payload"]["movieId"] for point in points] == ["space", "paris"]
    # Same point id the TypeScript client derives for "space" (computed with api/qdrant-client.ts's algorithm).
    assert points[0]["id"] == "3f49dbbf-e051-4b20-ac03-8923424fedf8"


def test_qdrant_falls_back_to_the_legacy_search_endpoint(services, monkeypatch):
    _openai_provider(monkeypatch)
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.test")
    monkeypatch.setenv("QDRANT_COLLECTION", "reel-circle-movies")
    legacy_hits = {"result": QDRANT_HITS["result"]["points"]}
    fake = services(FakeServices(qdrant={"POST query": (404, {}), "POST search": (200, legacy_hits)}))
    body = _turn()
    assert body["meta"]["semantic"]["retrievalMode"] == "qdrant"
    assert body["recommendations"][0]["movie"]["id"] == "paris"
    search = next(request for request in fake.requests if request.url.path.endswith("/points/search"))
    assert "query" not in json.loads(search.content) and json.loads(search.content)["vector"] == [1.0, 0.0]


def test_a_failing_qdrant_falls_back_to_in_memory_search_and_says_why(services, monkeypatch):
    _openai_provider(monkeypatch)
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.test")
    services(FakeServices(qdrant={"GET reel-circle-2": (500, {})}))
    body = _turn()
    assert body["meta"]["semantic"] == {
        "retrievalMode": "embedding-memory",
        "embeddingMode": "provider",
        "embeddingModel": "semantic-model",
        "dimensions": 2,
        "fallbackReason": "qdrant-failed",
    }
    assert body["recommendations"][0]["movie"]["id"] == "paris"
