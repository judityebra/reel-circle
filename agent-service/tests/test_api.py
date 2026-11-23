"""The recommendation agent's HTTP contract, outside the parity fixtures."""

from fastapi.testclient import TestClient

from reel_circle_agent.api import app

client = TestClient(app)


def test_health_check_reports_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_turn_endpoint_only_accepts_post():
    response = client.get("/agent")
    assert response.status_code == 405
    assert response.json() == {"error": "Method not allowed"}


def test_turn_requires_a_message_of_1_to_1000_characters():
    for body in [{}, {"message": "   "}, {"message": "x" * 1001}, {"message": 42}, []]:
        response = client.post("/agent", json=body)
        assert response.status_code == 400, body
        assert response.json() == {"error": "A message of 1-1000 characters is required"}


def test_turn_with_an_empty_candidate_pool_returns_no_recommendations_and_a_full_trace():
    response = client.post("/agent", json={"message": "Something funny under 90 minutes"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["recommendations"] == []
    assert body["preferences"]["genres"] == ["Comedy"]
    assert body["preferences"]["maxRuntime"] == 90
    assert body["meta"]["extractionMode"] == "deterministic"
    assert body["meta"]["modelStatus"] == "not-configured"
    assert [step["tool"] for step in body["meta"]["trace"]] == [
        "extract_preferences",
        "filter_candidates",
        "semantic_retrieval",
        "rank_candidates",
        "validate_results",
    ]
    assert [step["status"] for step in body["meta"]["trace"]] == [
        "completed",
        "skipped",
        "skipped",
        "skipped",
        "completed",
    ]


def test_message_length_is_counted_in_utf16_units_like_the_browser():
    assert client.post("/agent", json={"message": "😀" * 500}).status_code == 200
    assert client.post("/agent", json={"message": "😀" * 501}).status_code == 400


def test_a_non_finite_limit_falls_back_to_the_default_shortlist_size():
    response = client.post(
        "/agent",
        content='{"message": "Something funny", "limit": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200


def test_incoming_preferences_without_exclusions_are_accepted():
    response = client.post(
        "/agent", json={"message": "Under 90 minutes", "preferences": {"genres": ["Comedy"], "lastUpdatedAt": 0}}
    )
    assert response.status_code == 200
    assert response.json()["preferences"]["genres"] == ["Comedy"]
    assert response.json()["preferences"]["excludedGenres"] == []


def test_langsmith_tracing_is_off_unless_switched_on(monkeypatch):
    from langsmith.utils import get_env_var, tracing_is_enabled

    for name in ["LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"]:
        monkeypatch.delenv(name, raising=False)
    get_env_var.cache_clear()  # langsmith caches environment lookups
    assert not tracing_is_enabled()
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    get_env_var.cache_clear()
    assert tracing_is_enabled()
    get_env_var.cache_clear()
