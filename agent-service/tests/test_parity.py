"""Parity with the TypeScript recommendation agent, replaying fixtures/agent-parity.json over HTTP.

The fixture file states the rules: send each turn's request with the shared candidate pool, persons
and limit; compare preferences without lastUpdatedAt, recommendations in order with scores rounded
to `comparison.scoreDecimals`, and explanations word for word.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reel_circle_agent.api import app
from reel_circle_agent.scoring import broken_constraints

FIXTURES = json.loads((Path(__file__).parents[2] / "fixtures" / "agent-parity.json").read_text(encoding="utf-8"))
DECIMALS = FIXTURES["comparison"]["scoreDecimals"]
client = TestClient(app)


@pytest.fixture(autouse=True)
def deterministic_environment(monkeypatch):
    for name in [
        "OPENAI_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "QDRANT_URL",
        "LANGSMITH_TRACING",
    ]:
        monkeypatch.delenv(name, raising=False)


def _send(request: dict) -> dict:
    body = {
        "message": request["message"],
        "candidates": FIXTURES["candidatePool"],
        "profiles": FIXTURES["persons"],
        "limit": FIXTURES["limit"],
    }
    if request["preferences"] is not None:
        body["preferences"] = {**request["preferences"], "lastUpdatedAt": 0}
    response = client.post("/agent", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _observed(response: dict) -> dict:
    preferences = {key: value for key, value in response["preferences"].items() if key != "lastUpdatedAt"}
    recommendations = [
        {
            "movieId": recommendation["movie"]["id"],
            "score": round(recommendation["score"], DECIMALS),
            "matchedPreferences": recommendation["matchedPreferences"],
            "explanation": recommendation["explanation"],
            "semanticScore": round(recommendation["semanticScore"], DECIMALS),
            "semanticTerms": recommendation["semanticTerms"],
        }
        for recommendation in response["recommendations"]
    ]
    return {
        "preferences": preferences,
        "candidatesMeetingConstraints": response["meta"]["candidatesMeetingConstraints"],
        "recommendations": recommendations,
        "violations": [
            {"movieId": recommendation["movie"]["id"], "constraint": constraint}
            for recommendation in response["recommendations"]
            for constraint in broken_constraints(recommendation["movie"], preferences)
        ],
    }


@pytest.mark.parametrize("conversation", FIXTURES["conversations"], ids=lambda conversation: conversation["id"])
def test_reproduces_the_typescript_recommendation_agent(conversation):
    for index, turn in enumerate(conversation["turns"]):
        assert _observed(_send(turn["request"])) == turn["expected"], f"turn {index + 1}: {turn['request']['message']}"


def test_echoes_each_recommended_movie_exactly_as_the_browser_sent_it():
    response = _send(FIXTURES["conversations"][0]["turns"][0]["request"])
    pool = {movie["id"]: movie for movie in FIXTURES["candidatePool"]}
    assert response["recommendations"]
    assert all(
        recommendation["movie"] == pool[recommendation["movie"]["id"]] for recommendation in response["recommendations"]
    )
