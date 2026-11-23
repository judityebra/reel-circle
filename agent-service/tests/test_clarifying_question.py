"""The clarifying question pause: asked only when a model found nothing usable in the turn."""

import pytest
from fastapi.testclient import TestClient

from reel_circle_agent.api import app, configured_chat_model

MOVIES = [
    {
        "id": "clue",
        "title": "Clue",
        "year": 1985,
        "runtime": 94,
        "genres": ["Comedy", "Mystery"],
        "platforms": [],
        "rating": 3.8,
    },
    {
        "id": "aftersun",
        "title": "Aftersun",
        "year": 2022,
        "runtime": 102,
        "genres": ["Drama"],
        "platforms": [],
        "rating": 4.2,
    },
]
NOTHING = {
    "genres": [],
    "excludedGenres": [],
    "platforms": [],
    "moods": [],
    "maxRuntime": None,
    "yearRange": None,
    "languages": [],
    "peopleCount": None,
    "viewingContext": None,
    "clarificationQuestion": None,
}
QUESTION = "Do you want something light or something serious tonight?"


class ScriptedModel:
    """Answers each turn with the next scripted extraction."""

    def __init__(self, *answers):
        self.answers = list(answers)

    def with_structured_output(self, *_args, **_kwargs):
        return self

    async def ainvoke(self, _messages):
        return self.answers.pop(0)


@pytest.fixture
def model():
    def install(*answers):
        scripted = ScriptedModel(*answers)  # one model for every turn, so each turn takes the next answer
        app.dependency_overrides[configured_chat_model] = lambda: scripted

    yield install
    app.dependency_overrides.pop(configured_chat_model, None)


def _turn(client: TestClient, message: str, conversation_id: str | None = None) -> dict:
    body = {"message": message, "candidates": MOVIES, "profiles": [], "limit": 5}
    if conversation_id:
        body["conversationId"] = conversation_id
    response = client.post("/agent", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_an_unusable_turn_pauses_with_the_model_question(model):
    model({**NOTHING, "clarificationQuestion": QUESTION})
    with TestClient(app) as client:
        body = _turn(client, "hmm, not sure")
    assert body["pause"] == {"type": "clarifying_question", "question": QUESTION}
    assert body["clarificationQuestion"] == QUESTION
    assert body["recommendations"] == []
    assert body["meta"]["extractionMode"] == "model"


def test_the_reply_is_an_ordinary_turn_that_ends_the_pause(model):
    model({**NOTHING, "clarificationQuestion": QUESTION}, {**NOTHING, "genres": ["Comedy"]})
    with TestClient(app) as client:
        paused = _turn(client, "hmm, not sure")
        body = _turn(client, "Something light", paused["conversationId"])
    assert "pause" not in body
    assert body["preferences"]["genres"] == ["Comedy"]
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["clue"]


def test_a_usable_turn_never_pauses_even_if_the_model_has_a_question(model):
    model({**NOTHING, "genres": ["Drama"], "clarificationQuestion": QUESTION})
    with TestClient(app) as client:
        body = _turn(client, "a drama, maybe")
    assert "pause" not in body
    assert body["clarificationQuestion"] == QUESTION
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["aftersun"]


def test_deterministic_extraction_never_asks():
    with TestClient(app) as client:
        body = _turn(client, "hmm, not sure")
    assert "pause" not in body and "clarificationQuestion" not in body
    assert body["meta"]["extractionMode"] == "deterministic"


def test_a_model_that_falls_back_never_asks():
    class FailingModel(ScriptedModel):
        async def ainvoke(self, _messages):
            raise self.answers.pop(0)

    app.dependency_overrides[configured_chat_model] = lambda: FailingModel(*[ValueError("bad")] * 3)
    try:
        with TestClient(app) as client:
            body = _turn(client, "hmm, not sure")
    finally:
        app.dependency_overrides.pop(configured_chat_model, None)
    assert "pause" not in body
    assert body["meta"]["modelStatus"] == "failed"


def test_a_turn_that_only_repeats_earlier_preferences_still_asks(model):
    model({**NOTHING, "genres": ["Comedy"]}, {**NOTHING, "genres": ["Comedy"], "clarificationQuestion": QUESTION})
    with TestClient(app) as client:
        first = _turn(client, "something funny")
        body = _turn(client, "hmm, whatever", first["conversationId"])
    assert body["pause"] == {"type": "clarifying_question", "question": QUESTION}


def test_group_size_or_occasion_alone_is_not_enough_to_choose(model):
    model({**NOTHING, "peopleCount": 4, "viewingContext": "movie-night", "clarificationQuestion": QUESTION})
    with TestClient(app) as client:
        body = _turn(client, "movie night for four")
    assert body["pause"]["type"] == "clarifying_question"
    assert body["preferences"]["peopleCount"] == 4  # still remembered for the next turn


def test_a_decision_never_resumes_a_clarifying_question(model):
    model({**NOTHING, "clarificationQuestion": QUESTION}, {**NOTHING, "genres": ["Comedy"]})
    with TestClient(app) as client:
        paused = _turn(client, "hmm, not sure")
        response = client.post(
            "/agent",
            json={
                "message": "Something light",
                "conversationId": paused["conversationId"],
                "candidates": MOVIES,
                "decision": "accept",
                "answering": {},
            },
        )
    assert response.status_code == 200
    assert response.json()["preferences"]["genres"] == ["Comedy"]
