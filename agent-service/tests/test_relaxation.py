"""The relaxation pause: when nothing meets every hard constraint, loosen one at a time, with the user's consent."""

import pytest
from fastapi.testclient import TestClient

from reel_circle_agent.api import app, step_limit

MOVIES = [
    {
        "id": "memories",
        "title": "Memories of Murder",
        "year": 2003,
        "runtime": 131,
        "genres": ["Thriller", "Crime"],
        "platforms": ["Netflix"],
        "rating": 4.3,
        "language": "Korean",
    },
    {
        "id": "clue",
        "title": "Clue",
        "year": 1985,
        "runtime": 94,
        "genres": ["Comedy", "Mystery"],
        "platforms": ["Filmin"],
        "rating": 3.8,
        "language": "English",
    },
    {
        "id": "hereditary",
        "title": "Hereditary",
        "year": 2018,
        "runtime": 127,
        "genres": ["Horror"],
        "platforms": ["Filmin"],
        "rating": 4.0,
        "language": "English",
    },
]
EVERYTHING_BLOCKS = "A Korean thriller from the 1990s on Filmin under 90 minutes"


def _turn(client: TestClient, message: str, conversation_id: str | None = None, **extra) -> dict:
    body = {"message": message, "candidates": MOVIES, "profiles": [], "limit": 5, **extra}
    if conversation_id:
        body["conversationId"] = conversation_id
    response = client.post("/agent", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_nothing_fitting_pauses_with_a_proposal_to_loosen_one_constraint():
    with TestClient(app) as client:
        body = _turn(client, "A Korean thriller from the 1990s")
    assert body["recommendations"] == []
    assert body["pause"] == {"type": "relaxation", "constraint": "yearRange", "from": [1990, 1999], "to": None}
    assert body["meta"]["candidatesMeetingConstraints"] == 0
    assert "propose_relaxation" not in [step["tool"] for step in body["meta"]["trace"]]  # waiting, not done


def test_accepting_a_relaxation_returns_a_shortlist_that_meets_the_relaxed_constraints():
    with TestClient(app) as client:
        paused = _turn(client, "A Korean thriller from the 1990s")
        body = _turn(
            client, "Yes, any year is fine", paused["conversationId"], decision="accept", answering=paused["pause"]
        )
    assert "pause" not in body
    assert "yearRange" not in body["preferences"]
    assert body["preferences"]["languages"] == ["ko"]
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["memories"]
    assert [step["tool"] for step in body["meta"]["trace"]][:2] == ["propose_relaxation", "filter_candidates"]


def test_rejections_walk_the_agreed_order_then_say_nothing_fits():
    with TestClient(app) as client:
        body = _turn(client, EVERYTHING_BLOCKS)
        conversation_id = body["conversationId"]
        proposals = []
        while "pause" in body:
            proposals.append((body["pause"]["constraint"], body["pause"]["from"], body["pause"]["to"]))
            body = _turn(client, "No, keep that", conversation_id, decision="reject", answering=body["pause"])
    assert proposals == [
        ("yearRange", [1990, 1999], None),
        ("languages", ["ko"], []),
        ("genres", ["Thriller"], []),
        ("maxRuntime", 90, 105),
        ("platforms", ["Filmin"], []),
    ]
    assert body["recommendations"] == []
    assert body["nothingFits"] == {"tightestConstraint": "yearRange"}
    assert body["preferences"]["maxRuntime"] == 90  # every rejected constraint was kept


def test_excluded_genres_are_never_proposed_for_relaxation():
    only_horror = [movie for movie in MOVIES if movie["id"] == "hereditary"]
    with TestClient(app) as client:
        body = _turn(client, "No horror please", candidates=only_horror)
    assert "pause" not in body
    assert body["nothingFits"] == {"tightestConstraint": "excludedGenres"}


def test_runtime_is_loosened_fifteen_minutes_at_a_time():
    with TestClient(app) as client:
        body = _turn(client, "A thriller under 100 minutes")
        assert body["pause"] == {"type": "relaxation", "constraint": "genres", "from": ["Thriller"], "to": []}
        body = _turn(client, "Keep the thriller", body["conversationId"], decision="reject", answering=body["pause"])
        assert body["pause"] == {"type": "relaxation", "constraint": "maxRuntime", "from": 100, "to": 115}
        body = _turn(client, "Sure", body["conversationId"], decision="accept", answering=body["pause"])
        assert body["pause"] == {"type": "relaxation", "constraint": "maxRuntime", "from": 115, "to": 130}
        body = _turn(client, "Sure", body["conversationId"], decision="accept", answering=body["pause"])
        assert body["pause"] == {"type": "relaxation", "constraint": "maxRuntime", "from": 130, "to": 145}
        body = _turn(client, "Sure", body["conversationId"], decision="accept", answering=body["pause"])
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["memories"]
    assert body["preferences"]["maxRuntime"] == 145


def test_a_new_message_while_paused_ends_the_pause():
    with TestClient(app) as client:
        paused = _turn(client, "A Korean thriller from the 1990s")
        body = _turn(client, "Start over. Something funny", paused["conversationId"])
    assert "pause" not in body
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["clue"]


def test_a_decision_without_a_pending_pause_is_an_ordinary_turn():
    with TestClient(app) as client:
        body = _turn(client, "Something funny", decision="accept")
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["clue"]


@pytest.fixture
def tiny_step_limit():
    app.dependency_overrides[step_limit] = lambda: 3
    yield
    app.dependency_overrides.pop(step_limit, None)


def test_a_turn_that_exceeds_the_step_limit_ends_and_says_so(tiny_step_limit):
    with TestClient(app) as client:
        body = _turn(client, "Something funny")
    assert body["recommendations"] == []
    assert body["meta"]["trace"][-1] == {
        "tool": "step_limit",
        "status": "skipped",
        "durationMs": 0,
        "observation": "stopped after 3 steps",
    }


def test_a_decision_for_a_different_proposal_is_treated_as_an_ordinary_turn():
    with TestClient(app) as client:
        paused = _turn(client, "A Korean thriller from the 1990s")
        stale = {"type": "relaxation", "constraint": "languages", "from": ["ko"], "to": []}
        body = _turn(client, "Yes, loosen it", paused["conversationId"], decision="accept", answering=stale)
    assert body["pause"]["constraint"] == "yearRange"  # the same proposal again: nothing was loosened
    assert body["preferences"]["languages"] == ["ko"]


def test_runtime_is_not_proposed_once_it_already_covers_the_longest_movie():
    with TestClient(app) as client:
        body = _turn(client, "A Korean comedy under 140 minutes")
        constraints = []
        while "pause" in body:
            constraints.append(body["pause"]["constraint"])
            body = _turn(client, "No", body["conversationId"], decision="reject", answering=body["pause"])
    assert constraints == ["languages", "genres"]
    assert "nothingFits" in body


def test_accepting_keeps_the_model_extraction_metadata_of_the_turn():
    from reel_circle_agent.api import configured_chat_model

    class StuckModel:
        def with_structured_output(self, *_args, **_kwargs):
            return self

        async def ainvoke(self, _messages):
            return {
                "genres": ["Thriller"],
                "excludedGenres": [],
                "platforms": [],
                "moods": [],
                "maxRuntime": None,
                "yearRange": [1990, 1999],
                "languages": ["ko"],
                "peopleCount": None,
                "viewingContext": None,
                "clarificationQuestion": None,
            }

    app.dependency_overrides[configured_chat_model] = lambda: StuckModel()
    try:
        with TestClient(app) as client:
            paused = _turn(client, "a nineties korean thriller")
            body = _turn(client, "Yes", paused["conversationId"], decision="accept", answering=paused["pause"])
    finally:
        app.dependency_overrides.pop(configured_chat_model, None)
    assert (body["meta"]["extractionMode"], body["meta"]["modelStatus"]) == ("model", "succeeded")
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["memories"]
