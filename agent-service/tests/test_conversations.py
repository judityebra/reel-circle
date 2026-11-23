"""Conversations remembered on the server (ADR 0002): preferences only, per conversation, for 30 days."""

import asyncio
import re

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from reel_circle_agent.api import app, clock
from reel_circle_agent.conversations import ConversationStore

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
        "id": "hereditary",
        "title": "Hereditary",
        "year": 2018,
        "runtime": 127,
        "genres": ["Horror"],
        "platforms": [],
        "rating": 4.0,
    },
]
PERSONS = [
    {
        "id": "maya",
        "name": "Maya Quillfeather",
        "color": "#f36b35",
        "preferredGenres": ["Comedy"],
        "ratings": [],
        "watched": [],
    }
]
DAY_SECONDS = 24 * 60 * 60


@pytest.fixture
def sqlite_conversations(tmp_path, monkeypatch):
    path = tmp_path / "conversations.db"
    monkeypatch.setenv("CHECKPOINT_DATABASE_URL", f"sqlite:///{path.as_posix()}")
    return path


@pytest.fixture
def frozen_clock():
    now = {"seconds": 1_800_000_000.0}
    app.dependency_overrides[clock] = lambda: now["seconds"]
    yield now
    app.dependency_overrides.pop(clock, None)


def _turn(client: TestClient, message: str, conversation_id: str | None = None, **extra) -> dict:
    body = {"message": message, "candidates": MOVIES, "profiles": PERSONS, "limit": 5, **extra}
    if conversation_id:
        body["conversationId"] = conversation_id
    response = client.post("/agent", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_new_conversation_gets_an_id():
    with TestClient(app) as client:
        body = _turn(client, "Something funny")
    assert re.fullmatch(r"[0-9a-f-]{36}", body["conversationId"])


def test_earlier_exclusions_stay_in_force_without_resending_preferences():
    with TestClient(app) as client:
        first = _turn(client, "No horror tonight")
        second = _turn(client, "Something under two and a half hours", first["conversationId"])
    assert second["conversationId"] == first["conversationId"]
    assert second["preferences"]["excludedGenres"] == ["Horror"]
    assert [item["movie"]["id"] for item in second["recommendations"]] == ["clue"]


def test_conversations_are_kept_apart():
    with TestClient(app) as client:
        _turn(client, "No horror tonight")
        other = _turn(client, "Something scary")
    assert other["preferences"]["excludedGenres"] == []


def test_a_conversation_continues_after_a_service_restart(sqlite_conversations):
    with TestClient(app) as client:
        conversation_id = _turn(client, "No horror tonight")["conversationId"]
    with TestClient(app) as restarted:
        body = _turn(restarted, "Something funny", conversation_id)
    assert body["preferences"]["excludedGenres"] == ["Horror"]
    assert body["preferences"]["genres"] == ["Comedy"]


def test_messages_persons_and_the_candidate_pool_never_reach_storage(sqlite_conversations):
    with TestClient(app) as client:
        _turn(client, "No horror tonight, zebra-umbrella-secret")
    stored = sqlite_conversations.read_bytes()
    assert b"Horror" in stored  # the preferences are stored...
    for private in [b"zebra-umbrella-secret", b"Maya Quillfeather", b"Hereditary", b"Clue"]:
        assert private not in stored  # ...but nothing the user typed, nobody's profile and no movie


def test_a_conversation_older_than_30_days_starts_fresh(frozen_clock):
    with TestClient(app) as client:
        conversation_id = _turn(client, "No horror tonight")["conversationId"]
        frozen_clock["seconds"] += 29 * DAY_SECONDS
        assert _turn(client, "Something funny", conversation_id)["preferences"]["excludedGenres"] == ["Horror"]
        frozen_clock["seconds"] += 31 * DAY_SECONDS
        body = _turn(client, "Something funny", conversation_id)
    assert body["preferences"]["excludedGenres"] == []


def test_deleting_a_conversation_removes_its_preferences():
    with TestClient(app) as client:
        conversation_id = _turn(client, "No horror tonight")["conversationId"]
        assert client.delete(f"/agent/conversations/{conversation_id}").status_code == 204
        body = _turn(client, "Something funny", conversation_id)
    assert body["preferences"]["excludedGenres"] == []


@pytest.mark.parametrize("conversation_id", ["../etc", "x" * 65, "has space", 42])
def test_malformed_conversation_ids_are_rejected(conversation_id):
    with TestClient(app) as client:
        response = client.post("/agent", json={"message": "hi", "conversationId": conversation_id})
        assert response.status_code == 400
        assert client.delete("/agent/conversations/has%20space").status_code == 400


def test_the_sweep_deletes_every_conversation_idle_for_30_days():
    saver = InMemorySaver()
    store = ConversationStore(saver)

    async def scenario():
        from reel_circle_agent.graph import build_graph

        agent = build_graph(saver)
        for thread, last_turn in [("old", 0), ("recent", 25 * DAY_SECONDS * 1000)]:
            config = {"configurable": {"thread_id": thread}}
            values = {"preferences": {"excludedGenres": []}, "last_turn_at": last_turn}
            await agent.aupdate_state(config, values, as_node="extract_preferences")
        removed = await store.sweep(now_ms=31 * DAY_SECONDS * 1000)
        remaining = {item.config["configurable"]["thread_id"] async for item in saver.alist(None)}
        return removed, remaining

    removed, remaining = asyncio.run(scenario())
    assert removed == ["old"] and remaining == {"recent"}


def test_a_newer_browser_copy_wins_after_a_turn_the_service_missed(frozen_clock):
    """If a turn fell back to the TypeScript pipeline, the browser's preferences are newer than the stored ones."""
    with TestClient(app) as client:
        conversation_id = _turn(client, "No horror tonight")["conversationId"]
        newer = {
            "genres": [],
            "excludedGenres": [],
            "platforms": [],
            "moods": [],
            "lastUpdatedAt": int(frozen_clock["seconds"] * 1000) + 60_000,
        }
        stale = {**newer, "excludedGenres": ["Comedy"], "lastUpdatedAt": 0}
        assert _turn(client, "Anything", conversation_id, preferences=newer)["preferences"]["excludedGenres"] == []
        body = _turn(client, "Anything", conversation_id, preferences=stale)
    assert body["preferences"]["excludedGenres"] == []
