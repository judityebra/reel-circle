"""Model-based preference extraction, with a fake chat model and no network.

The model only interprets the message; its output is bounded to the allowed genres, moods and
value ranges, merged into the conversation's preferences the way the TypeScript agent does, and any
failure falls back to the deterministic extractor after a bounded number of retries.
"""

import json

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
EMPTY = {
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


class StatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeChatModel:
    """Answers structured-output calls from a script: each item is an output dict or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[list] = []
        self.structured_with: dict = {}

    def with_structured_output(self, schema, **options):
        self.structured_with = {"schema": schema, **options}
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def chat_model(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_RETRY_DELAY_SECONDS", "0")

    def install(model: FakeChatModel | None) -> FakeChatModel | None:
        app.dependency_overrides[configured_chat_model] = lambda: model
        return model

    yield install
    app.dependency_overrides.clear()


def _turn(message: str, preferences: dict | None = None) -> dict:
    body = {"message": message, "candidates": MOVIES, "profiles": [], "limit": 5}
    if preferences is not None:
        body["preferences"] = preferences
    response = TestClient(app).post("/agent", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_configured_model_interprets_the_turn(chat_model):
    model = chat_model(FakeChatModel({**EMPTY, "genres": ["Comedy"], "maxRuntime": 95}))
    body = _turn("something to laugh at, not too long")
    assert body["meta"]["extractionMode"] == "model"
    assert body["meta"]["modelStatus"] == "succeeded"
    assert "modelError" not in body["meta"]
    assert body["preferences"]["genres"] == ["Comedy"]
    assert body["preferences"]["maxRuntime"] == 95
    assert [item["movie"]["id"] for item in body["recommendations"]] == ["clue"]
    assert body["meta"]["trace"][0]["observation"] == "model structured extraction"
    assert model.structured_with["method"] == "json_schema"


def test_the_model_sees_the_current_preferences_and_the_latest_turn(chat_model):
    model = chat_model(FakeChatModel(EMPTY))
    current = {"genres": [], "excludedGenres": ["Horror"], "platforms": [], "moods": [], "lastUpdatedAt": 0}
    _turn("something darker", current)
    system, user = model.calls[0]
    assert "movie-viewing preferences" in system[1]
    assert json.loads(user[1]) == {"current": current, "latestTurn": "something darker"}


def test_model_output_is_bounded_to_allowed_values(chat_model):
    chat_model(
        FakeChatModel(
            {
                **EMPTY,
                "genres": ["Western", "Comedy", "Comedy"],
                "moods": ["Spooky", "Playful"],
                "maxRuntime": 999,
                "peopleCount": 50.7,
                "viewingContext": "x" * 120,
                "clarificationQuestion": "?" * 300,
            }
        )
    )
    body = _turn("anything")
    assert body["preferences"]["genres"] == ["Comedy"]
    assert body["preferences"]["moods"] == ["Playful"]
    assert body["preferences"]["maxRuntime"] == 600
    assert body["preferences"]["peopleCount"] == 20
    assert len(body["preferences"]["viewingContext"]) == 80
    assert len(body["clarificationQuestion"]) == 240


def test_empty_model_fields_keep_earlier_preferences(chat_model):
    chat_model(FakeChatModel({**EMPTY, "maxRuntime": 100}))
    current = {"genres": ["Comedy"], "excludedGenres": ["Horror"], "platforms": [], "moods": [], "lastUpdatedAt": 0}
    body = _turn("make it shorter", current)
    assert body["preferences"]["excludedGenres"] == ["Horror"]
    assert body["preferences"]["genres"] == ["Comedy"]
    assert body["preferences"]["maxRuntime"] == 100


def test_transient_failures_are_retried_before_giving_up(chat_model):
    model = chat_model(FakeChatModel(TimeoutError("slow"), StatusError(503), {**EMPTY, "genres": ["Drama"]}))
    body = _turn("a drama")
    assert len(model.calls) == 3
    assert (body["meta"]["extractionMode"], body["meta"]["modelStatus"]) == ("model", "succeeded")


@pytest.mark.parametrize(
    ("failure", "reason", "attempts"),
    [
        (StatusError(429), "http-429", 3),
        (TimeoutError("slow"), "timeout", 3),
        (ValueError("not JSON"), "invalid-output", 3),
        (StatusError(401), "http-401", 1),
    ],
    ids=["rate-limited", "timeout", "invalid-output", "unauthorised-is-not-retried"],
)
def test_a_failing_model_falls_back_to_deterministic_extraction(chat_model, failure, reason, attempts):
    model = chat_model(FakeChatModel(*[failure] * 3))
    body = _turn("Something funny under 90 minutes")
    assert len(model.calls) == attempts
    assert body["meta"]["extractionMode"] == "deterministic"
    assert body["meta"]["modelStatus"] == "failed"
    assert body["meta"]["modelError"] == reason
    assert body["preferences"]["genres"] == ["Comedy"]
    assert body["preferences"]["maxRuntime"] == 90
    assert body["meta"]["trace"][0]["observation"] == "deterministic structured extraction"


def test_output_that_is_not_an_object_counts_as_invalid(chat_model):
    chat_model(FakeChatModel("Comedy please", "still text", ["a list"]))
    body = _turn("Something funny")
    assert (body["meta"]["modelStatus"], body["meta"]["modelError"]) == ("failed", "invalid-output")


def test_without_a_configured_model_extraction_is_deterministic(chat_model):
    chat_model(None)
    body = _turn("Something funny")
    assert (body["meta"]["extractionMode"], body["meta"]["modelStatus"]) == ("deterministic", "not-configured")


@pytest.fixture
def clean_model_environment(monkeypatch):
    for name in [
        "AGENT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    configured_chat_model.cache_clear()
    yield monkeypatch
    configured_chat_model.cache_clear()


def test_no_model_is_configured_without_a_model_setting_or_openai_key(clean_model_environment):
    assert configured_chat_model() is None


def test_an_openai_key_alone_selects_gpt_4_1_mini_at_temperature_zero(clean_model_environment):
    clean_model_environment.setenv("OPENAI_API_KEY", "sk-test")
    clean_model_environment.setenv("OPENAI_BASE_URL", "https://llm.example.test/v1/")
    model = configured_chat_model()
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-4.1-mini"
    assert model.temperature == 0
    assert model.openai_api_base == "https://llm.example.test/v1"


def test_claude_is_selected_by_configuration_without_sampling_parameters(clean_model_environment):
    clean_model_environment.setenv("AGENT_MODEL", "anthropic:claude-haiku-5-5")
    clean_model_environment.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = configured_chat_model()
    assert type(model).__name__ == "ChatAnthropic"
    assert model.model == "claude-haiku-5-5"
    assert model.temperature is None


def test_gemini_is_selected_by_configuration(clean_model_environment):
    clean_model_environment.setenv("AGENT_MODEL", "google_genai:gemini-2.5-flash")
    clean_model_environment.setenv("GOOGLE_API_KEY", "google-test")
    model = configured_chat_model()
    assert type(model).__name__ == "ChatGoogleGenerativeAI"
    assert model.model.endswith("gemini-2.5-flash")
