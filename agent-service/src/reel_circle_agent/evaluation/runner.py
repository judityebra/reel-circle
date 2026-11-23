"""Replays scenarios through the recommendation agent's HTTP endpoint and scores every turn.

A scenario is a JSON file: an id, a language, tags, optional `candidatePool` and `persons`
(defaulting to the shared parity pool), and turns. A turn has a `message` and/or a `decision`
("accept" or "reject") that answers the pause of the turn before, and an `expect` block read by
the evaluators. Exclusions held after one turn are checked on the next automatically.

Runs are deterministic: no chat model, embedding provider, Qdrant, storage or tracing. The runner
clears those environment variables while it runs (and restores them), so it is meant for the CLI
and tests, not for concurrent use. It needs a source checkout: it reads the shared parity fixtures.
"""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from ..api import app, configured_chat_model
from .evaluators import EVALUATORS, evaluate_turn

REPO_ROOT = Path(__file__).resolve().parents[4]
SCENARIOS_DIR = REPO_ROOT / "agent-service" / "evals" / "scenarios"
PARITY_FIXTURES = REPO_ROOT / "fixtures" / "agent-parity.json"
_PROVIDER_VARIABLES = [
    "EMBEDDING_PROVIDER", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_API_KEY", "EMBEDDING_DIMENSIONS",
    "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION", "CHECKPOINT_DATABASE_URL",
    "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING",
]  # fmt: skip
_DEFAULT_MESSAGES = {"accept": "Yes, loosen it", "reject": "No, keep it"}


@dataclass
class Scenario:
    id: str
    lang: str
    tags: list[str]
    turns: list[dict[str, Any]]
    candidate_pool: list[dict[str, Any]] | None = None
    persons: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class Failure:
    scenario: str
    turn: int
    evaluator: str
    reason: str


@dataclass
class Scorecard:
    scenarios: int = 0
    turns: int = 0
    successful_turns: int = 0
    passed_scenarios: int = 0
    passes: dict[str, int] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)

    @property
    def scenario_pass_rate(self) -> float:
        return self.passed_scenarios / self.scenarios if self.scenarios else 0.0

    def render(self) -> str:
        def row(label: str, passed: int, total: int) -> str:
            share = f"{passed / total:.0%}" if total else "-"
            return f"  {label:<18} {passed:>3}/{total:<3} {share:>5}"

        lines = [f"Agent scorecard: {self.scenarios} scenarios, {self.turns} turns"]
        lines += [row(name, passed, self.turns) for name, passed in self.passes.items()]
        lines += [
            row("successful turns", self.successful_turns, self.turns),
            row("scenarios", self.passed_scenarios, self.scenarios),
        ]
        if self.failures:
            lines.append("Failures:")
            lines += [f"  {f.scenario} turn {f.turn} [{f.evaluator}]: {f.reason}" for f in self.failures]
        return "\n".join(lines)


def load_scenarios(directory: Path = SCENARIOS_DIR) -> list[Scenario]:
    scenarios = []
    for path in sorted(Path(directory).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        scenarios.append(
            Scenario(
                id=data["id"],
                lang=data["lang"],
                tags=data.get("tags", []),
                turns=data["turns"],
                candidate_pool=data.get("candidatePool"),
                persons=data.get("persons"),
            )
        )
    return scenarios


@contextmanager
def _deterministic_agent() -> Iterator[TestClient]:
    """The service in process, with no model, providers, Qdrant or persistent storage."""
    saved = {name: os.environ.pop(name) for name in _PROVIDER_VARIABLES if name in os.environ}
    previous_model = app.dependency_overrides.get(configured_chat_model)
    app.dependency_overrides[configured_chat_model] = lambda: None
    try:
        with TestClient(app) as client:
            yield client
    finally:
        if previous_model is None:
            app.dependency_overrides.pop(configured_chat_model, None)
        else:
            app.dependency_overrides[configured_chat_model] = previous_model
        os.environ.update(saved)


def _replay(client: TestClient, scenario: Scenario, defaults: dict[str, Any]) -> list[tuple[dict, dict, str | None]]:
    """Sends each turn in order; returns (turn, response, problem) triples, where a problem is a scenario
    mistake the runner found (a decision with no pause to answer)."""
    conversation_id, previous, exchanges = None, None, []
    for turn in scenario.turns:
        problem = None
        body = {
            "message": turn.get("message") or _DEFAULT_MESSAGES.get(turn.get("decision"), ""),
            "candidates": scenario.candidate_pool if scenario.candidate_pool is not None else defaults["candidatePool"],
            "profiles": scenario.persons if scenario.persons is not None else defaults["persons"],
            "limit": defaults["limit"],
        }
        if conversation_id:
            body["conversationId"] = conversation_id
        if "decision" in turn:
            if previous and previous.get("pause"):
                body |= {"decision": turn["decision"], "answering": previous["pause"]}
            else:
                problem = "a decision, but the turn before left no pause to answer"
        response = client.post("/agent", json=body)
        response.raise_for_status()
        previous = response.json()
        conversation_id = previous["conversationId"]
        exchanges.append((turn, previous, problem))
    return exchanges


def run_scenarios(scenarios: list[Scenario]) -> Scorecard:
    if not PARITY_FIXTURES.exists():
        raise FileNotFoundError(f"reel-eval needs a source checkout; {PARITY_FIXTURES} is missing")
    defaults = json.loads(PARITY_FIXTURES.read_text(encoding="utf-8"))
    scorecard = Scorecard(passes={evaluator.__name__.removeprefix("evaluate_"): 0 for evaluator in EVALUATORS})
    with _deterministic_agent() as client:
        for scenario in scenarios:
            scenario_passed, earlier_exclusions = True, []
            for index, (turn, response, problem) in enumerate(_replay(client, scenario, defaults), start=1):
                scorecard.turns += 1
                if problem:
                    scorecard.failures.append(Failure(scenario.id, index, "runner", problem))
                    scenario_passed = False
                    continue
                scores = evaluate_turn(turn.get("expect", {}), response, earlier_exclusions)
                earlier_exclusions = response["preferences"].get("excludedGenres", [])
                for score in scores:
                    scorecard.passes[score.name] += score.passed
                    if not score.passed:
                        scorecard.failures.append(Failure(scenario.id, index, score.name, score.reason))
                turn_passed = all(score.passed for score in scores)
                scorecard.successful_turns += turn_passed
                scenario_passed &= turn_passed
            scorecard.scenarios += 1
            scorecard.passed_scenarios += scenario_passed
    return scorecard
