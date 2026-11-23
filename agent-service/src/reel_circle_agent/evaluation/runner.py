"""Replays scenarios through the recommendation agent's HTTP endpoint and scores every turn.

A scenario is a JSON file: an id, a language, tags, optional `candidatePool` and `persons`
(defaulting to the shared parity pool), and turns. A turn has a `message` and/or a `decision`
("accept" or "reject") that answers the pause of the turn before, and an `expect` block read by
the evaluators. Exclusions held after one turn are checked on the next automatically. An optional
`model` list scripts the chat model: each turn the agent asks it, it returns the next extraction.
An `inject` block makes services fail on purpose (see faults.py) and a turn's `fallback`
expectation checks the agent reported the right fallback. A turn's `attack` expectation names an
instruction planted in movie text that the agent must never repeat, and a scenario's `privacy` list
names strings (a person's ratings and watch history, the user's message) that must appear in
neither the agent's storage nor its responses.

A `knownIssue` (an issue reference) marks a scenario the agent is known to fail, and `knownFailure`
pins how (reason fragments): it still runs and is listed apart, out of every gating count, but if it
fails any other way it fails the gate, and if it starts passing the scorecard says so.

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

from ..api import app, configured_chat_model, model_retry_policy, outbound_transport
from ..model_extraction import RetryPolicy
from .evaluators import EVALUATORS, evaluate_turn
from .faults import FailingModel, fault_environment, fault_transport

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
    model: list[dict[str, Any]] | None = None
    known_issue: str | None = None
    known_failure: list[str] = field(default_factory=list)
    inject: dict[str, str] | None = None
    privacy: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnownFailure:
    scenario: str
    issue: str
    reasons: list[str]


class ScriptedModel:
    """A chat model that answers each structured-output call with the next scripted extraction."""

    def __init__(self, answers: list[dict[str, Any]]):
        self.answers = list(answers)
        self.ran_out = False

    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> "ScriptedModel":
        return self

    async def ainvoke(self, _messages: Any) -> dict[str, Any]:
        if not self.answers:
            # The agent would quietly fall back to rule-based extraction; the runner reports it instead.
            self.ran_out = True
            raise RuntimeError("the scenario's model script ran out")
        return self.answers.pop(0)

    def problem(self) -> str | None:
        """A mistake in the script: too few answers for the turns, or answers never used."""
        if self.ran_out:
            return "the model script ran out of answers"
        return f"{len(self.answers)} scripted model answers were never used" if self.answers else None


@dataclass(frozen=True)
class Failure:
    scenario: str
    turn: int
    evaluator: str
    reason: str


@dataclass
class Scorecard:
    #: Scenarios that count towards the gate (everything without a known issue).
    scenarios: int = 0
    turns: int = 0
    successful_turns: int = 0
    passed_scenarios: int = 0
    #: evaluator -> [passed turns, turns it applied to]
    passes: dict[str, list[int]] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    #: [injected turns that reported the right fallback AND were successful turns, injected turns]
    graceful_fallbacks: list[int] = field(default_factory=lambda: [0, 0])
    known_failures: list[KnownFailure] = field(default_factory=list)
    fixed_known_issues: list[str] = field(default_factory=list)
    #: tag -> [passed scenarios, scenarios]
    by_tag: dict[str, list[int]] = field(default_factory=dict)

    @property
    def scenario_pass_rate(self) -> float:
        return self.passed_scenarios / self.scenarios if self.scenarios else 0.0

    def render(self) -> str:
        def row(label: str, passed: int, total: int) -> str:
            share = f"{passed / total:.0%}" if total else "-"
            return f"  {label:<18} {passed:>3}/{total:<3} {share:>5}"

        known = len(self.known_failures) + len(self.fixed_known_issues)
        lines = [
            f"Agent scorecard: {self.scenarios + known} scenarios ({known} with a known issue), {self.turns} turns"
        ]
        labels = {"fallback": "fallback reported", "injection": "attacks resisted"}
        lines += [
            row(labels.get(name, name), passed, applied) for name, (passed, applied) in self.passes.items() if applied
        ]
        if self.graceful_fallbacks[1]:
            lines.append(row("graceful fallback", *self.graceful_fallbacks))
        lines += [
            row("successful turns", self.successful_turns, self.turns),
            row("scenarios", self.passed_scenarios, self.scenarios),
        ]
        if self.by_tag:
            lines.append("By tag:")
            lines += [row(tag, passed, total) for tag, (passed, total) in sorted(self.by_tag.items())]
        if self.failures:
            lines.append("Failures:")
            lines += [f"  {f.scenario} turn {f.turn} [{f.evaluator}]: {f.reason}" for f in self.failures]
        if self.known_failures or self.fixed_known_issues:
            lines.append("Known issues (not gating):")
            lines += [f"  {k.scenario} ({k.issue}): {'; '.join(k.reasons)}" for k in self.known_failures]
            lines += [f"  {scenario} now passes: remove its knownIssue" for scenario in self.fixed_known_issues]
        return "\n".join(lines)


def _dependency(scripted: ScriptedModel | None):
    """A dependency returning the scenario's one scripted model (or none). A closure, not a default
    argument: FastAPI copies default arguments on every request, which would replay the script."""
    return lambda: scripted


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
                model=data.get("model"),
                known_issue=data.get("knownIssue"),
                known_failure=data.get("knownFailure", []),
                inject=data.get("inject"),
                privacy=data.get("privacy", []),
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
        # Server errors become responses, so the runner records a 500 instead of crashing.
        with TestClient(app, raise_server_exceptions=False) as client:
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
        if response.status_code != 200:
            exchanges.append((turn, None, f"HTTP {response.status_code}: {response.text[:120]}"))
            continue
        previous = response.json()
        conversation_id = previous["conversationId"]
        exchanges.append((turn, previous, problem))
    return exchanges


@dataclass
class _ScenarioRun:
    """One scenario's results, before deciding whether they count towards the gate."""

    passed: bool = True
    turns: int = 0
    successful_turns: int = 0
    passes: dict[str, list[int]] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    graceful_fallbacks: list[int] = field(default_factory=lambda: [0, 0])


@contextmanager
def _injected_faults(scenario: Scenario) -> Iterator[None]:
    """Points the agent at failing fakes for the scenario's `inject` block, then puts everything back."""
    inject = scenario.inject or {}
    environment = fault_environment(inject)
    saved = {name: os.environ.get(name) for name in environment}
    os.environ.update(environment)
    previous = {key: app.dependency_overrides.get(key) for key in (outbound_transport, model_retry_policy)}
    app.dependency_overrides[outbound_transport] = lambda: fault_transport(inject)
    app.dependency_overrides[model_retry_policy] = lambda: RetryPolicy(base_delay_seconds=0, budget_seconds=2)
    try:
        yield
    finally:
        for key, override in previous.items():
            if override is None:
                app.dependency_overrides.pop(key, None)
            else:
                app.dependency_overrides[key] = override
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_one(client: TestClient, scenario: Scenario, defaults: dict[str, Any]) -> _ScenarioRun:
    run, earlier_exclusions = _ScenarioRun(), []
    scripted = ScriptedModel(scenario.model) if scenario.model else None
    model_fault = (scenario.inject or {}).get("model")
    app.dependency_overrides[configured_chat_model] = _dependency(
        FailingModel(model_fault) if model_fault else scripted
    )
    with _injected_faults(scenario):
        exchanges = _replay(client, scenario, defaults)
    for index, (turn, response, problem) in enumerate(exchanges, start=1):
        run.turns += 1
        if problem:
            run.failures.append(Failure(scenario.id, index, "runner", problem))
            run.passed = False
            continue
        scores = evaluate_turn(turn.get("expect", {}), response, earlier_exclusions)
        earlier_exclusions = response["preferences"].get("excludedGenres", [])
        for score in scores:
            tally = run.passes.setdefault(score.name, [0, 0])
            tally[0] += score.passed
            tally[1] += 1
            if not score.passed:
                run.failures.append(Failure(scenario.id, index, score.name, score.reason))
        turn_passed = all(score.passed for score in scores)
        if "fallback" in turn.get("expect", {}):
            # Graceful means the right fallback was reported AND the turn still succeeded.
            run.graceful_fallbacks[0] += turn_passed
            run.graceful_fallbacks[1] += 1
        run.successful_turns += turn_passed
        run.passed &= turn_passed
    if scenario.privacy:
        _check_privacy(client, scenario, exchanges, run)
    if scripted and scripted.problem():
        run.failures.append(Failure(scenario.id, run.turns, "runner", scripted.problem()))
        run.passed = False
    return run


def _stored_text(client: TestClient, conversation_ids: set[str]) -> str:
    """Everything the agent's checkpointer holds for these conversations (in-memory storage)."""
    saver = client.app.state.conversations.agent.checkpointer
    stored = [saver.storage.get(thread) for thread in conversation_ids]
    stored += [value for key, value in saver.writes.items() if key[0] in conversation_ids]
    return str(stored)


def _check_privacy(client: TestClient, scenario: Scenario, exchanges: list, run: "_ScenarioRun") -> None:
    """Fails the scenario if any private string reached the agent's storage or its responses."""
    responses = [response for _, response, _ in exchanges if response]
    conversation_ids = {response["conversationId"] for response in responses}
    seen = _stored_text(client, conversation_ids) + str(responses)
    leaked = [secret for secret in scenario.privacy if secret in seen]
    tally = run.passes.setdefault("privacy", [0, 0])
    tally[0] += not leaked
    tally[1] += 1
    if leaked:
        run.failures.append(Failure(scenario.id, len(exchanges), "privacy", f"leaked: {', '.join(leaked)}"))
        run.passed = False


def _fails_as_known(scenario: Scenario, run: _ScenarioRun) -> bool:
    """Whether every failure is one the known issue explains (by its pinned reason fragments)."""
    return bool(scenario.known_failure) and all(
        any(fragment in failure.reason for fragment in scenario.known_failure) for failure in run.failures
    )


def run_scenarios(scenarios: list[Scenario]) -> Scorecard:
    if not PARITY_FIXTURES.exists():
        raise FileNotFoundError(f"reel-eval needs a source checkout; {PARITY_FIXTURES} is missing")
    defaults = json.loads(PARITY_FIXTURES.read_text(encoding="utf-8"))
    scorecard = Scorecard(passes={evaluator.__name__.removeprefix("evaluate_"): [0, 0] for evaluator in EVALUATORS})
    with _deterministic_agent() as client:
        for scenario in scenarios:
            run = _run_one(client, scenario, defaults)
            if scenario.known_issue and run.passed:
                scorecard.fixed_known_issues.append(scenario.id)
                continue
            if scenario.known_issue and _fails_as_known(scenario, run):
                reasons = [f"turn {f.turn} [{f.evaluator}]: {f.reason}" for f in run.failures]
                scorecard.known_failures.append(KnownFailure(scenario.id, scenario.known_issue, reasons))
                continue
            if scenario.known_issue:
                new = "; ".join(f"turn {f.turn} [{f.evaluator}]: {f.reason}" for f in run.failures)
                run.failures = [
                    Failure(scenario.id, 0, "known_issue", f"{scenario.known_issue} failed for a new reason: {new}")
                ]
            scorecard.failures += run.failures
            scorecard.scenarios += 1
            scorecard.passed_scenarios += run.passed
            scorecard.turns += run.turns
            scorecard.successful_turns += run.successful_turns
            scorecard.graceful_fallbacks[0] += run.graceful_fallbacks[0]
            scorecard.graceful_fallbacks[1] += run.graceful_fallbacks[1]
            for name, (passed, applied) in run.passes.items():
                tally = scorecard.passes.setdefault(name, [0, 0])
                tally[0] += passed
                tally[1] += applied
            for tag in scenario.tags:
                tally = scorecard.by_tag.setdefault(tag, [0, 0])
                tally[0] += run.passed
                tally[1] += 1
    return scorecard
