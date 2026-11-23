"""The recommendation agent as a LangGraph state graph.

    extract_preferences -> filter_candidates -> semantic_retrieval -> rank_candidates
        -> validate_results -> respond

Node names are the trace tool names the browser already knows. The graph is linear for now;
pauses (clarifying questions, relaxations) arrive in later tickets as conditional edges on this
same graph. The run config carries the outside world: the `http_client` for the embedding
provider and Qdrant, and the `chat_model` and `model_retry` policy for extraction, so callers and
tests choose them. Every node except `respond` appends one step
to the response trace, so the trace shows what actually ran.

ADR 0002: with a checkpointer, only `preferences` and `last_turn_at` are persisted per
conversation. The turn's input (message, persons, candidate pool) travels in the run config as
`turn`, because LangGraph checkpoints the graph input itself; intermediate results are
`UntrackedValue` channels that live for one run. Neither is ever written to storage.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .model_extraction import DEFAULT_RETRY_POLICY, extract_with_model, merge_model_extraction
from .preferences import Preferences, create_conversation_preferences, update_conversation_preferences
from .retrieval import retrieve_semantically
from .scoring import Movie, Person, Recommendation, broken_constraints, explain_preference_match, score_candidates

StepStatus = Literal["completed", "skipped"]


class TraceStep(TypedDict):
    tool: str
    status: StepStatus
    durationMs: int
    observation: str


class TurnInput(TypedDict):
    """Per-turn input, as the browser sent it. Passed in the run config, never as graph state."""

    message: str
    incoming_preferences: Preferences | None
    candidate_pool: list[Movie]
    persons: list[Person]
    limit: int
    started_at: float


def _turn(config: RunnableConfig) -> TurnInput:
    return config["configurable"]["turn"]


class TurnState(TypedDict, total=False):
    """What the nodes build up; each node reads only keys earlier nodes have set."""

    # Persisted per conversation.
    preferences: Preferences
    last_turn_at: int
    # Per run only.
    extraction: Annotated[dict[str, Any], UntrackedValue]
    clarifying_question: Annotated[str | None, UntrackedValue]
    ranked: Annotated[list[Recommendation], UntrackedValue]
    candidates_meeting_constraints: Annotated[int, UntrackedValue]
    semantic: Annotated[dict[str, Any], UntrackedValue]
    recommendations: Annotated[list[dict[str, Any]], UntrackedValue]
    trace: Annotated[list[TraceStep], UntrackedValue]
    response: Annotated[dict[str, Any], UntrackedValue]


NodeResult = tuple[dict[str, Any], StepStatus, str]


def _status(did_work: bool) -> StepStatus:
    return "completed" if did_work else "skipped"


TracedNode = Callable[[TurnState, RunnableConfig], Awaitable[NodeResult]]


def _traced(node: TracedNode) -> Callable[[TurnState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """Runs a node and records it as one trace step named after it."""

    async def run(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        update, status, observation = await node(state, config)
        duration = round((time.perf_counter() - started) * 1000)
        step: TraceStep = {"tool": node.__name__, "status": status, "durationMs": duration, "observation": observation}
        return {**update, "trace": [*state.get("trace", []), step]}

    run.__name__ = node.__name__
    return run


def _latest_preferences(stored: Preferences | None, sent: Preferences | None) -> Preferences | None:
    """The conversation's stored preferences, unless the browser's copy is newer (a turn the service
    missed because the TypeScript fallback answered it). Without a browser copy, the stored one wins."""
    if stored and sent and (sent.get("lastUpdatedAt") or 0) > (stored.get("lastUpdatedAt") or 0):
        return sent
    return stored or sent


async def extract_preferences(state: TurnState, config: RunnableConfig) -> NodeResult:
    """Model extraction when a chat model is configured; deterministic rules otherwise or when it fails."""
    now = config["configurable"]["now_ms"]
    turn = _turn(config)
    current = _latest_preferences(state.get("preferences"), turn["incoming_preferences"]) or (
        create_conversation_preferences(now)
    )
    settings = config["configurable"]
    outcome = await extract_with_model(
        settings.get("chat_model"), turn["message"], current, settings.get("model_retry", DEFAULT_RETRY_POLICY)
    )
    if outcome.extracted is not None:
        update = {
            "preferences": merge_model_extraction(current, outcome.extracted, now),
            "extraction": {"mode": "model", "status": outcome.status},
            "clarifying_question": outcome.extracted["clarificationQuestion"],
        }
        return {**update, "last_turn_at": now}, "completed", "model structured extraction"
    update = {
        "preferences": update_conversation_preferences(current, turn["message"], now),
        "extraction": {"mode": "deterministic", "status": outcome.status, "error": outcome.error},
    }
    return {**update, "last_turn_at": now}, "completed", "deterministic structured extraction"


async def filter_candidates(state: TurnState, config: RunnableConfig) -> NodeResult:
    turn = _turn(config)
    pool = turn["candidate_pool"]
    ranked = score_candidates(pool, turn["persons"], state["preferences"])
    update = {"ranked": ranked, "candidates_meeting_constraints": len(ranked)}
    return update, _status(bool(pool)), f"{len(ranked)}/{len(pool)} candidates satisfy hard constraints"


async def semantic_retrieval(state: TurnState, config: RunnableConfig) -> NodeResult:
    ranked = state["ranked"]
    result = await retrieve_semantically(_turn(config)["message"], ranked, config["configurable"]["http_client"])
    matches = sum(1 for recommendation in result["recommendations"] if recommendation["semanticScore"] > 0)
    source = f"{result['retrievalMode']} · {result['embeddingModel']} · {result['dimensions']}d"
    update = {"ranked": result["recommendations"], "semantic": result}
    return update, _status(bool(ranked)), f"{source} · {matches}/{len(ranked)} matches"


def _shortlist_entry(recommendation: Recommendation, preferences: Preferences) -> dict[str, Any]:
    movie, terms = recommendation["movie"], recommendation["semanticTerms"]
    signals = f" Semantic signals: {', '.join(terms[:3])}." if terms else ""
    return {
        "movie": movie,
        "score": recommendation["score"],
        "matchedPreferences": recommendation["reasons"],
        "explanation": f"{explain_preference_match(movie, preferences)}{signals}",
        "semanticScore": recommendation["semanticScore"],
        "semanticTerms": terms,
    }


async def rank_candidates(state: TurnState, config: RunnableConfig) -> NodeResult:
    limit = _turn(config)["limit"]
    shortlist = [_shortlist_entry(item, state["preferences"]) for item in state["ranked"][:limit]]
    return {"recommendations": shortlist}, _status(bool(shortlist)), f"{len(shortlist)} recommendations returned"


async def validate_results(state: TurnState, config: RunnableConfig) -> NodeResult:
    """Guardrail: drops any recommendation that breaks a hard constraint instead of re-ranking.

    Filtering runs before ranking, so a violation can only come from a bug; it is reported in the
    trace so the evals count it as a failure.
    """
    kept, dropped = [], []
    for recommendation in state["recommendations"]:
        broken = broken_constraints(recommendation["movie"], state["preferences"])
        (dropped if broken else kept).append(recommendation)
    observation = f"{len(dropped)} hard-constraint violations"
    if dropped:
        observation += f" dropped: {', '.join(recommendation['movie']['id'] for recommendation in dropped)}"
    return {"recommendations": kept}, "completed", observation


def respond(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    """Assembles the response in the shape the browser's agent client expects."""
    semantic, extraction = state["semantic"], state["extraction"]
    meta = {
        "extractionMode": extraction["mode"],
        "modelStatus": extraction["status"],
        "modelError": extraction.get("error"),
        "candidatesMeetingConstraints": state["candidates_meeting_constraints"],
        "semantic": {
            key: semantic[key]
            for key in ("retrievalMode", "embeddingMode", "embeddingModel", "dimensions", "fallbackReason")
            if key in semantic
        },
        "durationMs": round((time.perf_counter() - _turn(config)["started_at"]) * 1000),
        "trace": state["trace"],
    }
    response = {
        "preferences": state["preferences"],
        "recommendations": state["recommendations"],
        "clarificationQuestion": state.get("clarifying_question"),
        "meta": {key: value for key, value in meta.items() if value is not None},
    }
    # JSON.stringify drops undefined fields; dropping None keeps the same wire shape.
    return {"response": {key: value for key, value in response.items() if value is not None}}


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """The recommendation agent; with a checkpointer, each `thread_id` is one remembered conversation."""
    graph = StateGraph(TurnState)
    steps = [extract_preferences, filter_candidates, semantic_retrieval, rank_candidates, validate_results]
    for step in steps:
        graph.add_node(step.__name__, _traced(step))
    graph.add_node("respond", respond)
    graph.add_edge(START, steps[0].__name__)
    for current, following in zip(steps, steps[1:], strict=False):
        graph.add_edge(current.__name__, following.__name__)
    graph.add_edge(steps[-1].__name__, "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
