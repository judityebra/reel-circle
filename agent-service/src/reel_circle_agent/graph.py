"""The recommendation agent as a LangGraph state graph.

    extract_preferences -> filter_candidates -> semantic_retrieval -> rank_candidates
        -> validate_results -> respond

Node names are the trace tool names the browser already knows. The graph is linear for now;
pauses (clarifying questions, relaxations) and server-side conversation
state arrive in later tickets as conditional edges and a checkpointer on
this same graph. The run config carries the outside world: the `http_client` for the embedding
provider and Qdrant, and the `chat_model` and `model_retry` policy for extraction, so callers and
tests choose them. Every node except `respond` appends one step
to the response trace, so the trace shows what actually ran.

ADR 0002: `message`, `persons` and `candidate_pool` are per-turn input. When a checkpointer is
added, they must stay out of the persisted state.
"""

import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
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
    """Per-turn input, as the browser sent it."""

    message: str
    incoming_preferences: Preferences | None
    candidate_pool: list[Movie]
    persons: list[Person]
    limit: int
    started_at: float


class TurnState(TurnInput):
    """The input plus what the nodes build up; each node reads only keys earlier nodes have set."""

    preferences: Preferences
    extraction: dict[str, Any]
    clarifying_question: str | None
    ranked: list[Recommendation]
    candidates_meeting_constraints: int
    semantic: dict[str, Any]
    recommendations: list[dict[str, Any]]
    trace: Annotated[list[TraceStep], operator.add]
    response: dict[str, Any]


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
        return {**update, "trace": [step]}

    run.__name__ = node.__name__
    return run


async def extract_preferences(state: TurnState, config: RunnableConfig) -> NodeResult:
    """Model extraction when a chat model is configured; deterministic rules otherwise or when it fails."""
    now = int(time.time() * 1000)
    current = state.get("incoming_preferences") or create_conversation_preferences(now)
    settings = config["configurable"]
    outcome = await extract_with_model(
        settings.get("chat_model"), state["message"], current, settings.get("model_retry", DEFAULT_RETRY_POLICY)
    )
    if outcome.extracted is not None:
        update = {
            "preferences": merge_model_extraction(current, outcome.extracted, now),
            "extraction": {"mode": "model", "status": outcome.status},
            "clarifying_question": outcome.extracted["clarificationQuestion"],
        }
        return update, "completed", "model structured extraction"
    update = {
        "preferences": update_conversation_preferences(current, state["message"], now),
        "extraction": {"mode": "deterministic", "status": outcome.status, "error": outcome.error},
    }
    return update, "completed", "deterministic structured extraction"


async def filter_candidates(state: TurnState, config: RunnableConfig) -> NodeResult:
    pool = state["candidate_pool"]
    ranked = score_candidates(pool, state["persons"], state["preferences"])
    update = {"ranked": ranked, "candidates_meeting_constraints": len(ranked)}
    return update, _status(bool(pool)), f"{len(ranked)}/{len(pool)} candidates satisfy hard constraints"


async def semantic_retrieval(state: TurnState, config: RunnableConfig) -> NodeResult:
    ranked = state["ranked"]
    result = await retrieve_semantically(state["message"], ranked, config["configurable"]["http_client"])
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
    shortlist = [_shortlist_entry(item, state["preferences"]) for item in state["ranked"][: state["limit"]]]
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


def respond(state: TurnState) -> dict[str, Any]:
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
        "durationMs": round((time.perf_counter() - state["started_at"]) * 1000),
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


def build_graph():
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
    return graph.compile()


recommendation_agent = build_graph()
