"""The recommendation agent as a LangGraph state graph.

    extract_preferences -> filter_candidates -> semantic_retrieval -> rank_candidates
        -> validate_results -> respond
    extract_preferences --model found nothing usable--> ask_clarifying_question --pause-->
        (the user's reply arrives as a new turn)
    filter_candidates --nothing fits--> propose_relaxation --pause--> (user decides)
        accept: loosen one hard constraint -> filter_candidates
        reject: keep it -> filter_candidates -> next proposal, or "nothing fits"

Node names are the trace tool names the browser already knows. A pause is an `interrupt()`: the
run stops, the response carries the proposal, and the user's decision resumes the run with
`Command(resume=...)`. Any new message instead starts a fresh run, which ends the pause.

The run config carries the outside world: the `http_client` for the embedding provider and
Qdrant, and the `chat_model` and `model_retry` policy for extraction, so callers and tests choose
them. Every node except `respond` appends one step to the response trace, so the trace shows what
actually ran.

ADR 0002: with a checkpointer, only `preferences` and turn metadata (`last_turn_at`,
`rejected_relaxations`) are persisted per conversation, plus LangGraph's record of a pending pause.
The turn's input (message, persons, candidate pool) travels in the run config as `turn`, because
LangGraph checkpoints the graph input itself; intermediate results are `UntrackedValue` channels
that live for one run. Neither is ever written to storage.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .model_extraction import DEFAULT_RETRY_POLICY, extract_with_model, merge_model_extraction
from .preferences import Preferences, create_conversation_preferences, update_conversation_preferences
from .relaxation import Proposal, RelaxationDecision, apply_relaxation, next_relaxation, tightest_constraint
from .retrieval import retrieve_semantically
from .scoring import Movie, Person, Recommendation, broken_constraints, explain_preference_match, score_candidates
from .semantic import DIMENSIONS, LOCAL_MODEL

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

    # Persisted per conversation: preferences and turn metadata (ADR 0002).
    preferences: Preferences
    last_turn_at: int
    extraction: dict[str, Any]
    rejected_relaxations: list[str]
    # Per run only.
    clarifying_question: Annotated[str | None, UntrackedValue]
    needs_clarification: Annotated[bool, UntrackedValue]
    ranked: Annotated[list[Recommendation], UntrackedValue]
    candidates_meeting_constraints: Annotated[int, UntrackedValue]
    semantic: Annotated[dict[str, Any], UntrackedValue]
    recommendations: Annotated[list[dict[str, Any]], UntrackedValue]
    trace: Annotated[list[TraceStep], UntrackedValue]
    response: Annotated[dict[str, Any], UntrackedValue]


NodeResult = tuple[dict[str, Any], StepStatus, str]


def _status(did_work: bool) -> StepStatus:
    return "completed" if did_work else "skipped"


def _trace_step(tool: str, status: StepStatus, duration_ms: int, observation: str) -> TraceStep:
    return {"tool": tool, "status": status, "durationMs": duration_ms, "observation": observation}


TracedNode = Callable[[TurnState, RunnableConfig], Awaitable[NodeResult]]


def _traced(node: TracedNode) -> Callable[[TurnState, RunnableConfig], Awaitable[dict[str, Any]]]:
    """Runs a node and records it as one trace step named after it."""

    async def run(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        update, status, observation = await node(state, config)
        duration = round((time.perf_counter() - started) * 1000)
        return {**update, "trace": [*state.get("trace", []), _trace_step(node.__name__, status, duration, observation)]}

    run.__name__ = node.__name__
    return run


def _current_preferences(stored: Preferences | None, turn: TurnInput, now: int) -> Preferences:
    """The conversation's stored preferences, unless the browser's copy is newer (a turn the service
    missed because the TypeScript fallback answered it). Without either, a fresh conversation."""
    sent = turn["incoming_preferences"]
    if stored and sent and (sent.get("lastUpdatedAt") or 0) > (stored.get("lastUpdatedAt") or 0):
        return sent
    return stored or sent or create_conversation_preferences(now)


def _next_relaxation(state: TurnState, config: RunnableConfig) -> Proposal | None:
    """What the agent would offer to loosen next, given what the user already declined this turn."""
    rejected = state.get("rejected_relaxations", [])
    return next_relaxation(state["preferences"], rejected, _turn(config)["candidate_pool"])


CHOOSING_FIELDS = ("genres", "excludedGenres", "platforms", "moods", "maxRuntime", "yearRange", "languages")


def _adds_something_to_choose_by(before: Preferences, after: Preferences) -> bool:
    """Whether the turn changed a hard constraint or a mood. Group size and occasion are remembered
    but don't change which movies fit, so a turn with only those still gets a clarifying question."""
    return any(before.get(field) != after.get(field) for field in CHOOSING_FIELDS)


def _new_turn(now: int) -> dict[str, Any]:
    """Turn metadata that every new message resets, which also ends any pending pause."""
    return {"last_turn_at": now, "rejected_relaxations": []}


async def extract_preferences(state: TurnState, config: RunnableConfig) -> NodeResult:
    """Model extraction when a chat model is configured; deterministic rules otherwise or when it fails."""
    now = config["configurable"]["now_ms"]
    turn = _turn(config)
    current = _current_preferences(state.get("preferences"), turn, now)
    settings = config["configurable"]
    outcome = await extract_with_model(
        settings.get("chat_model"), turn["message"], current, settings.get("model_retry", DEFAULT_RETRY_POLICY)
    )
    if outcome.extracted is not None:
        question = outcome.extracted["clarificationQuestion"]
        merged = merge_model_extraction(current, outcome.extracted, now)
        update = {
            "preferences": merged,
            "extraction": {"mode": "model", "status": outcome.status},
            "clarifying_question": question,
            "needs_clarification": bool(question) and not _adds_something_to_choose_by(current, merged),
        }
        return {**update, **_new_turn(now)}, "completed", "model structured extraction"
    update = {
        "preferences": update_conversation_preferences(current, turn["message"], now),
        "extraction": {"mode": "deterministic", "status": outcome.status, "error": outcome.error},
    }
    return {**update, **_new_turn(now)}, "completed", "deterministic structured extraction"


def ask_clarifying_question(state: TurnState) -> dict[str, Any]:
    """Pauses with the model's question. The reply comes back as a new turn (the API never resumes
    this pause), which ends it; so a turn asks at most once, and only when a model interpreted it."""
    interrupt({"type": "clarifying_question", "question": state["clarifying_question"]})
    return {}


def route_after_extract(state: TurnState) -> str:
    return "ask_clarifying_question" if state.get("needs_clarification") else "filter_candidates"


async def filter_candidates(state: TurnState, config: RunnableConfig) -> NodeResult:
    turn = _turn(config)
    pool = turn["candidate_pool"]
    ranked = score_candidates(pool, turn["persons"], state["preferences"])
    update = {"ranked": ranked, "candidates_meeting_constraints": len(ranked)}
    return update, _status(bool(pool)), f"{len(ranked)}/{len(pool)} candidates satisfy hard constraints"


async def propose_relaxation(state: TurnState, config: RunnableConfig) -> NodeResult:
    """Pauses with one proposal to loosen a hard constraint; the user's decision resumes here."""
    proposal = _next_relaxation(state, config)
    decision: RelaxationDecision = interrupt({"type": "relaxation", **proposal})
    constraint = proposal["constraint"]
    if decision == "accept":
        update = {"preferences": apply_relaxation(state["preferences"], proposal)}
        return update, "completed", f"relaxed {constraint} from {proposal['from']} to {proposal['to']}"
    rejected = [*state.get("rejected_relaxations", []), constraint]
    return {"rejected_relaxations": rejected}, "completed", f"kept {constraint}"


def route_after_filter(state: TurnState, config: RunnableConfig) -> str:
    """Nothing fits and something can still be loosened: propose it. Otherwise carry on."""
    stuck = state["candidates_meeting_constraints"] == 0 and bool(_turn(config)["candidate_pool"])
    return "propose_relaxation" if stuck and _next_relaxation(state, config) else "semantic_retrieval"


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


LOCAL_SEMANTIC = {
    "retrievalMode": "local",
    "embeddingMode": "local",
    "embeddingModel": LOCAL_MODEL,
    "dimensions": DIMENSIONS,
}


def assemble_response(state: dict[str, Any], turn: TurnInput, pause: dict[str, Any] | None = None) -> dict[str, Any]:
    """The response in the shape the browser's agent client expects, for a finished or a paused turn."""
    semantic = state.get("semantic") or LOCAL_SEMANTIC
    extraction = state.get("extraction") or {"mode": "deterministic", "status": "not-configured"}
    candidates = state.get("candidates_meeting_constraints")
    nothing_fits = None
    if pause is None and candidates == 0 and turn["candidate_pool"]:
        tightest = tightest_constraint(turn["candidate_pool"], state["preferences"])
        nothing_fits = {"tightestConstraint": tightest} if tightest else None
    meta = {
        "extractionMode": extraction["mode"],
        "modelStatus": extraction["status"],
        "modelError": extraction.get("error"),
        "candidatesMeetingConstraints": candidates,
        "semantic": {
            key: semantic[key]
            for key in ("retrievalMode", "embeddingMode", "embeddingModel", "dimensions", "fallbackReason")
            if key in semantic
        },
        "durationMs": round((time.perf_counter() - turn["started_at"]) * 1000),
        "trace": state.get("trace", []),
    }
    response = {
        "preferences": state["preferences"],
        "recommendations": [] if pause else state.get("recommendations", []),
        "clarificationQuestion": state.get("clarifying_question"),
        "pause": pause,
        "nothingFits": nothing_fits,
        "meta": {key: value for key, value in meta.items() if value is not None},
    }
    # JSON.stringify drops undefined fields; dropping None keeps the same wire shape.
    return {key: value for key, value in response.items() if value is not None}


def respond(state: TurnState, config: RunnableConfig) -> dict[str, Any]:
    return {"response": assemble_response(state, _turn(config))}


def turn_response(result: dict[str, Any], turn: TurnInput) -> dict[str, Any]:
    """The browser's response for a run that finished, or one that paused (relaxation or question)."""
    interrupts = result.get("__interrupt__")
    return assemble_response(result, turn, pause=interrupts[0].value) if interrupts else result["response"]


def step_limit_response(stored: dict[str, Any], turn: TurnInput, limit: int, now: int) -> dict[str, Any]:
    """Ends a turn that ran past the step limit: no recommendations, and the trace says why."""
    preferences = _current_preferences(stored.get("preferences"), turn, now)
    step = _trace_step("step_limit", "skipped", 0, f"stopped after {limit} steps")
    return assemble_response({**stored, "preferences": preferences, "recommendations": [], "trace": [step]}, turn)


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """The recommendation agent; with a checkpointer, each `thread_id` is one remembered conversation."""
    graph = StateGraph(TurnState)
    for step in [
        extract_preferences,
        filter_candidates,
        propose_relaxation,
        semantic_retrieval,
        rank_candidates,
        validate_results,
    ]:
        graph.add_node(step.__name__, _traced(step))
    graph.add_node("ask_clarifying_question", ask_clarifying_question)
    graph.add_node("respond", respond)
    graph.add_edge(START, "extract_preferences")
    graph.add_conditional_edges(
        "extract_preferences", route_after_extract, ["ask_clarifying_question", "filter_candidates"]
    )
    graph.add_edge("ask_clarifying_question", END)
    graph.add_conditional_edges("filter_candidates", route_after_filter, ["propose_relaxation", "semantic_retrieval"])
    graph.add_edge("propose_relaxation", "filter_candidates")
    graph.add_edge("semantic_retrieval", "rank_candidates")
    graph.add_edge("rank_candidates", "validate_results")
    graph.add_edge("validate_results", "respond")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)
