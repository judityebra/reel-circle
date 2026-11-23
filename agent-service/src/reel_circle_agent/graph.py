"""The recommendation agent as a LangGraph state graph.

    extract -> filter -> semantic -> rank -> validate -> respond

This ticket's graph is linear and deterministic. Pauses (clarifying questions, relaxations), model
extraction and server-side conversation state are added by later tickets as conditional edges and
a checkpointer on this same graph. Each traced node appends one step to the response trace, so the
trace shows what actually ran.
"""

import operator
import time
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .preferences import Preferences, create_conversation_preferences, update_conversation_preferences
from .scoring import Movie, Person, Recommendation, broken_constraints, explain_preference_match, score_candidates
from .semantic import retrieve_semantically


class TraceStep(TypedDict):
    tool: str
    status: Literal["completed", "skipped"]
    durationMs: int
    observation: str


class TurnState(TypedDict, total=False):
    # Input, as the browser sent it.
    message: str
    incoming_preferences: Preferences | None
    candidates: list[Movie]
    persons: list[Person]
    limit: int
    # Built up node by node.
    preferences: Preferences
    ranked: list[Recommendation]
    candidates_meeting_constraints: int
    semantic: dict[str, Any]
    recommendations: list[dict[str, Any]]
    trace: Annotated[list[TraceStep], operator.add]


def _now_ms() -> int:
    return int(time.time() * 1000)


NodeResult = tuple[dict[str, Any], Literal["completed", "skipped"], str]


def _traced(tool: str, node: Callable[[TurnState], NodeResult]) -> Callable[[TurnState], dict[str, Any]]:
    """Runs a node and records it as one trace step under the tool name the browser expects."""

    def run(state: TurnState) -> dict[str, Any]:
        started = time.perf_counter()
        update, status, observation = node(state)
        duration = round((time.perf_counter() - started) * 1000)
        return {
            **update,
            "trace": [{"tool": tool, "status": status, "durationMs": duration, "observation": observation}],
        }

    return run


def extract(state: TurnState) -> NodeResult:
    now = _now_ms()
    current = state.get("incoming_preferences") or create_conversation_preferences(now)
    preferences = update_conversation_preferences(current, state["message"], now)
    return {"preferences": preferences}, "completed", "deterministic structured extraction"


def filter_candidates(state: TurnState) -> NodeResult:
    candidates = state["candidates"]
    ranked = score_candidates(candidates, state["persons"], state["preferences"])
    status = "completed" if candidates else "skipped"
    update = {"ranked": ranked, "candidates_meeting_constraints": len(ranked)}
    return update, status, f"{len(ranked)}/{len(candidates)} candidates satisfy hard constraints"


def semantic_retrieval(state: TurnState) -> NodeResult:
    ranked = state["ranked"]
    result = retrieve_semantically(state["message"], ranked)
    matches = sum(1 for recommendation in result["recommendations"] if recommendation["semanticScore"] > 0)
    source = f"{result['retrievalMode']} · {result['embeddingModel']} · {result['dimensions']}d"
    observation = f"{source} · {matches}/{len(ranked)} matches"
    return {"ranked": result["recommendations"], "semantic": result}, "completed" if ranked else "skipped", observation


def rank(state: TurnState) -> NodeResult:
    preferences = state["preferences"]
    recommendations = []
    for recommendation in state["ranked"][: state["limit"]]:
        movie, terms = recommendation["movie"], recommendation["semanticTerms"]
        signals = f" Semantic signals: {', '.join(terms[:3])}." if terms else ""
        recommendations.append(
            {
                "movie": movie,
                "score": recommendation["score"],
                "matchedPreferences": recommendation["reasons"],
                "explanation": f"{explain_preference_match(movie, preferences)}{signals}",
                "semanticScore": recommendation["semanticScore"],
                "semanticTerms": terms,
            }
        )
    status = "completed" if recommendations else "skipped"
    return {"recommendations": recommendations}, status, f"{len(recommendations)} recommendations returned"


def validate(state: TurnState) -> NodeResult:
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


def build_graph():
    graph = StateGraph(TurnState)
    graph.add_node("extract", _traced("extract_preferences", extract))
    graph.add_node("filter", _traced("filter_candidates", filter_candidates))
    graph.add_node("semantic", _traced("semantic_retrieval", semantic_retrieval))
    graph.add_node("rank", _traced("rank_candidates", rank))
    graph.add_node("validate", _traced("validate_results", validate))
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "filter")
    graph.add_edge("filter", "semantic")
    graph.add_edge("semantic", "rank")
    graph.add_edge("rank", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


recommendation_agent = build_graph()
