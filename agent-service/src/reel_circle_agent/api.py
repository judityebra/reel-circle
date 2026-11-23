"""HTTP API of the recommendation agent: the same request and response shape as the TypeScript `/api/agent`.

Model extraction is configured with AGENT_MODEL or OPENAI_API_KEY (see model_extraction.py), and
conversation storage with CHECKPOINT_DATABASE_URL (see conversations.py).
LangSmith tracing is off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set; LangGraph
then traces every node of every turn.
"""

import json
import math
import os
import ssl
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

import certifi
import httpx
from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from .conversations import ConversationStore, is_conversation_id, open_checkpointer, remembered_conversations
from .graph import TurnInput
from .model_extraction import DEFAULT_RETRY_POLICY, RetryPolicy, configured_chat_model


@dataclass(frozen=True)
class Conversations:
    """The recommendation agent compiled with the conversation checkpointer, and expiry for its threads."""

    agent: CompiledStateGraph
    store: ConversationStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    async with open_checkpointer(os.environ.get("CHECKPOINT_DATABASE_URL")) as saver:
        app.state.conversations = Conversations(*remembered_conversations(saver))
        try:
            yield
        finally:
            del app.state.conversations


app = FastAPI(title="Reel Circle recommendation agent", lifespan=lifespan)

_BAD_MESSAGE = {"error": "A message of 1-1000 characters is required"}
_DEFAULT_LIMIT = 5
_OUTBOUND_TIMEOUT = httpx.Timeout(10.0)
# Building an SSL context loads the CA bundle (~100 ms); build it once, not on every turn.
_OUTBOUND_SSL = ssl.create_default_context(cafile=certifi.where())


@lru_cache(maxsize=1)
def _in_memory_conversations() -> Conversations:
    return Conversations(*remembered_conversations(InMemorySaver()))


def conversations(request: Request) -> Conversations:
    """The service's conversations; in memory when the app runs without its lifespan (simple test clients)."""
    return getattr(request.app.state, "conversations", None) or _in_memory_conversations()


def clock() -> int:
    """Current time in milliseconds; tests override it to move through conversation expiry."""
    return int(time.time() * 1000)


def model_retry_policy() -> RetryPolicy:
    """Attempts, backoff and time budget for model extraction; tests override it."""
    return DEFAULT_RETRY_POLICY


def outbound_transport() -> httpx.AsyncBaseTransport | None:
    """Transport for calls to the embedding provider and Qdrant; tests override it with fakes."""
    return None


def _utf16_length(text: str) -> int:
    """JavaScript's `string.length`, which counts UTF-16 code units (an emoji counts as two)."""
    return len(text.encode("utf-16-le")) // 2


def _shortlist_size(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int | float) or not math.isfinite(limit):
        return _DEFAULT_LIMIT
    return max(1, min(10, math.floor(limit)))


def parse_conversation_id(body: Any) -> str | None:
    """The turn's conversation id: None starts a new conversation; a malformed id is a ValueError."""
    if not isinstance(body, dict) or body.get("conversationId") is None:
        return None
    if not is_conversation_id(body["conversationId"]):
        raise ValueError("conversationId must be 1-64 letters, digits, '-' or '_'")
    return body["conversationId"]


def parse_turn(body: Any) -> TurnInput | None:
    """Validates and bounds a turn the way the TypeScript handler does; None means a bad request."""
    if not isinstance(body, dict) or not isinstance(body.get("message"), str):
        return None
    message = body["message"].strip()
    if not message or _utf16_length(message) > 1000:
        return None
    return {
        "message": message,
        "incoming_preferences": body["preferences"] if isinstance(body.get("preferences"), dict) else None,
        "candidate_pool": body["candidates"][:250] if isinstance(body.get("candidates"), list) else [],
        "persons": body["profiles"][:12] if isinstance(body.get("profiles"), list) else [],
        "limit": _shortlist_size(body.get("limit")),
        "started_at": time.perf_counter(),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.api_route("/agent", methods=["GET", "PUT", "PATCH", "DELETE"])
async def agent_method_not_allowed() -> JSONResponse:
    return JSONResponse({"error": "Method not allowed"}, status_code=405)


@app.post("/agent")
async def agent_turn(
    request: Request,
    transport: Annotated[httpx.AsyncBaseTransport | None, Depends(outbound_transport)],
    chat_model: Annotated[BaseChatModel | None, Depends(configured_chat_model)],
    retry_policy: Annotated[RetryPolicy, Depends(model_retry_policy)],
    remembered: Annotated[Conversations, Depends(conversations)],
    now_ms: Annotated[int, Depends(clock)],
) -> JSONResponse:
    try:
        body = json.loads(await request.body())
        conversation_id = parse_conversation_id(body) or str(uuid.uuid4())
    except ValueError:
        return JSONResponse({"error": "Invalid request"}, status_code=400)
    turn = parse_turn(body)
    if turn is None:
        return JSONResponse(_BAD_MESSAGE, status_code=400)

    await remembered.store.sweep_hourly(now_ms)
    await remembered.store.forget_if_idle(conversation_id, now_ms)
    async with httpx.AsyncClient(transport=transport, timeout=_OUTBOUND_TIMEOUT, verify=_OUTBOUND_SSL) as client:
        settings = {
            "turn": turn,
            "thread_id": conversation_id,
            "now_ms": now_ms,
            "http_client": client,
            "chat_model": chat_model,
            "model_retry": retry_policy,
        }
        result = await remembered.agent.ainvoke({}, config={"configurable": settings}, durability="exit")
    payload = {**result["response"], "conversationId": conversation_id}
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


@app.delete("/agent/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, remembered: Annotated[Conversations, Depends(conversations)]
) -> Response:
    """Forgets a conversation's stored preferences (the browser calls this from Delete data)."""
    if not is_conversation_id(conversation_id):
        return JSONResponse({"error": "Invalid conversation id"}, status_code=400)
    await remembered.store.delete(conversation_id)
    return Response(status_code=204)
