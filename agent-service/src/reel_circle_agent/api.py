"""HTTP API of the recommendation agent: the same request and response shape as the TypeScript `/api/agent`.

LangSmith tracing is off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set; LangGraph
then traces every node of every turn.
"""

import json
import math
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .graph import TurnInput, recommendation_agent

app = FastAPI(title="Reel Circle recommendation agent")

_BAD_MESSAGE = {"error": "A message of 1-1000 characters is required"}
_DEFAULT_LIMIT = 5


def _utf16_length(text: str) -> int:
    """JavaScript's `string.length`, which counts UTF-16 code units (an emoji counts as two)."""
    return len(text.encode("utf-16-le")) // 2


def _shortlist_size(limit: Any) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int | float) or not math.isfinite(limit):
        return _DEFAULT_LIMIT
    return max(1, min(10, math.floor(limit)))


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
async def agent_turn(request: Request) -> JSONResponse:
    try:
        body = json.loads(await request.body())
    except ValueError:
        body = None
    turn = parse_turn(body)
    if turn is None:
        return JSONResponse(_BAD_MESSAGE, status_code=400)
    result = await recommendation_agent.ainvoke(turn)
    return JSONResponse(result["response"], headers={"Cache-Control": "no-store"})
