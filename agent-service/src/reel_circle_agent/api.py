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

from .graph import TurnState, recommendation_agent

app = FastAPI(title="Reel Circle recommendation agent")

_BAD_MESSAGE = {"error": "A message of 1-1000 characters is required"}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_turn(body: Any) -> TurnState | None:
    """Validates and bounds a turn the way the TypeScript handler does; None means a bad request."""
    if not isinstance(body, dict) or not isinstance(body.get("message"), str):
        return None
    message = body["message"].strip()
    if not message or len(message) > 1000:
        return None
    limit = body.get("limit")
    return {
        "message": message,
        "incoming_preferences": body["preferences"] if isinstance(body.get("preferences"), dict) else None,
        "candidates": body["candidates"][:250] if isinstance(body.get("candidates"), list) else [],
        "persons": body["profiles"][:12] if isinstance(body.get("profiles"), list) else [],
        "limit": max(1, min(10, math.floor(limit))) if _is_number(limit) else 5,
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

    started = time.perf_counter()
    result = await recommendation_agent.ainvoke(turn)
    semantic = result.get("semantic") or {
        "retrievalMode": "local",
        "embeddingMode": "local",
        "embeddingModel": "local-hash-v1",
        "dimensions": 384,
    }
    payload = {
        "preferences": result["preferences"],
        "recommendations": result["recommendations"],
        "meta": {
            "extractionMode": "deterministic",
            # Model extraction is not part of the Python service yet (ticket #7).
            "modelStatus": "not-configured",
            "candidatesMeetingConstraints": result["candidates_meeting_constraints"],
            "semantic": {
                key: semantic[key] for key in ("retrievalMode", "embeddingMode", "embeddingModel", "dimensions")
            },
            "durationMs": round((time.perf_counter() - started) * 1000),
            "trace": result["trace"],
        },
    }
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})
