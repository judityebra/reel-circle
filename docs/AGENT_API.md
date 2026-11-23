# Agent API

`POST /api/agent` is Reel Circle's server-side conversation and recommendation boundary.

## Request

```json
{
  "message": "Something darker, under 90 minutes",
  "preferences": {
    "genres": [],
    "excludedGenres": ["Horror"],
    "platforms": [],
    "moods": [],
    "lastUpdatedAt": 0
  },
  "candidates": [],
  "profiles": [],
  "limit": 5
}
```

The browser limits candidates to 250 and profiles to 12. Profiles contain preferred genres and learned feature weights; `ratings` and `watched` are replaced with empty arrays before transmission.

## Workflow

1. `extract_preferences`: request strict JSON-schema output from the configured model, or use deterministic extraction.
2. `filter_candidates`: enforce watched, genre exclusion, runtime, release period, language, platform, genre, and mood constraints.
3. `semantic_retrieval`: compare the request with rich movie documents using provider embeddings and Qdrant when configured, or local vectors and in-memory cosine search.
4. `rank_candidates`: blend semantic similarity with Reel Circle's explainable group scorer.
5. `validate_results`: verify excluded genres and runtime limits before returning recommendations.

## Relaxation pause (Python service only)

When nothing in the candidate pool meets every hard constraint, the Python service pauses instead of returning an empty list. The response has `recommendations: []` and a `pause` that proposes loosening one hard constraint, in the order year range, languages, requested genres, max runtime (+15 minutes per step), platforms. Excluded genres are never proposed.

```json
{ "pause": { "type": "relaxation", "constraint": "yearRange", "from": [1990, 1999], "to": null } }
```

The browser answers on the same `conversationId` with `"decision": "accept"` or `"decision": "reject"` and `"answering"`: the proposal it is answering, exactly as received. Accept loosens the constraint and filters again; reject keeps it and offers the next one. Runtime is offered 15 minutes at a time until the limit covers the longest movie in the candidate pool. After the last rejection the response carries `"nothingFits": { "tightestConstraint": "yearRange" }` instead of a pause (the tightest constraint is the one that alone lets the fewest movies through). A turn without a decision, or with a decision for a proposal that is no longer pending, ends any pause and runs as a normal turn. A turn stops after 12 graph steps and ends its trace with a `step_limit` step. If the Python service is down, the TypeScript fallback applies an accepted proposal itself, so the decision is not lost.

When a chat model interprets a turn but finds nothing usable in it (no hard constraint or preference) and suggests a question, the response pauses with `"pause": { "type": "clarifying_question", "question": "..." }` and no recommendations. The user's reply is an ordinary new turn on the same `conversationId`, which ends the pause, so a turn asks at most once. Rule-based extraction never asks.

## Response

```json
{
  "preferences": {},
  "recommendations": [
    {
      "movie": {},
      "score": 91,
      "matchedPreferences": [],
      "explanation": "The Puzzle matches the twisty mood and your under-90-minute limit."
    }
  ],
  "clarificationQuestion": null,
  "meta": {
    "extractionMode": "model",
    "modelStatus": "succeeded",
    "durationMs": 420,
    "trace": [
      {
        "tool": "extract_preferences",
        "status": "completed",
        "durationMs": 310,
        "observation": "model structured extraction"
      }
    ]
  }
}
```

## Configuration

- `OPENAI_API_KEY`: enables model extraction.
- `OPENAI_MODEL`: model name; defaults to `gpt-4.1-mini`.
- `OPENAI_BASE_URL`: optional OpenAI-compatible API base URL.
- `VITE_MOVIE_API_URL`: frontend API base, normally `/api` on Vercel.
- `EMBEDDING_PROVIDER`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_DIMENSIONS`: native Gemini or OpenAI-compatible embedding provider.
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`: optional persistent vector search.
- `AGENT_SERVICE_URL`: optional URL of the Python recommendation agent (`agent-service/`). Turns are forwarded there and fall back to this TypeScript pipeline on an HTTP error, an invalid answer, a network failure or a timeout.
- `AGENT_SERVICE_TIMEOUT_MS`: how long to wait for the Python service; defaults to 4000.

When `AGENT_SERVICE_URL` is set, `meta.agentService` says who answered: `{ "mode": "python" }`, or `{ "mode": "typescript-fallback", "fallbackReason": "timeout" | "unreachable" | "invalid-response" | "http-<status>" }`. The Python service also adds `meta.candidatesMeetingConstraints`.

Groq example:

```env
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-20b
```

Free Gemini embedding example:

```env
EMBEDDING_PROVIDER=gemini
EMBEDDING_API_KEY=your-google-ai-studio-key
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSIONS=768
```

All model configuration is server-only. If the model request fails or is not configured, the endpoint returns `extractionMode: "deterministic"` and continues through the same filtering, ranking, and validation workflow.

`modelStatus` is one of `not-configured`, `failed`, or `succeeded`; it never exposes credentials or provider response content.
When a request fails, `modelError` contains only a sanitized category such as `http-401`, `http-404`, `invalid-json`, or `request-error`.

## Current boundary

The endpoint semantically reranks the bounded catalog supplied by the browser. Without external configuration, its 384-dimensional local vectors use deterministic feature hashing and are not claimed as neural embeddings. With an embedding provider and Qdrant configured, response metadata reports `embeddingMode: "provider"` and `retrievalMode: "qdrant"`. Durable server memory, incremental indexing, and LangGraph orchestration remain future phases.
