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
