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
3. `rank_candidates`: apply Reel Circle's explainable group scorer.
4. `validate_results`: verify excluded genres and runtime limits before returning recommendations.

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

Groq example:

```env
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-20b
```

All model configuration is server-only. If the model request fails or is not configured, the endpoint returns `extractionMode: "deterministic"` and continues through the same filtering, ranking, and validation workflow.

`modelStatus` is one of `not-configured`, `failed`, or `succeeded`; it never exposes credentials or provider response content.
When a request fails, `modelError` contains only a sanitized category such as `http-401`, `http-404`, `invalid-json`, or `request-error`.

## Current boundary

The endpoint ranks the bounded catalog supplied by the browser. Embedding generation, semantic vector retrieval, durable server memory, and LangGraph orchestration are separate future phases and are not claimed by this contract.
