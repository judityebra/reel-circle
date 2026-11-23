# Reel Circle recommendation agent

The recommendation agent as a Python [LangGraph](https://langchain-ai.github.io/langgraph/) service. It
takes the same request and returns the same response as the TypeScript `/api/agent`, plus
`meta.candidatesMeetingConstraints`. Terms follow [GLOSSARY.md](../GLOSSARY.md); scope and server
state follow [ADR 0001](../docs/adr/0001-only-the-recommendation-agent-moves-to-python.md) and
[ADR 0002](../docs/adr/0002-server-state-holds-preferences-only.md).

```text
extract -> filter -> semantic -> rank -> validate
```

Today the graph is linear: rule-based preference extraction, the six hard constraints, group
scoring, semantic retrieval, and a validate guardrail that drops any movie breaking a hard
constraint. Model extraction, pauses and server-side conversation state arrive in later tickets
(see issue #2).

## Semantic retrieval

Configured with the same variables as the TypeScript service, read on every turn:

| Variables | Effect |
| --- | --- |
| none | Local 384-dimension hashed vectors, in-memory cosine search |
| `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY` | OpenAI-compatible provider embeddings |
| `EMBEDDING_PROVIDER=gemini`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY`, `EMBEDDING_DIMENSIONS` | Gemini embeddings (128-3072 dimensions, default 768) |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | Candidate-scoped search in Qdrant |

Each service falls back on its own and says so in `meta.semantic.fallbackReason`: a failing
provider means local vectors (`embedding-failed`), a failing Qdrant means in-memory search over the
provider's vectors (`qdrant-failed`). Tests replace both services with an `httpx.MockTransport`
by overriding the `outbound_transport` dependency.

## Run

```bash
uv sync
uv run uvicorn reel_circle_agent.api:app --reload      # http://127.0.0.1:8000/health
uv run pytest                                          # API contract + parity with TypeScript
uv run ruff check . && uv run ruff format --check .
```

Or with Docker:

```bash
docker build -t reel-circle-agent .
docker run -p 8000:8000 reel-circle-agent
```

## Parity with the TypeScript pipeline

`tests/test_parity.py` replays every turn of [fixtures/agent-parity.json](../fixtures/agent-parity.json)
over HTTP and expects the same preferences, ranking, scores and explanations. A few JavaScript
built-ins are reproduced exactly in `js_compat.py` (`Math.round`, `toFixed`, left-to-right
summation, 32-bit hashing). If the TypeScript behaviour changes on purpose, regenerate the fixtures
with `npm run fixtures:parity` and port the change.

## Tracing

LangSmith tracing is off by default. Set these to trace every node of every turn:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=reel-circle-agent
```
