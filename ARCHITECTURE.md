# Architecture

Reel Circle separates deterministic decisions, learned models, agent orchestration, and presentation so each layer can be tested independently.

```mermaid
flowchart LR
    CSV[Letterboxd CSV] --> Profiles[Local profiles]
    TMDB[TMDB server adapter] --> Catalog[Spain movie catalog]
    Profiles --> KG[Explainable taste graph]
    Catalog --> KG
    Prompt[Conversation turn or vibe] --> Parser[Structured preference extractor]
    Parser --> Memory[(Session conversation memory)]
    Memory --> Parser
    Parser --> Agent[Server agent orchestrator]
    LLM[Optional OpenAI-compatible LLM] --> Parser
    Agent --> Semantic[Local semantic vector index]
    Catalog --> Semantic
    Semantic --> Agent
    KG --> Baseline[Deterministic scorer]
    Feedback[Ratings and feedback] --> MLP[34-12-6-1 MLP]
    Feedback --> GCN[Two-layer GCN]
    Catalog --> MLP
    KG --> GCN
    Agent -->|typed tools| GCN
    Agent --> Compare[Model comparison]
    Baseline --> Compare
    MLP --> Compare
    GCN --> Compare
    Compare --> Approval{Human approval}
    Approval --> Result[Final recommendation]
    Profiles --> Store[(IndexedDB)]
    Feedback --> Store
    Compare --> Store
    Agent --> Trace[Execution trace]
    Trace --> Store
    Memory --> Store
```

## Conversational recommendation flow

Each turn is merged into a typed `ConversationPreferences` state. Positive genres, explicit exclusions, moods, runtime, release period, language, group size, and viewing context persist across refinements. For example, `I do not want horror` followed by `something darker, under 90 minutes` keeps horror excluded while adding a darker mood and runtime boundary. The same state drives filtering, recommendation explanations, and the visible interpretation chips.

Conversation messages and preferences are stored only in IndexedDB and can be cleared from the agent or with **Delete data**. Shared room URLs deliberately exclude conversation text.

When `VITE_MOVIE_API_URL` is configured, turns are sent to `POST /api/agent`. The server attempts strict JSON-schema extraction with an OpenAI-compatible model, then runs deterministic filtering, ranking, and hard-constraint validation. If the model is absent or fails, the same endpoint falls back to the deterministic extractor. The response includes extraction mode, latency, and an observable tool trace.

After hard filtering, the agent embeds each movie's synopsis, genres, keywords, cast, director, language, tone, and moods. A configured OpenAI-compatible embedding provider supplies neural vectors; otherwise the service builds 384-dimensional local vectors. Configured Qdrant collections receive bounded candidate upserts and candidate-scoped similarity searches. Without Qdrant, cosine search runs in memory. Semantic similarity contributes a bounded 28% of the final score, so vibe phrases can reorder valid candidates but cannot bypass explicit exclusions.

Raw Letterboxd ratings and watched-film history are stripped by the client adapter. The server receives bounded movie candidates and privacy-safe learned preference weights.

## Recommendation agent (Python LangGraph service)

`agent-service/` runs the recommendation agent as one LangGraph graph ([ADR 0003](docs/adr/0003-one-graph-not-a-multi-agent-system.md)). `/api/agent` forwards each turn there when `AGENT_SERVICE_URL` is set and falls back to the TypeScript pipeline on any failure.

```mermaid
flowchart LR
    Turn[Conversation turn] --> Extract[extract_preferences]
    Extract -->|model found nothing usable| Ask[ask_clarifying_question]
    Ask -->|pause: the reply is a new turn| Done((end))
    Extract --> Filter[filter_candidates]
    Filter -->|nothing fits| Relax[propose_relaxation]
    Relax -->|pause: accept or reject| Filter
    Filter --> Semantic[semantic_retrieval]
    Semantic --> Rank[rank_candidates]
    Rank --> Validate[validate_results]
    Validate --> Respond[respond]
```

- **Pauses** are `interrupt()`s. A relaxation is resumed with the user's decision, and only if the decision names the pending proposal; a clarifying question is never resumed, because the reply is a new turn. Any new message ends a pause.
- **Memory** is one checkpointed thread per conversation (in memory, SQLite or Postgres). Following [ADR 0002](docs/adr/0002-server-state-holds-preferences-only.md), only typed preferences and turn metadata are stored: the turn's input travels in the run config, intermediate results are untracked channels, and a test reads the raw SQLite file to prove the message, persons and candidate pool never reach storage.
- **Outside services** (the chat model, the embedding provider, Qdrant) come in through FastAPI dependencies and the run config, so tests swap in fakes, and each falls back on its own.
- **Parity:** `fixtures/agent-parity.json` freezes the TypeScript pipeline's output; the Python service reproduces every turn exactly.

## Lab agent loop

The in-browser lab agent follows a bounded plan-act-observe policy with at most eight steps. It can parse constraints, request active-learning input, train the GCN, compare models, request constraint relaxation, and propose a final pick. Constraint relaxation and final selection require explicit approval.

## Recommendation layers

1. Candidate filtering applies runtime, genre, mood, platform, and watched-history constraints.
2. The explainable scorer computes individual scores and applies the selected group-fairness strategy.
3. A local MLP or GCN can contribute a confidence-scaled score.
4. The experiment layer compares baseline and hybrid rankings before agent use.

## Privacy

Profiles, feedback, model outputs, experiments, and traces are stored in IndexedDB. Share links omit raw ratings, watched-film history, model results, experiments, and agent traces. TMDB credentials remain server-side.

## Performance boundaries

TensorFlow.js is lazy-loaded on first training. The current GCN uses a dense adjacency matrix and is intended for a small prototype catalog. A production-scale graph should use sparse tensors, sampled neighborhoods, offline embeddings, and a retrieval/reranking architecture.

TMDB discovery is paginated through the browser in bounded 20-film server calls. Each server call enriches one discovery page, while the client deduplicates up to 200 initial films and can load later pages incrementally. Only the ranked top candidates are rendered in the graph and shortlist.

## Production service boundary

The deployment remains local-first and includes optional hosted LLM extraction, OpenAI-compatible embeddings, and Qdrant persistence behind `/api/agent`. Each capability falls back independently: model extraction to deterministic parsing, embeddings to local vectors, and Qdrant to in-memory cosine search. Durable conversation checkpoints now live in the Python service. The next backend phase should add incremental background indexing so unchanged movies are not re-embedded on each request.
