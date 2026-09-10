# Reel Circle: Agentic Graph Recommender

## Project summary

Reel Circle is a local-first, multi-person movie recommender built from Letterboxd exports. It combines an explainable knowledge graph, an active-learning agent, a multilayer perceptron, and a two-layer graph convolutional network.

## What is genuinely agentic

The application runs a closed feedback loop:

1. It observes ratings, skips, selections, and post-watch reactions.
2. It identifies uncertain taste regions and asks a contrasting pairwise question.
3. It updates positive and negative preference signals.
4. It invalidates stale models and requests retraining.
5. It re-ranks the group shortlist and explains the strongest graph path.

The language interface also converts English or Spanish requests into deterministic runtime, genre, mood, platform, and fairness constraints.

### Autonomous decision loop

The decision agent uses a bounded plan-act-observe loop with a maximum of eight steps. Its policy observes shortlist size, model readiness, label coverage, model-comparison state, and group disagreement before selecting a tool.

Available tools:

- `parse_constraints`
- `ask_preference`
- `train_gcn`
- `compare_models`
- `request_relaxation`
- `propose_pick`

Every tool call records its rationale, observation, timestamp, and completion state in a persistent execution trace. The agent pauses for user input when labels are sparse and requires explicit human approval before relaxing hard constraints or finalizing a recommendation. This makes the autonomy observable and bounded rather than an opaque chatbot workflow.

### Agent evaluation

The evaluation dashboard groups trace events by stable run ID and reports:

- Run and tool-step counts
- Average steps per run
- Completed-tool rate
- Human-intervention rate
- Approval rate
- Per-tool usage
- Run latency
- A bounded benchmark score rewarding completion, safety gates, and step efficiency

A privacy-safe JSON report exports configuration, model metrics, A/B experiments, agent evaluation, and traces while excluding raw ratings and watch history. Legacy traces without run IDs are migrated by step-boundary grouping.

## Models

### Explainable scorer

A deterministic baseline combines community rating, preferred genres, learned feature weights, runtime, and group fairness. It supports average, balanced, least-misery, and Nash aggregation.

### MLP

```text
34 inputs -> 12 ReLU -> 6 ReLU -> 1 sigmoid
```

The model uses 32 deterministic feature-hash buckets plus normalized runtime and rating. The interface exports and visualizes actual hidden-layer activations for the selected movie.

### Graph convolutional network

The GCN builds a heterogeneous bipartite graph containing movie and feature nodes. Features include genres, platforms, directors, languages, moods, and decades. It applies two normalized message-passing layers:

$$
H^{(1)} = \operatorname{ReLU}(\hat{A} X W^{(0)})
$$

$$
H^{(2)} = \operatorname{ReLU}(\hat{A} H^{(1)} W^{(1)})
$$

Movie-node outputs are supervised with a masked mean-squared-error loss. Feature nodes participate in propagation without being treated as rating labels. The UI exposes learned 12-dimensional movie embeddings and message-passing edges.

This is an actual GCN prototype rather than a graph-shaped visualization: the normalized adjacency matrix participates in both trainable forward layers. It remains a small local experiment and is not presented as a production-scale GNN.

## Experiment methodology

- Compare deterministic, MLP, and GCN rankings on the same filtered catalog.
- Record confidence, training loss, top-pick changes, top-three overlap, and mean rank shift.
- Visualize group disagreement as a candidate heatmap.
- Inspect the strongest person-to-feature-to-movie path.
- Persist experiment records locally for repeatable demonstrations.
- Run benchmark missions through the autonomous agent and inspect the complete tool trace.
- Verify safety behavior with impossible constraints and final-pick approval gates.

The current catalog and sample size are intentionally small, so confidence values are displayed and neural influence is capped. This is a prototype for experimentation, not evidence of production recommendation quality.

## Demo script

1. Select a person and provide at least four distinct feedback labels.
2. Train the MLP and open **MLP** in the graph toolbar.
3. Select different movies and observe real hidden-neuron activation values change.
4. Train the GCN and open **GCN** to inspect feature-to-movie message passing and embedding magnitudes.
5. Switch the active model between MLP and GCN.
6. Compare confidence, loss, rank shift, top-three overlap, disagreement, and explanation path in the experiment lab.
7. Toggle neural blending and conflict heatmap to compare outcomes.
8. Run an autonomous benchmark mission and explain the parse → observe → ask/train → compare → approval sequence from the execution trace.
9. Run an impossible Filmin/under-90 comedy mission and show that the agent requests approval before relaxing constraints.
10. Open the evaluation metrics and export a JSON evidence report for the session.

## Engineering evidence

- Unit tests cover recommendation logic, active learning, MLP training, GCN message passing, agent policy, trace evaluation, persistence, constraints, and TMDB normalization.
- GitHub Actions runs tests, lint, production build, and production dependency audit.
- TensorFlow.js is lazy-loaded so the large ML runtime does not block the initial application bundle.
- The UI supports persistent light and dark themes and responsive desktop/mobile graph views.

## LinkedIn-ready description

> Built Reel Circle, a local-first agentic movie recommender using React, TypeScript, TensorFlow.js, an explainable knowledge graph, and a two-layer graph convolutional network. Implemented a bounded plan-act-observe agent with typed tools, persistent execution traces, active-learning questions, and human approval gates. Added bilingual constraint parsing, normalized movie-feature message passing, masked node supervision, activation and embedding inspection, confidence-weighted hybrid ranking, group fairness strategies, and reproducible A/B metrics.

Use **graph convolutional network prototype**, not **production GNN recommendation system**. The distinction is technically important and gives you a good interview discussion point.
