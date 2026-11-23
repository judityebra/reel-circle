# The recommendation agent is one LangGraph graph, not a multi-agent system

The recommendation agent is a single LangGraph state graph: fixed nodes (extract, filter, semantic retrieval, rank, validate, respond) with conditional edges for the two pauses (a clarifying question, a relaxation proposal). We considered a supervisor with specialist agents (an extraction agent, a retrieval agent, a critic) and a tool-calling ReAct loop, and rejected both: every step here is either deterministic or a single structured-output call, the order never varies, and the hard constraints must hold on every path. A graph keeps that order explicit and testable, gives each step a trace entry and a parity test against the TypeScript pipeline, lets `interrupt()` pause for the user at exactly two known points, and keeps cost at one model call per turn. A multi-agent design would add model calls, latency and nondeterminism without giving the agent any decision it can't already make.

## Consequences

- New behaviour arrives as a node or a conditional edge, each with its own test at the HTTP seam, not as a new agent.
- The step limit (12 graph steps per request) is a safety net for future loops, not a budget the current graph comes close to.
- If a future feature needs open-ended reasoning (for example, researching films outside the candidate pool), that is the point to revisit this decision, as a subgraph rather than a rewrite.
