# Only the recommendation agent moves to the Python LangGraph service

The server-side recommendation agent (`/api/agent`) is rebuilt as a Python LangGraph service, while the in-browser lab agent (GCN training, taste questions, final-pick approval) stays in TypeScript. The lab agent trains on raw ratings, and moving it to a server would break the rule that ratings and watch history never leave the browser. Server-side `interrupt()` pauses are therefore new pauses (clarify, relax), not a replacement for the lab agent's approval gates.
