"""Services that fail on purpose, for the failure-injection scenarios.

A scenario's `inject` names one failure per service: `{"embedding": "timeout"}`,
`{"qdrant": "http-500"}`, `{"model": "invalid-json"}`. A failure is `http-<status>`, `timeout` or
`invalid-json`. The fakes go in through the service's own dependency overrides (the outbound
transport and the chat model), so the agent's real fallback code runs; only the outside world is fake.
"""

import json
from typing import Any

import httpx

EMBEDDINGS_URL = "https://embeddings.test/v1"
QDRANT_URL = "https://qdrant.test"


def fault_environment(inject: dict[str, str]) -> dict[str, str]:
    """Configuration that points the agent at the fake services the scenario needs."""
    environment = {}
    if "embedding" in inject or "qdrant" in inject:
        environment |= {"EMBEDDING_BASE_URL": EMBEDDINGS_URL, "EMBEDDING_MODEL": "fake-embeddings"}
    if "qdrant" in inject:
        environment["QDRANT_URL"] = QDRANT_URL
    return environment


def _fail(failure: str, request: httpx.Request) -> httpx.Response:
    if failure == "timeout":
        raise httpx.ReadTimeout("injected timeout", request=request)
    if failure == "invalid-json":
        return httpx.Response(200, content=b"this is not JSON")
    if failure.startswith("http-"):
        return httpx.Response(int(failure.removeprefix("http-")), json={"error": "injected"})
    raise ValueError(f"unknown injected failure {failure!r}")


def fault_transport(inject: dict[str, str]) -> httpx.MockTransport:
    """Answers like a healthy embedding provider and Qdrant, except where the scenario injects a failure."""

    def handle(request: httpx.Request) -> httpx.Response:
        service = "embedding" if request.url.host == "embeddings.test" else "qdrant"
        if service in inject:
            return _fail(inject[service], request)
        if service == "embedding":
            texts = json.loads(request.content)["input"]
            return httpx.Response(
                200, json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(texts))]}
            )
        return httpx.Response(200, json={"result": {"points": []}})

    return httpx.MockTransport(handle)


class _ProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"injected HTTP {status_code}")
        self.status_code = status_code


class FailingModel:
    """A chat model that fails every call the way the scenario injects."""

    def __init__(self, failure: str):
        self.failure = failure

    def with_structured_output(self, *_args: Any, **_kwargs: Any) -> "FailingModel":
        return self

    async def ainvoke(self, _messages: Any) -> Any:
        if self.failure == "timeout":
            raise TimeoutError("injected timeout")
        if self.failure == "invalid-json":
            raise ValueError("injected invalid JSON")
        if self.failure.startswith("http-"):
            raise _ProviderError(int(self.failure.removeprefix("http-")))
        raise ValueError(f"unknown injected failure {self.failure!r}")
