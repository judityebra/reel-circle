"""Server-side conversation state (ADR 0002): one LangGraph thread per conversation.

The checkpointer keeps only the graph's tracked state, which is the typed preferences and the time
of the last turn; the turn's input travels in the run config and intermediate results are untracked
channels, so neither is ever written. Turns run with `durability="exit"`: one checkpoint per run.
Conversations idle for 30 days are forgotten when next used and by an hourly sweep.

CHECKPOINT_DATABASE_URL picks the storage:
- unset: in memory (lost on restart; fine for development and tests)
- `sqlite:///path/to/file.db`: a local file
- `postgresql://...`: Postgres, for deployments (Neon, Supabase, ...)
"""

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

CONVERSATION_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
MAX_IDLE_MS = 30 * 24 * 60 * 60 * 1000
SWEEP_INTERVAL_MS = 60 * 60 * 1000


@asynccontextmanager
async def open_checkpointer(url: str | None) -> AsyncIterator[BaseCheckpointSaver]:
    """The checkpointer CHECKPOINT_DATABASE_URL selects, open for the life of the service."""
    if not url:
        yield InMemorySaver()
    elif url.startswith("sqlite:///"):
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with AsyncSqliteSaver.from_conn_string(url.removeprefix("sqlite:///")) as saver:
            yield saver
    elif url.startswith(("postgres://", "postgresql://")):
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        # A pool that checks connections before use: hosted Postgres (Neon, Supabase) drops idle
        # connections. No prepared statements, so it also works behind pgbouncer.
        options = {"autocommit": True, "prepare_threshold": None, "row_factory": dict_row}
        async with AsyncConnectionPool(
            url, kwargs=options, min_size=1, max_size=10, check=AsyncConnectionPool.check_connection, open=False
        ) as pool:
            saver = AsyncPostgresSaver(pool)
            await saver.setup()
            yield saver
    else:
        raise ValueError("CHECKPOINT_DATABASE_URL must start with sqlite:/// or postgresql://")


def remembered_conversations(saver: BaseCheckpointSaver):
    """The recommendation agent compiled with this checkpointer, and expiry for its conversations."""
    from .graph import build_graph

    return build_graph(saver), ConversationStore(saver)


def is_conversation_id(value: object) -> bool:
    return isinstance(value, str) and bool(CONVERSATION_ID.fullmatch(value))


class ConversationStore:
    """Expiry and deletion for the conversations kept by a checkpointer."""

    def __init__(self, saver: BaseCheckpointSaver):
        self.saver = saver
        self._last_sweep_ms: int | None = None

    async def _last_turn_at(self, conversation_id: str) -> int | None:
        latest = await self.saver.aget_tuple({"configurable": {"thread_id": conversation_id}})
        return latest.checkpoint["channel_values"].get("last_turn_at") if latest else None

    async def forget_if_idle(self, conversation_id: str, now_ms: int) -> None:
        """Deletes the conversation before it is used again if it has been idle for 30 days."""
        last_turn = await self._last_turn_at(conversation_id)
        if last_turn is not None and now_ms - last_turn > MAX_IDLE_MS:
            await self.delete(conversation_id)

    async def delete(self, conversation_id: str) -> None:
        await self.saver.adelete_thread(conversation_id)

    async def sweep(self, now_ms: int) -> None:
        """Deletes every conversation idle for 30 days. It reads every checkpoint, so it runs at most hourly;
        with one checkpoint per run (`durability="exit"`) that stays small at this app's scale."""
        latest_turns: dict[str, int] = {}
        async for item in self.saver.alist(None):
            thread = item.config["configurable"]["thread_id"]
            last_turn = item.checkpoint["channel_values"].get("last_turn_at", 0)
            latest_turns[thread] = max(latest_turns.get(thread, 0), last_turn)
        idle = [thread for thread, last_turn in latest_turns.items() if now_ms - last_turn > MAX_IDLE_MS]
        for thread in idle:
            await self.delete(thread)

    async def sweep_hourly(self, now_ms: int) -> None:
        """Runs the sweep at most once an hour, from the request path (no background worker needed)."""
        if self._last_sweep_ms is None or now_ms - self._last_sweep_ms >= SWEEP_INTERVAL_MS:
            self._last_sweep_ms = now_ms
            await self.sweep(now_ms)
