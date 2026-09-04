"""Observability helpers.

Two layers are provided, both optional:

1. **Langfuse** — if ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are
   set in ``.env``, ``get_langfuse_handler()`` returns a real Langfuse
   callback handler. Pass it in the invocation config and every run, node,
   superstep and state update is traced to a Langfuse trace.

2. **Console callback** — if Langfuse is not configured (or on top of it),
   ``ConsoleTraceHandler`` prints every graph/node event. It is a thin
   ``langchain_core`` callback and demonstrates that LangGraph's "magic" is
   just the standard LangChain callback interface.

See ``examples/04_langfuse_observability.py`` for usage.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def get_langfuse_handler(
    *,
    trace_id: str | None = None,
    trace_name: str | None = None,
    metadata: dict | None = None,
):
    """Return a Langfuse LangChain callback handler, or None if unconfigured.

    ``trace_id``/``trace_name``/``metadata`` link the whole graph run into a
    single, named Langfuse trace (each node then becomes a span inside it).

    Never raises: missing credentials or broken imports degrade gracefully to
    console-only observability so the examples always run.
    """
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print(
            "[observability] LANGFUSE keys missing — skipping Langfuse. "
            "Copy .env.example to .env and fill in your keys."
        )
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        from langfuse.callback import CallbackHandler  # langfuse < 4.x
    except Exception as err:  # pragma: no cover - defensive
        print(f"[observability] cannot init Langfuse: {err}")
        return None

    trace_context = {}
    if trace_id:
        trace_context["id"] = trace_id
    if trace_name:
        trace_context["name"] = trace_name
    if metadata:
        trace_context["metadata"] = metadata

    try:
        handler = (
            CallbackHandler(trace_context=trace_context)
            if trace_context
            else CallbackHandler()
        )
        print("[observability] Langfuse tracing enabled.")
        return handler
    except Exception as err:  # pragma: no cover - defensive
        print(f"[observability] Langfuse init failed (host unreachable?): {err}")
        return None


def langfuse_trace_url(trace_id: str) -> str | None:
    """Build the clickable URL for a trace id (no network required)."""
    try:
        from langfuse import Langfuse

        return Langfuse().get_trace_url(trace_id=trace_id)
    except Exception as err:  # pragma: no cover - depends on backend
        print(f"[langfuse] could not build trace URL: {err}")
        return None


class ConsoleTraceHandler(BaseCallbackHandler):
    """Print every LangGraph engine event so you can *see* the graph run.

    LangGraph routes all of its lifecycle (on_chain_start / on_chain_end /
    on_chain_error plus intermediate node events and LLM calls) through
    registry callbacks. Langfuse plugs in through the very same mechanism —
    that is why adding your own handler here needs no changes to the graph.
    """

    def __init__(self, name: str = "console-trace") -> None:
        self.name = name
        self._depth = 0

    def on_chain_start(
        self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs
    ):
        tag = serialized.get("name") if isinstance(serialized, dict) else self.name
        print(f"  {'  ' * self._depth}>> start       {tag}")
        self._depth += 1

    def on_chain_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        self._depth = max(0, self._depth - 1)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self._depth = max(0, self._depth - 1)
        # A pause via interrupt() surfaces to this callback as an "error".
        if type(error).__name__ == "GraphInterrupt":
            print("  II interrupt  graph paused at a human-in-the-loop gate")
            return
        print(f"  !! error      {error!r}")

    def on_custom_event(self, name: str, data: Any, **kwargs) -> None:
        print(f"  -- custom     {name}: {data}")

    def on_llm_start(
        self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs
    ):
        print(f"  {'  ' * self._depth}llm      # {len(prompts)} prompt(s)")
