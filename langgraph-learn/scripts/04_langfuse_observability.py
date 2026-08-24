"""04 — Langfuse observability.

Every LangGraph run already emits standard LangChain callback events
(on_chain_start / on_chain_end per node, LLM calls, state updates). Langfuse
hooks into exactly that pipeline — no graph changes are needed, you just pass
its callback handler through the invocation `config`.

    1. The console handler prints the events (works with zero setup).
    2. If `LANGFUSE_*` keys exist (copy `.env.example` to `.env`), the same
       run is pushed to Langfuse as a single trace whose spans are the graph
       nodes; the SDK then derives a clickable trace URL for you.
    3. `callbacks`/`run_name`/`metadata` on the config flow straight into the
       trace, which is exactly the search surface you will use in the UI.

Run:
    uv run python scripts/04_langfuse_observability.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is installed
    pass

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_learn.checkpointer import thread_config
from langgraph_learn.graph import build_graph
from langgraph_learn.observability import (
    ConsoleTraceHandler,
    get_langfuse_handler,
    langfuse_trace_url,
)


def main() -> None:
    run_name = "langgraph-learn report (fan-out/fan-in + HITL)"
    trace_id = uuid.uuid4().hex

    langfuse_handler = get_langfuse_handler(
        trace_id=trace_id,
        trace_name=run_name,
        metadata={"course": "langgraph-learn", "user_id": "demo-user"},
    )

    callbacks = [langfuse_handler] if langfuse_handler else []
    if not callbacks:
        callbacks.append(ConsoleTraceHandler())  # offline text trace

    config = {
        "configurable": {"thread_id": "demo-langfuse"},
        "callbacks": callbacks,
        "run_name": run_name,
        "tags": ["langgraph-learn", "demo", "fan-out-fan-in"],
        "metadata": {"user_id": "demo-user"},
    }

    graph = build_graph(MemorySaver())
    print("\n--- running the fan-out/fan-in graph with tracing enabled ---\n")
    graph.invoke({"topic": "graph RAG pipelines"}, config)
    graph.invoke(Command(resume="approve"), config)

    final = graph.get_state(config).values
    print(f"\nrun finished -> approved={final['approved']}, "
          f"report={len(final['full_report'])} chars")

    if langfuse_handler:
        url = langfuse_trace_url(trace_id)
        print(f"\nLangfuse trace id: {trace_id}")
        print(f"Langfuse trace url: {url or '(host unreachable — still exported async)'}")
    else:
        print("\n> Langfuse not configured - set LANGFUSE_PUBLIC_KEY / "
              "LANGFUSE_SECRET_KEY in .env and re-run to upload traces.")


if __name__ == "__main__":
    main()