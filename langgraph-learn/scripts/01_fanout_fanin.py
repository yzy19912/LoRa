"""01 — Fan-out / Fan-in.

Watch a single graph invocation split into N parallel workers and merge
back into one aggregation step.

    plan -> [Send x4] -> research_section (parallel) -> aggregate

What to look for in the output:

* **fan-out**  — one `plan` event, then FOUR `research_section` events in the
  same superstep (they run *concurrently*);
* **fan-in**   — exactly ONE `aggregate` event after the four workers, even
  though four branches reach it. That is Pregel barrier syncing: the
  aggregator runs once all incoming branches of a superstep have finished;
* **reducer**  — each worker writes `{"sections": [one item]}`; the
  `operator.add` reducer merges them into a single list of 4.

Run:
    uv run python examples/01_fanout_fanin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_learn.checkpointer import thread_config
from langgraph_learn.graph import AGGREGATE, RESEARCH, build_graph


def main() -> None:
    graph = build_graph(MemorySaver())
    config = thread_config("demo-01-fanout")

    print("invoking — streaming every partial state update as it happens")
    print("-" * 72)

    for update in graph.stream({"topic": "matching engines"}, config, stream_mode="updates"):
        for node_name, payload in update.items():
            label = node_name
            if node_name == RESEARCH and payload.get("sections"):
                section = payload["sections"][0]["section"]
                label = f"{node_name}: {section!r}"
            elif node_name == AGGREGATE:
                label = f"{node_name} (fan-in: {len(payload['full_report'])} chars)"
            print(f"[stream] {label}")

    print("-" * 70)
    print("graph paused at the human gate (see example 02).")

    # Finish the run so the thread ends cleanly.
    graph.invoke(Command(resume="approve"), config)

    final = graph.get_state(config).values
    print(f"\nfinal report -> {len(final['full_report'])} chars,")
    print(f"  sections stored by reducer: {len(final['sections'])} "
          f"(one per worker per round)")


if __name__ == "__main__":
    main()