"""Helper for example 03: resume a graph from a *different OS process*.

State lives in the sqlite checkpointer file — not in memory, not in this
process. `python -m langgraph_learn.cross_process <thread> <decision>`
builds a fresh graph on a fresh checkpointer connection and continues the
thread where the other process stopped.

Run directly:
    uv run python -m langgraph_learn.cross_process demo-03 approve
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command

from .checkpointer import create_checkpointer
from .graph import build_graph


def resume_process(thread_id: str, decision: str) -> None:
    checkpointer = create_checkpointer("checkpoints.db")
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    snap = graph.get_state(config)
    if not snap.interrupts:
        print(f"[pid {os.getpid()}] no paused interrupt for thread {thread_id!r} — nothing to do")
        return

    print(f"[pid {os.getpid()}] found a paused run (thread {thread_id!r}), "
          f"node={snap.next!r}, pending interrupts={len(snap.interrupts)}")
    print(f"[pid {os.getpid()}] resuming with decision {decision!r} …")
    out = graph.invoke(Command(resume=decision), config)
    print(f"[pid {os.getpid()}] done -> approved={out.get('approved')}, "
          f"report={len(out.get('full_report', ''))} chars")


def main() -> None:
    thread = sys.argv[1] if len(sys.argv) > 1 else "demo-03"
    decision = sys.argv[2] if len(sys.argv) > 2 else "approve"
    resume_process(thread, decision)


if __name__ == "__main__":
    main()