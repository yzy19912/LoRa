"""02 — Human-in-the-loop.

The graph pauses mid-run until a human answers. `interrupt()` returns control
to the caller; the caller inspects the *paused snapshot* and later resumes
with `Command(resume=...)`.

The full loop:

    invoke -> ... -> review -> INTERRUPT (state persisted)
    caller inspects `graph.get_state(config)`
    caller replies -> `graph.invoke(Command(resume="approve"), config)`

Defaults to a scripted auto-approve/feedback flow so it runs anywhere.
Pass `--interactive` to actually type your answers on a terminal.

Run:
    uv run python scripts/02_human_in_the_loop.py
    uv run python scripts/02_human_in_the_loop.py --interactive
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph_learn.checkpointer import thread_config
from langgraph_learn.graph import build_graph

INTERACTIVE = "--interactive" in sys.argv


def show_paused_snapshot(graph, config) -> None:
    snap = graph.get_state(config)
    print(f"[paused] next nodes         : {snap.next}")
    print(f"[paused] revision round     : {snap.values.get('revision_round')}")
    print(f"[paused] pending interrupts : {len(snap.interrupts)}")
    if snap.interrupts:
        payload = snap.interrupts[0].value
        print(f"[paused] ask                : {payload['ask']}")
        report = payload["report"]
        print(f"[paused] report head        : {report.splitlines()[3][:60]!r}…")


def get_decision() -> str:
    if INTERACTIVE:
        print("\n  (type 'approve' to accept, or paste feedback for a revision)")
        return input("  human> ").strip() or "approve"
    return "approve"


def main() -> None:
    graph = build_graph(MemorySaver())
    config = thread_config("ex02-hitl")

    print("--- invocation starts, graph will pause at the review gate ---\n")
    graph.invoke({"topic": "vector search indexing"}, config)

    if graph.get_state(config).interrupts:
        show_paused_snapshot(graph, config)

    decision = get_decision()
    graph.invoke(Command(resume=decision), config)
    print(
        f"[after {decision!r:<18}] approved={graph.get_state(config).values['approved']!r}"
    )

    if graph.get_state(config).interrupts:
        print("\n(requested a revision — the graph looped back to the human)")
        show_paused_snapshot(graph, config)
        graph.invoke(Command(resume="approve"), config)

    final = graph.get_state(config).values
    print(
        f"\nfinal state   -> approved={final['approved']}, "
        f"report={len(final['full_report'])} chars, "
        f"revision_round={final['revision_round']}"
    )
    print("note: every step above was persisted as a checkpoint (see example 03).")


if __name__ == "__main__":
    main()
