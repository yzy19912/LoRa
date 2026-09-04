"""03 — Checkpointing: persistence, history and time travel.

The checkpointer is what makes everything else in this repo possible:

  1. Persistence across processes — the run pauses; a brand-new uv subprocess
     (fresh graph, fresh checkpointer) resumes the same thread from the
     `checkpoints.db` sqlite file.
  2. History — every superstep is a checkpoint; print the whole timeline.
  3. Time travel — jump back to an old checkpoint (`invoke(None, <ckpt>)`
     replays it forward), answer the review differently, and watch a new,
     divergent branch grow in the same thread.
  4. State editing — `update_state` injects a value with zero node runs.

Run:
    uv run python scripts/03_checkpoints.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.types import Command
from langgraph_learn.checkpointer import create_checkpointer, thread_config
from langgraph_learn.graph import build_graph

DB_PATH = "checkpoints.db"


def run_to_interrupt(graph, config, topic: str) -> None:
    if not graph.get_state(config).interrupts:
        graph.invoke({"topic": topic}, config)
    snap = graph.get_state(config)
    print(f"  -> {'paused' if snap.interrupts else 'finished'}, next={snap.next!r}")


def show_timeline(graph, config) -> None:
    history = list(graph.get_state_history(config))
    print(
        f"\n--- checkpoint timeline ({len(history)} checkpoints, oldest -> newest) ---"
    )
    for i, snap in enumerate(reversed(history)):
        rev = snap.values.get("revision_round", "?")
        where = ", ".join(snap.next) or "END"
        intr = " (INTERRUPT)" if snap.interrupts else ""
        print(f"  #{i:<3} round={rev:<2} {where}{intr}")


def main() -> None:
    cp = create_checkpointer(DB_PATH)
    graph = build_graph(cp)

    # ---- 1. persistence across processes --------------------------------
    persist_id = f"demo-persist-{int(time.time())}"
    persist_cfg = thread_config(persist_id)
    print(
        f"[1] thread {persist_id!r} pauses at the human gate; "
        "state is saved to checkpoints.db"
    )
    run_to_interrupt(graph, persist_cfg, "recommendation systems")

    print("\n    a NEW process (fresh graph, fresh checkpointer) resumes it:")
    proc = subprocess.run(
        [sys.executable, "-m", "langgraph_learn.cross_process", persist_id, "approve"],
        capture_output=True,
        text=True,
    )
    print("   " + proc.stdout.replace("\n", "\n   ").strip())
    if proc.returncode != 0:
        print("subprocess error:\n", proc.stderr[:1500])
        sys.exit(1)

    # ---- 2. one revision round + approval on a clean thread --------------
    rev_id = f"demo-rev-{int(time.time())}"
    cfg = thread_config(rev_id)
    print(f"\n[2] thread {rev_id!r}: one revision round, then approval")
    run_to_interrupt(graph, cfg, "recsys cold start")  # round 0
    graph.invoke(Command(resume="be more concise"), cfg)  # round 1
    run_to_interrupt(graph, cfg, "")  # round 1 review
    graph.invoke(Command(resume="approve"), cfg)

    show_timeline(graph, cfg)

    # ---- 3. time travel: replay from round-0, answer differently --------
    pause0 = None
    for snap in graph.get_state_history(cfg):
        if (
            snap.interrupts
            and "review_node" in snap.next
            and snap.values.get("revision_round") == 0
        ):
            pause0 = snap
            break

    if pause0 is None:
        print("\n(no round-0 review checkpoint found - skipping time travel)")
    else:
        ckpt = pause0.config["configurable"]["checkpoint_id"]
        print(
            f"\n[3] time travel to round-0 review (ckpt {ckpt[:8]}\u2026): "
            "replay, then answer differently"
        )
        rt = {"configurable": {"thread_id": rev_id, "checkpoint_id": ckpt}}
        graph.invoke(None, rt)  # replay -> new tail interrupt
        alt = graph.invoke(Command(resume="add real metrics"), cfg)  # tail resume

        original = None
        for snap in graph.get_state_history(cfg):
            if (
                not snap.interrupts
                and snap.next == ()
                and snap.values.get("revision_round") == 1
                and snap.values["full_report"] != alt["full_report"]
            ):
                original = snap.values["full_report"]
                break

        print(
            f"    alternate report mentions 'add real metrics': "
            f"{'add real metrics' in alt['full_report']}"
        )
        print(
            f"    original report  mentions 'be more concise':   "
            f"{(original or '').__contains__('be more concise')}"
        )
        print("    -> two divergent v1 reports coexist in the same timeline.")

    # ---- 4. edit state with no node involved ----------------------------
    print("\n[4] update_state: inject a value as if a human pasted it")
    graph.update_state(cfg, {"feedback": "pasted by a human"})
    print("    feedback now =", repr(graph.get_state(cfg).values["feedback"]))
    print("\nall checkpoints above live in", DB_PATH)


if __name__ == "__main__":
    main()
