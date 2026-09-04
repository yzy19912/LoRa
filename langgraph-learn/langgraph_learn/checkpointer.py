"""Checkpointer factory.

A checkpointer is what turns LangGraph from "recompute every run" into
"remember state". It gives you three superpowers used in the examples:

  1. human-in-the-loop — a graph with interrupts can only pause if there is
     somewhere to persist the paused snapshot;
  2. resumability across processes — a SQLite file survives restarts, so a
     different process can pick a thread back up exactly where it stopped;
  3. time travel — every step is stored ("checkpoint history"), so you can
     inspect, roll back to, or fork any past state.

:memory:   in-process only, good for tests.
file path  persisted on disk, good for demos (e.g. ``checkpoints.db``).
"""

from __future__ import annotations

from typing import Any


def create_checkpointer(db_path: str = "checkpoints.db") -> Any:
    """Return a sqlite-backed LangGraph checkpointer.

    Uses the stdlib sqlite3 through ``langgraph-checkpoint-sqlite`` — no
    extra services required. Pass ``":memory:"`` to skip persistence.
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    if db_path == ":memory:":
        return SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))

    # Keep the connection open for the lifetime of the saver (that is what
    # makes state accessible across processes reading the same file).
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def thread_config(thread_id: str) -> dict:
    """The minimal config a user must pass: which conversation thread."""
    return {"configurable": {"thread_id": thread_id}}


def list_threads(checkpointer: Any) -> list[str]:
    """Enumerate every thread id persisted in the given checkpointer."""
    ids = []
    try:
        for snapshot in checkpointer.list({}):
            cid = snapshot.config["configurable"].get("thread_id")
            if cid and cid not in ids:
                ids.append(cid)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[checkpointer] could not list threads: {e}")
    return ids
