"""End-to-end tests for the fan-out/fan-in + human-in-the-loop graph."""

from __future__ import annotations

import pytest
from langgraph.types import Command
from langgraph_learn.checkpointer import create_checkpointer, thread_config
from langgraph_learn.graph import AGGREGATE, build_graph


@pytest.fixture
def graph():
    cp = create_checkpointer(":memory:")
    return build_graph(cp)


def run_to_pause(graph, topic: str, thread_id: str = "test-thread"):
    config = thread_config(thread_id)
    out = graph.invoke({"topic": topic}, config)
    assert graph.get_state(config).interrupts, "graph should pause at review"
    return config, out


def test_fan_out_fan_in(graph):
    config, _ = run_to_pause(graph, "LoRA")
    snap = graph.get_state(config).values
    assert len(snap["plan"]) == 4
    assert len(snap["sections"]) == 4
    assert snap["full_report"]


def test_aggregate_runs_once(graph):
    """Fan-in barrier: exactly ONE aggregate event for the 4 workers."""
    config = thread_config("agg-once")
    aggregate_events = 0
    for update in graph.stream({"topic": "agri"}, config, stream_mode="updates"):
        for node_name, payload in update.items():
            if node_name == AGGREGATE:
                aggregate_events += 1
    assert aggregate_events == 1


def test_four_workers_in_parallel(graph):
    """Each worker writes one reducer item; total exactly 4 in round 0."""
    config = thread_config("workers")
    graph.invoke({"topic": "robots"}, config)
    values = graph.get_state(config).values
    round0 = [s for s in values["sections"] if s["revision_round"] == 0]
    assert len(round0) == 4


def test_hitl_approve(graph):
    config, _ = run_to_pause(graph, "cooling")
    graph.invoke(Command(resume="approve"), config)
    values = graph.get_state(config).values
    assert values["approved"] is True


def test_hitl_revision_loop(graph):
    """Feedback -> one more round -> approve. Both rounds stay in history."""
    config, _ = run_to_pause(graph, "batteries")
    graph.invoke(Command(resume="be more concise"), config)
    # now paused again at round 1
    assert graph.get_state(config).interrupts
    assert graph.get_state(config).values["revision_round"] == 1

    graph.invoke(Command(resume="approve"), config)
    values = graph.get_state(config).values
    assert values["approved"] is True
    assert values["revision_round"] == 1
    assert len(values["sections"]) == 8  # 4 per round, reducer keeps both


def test_feedback_reaches_workers(graph):
    """Feedback from the human must influence the revised report."""
    config, _ = run_to_pause(graph, "ranking")
    graph.invoke(Command(resume="add real metrics"), config)
    report = graph.get_state(config).values["full_report"]
    assert "add real metrics" in report
    assert "Revision note" in report


def test_checkpoint_history_and_rollback(graph):
    config, _ = run_to_pause(graph, "memory")
    history_before = list(graph.get_state_history(config))
    assert len(history_before) >= 3  # start + plan + workers + aggregate + review

    graph.invoke(Command(resume="approve"), config)
    history_after = list(graph.get_state_history(config))
    assert len(history_after) > len(history_before)

    # newest checkpoint is the terminal one
    newest = history_after[0]
    assert newest.next == ()


def test_time_travel_fork(graph):
    config, _ = run_to_pause(graph, "encoders")
    graph.invoke(Command(resume="be more concise"), config)
    graph.invoke(Command(resume="approve"), config)

    # find the round-0 review pause
    pause = None
    for snap in graph.get_state_history(config):
        if (
            snap.interrupts
            and "review_node" in snap.next
            and snap.values.get("revision_round") == 0
        ):
            pause = snap
            break
    assert pause is not None
    ckpt = pause.config["configurable"]["checkpoint_id"]

    # time travel: replay from that checkpoint, then answer differently
    rt = {"configurable": {"thread_id": "test-thread", "checkpoint_id": ckpt}}
    graph.invoke(None, rt)
    alt = graph.invoke(Command(resume="cite sources"), config)
    assert "cite sources" in alt["full_report"]

    # the original outcome is still in the same thread history
    original = None
    for snap in graph.get_state_history(config):
        if (
            not snap.interrupts
            and snap.next == ()
            and snap.values["full_report"] != alt["full_report"]
        ):
            original = snap.values["full_report"]
            break
    assert original is not None
    assert "be more concise" in original


def test_no_langfuse_keys_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    from langgraph_learn.observability import get_langfuse_handler

    assert get_langfuse_handler() is None
