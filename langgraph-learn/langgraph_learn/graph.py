"""Graph assembly. Single source of truth for the topology.

    START -> plan -> [fan-out: one Send per subtopic]
         -> research_section xN (parallel workers)
         -> aggregate (fan-in, runs once) -> review (human gate)
         -> approve? END | revision -> [fan-out again] -> loop

Nodes are added one-by-one (not via ``add_sequence``) so each concept —
parallel branching, reducer-based fan-in, the interrupt gate — is easy to
point at in code.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState

# Node names == function names; kept as constants so edges stay typo-free.
PLAN = nodes.plan_node.__name__
RESEARCH = nodes.research_section.__name__
AGGREGATE = nodes.aggregate_node.__name__
REVIEW = nodes.review_node.__name__
REVISION = nodes.revision_node.__name__


def build_graph(checkpointer: Any | None = None):
    """Build and compile the demo graph.

    ``checkpointer`` is required for human-in-the-loop — interrupts cannot
    pause a run without one. Pass the sqlite saver for persistence across
    processes, or None to fall back to an in-memory saver.
    """
    builder = StateGraph(GraphState)

    builder.add_node(nodes.plan_node)
    builder.add_node(nodes.research_section)
    builder.add_node(nodes.aggregate_node)
    builder.add_node(nodes.review_node)
    builder.add_node(nodes.revision_node)

    # START -> planner
    builder.add_edge(START, PLAN)

    # Fan-out: planner -> Send(research_section xN), executed in parallel.
    builder.add_conditional_edges(PLAN, nodes.fan_out, [RESEARCH])

    # Fan-in: every parallel worker funnels into one aggregate node.
    builder.add_edge(RESEARCH, AGGREGATE)

    # aggregate -> human-in-the-loop gate
    builder.add_edge(AGGREGATE, REVIEW)

    # HITL gate: approve -> finish, otherwise loop into a new revision round.
    builder.add_conditional_edges(
        REVIEW,
        nodes.approve_decider,
        {"revision": REVISION, "end": END},
    )

    # Revision round -> fan-out again (loop). This makes fan-out/fan-in run
    # repeatedly and builds up a rich checkpoint history to inspect in ex. 03.
    builder.add_conditional_edges(REVISION, nodes.fan_out, [RESEARCH])

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


__all__ = ["AGGREGATE", "PLAN", "RESEARCH", "REVIEW", "REVISION", "build_graph"]
