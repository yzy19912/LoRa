"""State shape of the demo graph.

The key teaching point: `sections` uses a **reducer** (`operator.add`).
Multiple parallel worker nodes each write one item, and the reducer merges
all partial writes into a single list. That is exactly what makes the
"fan-in" (aggregate) step possible — without a reducer, parallel writes to
the same key would overwrite each other.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Section(TypedDict):
    """One researched section produced by a single fan-out worker."""

    section: str
    content: str
    revision_round: int


class GraphState(TypedDict):
    # User/global inputs
    topic: str
    # The plan: list of subtopics produced by the planner. One worker runs
    # per entry (fan-out).
    plan: list[str]
    # Fan-in: every worker appends here; the reducer accumulates them.
    sections: Annotated[list[Section], operator.add]
    # Built by the aggregate (fan-in) node from `sections`.
    full_report: str
    # Cycle counter that drives the revision loop.
    revision_round: int
    # Written by the human-in-the-loop review node.
    feedback: str
    approved: bool
