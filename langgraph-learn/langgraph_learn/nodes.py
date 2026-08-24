"""Node implementations for the demo graph.

Every function here takes the graph state (or a Send payload) and returns a
partial state update. None of them have side effects other than printing —
this makes the data flow easy to follow and the code fully runnable offline.

By default the graph uses **mock** functions so it runs with zero API keys.
If you set ``OPENAI_API_KEY`` and ``USE_LLM=1``, the planner and each worker
are routed through ``langchain-openai`` instead (same graph, real LLM).
"""

from __future__ import annotations

import os

from langgraph.types import interrupt

from .state import GraphState, Section

# ---------------------------------------------------------------------------
# Optional LLM backend (used only when OPENAI_API_KEY + USE_LLM=1)
# ---------------------------------------------------------------------------


def _llm_enabled() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_LLM") == "1"


def _chat():
    from langchain_openai import ChatOpenAI  # imported lazily

    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# ---------------------------------------------------------------------------
# Mock content generation (default, offline friendly)
# ---------------------------------------------------------------------------

_MOCK_SECTIONS = [
    "History & origins",
    "How it works today",
    "Notable people & culture",
    "Future directions",
]

_MOCK_TEMPLATES = [
    "{section}: the origins of {topic} date back decades, and the field has "
    "evolved in response to real-world demand.",
    "{section}: practitioners today combine techniques, tooling and process — "
    "and measure outcomes rather than effort.",
    "{section}: the community around {topic} is its real superpower; "
    "gatherings, papers and open source keep it moving.",
    "{section}: looking forward, automation and smaller, faster tools are the "
    "clearest signals for where {topic} goes next.",
]


def _mock_text(topic: str, section: str, feedback: str | None = None) -> str:
    idx = abs(hash(section)) % len(_MOCK_TEMPLATES)
    body = _MOCK_TEMPLATES[idx].format(topic=topic, section=section.title())
    if feedback:
        body += f"\nRevision note: {feedback}"
    return body


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def plan_node(state: GraphState) -> dict:
    """Fan-out *source*: break the topic into subtopics to research in parallel."""
    if _llm_enabled():
        import json

        raw = _chat().invoke(
            "Return a JSON list of 3-5 short subtopics to research for the topic "
            f'"{state["topic"]}". Only the JSON array, e.g. ["a", "b"].'
        ).content
        plan = json.loads(raw)
    else:
        plan = _MOCK_SECTIONS[:]

    print(f"[plan]     -> subtopics: {plan}")
    return {
        "plan": plan,
        "revision_round": state.get("revision_round", 0),
    }


def research_section(state: GraphState) -> dict:
    """One parallel worker. `state` here is the Send payload merged onto the
    global state, so `section` comes from the payload while `topic` and
    `revision_round` are inherited from the parent state.

    Writes a single item into the reducer key `sections` — this is the
    fan-in primitive.
    """
    revision = state.get("revision_round", 0)
    section = state["section"]
    topic = state["topic"]
    feedback = state.get("feedback")

    if _llm_enabled():
        prompt = (
            f"Write a short paragraph about '{section}' of '{topic}'. "
            f"This is revision round {revision} of the report."
        )
        if feedback:
            prompt += f" Reviewer feedback to incorporate: {feedback}"
        content = _chat().invoke(prompt).content
    else:
        content = _mock_text(topic, section, feedback)

    item: Section = {
        "section": section,
        "content": content,
        "revision_round": revision,
    }
    print(f"[worker:{section!r:<12}] wrote {len(content)} chars (rev {revision})")
    return {"sections": [item]}


def fan_out(state: GraphState) -> list:
    """Fan-out *edge*: return one ``Send`` per plan item. LangGraph schedules
    a separate parallel `research_section` run for each Send.

    Note (LangGraph 1.x): the Send payload *is* the worker's state — it is
    not merged with the parent state, so everything the worker needs
    (`section`, `topic`, `revision_round`, `feedback`) is passed explicitly.
    """
    from langgraph.types import Send

    return [
        Send("research_section", {
            "section": s,
            "topic": state["topic"],
            "revision_round": state.get("revision_round", 0),
            "feedback": state.get("feedback"),
        })
        for s in state["plan"]
    ]


def aggregate_node(state: GraphState) -> dict:
    """Fan-in target. Runs once when *all* parallel workers of the current
    superstep are done. Picks only the sections belonging to the active
    revision round (the reducer list keeps older rounds around).
    """
    revision = state.get("revision_round", 0)
    sections = [
        s for s in state.get("sections", [])
        if s["revision_round"] == revision
    ]
    sections.sort(key=lambda s: s["section"])

    report = "\n\n".join(f"## {s['section']}\n{s['content']}" for s in sections)
    report = f"# {state['topic']} — research report (v{revision})\n\n{report}"
    print(f"[aggregate] merged {len(sections)} sections -> {len(report)} chars")
    return {"full_report": report}


def review_node(state: GraphState) -> dict:
    """Human-in-the-loop gate.

    ``interrupt()`` pauses execution *inside* the graph and hands control back
    to the caller with the payload below. The rest of the graph resumes later
    via ``Command(resume="...")``, at which point the call returns whatever
    the human replied. The paused snapshot is persisted by the checkpointer.

    Convention:
      * human replies ``approve``             -> report accepted.
      * anything else is revision feedback  -> loop back to the rewrite cycle.
    """
    decision = interrupt({
        "ask": "Review the report. Reply 'approve' to accept, or paste "
               "feedback for another revision round.",
        "report": state["full_report"],
        "revision_round": state.get("revision_round", 0),
    })

    if isinstance(decision, str):
        decision = decision.strip() or "approve"

    if isinstance(decision, str) and decision.lower() == "approve":
        print("[review] human approved the report.")
        return {"approved": True}
    print(f"[review] human requested a revision: {decision!r}")
    return {"approved": False, "feedback": decision}


def revision_node(state: GraphState) -> dict:
    """Bump the revision round — that alone drives the loop back to fan-out."""
    return {
        "revision_round": state.get("revision_round", 0) + 1,
        "full_report": "",
    }


def approve_decider(state: GraphState) -> str:
    return "end" if state.get("approved") else "revision"