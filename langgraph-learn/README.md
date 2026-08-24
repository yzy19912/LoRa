# langgraph-learn

A self-contained project for learning **LangGraph** through a single working
graph that combines the four most important pieces of a production agent:

1. **Fan-out / fan-in** — split one task into N parallel workers, then merge.
2. **Human-in-the-loop** — pause mid-run with `interrupt()`, resume with
   `Command(resume=...)`.
3. **Checkpointing** — SQLite-backed state that survives process restarts,
   gives you a full history, and enables time travel.
4. **Observability** — trace every run with **Langfuse** (plus a zero-config
   console trace so it works offline).

Everything runs with **no API keys** (the planner/workers are mocks). Point
them at a real LLM and Langfuse only if you want to.

## The graph

A "research report" pipeline: plan subtopics, research each in parallel,
aggregate, then ask a human to review — looping back for revisions if needed.

```text
   START
     │
     ▼
  plan ──────── (fan-out: one Send per subtopic)
     │
     ├──► research_section   ◄──┐
     ├──► research_section      │  run in parallel
     ├──► research_section      │
     └──► research_section   ───┘
                    │
                    ▼ (fan-in: reducer merges the 4 sections)
              aggregate
                    │
                    ▼
   review ◄── interrupt()  ── human answers ──► approve ──► END
     │                                                  ▲
     └── feedback ──► revision (round+1) ──► fan-out ────┘
```

## Layout

```
langgraph-learn/
├── pyproject.toml            # uv project + deps
├── .env.example              # Langfuse / OpenAI keys (optional)
├── langgraph_learn/
│   ├── state.py              # state shape + the reducer (fan-in primitive)
│   ├── nodes.py              # node functions (mock by default, LLM optional)
│   ├── graph.py              # topology + compile
│   ├── checkpointer.py       # sqlite-backed persistence
│   ├── observability.py      # Langfuse + console tracing handlers
│   └── cross_process.py      # helper: resume a thread from a new process
├── scripts/
│   ├── 01_fanout_fanin.py
│   ├── 02_human_in_the_loop.py
│   ├── 03_checkpoints.py
│   └── 04_langfuse_observability.py
└── tests/test_graph.py
```

## Setup

```bash
cd langgraph-learn
uv sync                # create .venv and install dependencies
uv run pytest          # run the test suite (fast sanity check)
```

Optional — enable a real LLM and/or Langfuse:

```bash
cp .env.example .env   # then fill in LANGFUSE_* / OPENAI_API_KEY, set USE_LLM=1
```

## Run the demos

| # | Script | What it shows |
|---|--------|---------------|
| 1 | `uv run python scripts/01_fanout_fanin.py` | stream the parallel workers; note that `aggregate` runs **once** (fan-in barrier) |
| 2 | `uv run python scripts/02_human_in_the_loop.py` | pause at the review gate, inspect the snapshot, resume (add `--interactive` to type answers) |
| 3 | `uv run python scripts/03_checkpoints.py` | cross-process resume, checkpoint timeline, time travel, state editing |
| 4 | `uv run python scripts/04_langfuse_observability.py` | trace the run (Langfuse + console fallback) |

## The four concepts, mapped to code

### 1. Fan-out / fan-in
- `nodes.fan_out` returns a list of `Send("research_section", {...})` — one
  per subtopic. LangGraph schedules them **in parallel**.
- The `Send` payload **is** the worker's state (LangGraph 1.x), so the worker
  reads `state["section"]` and writes `{"sections": [one item]}`.
- `sections` is `Annotated[list[Section], operator.add]` in `state.py`. That
  **reducer** merges each worker's write into one list — it is what makes the
  fan-in possible. Without a reducer, parallel writes would overwrite.
- `aggregate` runs **once** for the whole batch (Pregel barrier syncing),
  which you can verify in example 01's output.

### 2. Human-in-the-loop
- `nodes.review_node` calls `interrupt({...})`. The graph stops there and
  returns control to the caller; the **checkpointer persists the snapshot**.
- The caller inspects `graph.get_state(config).interrupts`, then either
  resumes with `graph.invoke(Command(resume="approve"), config)` or edits
  state while paused (`graph.update_state`).
- Requires a checkpointer (that's why `build_graph` always compiles with one).

### 3. Checkpointing
- `create_checkpointer("checkpoints.db")` gives a persistent `SqliteSaver`.
- `graph.get_state_history(config)` is the full timeline — every superstep is
  a checkpoint, so you can roll back or fork.
- Time travel (example 03): `graph.invoke(None, {..., "checkpoint_id": ...})`
  replays from an old checkpoint; answering the review differently afterwards
  grows a **divergent branch** next to the original in the same thread.
- Persistence across processes: `uv run python -m langgraph_learn.cross_process
  <thread_id> approve` resumes a thread that another process left paused.

### 4. Observability (Langfuse)
- LangGraph emits standard LangChain callback events; Langfuse is just a
  callback handler passed in the invocation config — **no graph changes**.
- `get_langfuse_handler(trace_id=..., trace_name=..., metadata=...)` returns a
  handler that groups the whole run into one named trace, one span per node.
- `langfuse_trace_url(trace_id)` derives the clickable link.
- If keys are missing, `ConsoleTraceHandler` prints the same events so you can
  still watch the run; the examples never crash on missing credentials.

## Notes / gotchas hit while building this

- In LangGraph **1.x**, `Send` imports from `langgraph.types` (not
  `langgraph.constants`).
- The `Send` payload is not merged with the parent state — pass the worker
  everything it needs.
- `SqliteSaver.from_conn_string` is a **context manager** in
  `langgraph-checkpoint-sqlite` 3.x; this repo constructs `SqliteSaver(conn)`
  directly (see `checkpointer.py`).
- Langfuse **4.x** removed `langfuse.callback` and `fetch_traces`; use
  `langfuse.langchain.CallbackHandler` with a `trace_context`, and read URLs
  via `get_trace_url`.
- Resuming a *stale* checkpoint with `Command(resume=...)` after a sibling
  branch exists can drop the value; the robust time-travel pattern is
  **replay** (`invoke(None, ckpt)`), then resume against the current tail.