"""langgraph-learn: demo LangGraph pipeline with fan-out/fan-in,
human-in-the-loop, checkpointing and Langfuse observability.

Modules
-------
- state          : graph state shape + reducers (the fan-in mechanism)
- nodes          : node functions (mock by default, LLM optional)
- graph          : topology / compile
- checkpointer   : sqlite-backed persistence
- observability  : Langfuse + console tracing handlers
"""

from .checkpointer import create_checkpointer, list_threads, thread_config
from .graph import build_graph
from .observability import ConsoleTraceHandler, get_langfuse_handler, langfuse_trace_url

__all__ = [
    "ConsoleTraceHandler",
    "build_graph",
    "create_checkpointer",
    "get_langfuse_handler",
    "langfuse_trace_url",
    "list_threads",
    "thread_config",
]
