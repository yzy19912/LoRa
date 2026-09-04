import asyncio
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langsmith import traceable


class State(TypedDict):
    text: str
    length: int


async def clean_text(state: State):
    print(f"current len {len(state['text'])}")
    text = state["text"][:-1]
    await asyncio.sleep(2)
    return {"text": text}


async def count_length(state: State):
    await asyncio.sleep(3)
    return {"length": len(state["text"])}


def router(state: State) -> Literal["clean_text", "count_length"]:
    if len(state["text"]) > 5:
        return "clean_text"
    return "count_length"


# HITL


def review(state: State):
    interrupt(
        {
            "message": "Approve this text?",
            "text": state["text"],
        }
    )


builder = StateGraph(State)

builder.add_node("clean_text", clean_text)
builder.add_node("count_length", count_length)

builder.add_conditional_edges(START, router)

builder.add_conditional_edges(
    "clean_text",
    router,
)

builder.add_edge("count_length", END)

graph = builder.compile()

print(graph.get_graph().draw_mermaid())


@traceable(name="agent_request")
async def main():
    try:
        result = await graph.ainvoke(
            {
                "text": "1234567890",
                "length": 0,
            },
            config={"recursion_limit": 3},
        )
    except Exception as e:
        print(e)
    print("finished")


asyncio.run(main())
