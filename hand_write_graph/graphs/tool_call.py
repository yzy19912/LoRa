import asyncio
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from ohmyrouter import ModelClient


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
async def get_weather(city: str):
    """Return weather info"""
    return f"{city}  is 26 C"


@tool
async def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


tools = [calculate, get_weather]


client = ModelClient().get_model().bind_tools(tools)

tool_node = ToolNode(tools)


async def agent(state: State):
    response = await client.ainvoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(State)

builder.add_node("tools", tool_node)
builder.add_node("agent", agent)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()


async def main():
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="What's the weather in San Francisco, and what is 10 + 20?"
                )
            ]
        }
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
