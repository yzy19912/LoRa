from typing import Annotated
from typing_extensions import TypedDict, Literal

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage

from ohmyrouter import ModelClient
import asyncio

client = ModelClient().get_model()



IntentType = Literal["refund", "tracking"]


SKILL_REGISTRY = {
    "refund_skill": """
 call bank tool to refund
""",
    "tracking_skill": """
 call usps tool to get tracking number.
""",
    "intent": """
 based on user's input, judge if user wanna to refund, tracking, or not related.
""",
}

BASIC_PROMPT = "You are a customer after service client, provide professional service including refund or tracking"


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    summary: str
    # order
    intent: IntentType
    order_id: int
    order_info: dict
    tracking_id: int
    human_approve: bool
    approve_info: str
    product_info: dict


def _construct(
    skill: str, user_input: str, history: list[AnyMessage], summary: str
) -> list[AnyMessage]:

    system_parts = [
        BASIC_PROMPT,
        SKILL_REGISTRY.get(skill) if skill else None,
        f"history_summary : {summary}" if summary else None,
    ]

    return [
        SystemMessage(content="\n\n".join(filter(None, system_parts))),
        *(history or []),
        *([HumanMessage(content=user_input)] if user_input else []),
    ]


class Intent(BaseModel):
    intent: IntentType

intent_llm = client.with_structured_output(Intent)

async def intent_node(state:State):
    intent_msg = {
        "skill": "intent",
        "user_input" : state["user_input"],
        "history": None,
        "summary": None,
    }


    message = _construct(**intent_msg)
    resp = await intent_llm.ainvoke(message)
    return {"intent": resp.intent}

builder = StateGraph(State)

builder.add_node("intent_node", intent_node)

builder.add_edge(START,"intent_node")
builder.add_edge("intent_node", END)

graph  = builder.compile()


async def main():
    res = await graph.ainvoke({"user_input": "where is my order"})
    print(res)


if __name__ == "__main__":
    asyncio.run(main())