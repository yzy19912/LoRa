import asyncio
import os
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-first-project"


class Profile(TypedDict):
    name: str
    age: int


class State(TypedDict):
    user_id: int
    profile: Profile
    order_history: Annotated[list[str], add]
    result: Literal["Pass", "Fail"]


async def get_user_profile(state: State):
    print(f"fetching user: {state['user_id']}")
    await asyncio.sleep(0.1)
    return {"profile": {"name": "大脑袋", "age": 22}}


async def get_order_history(state: State):
    print(f"fetching user history for: {state['user_id']}")
    await asyncio.sleep(0.3)
    return {"order_history": ["refund"]}


def human_review(state: State):
    decision = interrupt(
        {
            "message": f"{state['profile']['name']} 要买500w 的房子",
            "question": "同意贷款吗？",
        }
    )

    if decision["result"] == "Approve":
        return Command(goto="check_out", update={"result": "Pass"})

    else:
        return Command(goto="good_bye", update={"result": "Fail"})


def good_bye(state: State):
    print(f"{state['profile']['name']} 好好上班，下次再来")
    return {"result": "Fail"}


async def check_out(state: State):
    await asyncio.sleep(1)
    print(f" {state['profile']['name']} 成功贷款，上牛马")
    return {"result": "Pass"}


graph = StateGraph(State)

graph.add_node("get_user_profile", get_user_profile)
graph.add_node("get_order_history", get_order_history)
graph.add_node("human_review", human_review)
graph.add_node("good_bye", good_bye)
graph.add_node("check_out", check_out)

graph.add_edge(START, "get_user_profile")
graph.add_edge(START, "get_order_history")
graph.add_edge(["get_order_history", "get_user_profile"], "human_review")
graph.add_edge("check_out", END)
graph.add_edge("good_bye", END)


bot = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "3154"}}


async def main():
    result = await bot.ainvoke({"user_id": 123}, config=config)

    print(f"first : {result}")

    manager_review = input("Approve?")

    result = await bot.ainvoke(
        Command(resume={"result": manager_review.capitalize()}), config=config
    )

    print(f"second : {result}")


asyncio.run(main())
