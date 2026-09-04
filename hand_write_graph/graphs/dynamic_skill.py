import asyncio
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ohmyrouter import ModelClient


class State(TypedDict):
    user_input: str
    skills: Annotated[list, operator.add]


def handle_refund(text) -> bool:
    return "refund" in text


def handle_general_question(text) -> bool:
    return "product" in text


def handle_return(text) -> bool:
    return "return" in text


handler = {
    "return": handle_return,
    "product": handle_general_question,
    "refund": handle_refund,
}

SKILL_REGISTRY = {
    "refund": "call refund tools",
    "product": "use rag to pull product info",
    "return": "use tracking tool to create label",
}


async def deter_skill_adder(state: State):
    text = state["user_input"]
    await asyncio.sleep(0.1)
    return {"skills": [skill for skill, handle in handler.items() if handle(text)]}


#   llm select


class LlmSkillSelector(BaseModel):
    skills: list[Literal["refund", "product", "return"]]
    reason: str


client = ModelClient().get_model()

skill_llm = client.with_structured_output(LlmSkillSelector)


async def get_skills_from_llm(state: State):
    prompt = f"""
    Select the relevant skills for the user request.

    giving the reason why you pick 

Available skills:
refund:
if user wanna his money back.

return:
if user don't like item and trying to keep it.

product:
if user ask question about the prodcut itself.

User request:
{state["user_input"]}
"""
    try:
        result = await skill_llm.ainvoke(prompt)
    except Exception as e:
        print("TYPE:", type(e))
        print("ERROR:", repr(e))

        if hasattr(e, "response"):
            print("RESPONSE:", e.response)

        if hasattr(e, "body"):
            print("BODY:", e.body)

        if hasattr(e, "message"):
            print("MESSAGE:", e.message)

        raise
    result = await skill_llm.ainvoke(prompt)
    print("llm", result)
    return {"skills": result.skills}


builder = StateGraph(State)

builder.add_node("deter_skill_adder", deter_skill_adder)
builder.add_node("get_skills_from_ll", get_skills_from_llm)

builder.add_edge(
    START,
    "deter_skill_adder",
)
builder.add_edge(START, "get_skills_from_ll")

builder.add_edge(["get_skills_from_ll", "deter_skill_adder"], END)

graph = builder.compile()


async def target(inputs: dict):
    result = await graph.ainvoke(
        {
            "user_input": inputs["user_input"],
        }
    )

    return {
        "skills": result["skills"],
    }


async def main():
    result = await graph.ainvoke(
        {
            "user_input": "I don't like this item, and I no longer need it, could I send it back, money back"
        }
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
