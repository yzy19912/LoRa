import asyncio

from langsmith import Client

from hand_write_graph.graphs.dynamic_skill import graph

client = Client()


async def target(inputs: dict):
    result = await graph.ainvoke({"user_input": inputs["user_input"]})

    return {"skills": result["skills"]}


def skill_match(inputs, outputs, reference_outputs):
    return set(outputs["skills"]) == set(reference_outputs["skills"])


async def main():
    await client.aevaluate(
        target,
        data="skill-selector-eval",
        evaluators=[skill_match],
        experiment_prefix="skill-selector-v1",
    )


if __name__ == "__main__":
    asyncio.run(main())
