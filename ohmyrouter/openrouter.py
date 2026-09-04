from langchain_openrouter import ChatOpenRouter


class ModelClient:
    def __init__(
        self,
        model: str = "deepseek/deepseek-v4-flash-0731",
        temperature: float = 0,
        thinking: str = "low",
    ):
        # Read OpenRouter API key from environment

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set.")

        self.client = ChatOpenRouter(
            model=model,
            temperature=temperature,
            api_key=api_key,
            max_retries=2,
            reasoning={
                "effort": thinking,
            },
        )

    def get_model(self):
        return self.client

    async def ainvoke(self, messages):
        return await self.client.ainvoke(messages)

    def invoke(self, messages):
        return self.client.invoke(messages)
