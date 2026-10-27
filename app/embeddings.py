from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

client = AsyncOpenAI()


async def create_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    # Match each returned vector to the corresponding input position.
    return [
        item.embedding for item in sorted(response.data, key=lambda item: item.index)
    ]
