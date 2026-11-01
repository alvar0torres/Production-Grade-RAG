from dotenv import load_dotenv
from openai import AsyncOpenAI

from app.schemas import GeneratedAnswer

load_dotenv()

client = AsyncOpenAI()


async def generate_answer(
    question: str,
    sources: list[dict[str, str | int]],
) -> GeneratedAnswer:
    source_text = "\n\n".join(
        f"[S{index}] {source['text']}" for index, source in enumerate(sources, start=1)
    )

    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided sources. "
                    "If they don't contain the answer, say so and return no "
                    "source IDs. Include only IDs of sources that support "
                    "your answer."
                ),
            },
            {
                "role": "user",
                "content": (f"Question: {question}\n\n" f"Sources:\n{source_text}"),
            },
        ],
        response_format=GeneratedAnswer,
    )

    message = response.choices[0].message

    if message.parsed is None:
        raise ValueError("The model did not return a structured answer.")

    return message.parsed
