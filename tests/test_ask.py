from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app import main
from app.schemas import GeneratedAnswer

client = TestClient(main.app)


def test_ask_returns_answer_with_citation(monkeypatch):
    sources = [
        {
            "text": "The capital of France is Paris.",
            "file_name": "geography.pdf",
            "page_number": 1,
            "chunk_index": 1,
        },
        {
            "text": "Berlin is the capital of Germany.",
            "file_name": "geography.pdf",
            "page_number": 2,
            "chunk_index": 1,
        },
    ]

    embeddings_mock = AsyncMock(return_value=[[0.1, 0.2]])
    search_results = [Mock(payload=source) for source in sources]
    search_mock = Mock(return_value=search_results)
    generated_answer = GeneratedAnswer(
        answer="The capital of France is Paris.",
        source_ids=["S1"],
    )
    generation_mock = AsyncMock(return_value=generated_answer)

    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "search_chunks", search_mock)
    monkeypatch.setattr(main, "generate_answer", generation_mock)

    response = client.post(
        "/ask",
        json={"question": "What is the capital of France?", "limit": 5},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The capital of France is Paris.",
        "citations": [
            {
                "file_name": "geography.pdf",
                "page_number": 1,
                "quote": "The capital of France is Paris.",
            }
        ],
    }

    embeddings_mock.assert_awaited_once_with(["What is the capital of France?"])
    search_mock.assert_called_once_with([0.1, 0.2], limit=5)
    generation_mock.assert_awaited_once_with("What is the capital of France?", sources)


def test_ask_rejects_unknown_source_id(monkeypatch):
    source = {
        "text": "The capital of France is Paris.",
        "file_name": "geography.pdf",
        "page_number": 1,
        "chunk_index": 1,
    }

    monkeypatch.setattr(
        main,
        "create_embeddings",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    monkeypatch.setattr(
        main,
        "search_chunks",
        Mock(return_value=[Mock(payload=source)]),
    )
    monkeypatch.setattr(
        main,
        "generate_answer",
        AsyncMock(
            return_value=GeneratedAnswer(
                answer="The capital is Paris.",
                source_ids=["S2"],  # Only S1 exists for this single result.
            )
        ),
    )

    response = client.post(
        "/ask",
        json={"question": "What is the capital of France?", "limit": 5},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "The answer model returned an unknown source ID: S2"
    }


def test_ask_handles_no_sources(monkeypatch):
    generation_mock = AsyncMock(
        return_value=GeneratedAnswer(
            answer="I could not find an answer in the provided sources.",
            source_ids=[],
        )
    )

    monkeypatch.setattr(
        main,
        "create_embeddings",
        AsyncMock(return_value=[[0.1, 0.2]]),
    )
    monkeypatch.setattr(
        main,
        "search_chunks",
        Mock(return_value=[]),  # No sources found.
    )
    monkeypatch.setattr(
        main,
        "generate_answer",
        generation_mock,
    )

    response = client.post(
        "/ask",
        json={"question": "What is the capital of France?", "limit": 5},
    )

    generation_mock.assert_not_awaited()
    assert response.status_code == 200
    assert response.json() == {
        "answer": "I could not find an answer in the provided sources.",
        "citations": [],
    }
