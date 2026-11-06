import asyncio
from unittest.mock import AsyncMock, Mock

from app import main
from app.schemas import SearchRequest


def test_search_returns_results_from_vector_store(monkeypatch):
    question = "What is the retention period?"
    payload = {
        "text": "Logs are kept for 30 days.",
        "file_name": "retention.pdf",
        "page_number": 2,
    }
    embeddings_mock = AsyncMock(return_value=[[0.1, 0.2]])
    search_mock = Mock(
        return_value=[
            Mock(score=0.95, payload=payload),
            Mock(score=0.4, payload=None),
        ]
    )
    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "search_chunks", search_mock)

    response = asyncio.run(main.search(SearchRequest(question=question, limit=3)))

    assert response == {
        "question": question,
        "results": [
            {
                "score": 0.95,
                "text": "Logs are kept for 30 days.",
                "file_name": "retention.pdf",
                "page_number": 2,
            },
            {
                "score": 0.4,
                "text": None,
                "file_name": None,
                "page_number": None,
            },
        ],
    }
    embeddings_mock.assert_awaited_once_with([question])
    search_mock.assert_called_once_with([0.1, 0.2], limit=3)


def test_search_returns_empty_results_when_nothing_matches(monkeypatch):
    question = "No matching information?"
    embeddings_mock = AsyncMock(return_value=[[0.3, 0.4]])
    search_mock = Mock(return_value=[])
    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "search_chunks", search_mock)

    response = asyncio.run(main.search(SearchRequest(question=question)))

    assert response == {"question": question, "results": []}
    embeddings_mock.assert_awaited_once_with([question])
    search_mock.assert_called_once_with([0.3, 0.4], limit=5)
