from unittest.mock import AsyncMock, Mock

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_ingest_pdf_embeds_and_stores_chunks(monkeypatch):
    pages = [
        {
            "page_number": 1,
            "text": "Logs are kept for 30 days.",
            "chunks": [
                {"chunk_index": 1, "text": "Logs are kept for 30 days."},
            ],
        },
        {
            "page_number": 2,
            "text": "Backups are encrypted.",
            "chunks": [
                {"chunk_index": 1, "text": "Backups are encrypted."},
            ],
        },
    ]

    extract_mock = AsyncMock(return_value=("demo.pdf", pages))
    embeddings = [[0.1, 0.2], [0.3, 0.4]]
    embeddings_mock = AsyncMock(return_value=embeddings)

    call_order = []
    delete_mock = Mock(side_effect=lambda filename: call_order.append("delete"))
    store_mock = Mock(side_effect=lambda chunks, vectors: call_order.append("store"))

    monkeypatch.setattr(main, "extract_pdf_data", extract_mock)
    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)
    monkeypatch.setattr(main, "store_chunks", store_mock)

    response = client.post(
        "/documents/ingest",
        files={"file": ("demo.pdf", b"fake PDF content", "application/pdf")},
    )

    expected_chunks = [
        {
            "text": "Logs are kept for 30 days.",
            "file_name": "demo.pdf",
            "page_number": 1,
            "chunk_index": 1,
        },
        {
            "text": "Backups are encrypted.",
            "file_name": "demo.pdf",
            "page_number": 2,
            "chunk_index": 1,
        },
    ]

    assert response.status_code == 200
    assert response.json() == {
        "filename": "demo.pdf",
        "page_count": 2,
        "chunk_count": 2,
        "indexed_count": 2,
    }

    embeddings_mock.assert_awaited_once_with(
        ["Logs are kept for 30 days.", "Backups are encrypted."]
    )
    delete_mock.assert_called_once_with("demo.pdf")
    store_mock.assert_called_once_with(expected_chunks, embeddings)
    assert call_order == ["delete", "store"]


def test_ingest_pdf_with_no_chunks_returns_400(monkeypatch):
    pages = [
        {
            "page_number": 1,
            "text": "",
            "chunks": [],
        },
    ]

    extract_mock = AsyncMock(return_value=("empty.pdf", pages))
    embeddings_mock = AsyncMock()
    delete_mock = Mock()
    store_mock = Mock()

    monkeypatch.setattr(main, "extract_pdf_data", extract_mock)
    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)
    monkeypatch.setattr(main, "store_chunks", store_mock)

    response = client.post(
        "/documents/ingest",
        files={"file": ("empty.pdf", b"fake PDF content", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "No text could be extracted from this PDF."}
    embeddings_mock.assert_not_awaited()
    delete_mock.assert_not_called()
    store_mock.assert_not_called()
