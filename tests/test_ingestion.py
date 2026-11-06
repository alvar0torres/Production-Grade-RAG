from unittest.mock import AsyncMock, Mock, call

from fastapi.testclient import TestClient
import pytest

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


def test_ingestion_batches_embeddings_in_groups_of_100(monkeypatch):
    chunks = [{"chunk_index": i, "text": f"Chunk {i}"} for i in range(1, 102)]
    pages = [
        {
            "page_number": 1,
            "text": "Page 1 text.",
            "chunks": chunks,
        }
    ]

    extract_mock = AsyncMock(return_value=("boundary.pdf", pages))
    first_batch_vectors = [[0.1, 0.2] for _ in range(100)]
    last_batch_vector = [[0.3, 0.4]]
    embeddings_mock = AsyncMock(side_effect=[first_batch_vectors, last_batch_vector])
    delete_mock = Mock()
    store_mock = Mock()

    monkeypatch.setattr(main, "extract_pdf_data", extract_mock)
    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)
    monkeypatch.setattr(main, "store_chunks", store_mock)

    response = client.post(
        "/documents/ingest",
        files={"file": ("boundary.pdf", b"fake PDF content", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["indexed_count"] == 101

    assert embeddings_mock.await_args_list == [
        call([f"Chunk {i}" for i in range(1, 101)]),
        call(["Chunk 101"]),
    ]

    store_mock.assert_called_once()
    stored_chunks, stored_embeddings = store_mock.call_args.args
    assert len(stored_chunks) == 101
    assert stored_embeddings == first_batch_vectors + last_batch_vector


def test_embedding_failure_does_not_delete_or_store_chunks(monkeypatch):
    pages = [
        {
            "page_number": 1,
            "text": "Some text.",
            "chunks": [{"chunk_index": 1, "text": "Some text."}],
        }
    ]

    extract_mock = AsyncMock(return_value=("fail.pdf", pages))
    embeddings_mock = AsyncMock(side_effect=RuntimeError("Embedding service failed"))
    delete_mock = Mock()
    store_mock = Mock()

    monkeypatch.setattr(main, "extract_pdf_data", extract_mock)
    monkeypatch.setattr(main, "create_embeddings", embeddings_mock)
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)
    monkeypatch.setattr(main, "store_chunks", store_mock)

    with pytest.raises(RuntimeError, match="Embedding service failed"):
        client.post(
            "/documents/ingest",
            files={"file": ("fail.pdf", b"fake PDF content", "application/pdf")},
        )

    embeddings_mock.assert_awaited_once_with(["Some text."])
    delete_mock.assert_not_called()
    store_mock.assert_not_called()
