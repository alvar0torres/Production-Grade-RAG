from uuid import UUID
from unittest.mock import Mock

from qdrant_client import models
import pytest

from app import vector_store


def test_store_chunks_upserts_points_with_embeddings_and_payloads(monkeypatch):
    upsert_mock = Mock()
    monkeypatch.setattr(vector_store, "client", Mock(upsert=upsert_mock))
    chunks = [
        {"text": "First chunk", "file_name": "demo.pdf", "page_number": 1},
        {"text": "Second chunk", "file_name": "demo.pdf", "page_number": 2},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    vector_store.store_chunks(chunks, embeddings)

    upsert_mock.assert_called_once()
    kwargs = upsert_mock.call_args.kwargs
    assert kwargs["collection_name"] == vector_store.COLLECTION_NAME
    points = kwargs["points"]
    assert len(points) == 2
    assert [point.vector for point in points] == embeddings
    assert [point.payload for point in points] == chunks
    for point in points:
        UUID(point.id)


def test_store_chunks_rejects_mismatched_chunk_and_embedding_counts(monkeypatch):
    upsert_mock = Mock()
    monkeypatch.setattr(vector_store, "client", Mock(upsert=upsert_mock))

    with pytest.raises(ValueError, match="There must be one embedding for each chunk."):
        vector_store.store_chunks(
            [{"text": "A chunk", "file_name": "demo.pdf", "page_number": 1}],
            [],
        )

    upsert_mock.assert_not_called()


def test_search_chunks_passes_query_options_and_returns_results(monkeypatch):
    results = [Mock(), Mock()]
    query_points_mock = Mock(return_value=Mock(points=results))
    monkeypatch.setattr(
        vector_store,
        "client",
        Mock(query_points=query_points_mock),
    )
    embedding = [0.1, 0.2]

    response = vector_store.search_chunks(embedding, limit=3)

    assert response is results
    query_points_mock.assert_called_once_with(
        collection_name=vector_store.COLLECTION_NAME,
        query=embedding,
        limit=3,
        with_payload=True,
    )


def test_delete_document_chunks_sends_file_name_filter(monkeypatch):
    delete_mock = Mock()
    monkeypatch.setattr(vector_store, "client", Mock(delete=delete_mock))

    vector_store.delete_document_chunks("demo.pdf")

    delete_mock.assert_called_once()
    kwargs = delete_mock.call_args.kwargs
    assert kwargs["collection_name"] == vector_store.COLLECTION_NAME

    selector = kwargs["points_selector"]
    assert isinstance(selector, models.FilterSelector)
    assert len(selector.filter.must) == 1

    condition = selector.filter.must[0]
    assert isinstance(condition, models.FieldCondition)
    assert condition.key == "file_name"
    assert condition.match == models.MatchValue(value="demo.pdf")
