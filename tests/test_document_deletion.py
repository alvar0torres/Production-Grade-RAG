from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_delete_document_removes_chunks(monkeypatch):
    delete_mock = Mock()
    monkeypatch.setattr(main, "delete_document_chunks", delete_mock)

    response = client.delete(
        "/documents/delete",
        params={"filename": "demo.pdf"},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Document deleted: demo.pdf"}
    delete_mock.assert_called_once_with("demo.pdf")
