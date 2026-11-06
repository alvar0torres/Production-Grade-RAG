from io import BytesIO
from unittest.mock import Mock

from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.errors import PdfReadError

from app import document_processing, main

client = TestClient(main.app)


def test_extract_pdf_returns_text_and_chunks(monkeypatch):
    first_page = Mock()
    first_page.extract_text.return_value = "Logs are kept for 30 days."

    second_page = Mock()
    second_page.extract_text.return_value = "Backups are encrypted."

    fake_reader = Mock(
        pages=[first_page, second_page],
        is_encrypted=False,
    )
    reader_mock = Mock(return_value=fake_reader)
    monkeypatch.setattr(document_processing, "PdfReader", reader_mock)

    response = client.post(
        "/documents/extract",
        files={"file": ("demo.pdf", b"fake PDF bytes", "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "demo.pdf",
        "page_count": 2,
        "chunk_count": 2,
        "pages": [
            {
                "page_number": 1,
                "text": "Logs are kept for 30 days.",
                "chunks": [
                    {
                        "chunk_index": 1,
                        "text": "Logs are kept for 30 days.",
                    }
                ],
            },
            {
                "page_number": 2,
                "text": "Backups are encrypted.",
                "chunks": [
                    {
                        "chunk_index": 1,
                        "text": "Backups are encrypted.",
                    }
                ],
            },
        ],
    }


def test_extract_parses_a_real_pdf():
    pdf_buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(pdf_buffer)

    response = client.post(
        "/documents/extract",
        files={"file": ("blank.pdf", pdf_buffer.getvalue(), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "blank.pdf",
        "page_count": 1,
        "chunk_count": 0,
        "pages": [
            {
                "page_number": 1,
                "text": "",
                "chunks": [],
            }
        ],
    }


def test_extract_rejects_non_pdf_filename(monkeypatch):
    reader_mock = Mock()
    monkeypatch.setattr(document_processing, "PdfReader", reader_mock)

    response = client.post(
        "/documents/extract",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Please upload a PDF file."}
    reader_mock.assert_not_called()


def test_extract_rejects_unreadable_pdf(monkeypatch):
    reader_mock = Mock(side_effect=PdfReadError("Invalid PDF"))
    monkeypatch.setattr(document_processing, "PdfReader", reader_mock)

    response = client.post(
        "/documents/extract",
        files={"file": ("broken.pdf", b"not really a PDF", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "The uploaded file could not be read as a PDF."
    }


def test_extract_rejects_encrypted_pdf(monkeypatch):
    fake_reader = Mock(pages=[], is_encrypted=True)
    reader_mock = Mock(return_value=fake_reader)
    monkeypatch.setattr(document_processing, "PdfReader", reader_mock)

    response = client.post(
        "/documents/extract",
        files={"file": ("locked.pdf", b"fake PDF bytes", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Encrypted PDFs are not supported yet."}
