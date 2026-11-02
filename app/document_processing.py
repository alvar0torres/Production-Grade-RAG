from io import BytesIO
from pathlib import Path

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.chunking import chunk_text
from app.schemas import ExtractedPage


async def extract_pdf_data(
    file: UploadFile,
) -> tuple[str, list[ExtractedPage]]:
    filename = file.filename

    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read()

    try:
        reader = PdfReader(BytesIO(contents))
    except PdfReadError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF.",
        ) from error

    if reader.is_encrypted:
        raise HTTPException(
            status_code=400,
            detail="Encrypted PDFs are not supported yet.",
        )

    pages: list[ExtractedPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks = chunk_text(text)

        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "chunks": [
                    {"chunk_index": chunk_index, "text": chunk}
                    for chunk_index, chunk in enumerate(chunks, start=1)
                ],
            }
        )

    return filename, pages
