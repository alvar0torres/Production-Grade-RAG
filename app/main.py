from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.embeddings import create_embedding

class EmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)

app = FastAPI(title="Production-Grade RAG API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/embeddings")
async def create_embedding_endpoint(request: EmbeddingRequest):
    embedding = await create_embedding(request.text)
    return {
        "dimensions": len(embedding), 
        "preview": embedding[:5]
    }

@app.post("/documents/extract")
async def extract_pdf(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    contents = await file.read()

    try:
        reader = PdfReader(BytesIO(contents))
    except PdfReadError as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be read as a PDF."
        ) from error

    if reader.is_encrypted:
        raise HTTPException(
            status_code=400,
            detail="Encrypted PDFs are not supported yet."
        )

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks = chunk_text(text)

        pages.append(
            {
                "page_number": page_number,
                "text": text,
                "chunks": [
                    {
                        "chunk_index": chunk_index,
                        "text": chunk,
                    }
                    for chunk_index, chunk in enumerate(chunks, start=1)
                ],
            }
        )

    return {
        "filename": file.filename,
        "page_count": len(pages),
        "chunk_count": sum(len(page["chunks"]) for page in pages),
        "pages": pages,
    }
