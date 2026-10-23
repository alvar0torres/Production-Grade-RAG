from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

app = FastAPI(title="Production-Grade RAG API")

@app.get("/health")
def health_check():
    return {"status": "ok"}

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

    pages = [
        {
            "page_number": page_number,
            "text": page.extract_text() or ""
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]

    return {
        "filename": file.filename, 
        "page_count": len(pages), 
        "pages": pages
    }
