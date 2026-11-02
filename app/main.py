from fastapi import FastAPI, File, HTTPException, UploadFile

from app.document_processing import extract_pdf_data
from app.embeddings import create_embeddings
from app.generation import generate_answer
from app.vector_store import search_chunks, store_chunks
from app.schemas import (
    AskRequest,
    AskResponse,
    Citation,
    EmbeddingRequest,
    IndexChunksRequest,
    SearchRequest,
)

app = FastAPI(title="Production-Grade RAG API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/embeddings")
async def create_embedding_endpoint(request: EmbeddingRequest):
    embeddings = await create_embeddings(request.texts)

    return {
        "embedding_count": len(embeddings),
        "dimensions": len(embeddings[0]),
        "previews": [embedding[:5] for embedding in embeddings],
    }


@app.post("/documents/extract")
async def extract_pdf(file: UploadFile = File(...)):
    filename, pages = await extract_pdf_data(file)

    return {
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": sum(len(page["chunks"]) for page in pages),
        "pages": pages,
    }


@app.post("/documents/index")
async def index_chunks(request: IndexChunksRequest):
    chunks = [chunk.model_dump() for chunk in request.chunks]
    texts = [chunk["text"] for chunk in chunks]

    embeddings = await create_embeddings(texts)
    store_chunks(chunks, embeddings)

    return {"indexed_count": len(chunks)}


@app.post("/search")
async def search(request: SearchRequest):
    embeddings = await create_embeddings([request.question])
    results = search_chunks(embeddings[0], limit=request.limit)

    return {
        "question": request.question,
        "results": [
            {
                "score": result.score,
                "text": result.payload.get("text") if result.payload else None,
                "file_name": (
                    result.payload.get("file_name") if result.payload else None
                ),
                "page_number": (
                    result.payload.get("page_number") if result.payload else None
                ),
            }
            for result in results
        ],
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    question_embedding = (await create_embeddings([request.question]))[0]
    results = search_chunks(question_embedding, limit=request.limit)

    sources = [result.payload for result in results if result.payload is not None]

    generated = await generate_answer(request.question, sources)

    # Map the model's source IDs back to retrieved chunks.
    sources_by_id = {
        f"S{index}": source for index, source in enumerate(sources, start=1)
    }

    citations = []
    for source_id in generated.source_ids:
        source = sources_by_id.get(source_id)

        if source is None:
            raise HTTPException(
                status_code=502,
                detail=f"The answer model returned an unknown source ID: {source_id}",
            )

        citations.append(
            Citation(
                file_name=source["file_name"],
                page_number=source["page_number"],
                quote=source["text"],
            )
        )

    return AskResponse(answer=generated.answer, citations=citations)


@app.post("/documents/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    filename, pages = await extract_pdf_data(file)

    # Add source details to every chunk before storing it.
    chunks: list[dict[str, str | int]] = []
    for page in pages:
        for chunk in page["chunks"]:
            chunks.append(
                {
                    "text": chunk["text"],
                    "file_name": filename,
                    "page_number": page["page_number"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this PDF.",
        )

    # Send chunks in batches of at most 100 to the embedding API.
    embeddings: list[list[float]] = []
    for batch_start in range(0, len(chunks), 100):
        batch = chunks[batch_start : batch_start + 100]
        batch_embeddings = await create_embeddings([chunk["text"] for chunk in batch])
        embeddings.extend(batch_embeddings)

    store_chunks(chunks, embeddings)

    return {
        "filename": filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "indexed_count": len(chunks),
    }
