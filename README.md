# PDF RAG API

A learning project that demonstrates a Retrieval-Augmented Generation (RAG) API for PDF documents. It extracts and chunks PDF text, creates embeddings, stores chunks in a local Qdrant database, and answers questions with citations linked to retrieved source chunks.

## How it works

1. Upload a PDF for extraction or ingestion.
2. Extract text page by page and split it into overlapping, word-based chunks.
3. Generate embeddings with OpenAI's `text-embedding-3-small` model and store them in Qdrant.
4. Search for relevant chunks and pass them to `gpt-4o-mini` to generate an answer.
5. Map the model's source IDs back to retrieved chunks to return citations.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with access to the embedding and chat models used by the app

OpenAI API calls may incur charges.

## Setup

From the project directory, create a virtual environment:

```bash
python -m venv .venv
```

Activate it using the command for your shell:

```bash
source .venv/bin/activate
```

Install dependencies and configure the API key:

```bash
python -m pip install -r requirements.txt
cp .env.template .env
```

Replace the placeholder value for `OPENAI_API_KEY` in `.env` with your key. Do not commit `.env`.

## Run the API

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Main endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the API is running. |
| `POST` | `/documents/extract` | Extract page text and chunks from an uploaded PDF without indexing it. |
| `POST` | `/documents/ingest` | Extract a PDF, create embeddings, and replace indexed chunks with the same filename. |
| `POST` | `/documents/index` | Create embeddings and index supplied chunks. |
| `POST` | `/search` | Return the top matching chunks for a question. |
| `POST` | `/ask` | Generate an answer using retrieved chunks and return source citations. |
| `DELETE` | `/documents/delete?filename=example.pdf` | Delete indexed chunks for a filename. |

### Example: ingest a PDF and ask a question

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
  -F "file=@example.pdf"
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","limit":5}'
```

An `/ask` response has this shape:

```json
{
  "answer": "The answer based on retrieved sources.",
  "citations": [
    {
      "file_name": "example.pdf",
      "page_number": 1,
      "quote": "Text from the retrieved source chunk."
    }
  ]
}
```

## Tests

Run the test suite with:

```bash
python -m pytest
```

Tests mock external services, so they don't make OpenAI calls or modify the local Qdrant database.

## Current scope and limitations

- PDF documents only; Markdown and other formats are not implemented.
- Chunking is a simple word-based splitter with overlap, not semantic chunking.
- Retrieval uses dense vector search; hybrid BM25 search and reranking are not implemented.
- The Qdrant database is local and persistent in `qdrant_data/`.
- The answer response includes an answer and citations; a confidence score and frontend are not implemented.
- This is a learning project, not a hardened production deployment. Authentication, authorization, deployment configuration, and operational monitoring are outside its current scope.
