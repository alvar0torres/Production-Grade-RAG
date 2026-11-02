from pydantic import BaseModel, Field
from typing import TypedDict


class ChunkInput(BaseModel):
    text: str = Field(min_length=1)
    file_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)


class IndexChunksRequest(BaseModel):
    chunks: list[ChunkInput] = Field(min_length=1, max_length=100)


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class SearchRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=10)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=10)


class Citation(BaseModel):
    file_name: str
    page_number: int
    quote: str


class GeneratedAnswer(BaseModel):
    answer: str
    source_ids: list[str]


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]


class ExtractedChunk(TypedDict):
    chunk_index: int
    text: str


class ExtractedPage(TypedDict):
    page_number: int
    text: str
    chunks: list[ExtractedChunk]
