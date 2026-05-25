from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str
    user_api_key: str
    document_ids: list[str] = Field(default_factory=list)


class ReferenceItem(BaseModel):
    doc_id: str
    source: str
    chunk_id: int
    score: float | None = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    references: list[ReferenceItem] = Field(default_factory=list)
    cache_hit: bool = False
    topic: str = ""
    scope: str = "all"


class UploadResponse(BaseModel):
    id: str
    name: str
    file_type: str
    uploaded_at: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    documents: list[dict[str, Any]] = Field(default_factory=list)


class ConversationsResponse(BaseModel):
    conversations: list[dict[str, Any]] = Field(default_factory=list)


class ClearCacheResponse(BaseModel):
    message: str
    cleared_count: int
