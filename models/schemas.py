from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# =========================================================
# INGESTION
# =========================================================

class IngestionResponse(BaseModel):
    doc_id: str
    filename: str
    total_chunks: int
    images_extracted: int
    pages_processed: int
    message: str = "Document ingested successfully"


# =========================================================
# CHAT
# =========================================================

class ChatRequest(BaseModel):
    session_id: UUID
    message: str


class SourceDoc(BaseModel):
    filename: str
    page_number: int
    chunk_index: int
    score: float


class ResponseImage(BaseModel):
    filename: str
    mime_type: str
    data: str


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    agent_used: str
    sources: List[SourceDoc] = []
    images: List[ResponseImage] = []
    timestamp: datetime


# =========================================================
# HISTORY
# =========================================================

class ChatMessage(BaseModel):
    role: str
    content: str
    agent_used: Optional[str] = ""
    timestamp: datetime


class HistoryResponse(BaseModel):
    session_id: UUID
    messages: List[ChatMessage]
    total: int


class SessionsResponse(BaseModel):
    sessions: List[str]
    total: int


# =========================================================
# HEALTH
# =========================================================

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    weaviate_connected: bool
    model_loaded: bool
    graph_compiled: bool


# =========================================================
# DOCUMENTS
# =========================================================

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    doc_summary: str = ""
    topics: List[str] = []
    uploaded_at: str = ""
    chunk_count: int = 0


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


# =========================================================
# ERROR
# =========================================================

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)