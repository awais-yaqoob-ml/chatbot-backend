# ============================================================
# Multi-Agent Company Knowledge Chatbot — FastAPI Backend
# ============================================================
# Setup:
#   1. pip install -r requirements.txt
#   2. cp .env.example .env  →  fill in GROQ_API_KEY
#   3. uvicorn main:app --reload --port 8000
#   4. Swagger UI: http://localhost:8000/docs
#
# Architecture:
#   FastAPI → LangGraph StateGraph → [retrieve → route → agent]
#   Vector DB: Weaviate (embedded)
#   LLM: Groq (llama3-70b-8192)
#   Agents: KnowledgeAgent (RAG) | GeneralAgent (fallback)
# ============================================================

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging import setup_logging
from core.model_loader import load_embedding_model
from core.weaviate_client import (
    get_weaviate_client,
    initialize_weaviate,
)
from graph.graph_builder import get_compiled_graph

from routers.ingestion import router as ingestion_router
from routers.chat import router as chat_router
from routers.history import router as history_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle.
    """

    setup_logging()
    logger.info("Starting application...")

    # Load embedding model
    embed_model = load_embedding_model()
    app.state.embed_model = embed_model
    logger.info("Embedding model loaded")

    # Init Weaviate
    client = get_weaviate_client()
    initialize_weaviate(client)
    app.state.weaviate_client = client
    logger.info("Weaviate initialized")

    # Build graph
    graph = get_compiled_graph()
    app.state.graph = graph
    logger.info("LangGraph compiled")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    try:
        client.close()
    except Exception:
        pass


app = FastAPI(
    title="Multi-Agent Company Knowledge Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ingestion_router)
app.include_router(chat_router)
app.include_router(history_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "weaviate_connected": app.state.weaviate_client.is_ready(),
        "model_loaded": hasattr(app.state, "embed_model"),
        "graph_compiled": hasattr(app.state, "graph"),
    }