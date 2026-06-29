from fastapi import Request

from core.model_loader import load_embedding_model
from core.weaviate_client import get_weaviate_client
from services.llm_service import get_llm


def get_embed_model(request: Request):
    """
    Dependency to fetch embedding model from app state.
    """
    return request.app.state.embed_model


def get_weaviate_client_dep(request: Request):
    """
    Dependency to fetch Weaviate client from app state.
    """
    return request.app.state.weaviate_client


def get_graph(request: Request):
    """
    Dependency to fetch compiled LangGraph from app state.
    """
    return request.app.state.graph


def get_llm_dep():
    """
    Dependency to create an LLM instance.
    """
    return get_llm()