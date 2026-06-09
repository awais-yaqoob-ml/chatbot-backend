from fastapi import APIRouter, Depends, HTTPException

from core.dependencies import get_weaviate_client_dep
from models.schemas import HistoryResponse, SessionsResponse, ChatMessage
from services.history_service import get_history, delete_history, list_sessions

router = APIRouter(prefix="/api/v1", tags=["history"])


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def fetch_history(session_id: str, client=Depends(get_weaviate_client_dep)):
    messages = get_history(client, session_id)

    if messages is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
    )


@router.delete("/history/{session_id}")
async def remove_history(session_id: str, client=Depends(get_weaviate_client_dep)):
    deleted = delete_history(client, session_id)

    return {
        "session_id": session_id,
        "deleted": deleted,
    }


@router.get("/sessions", response_model=SessionsResponse)
async def get_all_sessions(client=Depends(get_weaviate_client_dep)):
    sessions = list_sessions(client)

    return SessionsResponse(
        sessions=sessions,
        total=len(sessions),
    )