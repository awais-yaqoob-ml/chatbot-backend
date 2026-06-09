from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from core.dependencies import get_weaviate_client_dep, get_graph, get_embed_model
from models.schemas import ChatRequest, ChatResponse, SourceDoc
from services.history_service import save_message, get_history

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client=Depends(get_weaviate_client_dep),
    graph=Depends(get_graph),
):
    session_id = str(request.session_id)

    history = get_history(client, session_id, limit=5)

    state = {
        "session_id": session_id,
        "user_message": request.message,
        "chat_history": history,
        "retrieved_chunks": [],
        "retrieval_score": 0.0,
        "agent_decision": "",
        "final_answer": "",
        "sources": [],
        "agent_used": "",
        "error": None,
    }

    try:
        result = await graph.ainvoke(state)

        answer = result.get("final_answer", "")
        agent_used = result.get("agent_used", "")
        sources = result.get("sources", [])

        save_message(client, session_id, "user", request.message, "")
        save_message(client, session_id, "assistant", answer, agent_used)

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            agent_used=agent_used,
            sources=[
                SourceDoc(
                    filename=s.get("filename", ""),
                    page_number=s.get("page_number", 0),
                    chunk_index=s.get("chunk_index", 0),
                    score=s.get("score", 0.0),
                )
                for s in sources
            ],
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))