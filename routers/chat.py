import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from core.dependencies import get_weaviate_client_dep, get_graph, get_embed_model
from models.schemas import ChatRequest, ChatResponse, SourceDoc
from services.history_service import save_message, get_history

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client=Depends(get_weaviate_client_dep),
    graph=Depends(get_graph),
):
    """
    Chat endpoint with intent-based routing.

    Flow:
    1. Retrieve chat history
    2. Initialize state with intent-based routing support
    3. Execute graph (intent classification → agent dispatch)
    4. Save conversation and return response
    """
    session_id = str(request.session_id)

    logger.info(f"Chat request from session: {session_id}")
    logger.info(f"User message: {request.message}")

    history = get_history(client, session_id, limit=5)

    # Initialize state with all required fields for intent-based routing
    state = {
        "session_id": session_id,
        "user_message": request.message,
        "chat_history": history,
        # Intent classification
        "intent": None,
        # Retrieval (for COMPANY_QA)
        "retrieved_chunks": [],
        "retrieval_score": 0.0,
        # Routing (legacy support)
        "agent_decision": None,
        # Output
        "final_answer": "",
        "sources": [],
        "summary": None,
        # Metadata
        "agent_used": "",
        "error": None,
    }

    try:
        logger.info("Invoking graph with intent-based routing")
        result = await graph.ainvoke(state)

        intent = result.get("intent", "UNKNOWN")
        agent_used = result.get("agent_used", "")
        answer = result.get("final_answer", "")
        sources = result.get("sources", [])

        logger.info(f"Intent classified as: {intent}")
        logger.info(f"Routed to agent: {agent_used}")

        # Save conversation to history
        save_message(client, session_id, "user", request.message, "")
        save_message(client, session_id, "assistant", answer, agent_used)

        logger.info(f"Response saved to history for session: {session_id}")

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
        logger.exception(f"Error processing chat request for session {session_id}")
        raise HTTPException(status_code=500, detail=str(e))