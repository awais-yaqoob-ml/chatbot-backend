import logging

from core.config import settings
from services.embedding_service import embed_query
from services.retrieval_service import retrieve_chunks
from services.llm_service import get_llm, build_messages, run_llm

logger = logging.getLogger(__name__)


def retrieve_node(state, embed_model, client):
    try:
        logger.info("Running retrieve_node")

        query = state["user_message"]

        query_vector = embed_query(embed_model, query)

        chunks = retrieve_chunks(
            client=client,
            query_vector=query_vector,
            top_k=settings.top_k,
            )

        max_score = (
            max(
                [c["score"] for c in chunks],
                default=0.0,
            )
        )

        return {
            "retrieved_chunks": chunks,
            "retrieval_score": max_score,
        }

    except Exception as e:
        logger.exception("retrieve_node failed")
        return {"error": str(e)}


def router_node(state):
    try:
        logger.info("Running router_node")

        score = state.get("retrieval_score", 0.0)

        threshold = settings.retrieval_threshold

        decision = (
            "knowledge_agent"
            if score >= threshold
            else "general_agent"
        )

        return {"agent_decision": decision}

    except Exception as e:
        logger.exception("router_node failed")
        return {"error": str(e)}


def knowledge_agent_node(state, llm):
    try:
        logger.info("Running knowledge_agent_node")

        chunks = state.get("retrieved_chunks", [])

        context = "\n\n".join(
            [
                f"[{c.get('filename')} | page {c.get('page_number')}]\n{c.get('text')}"
                for c in chunks
            ]
        )

        system_prompt = (
            "You are a precise company user-guide assistant. "
            "Answer ONLY using the provided context. "
            "If the answer is not in the context, say: "
            "'I could not find this information in the company documents.' "
            "Always cite source filename and page number."
            f"\n\nContext:\n{context}"
        )

        messages = build_messages(
            system_prompt=system_prompt,
            chat_history=state.get("chat_history", []),
            user_message=state["user_message"],
        )

        answer = run_llm(llm, messages)

        sources = [
            {
                "filename": c.get("filename"),
                "page_number": c.get("page_number"),
                "chunk_index": c.get("chunk_index"),
                "score": c.get("score", 0.0),
            }
            for c in chunks
        ]

        return {
            "final_answer": answer,
            "sources": sources,
            "agent_used": "KnowledgeAgent",
        }

    except Exception as e:
        logger.exception("knowledge_agent_node failed")
        return {"error": str(e)}


def general_agent_node(state, llm):
    try:
        logger.info("Running general_agent_node")

        system_prompt = (
            "You are a friendly general assistant for this company. "
            "Answer helpfully. You do not have access to internal documents."
        )

        messages = build_messages(
            system_prompt=system_prompt,
            chat_history=state.get("chat_history", []),
            user_message=state["user_message"],
        )

        answer = run_llm(llm, messages)

        return {
            "final_answer": answer,
            "sources": [],
            "agent_used": "GeneralAgent",
        }

    except Exception as e:
        logger.exception("general_agent_node failed")
        return {"error": str(e)}


def error_handler_node(state):
    logger.error("Running error_handler_node")

    return {
        "final_answer": "An error occurred while processing your request.",
        "sources": [],
        "agent_used": "ErrorHandler",
    }