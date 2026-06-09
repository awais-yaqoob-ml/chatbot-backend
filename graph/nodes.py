import logging
from typing import Optional

from core.config import settings
from services.embedding_service import embed_query
from services.retrieval_service import retrieve_chunks
from services.llm_service import get_llm, build_messages, run_llm

logger = logging.getLogger(__name__)

# Intent constants
INTENT_GREETING = "GREETING"
INTENT_SUMMARIZATION = "SUMMARIZATION"
INTENT_COMPANY_QA = "COMPANY_QA"
INTENT_FALLBACK = "FALLBACK"


# ==============================================================================
# INTENT CLASSIFICATION NODE
# ==============================================================================


def intent_classifier_node(state, llm) -> dict:
    """
    Classify user intent into one of: GREETING, SUMMARIZATION, COMPANY_QA, FALLBACK.

    Args:
        state: AgentState containing user_message
        llm: Language model instance

    Returns:
        dict with intent field set
    """
    try:
        logger.info("Running intent_classifier_node")

        user_message = state["user_message"]

        system_prompt = (
            "You are an intent classification system. "
            "Classify the user query into exactly one of: GREETING, SUMMARIZATION, COMPANY_QA, FALLBACK.\n\n"
            "Classification rules:\n"
            "- GREETING: Greetings, pleasantries, small talk (Hi, Hello, Good morning, How are you?, Thanks, Nice to meet you)\n"
            "- SUMMARIZATION: Requests to summarize conversation (Summarize our conversation, What have we discussed?, Give me a summary, Recap)\n"
            "- COMPANY_QA: Questions about company policies, documents, procedures, or business-specific inquiries answerable from knowledge base\n"
            "- FALLBACK: Anything unrelated to company information (Who won the World Cup?, Explain quantum mechanics, Write Python code, General knowledge)\n\n"
            "Return ONLY the label, no explanation."
        )

        messages = build_messages(
            system_prompt=system_prompt,
            chat_history=[],
            user_message=user_message,
        )

        response = run_llm(llm, messages).strip().upper()

        # Validate intent
        valid_intents = [INTENT_GREETING, INTENT_SUMMARIZATION, INTENT_COMPANY_QA, INTENT_FALLBACK]
        intent = response if response in valid_intents else INTENT_FALLBACK

        logger.info(f"Intent classified as: {intent}")

        return {"intent": intent}

    except Exception as e:
        logger.exception("intent_classifier_node failed")
        return {"error": str(e), "intent": INTENT_FALLBACK}


# ==============================================================================
# RETRIEVAL NODE (used for COMPANY_QA)
# ==============================================================================


def retrieve_node(state, embed_model, client) -> dict:
    """
    Retrieve relevant chunks from vector database.

    Args:
        state: AgentState containing user_message
        embed_model: Embedding model for vectorizing queries
        client: Weaviate client

    Returns:
        dict with retrieved_chunks and retrieval_score
    """
    try:
        logger.info("Running retrieve_node")

        query = state["user_message"]

        query_vector = embed_query(embed_model, query)

        chunks = retrieve_chunks(
            client=client,
            query_vector=query_vector,
            top_k=settings.top_k,
        )

        max_score = max([c["score"] for c in chunks], default=0.0)

        return {
            "retrieved_chunks": chunks,
            "retrieval_score": max_score,
        }

    except Exception as e:
        logger.exception("retrieve_node failed")
        return {"error": str(e)}


# ==============================================================================
# GREETING AGENT NODE
# ==============================================================================


def greeting_agent_node(state: dict) -> dict:
    """
    Handle greeting and small talk requests.

    Args:
        state: AgentState

    Returns:
        dict with final_answer, sources, and agent_used
    """
    try:
        logger.info("Running greeting_agent_node")

        greeting_response = (
            "Hello! I'm your company assistant. How can I help you with "
            "company-related information, policies, or procedures today?"
        )

        return {
            "final_answer": greeting_response,
            "sources": [],
            "agent_used": "GreetingAgent",
        }

    except Exception as e:
        logger.exception("greeting_agent_node failed")
        return {
            "error": str(e),
            "final_answer": "Hello! How can I help you?",
            "sources": [],
            "agent_used": "GreetingAgent",
        }


# ==============================================================================
# SUMMARIZATION AGENT NODE
# ==============================================================================


def summarization_agent_node(state, llm) -> dict:
    """
    Generate a summary of the conversation history.

    Args:
        state: AgentState containing chat_history
        llm: Language model instance

    Returns:
        dict with final_answer (summary), summary field, and agent_used
    """
    try:
        logger.info("Running summarization_agent_node")

        chat_history = state.get("chat_history", [])

        if not chat_history:
            summary = "No conversation history to summarize."
            return {
                "final_answer": summary,
                "summary": summary,
                "sources": [],
                "agent_used": "SummarizationAgent",
            }

        # Format history into text
        history_text = "\n".join(
            [f"{msg.get('role', '').upper()}: {msg.get('content', '')}" for msg in chat_history]
        )

        system_prompt = (
            "You are a concise conversation summarizer. "
            "Generate a brief summary of the conversation with the following structure:\n\n"
            "Summary:\n[1-2 sentences summarizing the main topics]\n\n"
            "Key Topics:\n[Bullet list of main topics discussed]\n\n"
            "Open Questions:\n[Any unanswered questions or topics to follow up on]"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this conversation:\n\n{history_text}"},
        ]

        summary = run_llm(llm, messages)

        logger.info("Conversation summarized successfully")

        return {
            "final_answer": summary,
            "summary": summary,
            "sources": [],
            "agent_used": "SummarizationAgent",
        }

    except Exception as e:
        logger.exception("summarization_agent_node failed")
        return {
            "error": str(e),
            "final_answer": "Could not generate summary at this time.",
            "summary": None,
            "sources": [],
            "agent_used": "SummarizationAgent",
        }


# ==============================================================================
# COMPANY QA AGENT NODE (formerly knowledge_agent_node)
# ==============================================================================


def company_qa_agent_node(state, llm) -> dict:
    """
    Answer questions using company knowledge base (RAG).

    Args:
        state: AgentState containing user_message, chat_history, retrieved_chunks
        llm: Language model instance

    Returns:
        dict with final_answer, sources, and agent_used
    """
    try:
        logger.info("Running company_qa_agent_node")

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

        logger.info(f"Company QA completed with {len(sources)} sources")

        return {
            "final_answer": answer,
            "sources": sources,
            "agent_used": "CompanyQAAgent",
        }

    except Exception as e:
        logger.exception("company_qa_agent_node failed")
        return {"error": str(e)}


# ==============================================================================
# FALLBACK AGENT NODE
# ==============================================================================


def fallback_agent_node(state: dict) -> dict:
    """
    Handle unsupported requests outside company knowledge domain.

    Args:
        state: AgentState

    Returns:
        dict with final_answer, sources, and agent_used
    """
    try:
        logger.info("Running fallback_agent_node")

        fallback_response = (
            "I'm designed to assist with company-related information and support. "
            "I cannot answer questions outside the company knowledge domain. "
            "Please ask me about company policies, procedures, or other company-related topics."
        )

        logger.info("Fallback agent invoked")

        return {
            "final_answer": fallback_response,
            "sources": [],
            "agent_used": "FallbackAgent",
        }

    except Exception as e:
        logger.exception("fallback_agent_node failed")
        return {
            "error": str(e),
            "final_answer": "I'm unable to assist with that request.",
            "sources": [],
            "agent_used": "FallbackAgent",
        }


# ==============================================================================
# ERROR HANDLER NODE
# ==============================================================================


def error_handler_node(state: dict) -> dict:
    """
    Handle errors that occur during graph execution.

    Args:
        state: AgentState containing error information

    Returns:
        dict with error response
    """
    error_msg = state.get("error", "Unknown error occurred")
    logger.error(f"Error handler invoked: {error_msg}")

    return {
        "final_answer": "An error occurred while processing your request. Please try again.",
        "sources": [],
        "agent_used": "ErrorHandler",
    }


# ==============================================================================
# LEGACY NODES (for backward compatibility)
# ==============================================================================


def router_node(state: dict) -> dict:
    """
    Legacy router node for backward compatibility.
    Routes based on retrieval score.

    Args:
        state: AgentState

    Returns:
        dict with agent_decision
    """
    try:
        logger.info("Running router_node (legacy)")

        score = state.get("retrieval_score", 0.0)
        threshold = settings.retrieval_threshold

        decision = "knowledge_agent" if score >= threshold else "general_agent"

        return {"agent_decision": decision}

    except Exception as e:
        logger.exception("router_node failed")
        return {"error": str(e)}


def knowledge_agent_node(state, llm) -> dict:
    """
    Legacy knowledge agent node. Use company_qa_agent_node instead.

    Args:
        state: AgentState
        llm: Language model instance

    Returns:
        dict with final_answer and sources
    """
    logger.warning("knowledge_agent_node is deprecated. Use company_qa_agent_node instead.")
    return company_qa_agent_node(state, llm)


def general_agent_node(state, llm) -> dict:
    """
    Legacy general agent node. Use fallback_agent_node instead.

    Args:
        state: AgentState
        llm: Language model instance

    Returns:
        dict with final_answer
    """
    logger.warning("general_agent_node is deprecated. Use fallback_agent_node instead.")
    return fallback_agent_node(state)