import logging

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    intent_classifier_node,
    retrieve_node,
    greeting_agent_node,
    summarization_agent_node,
    company_qa_agent_node,
    fallback_agent_node,
    error_handler_node,
    # Legacy nodes for backward compatibility
    router_node,
    knowledge_agent_node,
    general_agent_node,
)
from graph.edges import route_by_intent, route_after_retrieval

from services.llm_service import get_llm
from core.model_loader import load_embedding_model
from core.weaviate_client import get_weaviate_client

logger = logging.getLogger(__name__)


def get_compiled_graph():
    """
    Build and compile LangGraph StateGraph with intent-based routing.

    Flow:
        START
          ↓
      intent_classifier
          ↓
          ├── GREETING → greeting_agent → END
          ├── SUMMARIZATION → summarization_agent → END
          ├── COMPANY_QA → retrieve → company_qa_agent → END
          ├── FALLBACK → fallback_agent → END
          └── ERROR → error_handler → END
    """

    logger.info("Building compiled graph with intent-based routing")

    llm = get_llm()
    embed_model = load_embedding_model()
    client = get_weaviate_client()

    workflow = StateGraph(AgentState)

    # ===========================================================================
    # NODES
    # ===========================================================================

    # Intent Classification
    workflow.add_node(
        "intent_classifier",
        lambda state: intent_classifier_node(state, llm),
    )

    # Retrieval (for COMPANY_QA intent)
    workflow.add_node(
        "retrieve_node",
        lambda state: retrieve_node(state, embed_model, client),
    )

    # Agent Nodes
    workflow.add_node(
        "greeting_agent",
        greeting_agent_node,
    )

    workflow.add_node(
        "summarization_agent",
        lambda state: summarization_agent_node(state, llm),
    )

    workflow.add_node(
        "company_qa_agent",
        lambda state: company_qa_agent_node(state, llm),
    )

    workflow.add_node(
        "fallback_agent",
        fallback_agent_node,
    )

    workflow.add_node(
        "error_handler",
        error_handler_node,
    )

    # ===========================================================================
    # EDGES
    # ===========================================================================

    # Entry point: classify intent
    workflow.set_entry_point("intent_classifier")

    # Intent classification routing
    workflow.add_conditional_edges(
        "intent_classifier",
        route_by_intent,
        {
            "greeting_agent": "greeting_agent",
            "summarization_agent": "summarization_agent",
            "company_qa_agent": "retrieve_node",  # COMPANY_QA requires retrieval first
            "fallback_agent": "fallback_agent",
            "error_handler": "error_handler",
        },
    )

    # Retrieval → Company QA Agent
    workflow.add_edge("retrieve_node", "company_qa_agent")

    # End states
    workflow.add_edge("greeting_agent", END)
    workflow.add_edge("summarization_agent", END)
    workflow.add_edge("company_qa_agent", END)
    workflow.add_edge("fallback_agent", END)
    workflow.add_edge("error_handler", END)

    logger.info("Graph compiled successfully with intent-based routing")

    return workflow.compile()


def get_legacy_compiled_graph():
    """
    Build and compile legacy LangGraph StateGraph for backward compatibility.

    This maintains the old retrieve → router → knowledge_agent/general_agent flow.
    """

    logger.warning("Using legacy graph. Consider migrating to intent-based routing.")

    llm = get_llm()
    embed_model = load_embedding_model()
    client = get_weaviate_client()

    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node(
        "retrieve_node",
        lambda state: retrieve_node(state, embed_model, client),
    )

    workflow.add_node("router_node", router_node)

    workflow.add_node(
        "knowledge_agent",
        lambda state: knowledge_agent_node(state, llm),
    )

    workflow.add_node(
        "general_agent",
        lambda state: general_agent_node(state, llm),
    )

    workflow.add_node("error_handler", error_handler_node)

    # Flow
    workflow.set_entry_point("retrieve_node")

    workflow.add_edge("retrieve_node", "router_node")

    workflow.add_conditional_edges(
        "router_node",
        route_after_retrieval,
        {
            "knowledge_agent": "knowledge_agent",
            "general_agent": "general_agent",
            "error_handler": "error_handler",
        },
    )

    workflow.add_edge("knowledge_agent", END)
    workflow.add_edge("general_agent", END)
    workflow.add_edge("error_handler", END)

    return workflow.compile()