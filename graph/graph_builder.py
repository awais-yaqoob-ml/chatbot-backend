import logging

from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    intent_classifier_node,
    rewrite_query_node,
    retrieve_node,
    greeting_agent_node,
    summarization_agent_node,
    company_qa_agent_node,
    company_info_agent_node,
    fallback_agent_node,
    error_handler_node,
)
from graph.edges import route_by_intent

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
        lambda state: intent_classifier_node(state, llm, client),
    )

    # Query Rewriting (for COMPANY_QA intent)
    workflow.add_node(
        "rewrite_query",
        lambda state: rewrite_query_node(state, llm),
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
        "company_info_agent",
        lambda state: company_info_agent_node(state, client, llm),
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
            "company_info_agent": "company_info_agent",
            "company_qa_agent": "rewrite_query",  # COMPANY_QA: rewrite → retrieve → answer
            "fallback_agent": "fallback_agent",
            "error_handler": "error_handler",
        },
    )

    # Rewrite → Retrieval → Company QA Agent
    workflow.add_edge("rewrite_query", "retrieve_node")
    workflow.add_edge("retrieve_node", "company_qa_agent")

    # End states
    workflow.add_edge("greeting_agent", END)
    workflow.add_edge("summarization_agent", END)
    workflow.add_edge("company_qa_agent", END)
    workflow.add_edge("company_info_agent", END)
    workflow.add_edge("fallback_agent", END)
    workflow.add_edge("error_handler", END)

    logger.info("Graph compiled successfully with intent-based routing")

    return workflow.compile()