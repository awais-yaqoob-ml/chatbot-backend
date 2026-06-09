from langgraph.graph import StateGraph, END

from graph.state import AgentState
from graph.nodes import (
    retrieve_node,
    router_node,
    knowledge_agent_node,
    general_agent_node,
    error_handler_node,
)
from graph.edges import route_after_retrieval

from services.llm_service import get_llm
from core.model_loader import load_embedding_model
from core.weaviate_client import get_weaviate_client


def get_compiled_graph():
    """
    Build and compile LangGraph StateGraph.
    """

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