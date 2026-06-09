import logging

from graph.nodes import (
    INTENT_GREETING,
    INTENT_SUMMARIZATION,
    INTENT_COMPANY_QA,
    INTENT_FALLBACK,
)
from graph.state import AgentState

logger = logging.getLogger(__name__)


def route_by_intent(state: AgentState) -> str:
    """
    Route next node based on classified intent.

    Args:
        state: AgentState with intent field

    Returns:
        str: Next node name
    """
    if state.get("error"):
        logger.info("Error detected, routing to error_handler")
        return "error_handler"

    intent = state.get("intent", INTENT_FALLBACK)

    logger.info(f"Routing by intent: {intent}")

    intent_node_map = {
        INTENT_GREETING: "greeting_agent",
        INTENT_SUMMARIZATION: "summarization_agent",
        INTENT_COMPANY_QA: "company_qa_agent",
        INTENT_FALLBACK: "fallback_agent",
    }

    return intent_node_map.get(intent, "fallback_agent")


def route_after_retrieval(state: AgentState) -> str:
    """
    Legacy edge function for backward compatibility.
    Decide next node after retrieval + routing step.

    Args:
        state: AgentState

    Returns:
        str: Next node name
    """
    if state.get("error"):
        return "error_handler"

    return state.get("agent_decision", "general_agent")