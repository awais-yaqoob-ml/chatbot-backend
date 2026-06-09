from graph.state import AgentState


def route_after_retrieval(state: AgentState) -> str:
    """
    Decide next node after retrieval + routing step.
    """
    if state.get("error"):
        return "error_handler"

    return state.get("agent_decision", "general_agent")