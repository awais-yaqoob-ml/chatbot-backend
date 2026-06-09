from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict):
    session_id: str
    user_message: str
    chat_history: List[Dict[str, str]]

    retrieved_chunks: List[Dict]
    retrieval_score: float

    agent_decision: str  # "knowledge_agent" | "general_agent"

    final_answer: str
    sources: List[Dict]

    agent_used: str

    error: Optional[str]