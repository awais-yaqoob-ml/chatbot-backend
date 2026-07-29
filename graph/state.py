from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict):
    # Session and input
    session_id: str
    user_message: str
    chat_history: List[Dict[str, str]]

    # Intent classification
    intent: Optional[str]  # GREETING | SUMMARIZATION | COMPANY_QA | FALLBACK

    # Query rewriting (for COMPANY_QA)
    rewritten_query: Optional[str]  # Search-optimized query after rewriting

    # Retrieval (for COMPANY_QA)
    retrieved_chunks: List[Dict]
    retrieval_score: float

    # Output
    final_answer: str
    sources: List[Dict]
    summary: Optional[str]  # For SUMMARIZATION intent

    # Company profile
    company_profile: Optional[str]

    # Metadata
    agent_used: str
    error: Optional[str]