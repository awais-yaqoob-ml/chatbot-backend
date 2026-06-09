from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from core.config import settings


def get_llm():
    """
    Initialize Groq LLM.
    """
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.2,
    )


def build_messages(system_prompt: str, chat_history: list, user_message: str):
    """
    Convert history + input into LangChain message format.
    """
    messages = [SystemMessage(content=system_prompt)]

    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_message))
    return messages


def run_llm(llm, messages):
    """
    Invoke LLM safely.
    """
    response = llm.invoke(messages)
    return response.content