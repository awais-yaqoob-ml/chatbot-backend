"""Unit tests for agent nodes."""

import pytest
from unittest.mock import Mock

from graph.nodes import (
    greeting_agent_node,
    fallback_agent_node,
    summarization_agent_node,
    company_qa_agent_node,
)


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str):
        self.response = response

    def invoke(self, messages):
        return Mock(content=self.response)


class TestGreetingAgent:
    """Test suite for greeting agent node."""

    def test_greeting_agent_returns_valid_response(self):
        """Test that greeting agent returns a valid response."""
        state = {"user_message": "Hi"}

        result = greeting_agent_node(state)

        assert "final_answer" in result
        assert result["agent_used"] == "GreetingAgent"
        assert result["sources"] == []
        assert len(result["final_answer"]) > 0

    def test_greeting_agent_response_contains_company_focus(self):
        """Test that greeting response maintains company focus."""
        state = {"user_message": "Hello"}

        result = greeting_agent_node(state)

        response = result["final_answer"].lower()
        assert "company" in response or "help" in response

    def test_greeting_agent_exception_handling(self):
        """Test exception handling in greeting agent."""
        state = {}

        result = greeting_agent_node(state)

        assert "final_answer" in result
        assert result["agent_used"] == "GreetingAgent"


class TestFallbackAgent:
    """Test suite for fallback agent node."""

    def test_fallback_agent_returns_valid_response(self):
        """Test that fallback agent returns a valid response."""
        state = {"user_message": "Tell me a joke"}

        result = fallback_agent_node(state)

        assert "final_answer" in result
        assert result["agent_used"] == "FallbackAgent"
        assert result["sources"] == []

    def test_fallback_agent_response_refers_to_company_domain(self):
        """Test that fallback response indicates company focus."""
        state = {"user_message": "Random question"}

        result = fallback_agent_node(state)

        response = result["final_answer"].lower()
        assert "company" in response or "domain" in response or "knowledge" in response

    def test_fallback_agent_response_does_not_hallucinate(self):
        """Test that fallback agent doesn't provide unrelated answers."""
        state = {"user_message": "What is the capital of France?"}

        result = fallback_agent_node(state)

        response = result["final_answer"].lower()
        # Should not answer the question directly
        assert "paris" not in response

    def test_fallback_agent_exception_handling(self):
        """Test exception handling in fallback agent."""
        state = {}

        result = fallback_agent_node(state)

        assert "final_answer" in result
        assert result["agent_used"] == "FallbackAgent"


class TestSummarizationAgent:
    """Test suite for summarization agent node."""

    def test_summarization_agent_with_empty_history(self):
        """Test summarization agent with empty chat history."""
        llm = MockLLM("No conversation to summarize.")
        state = {"chat_history": []}

        result = summarization_agent_node(state, llm)

        assert "final_answer" in result
        assert result["agent_used"] == "SummarizationAgent"
        assert result["sources"] == []

    def test_summarization_agent_with_conversation_history(self):
        """Test summarization agent with actual conversation history."""
        llm = MockLLM(
            "Summary: Discussed company policies.\n\n"
            "Key Topics:\n- PTO Policy\n- Work Hours\n\n"
            "Open Questions:\n- Flexible work hours"
        )
        state = {
            "chat_history": [
                {"role": "user", "content": "What is the PTO policy?"},
                {"role": "assistant", "content": "Our company offers 20 days of PTO per year."},
                {"role": "user", "content": "What about work hours?"},
                {"role": "assistant", "content": "Standard hours are 9-5."},
            ]
        }

        result = summarization_agent_node(state, llm)

        assert "final_answer" in result
        assert "summary" in result
        assert result["agent_used"] == "SummarizationAgent"
        assert len(result["final_answer"]) > 0

    def test_summarization_agent_sets_summary_field(self):
        """Test that summarization agent sets both final_answer and summary fields."""
        llm = MockLLM("Test summary")
        state = {
            "chat_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        result = summarization_agent_node(state, llm)

        assert result["summary"] is not None
        assert result["final_answer"] == result["summary"]

    def test_summarization_agent_exception_handling(self):
        """Test exception handling in summarization agent."""
        llm = Mock()
        llm.invoke.side_effect = Exception("LLM error")
        state = {"chat_history": [{"role": "user", "content": "Test"}]}

        result = summarization_agent_node(state, llm)

        assert result["error"] is not None
        assert "final_answer" in result


class TestCompanyQAAgent:
    """Test suite for company QA agent node."""

    def test_company_qa_agent_with_retrieved_chunks(self):
        """Test company QA agent with retrieved chunks."""
        llm = MockLLM("The company offers 20 days of PTO annually.")
        state = {
            "user_message": "How much PTO does the company offer?",
            "chat_history": [],
            "retrieved_chunks": [
                {
                    "filename": "pto_policy.pdf",
                    "page_number": 1,
                    "text": "Our company provides 20 days of PTO per year.",
                    "chunk_index": 0,
                    "score": 0.95,
                }
            ],
        }

        result = company_qa_agent_node(state, llm)

        assert "final_answer" in result
        assert result["agent_used"] == "CompanyQAAgent"
        assert len(result["sources"]) > 0
        assert result["sources"][0]["filename"] == "pto_policy.pdf"

    def test_company_qa_agent_with_no_chunks(self):
        """Test company QA agent with no retrieved chunks."""
        llm = MockLLM("I could not find this information in the company documents.")
        state = {
            "user_message": "Unknown question",
            "chat_history": [],
            "retrieved_chunks": [],
        }

        result = company_qa_agent_node(state, llm)

        assert "final_answer" in result
        assert result["sources"] == []

    def test_company_qa_agent_includes_sources(self):
        """Test that company QA agent properly formats sources."""
        llm = MockLLM("Answer based on the documents.")
        state = {
            "user_message": "Test question",
            "chat_history": [],
            "retrieved_chunks": [
                {
                    "filename": "doc1.pdf",
                    "page_number": 5,
                    "text": "Some content",
                    "chunk_index": 1,
                    "score": 0.85,
                },
                {
                    "filename": "doc2.pdf",
                    "page_number": 10,
                    "text": "More content",
                    "chunk_index": 2,
                    "score": 0.75,
                },
            ],
        }

        result = company_qa_agent_node(state, llm)

        assert len(result["sources"]) == 2
        assert result["sources"][0]["filename"] == "doc1.pdf"
        assert result["sources"][1]["filename"] == "doc2.pdf"
        assert result["sources"][0]["page_number"] == 5
        assert result["sources"][1]["page_number"] == 10

    def test_company_qa_agent_exception_handling(self):
        """Test exception handling in company QA agent."""
        llm = Mock()
        llm.invoke.side_effect = Exception("LLM error")
        state = {
            "user_message": "Test",
            "chat_history": [],
            "retrieved_chunks": [],
        }

        result = company_qa_agent_node(state, llm)

        assert result["error"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
