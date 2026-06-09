"""Unit tests for intent classification."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from graph.nodes import intent_classifier_node, INTENT_GREETING, INTENT_SUMMARIZATION, INTENT_COMPANY_QA, INTENT_FALLBACK


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str):
        self.response = response

    def invoke(self, messages):
        return Mock(content=self.response)


class TestIntentClassifier:
    """Test suite for intent classifier node."""

    def test_intent_classifier_greeting(self):
        """Test classification of greeting intent."""
        llm = MockLLM("GREETING")
        state = {"user_message": "Hello there!"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_GREETING
        assert "error" not in result

    def test_intent_classifier_summarization(self):
        """Test classification of summarization intent."""
        llm = MockLLM("SUMMARIZATION")
        state = {"user_message": "Summarize our conversation so far"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_SUMMARIZATION

    def test_intent_classifier_company_qa(self):
        """Test classification of company QA intent."""
        llm = MockLLM("COMPANY_QA")
        state = {"user_message": "What is the company PTO policy?"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_COMPANY_QA

    def test_intent_classifier_fallback(self):
        """Test classification of fallback intent."""
        llm = MockLLM("FALLBACK")
        state = {"user_message": "Tell me a joke"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_FALLBACK

    def test_intent_classifier_invalid_response_defaults_to_fallback(self):
        """Test that invalid LLM responses default to FALLBACK."""
        llm = MockLLM("INVALID_INTENT")
        state = {"user_message": "Some message"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_FALLBACK

    def test_intent_classifier_whitespace_handling(self):
        """Test that whitespace in LLM response is handled correctly."""
        llm = MockLLM("  GREETING  \n")
        state = {"user_message": "Hi!"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_GREETING

    def test_intent_classifier_lowercase_handling(self):
        """Test that lowercase responses are converted to uppercase."""
        llm = MockLLM("greeting")
        state = {"user_message": "Hello!"}

        result = intent_classifier_node(state, llm)

        assert result["intent"] == INTENT_GREETING

    def test_intent_classifier_exception_handling(self):
        """Test exception handling in intent classifier."""
        llm = Mock()
        llm.invoke.side_effect = Exception("LLM error")
        state = {"user_message": "Test"}

        result = intent_classifier_node(state, llm)

        assert result["error"] is not None
        assert result["intent"] == INTENT_FALLBACK

    def test_intent_classifier_with_various_greeting_examples(self):
        """Test intent classifier with various greeting examples."""
        greeting_examples = [
            "Hi",
            "Hello",
            "Good morning",
            "How are you?",
            "Thanks",
            "Nice to meet you",
        ]

        for example in greeting_examples:
            llm = MockLLM(INTENT_GREETING)
            state = {"user_message": example}

            result = intent_classifier_node(state, llm)

            assert result["intent"] == INTENT_GREETING, f"Failed for: {example}"

    def test_intent_classifier_with_various_summarization_examples(self):
        """Test intent classifier with various summarization examples."""
        summary_examples = [
            "Summarize our conversation",
            "What have we discussed so far?",
            "Give me a summary",
            "Recap the chat",
        ]

        for example in summary_examples:
            llm = MockLLM(INTENT_SUMMARIZATION)
            state = {"user_message": example}

            result = intent_classifier_node(state, llm)

            assert result["intent"] == INTENT_SUMMARIZATION, f"Failed for: {example}"

    def test_intent_classifier_with_various_fallback_examples(self):
        """Test intent classifier with various fallback examples."""
        fallback_examples = [
            "Who won the World Cup?",
            "Explain quantum mechanics",
            "Write Python code",
            "Tell me a joke",
        ]

        for example in fallback_examples:
            llm = MockLLM(INTENT_FALLBACK)
            state = {"user_message": example}

            result = intent_classifier_node(state, llm)

            assert result["intent"] == INTENT_FALLBACK, f"Failed for: {example}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
