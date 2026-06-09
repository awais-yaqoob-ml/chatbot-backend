"""Unit tests for graph edges and routing."""

import pytest

from graph.edges import route_by_intent
from graph.nodes import INTENT_GREETING, INTENT_SUMMARIZATION, INTENT_COMPANY_QA, INTENT_FALLBACK


class TestRouteByIntent:
    """Test suite for intent-based routing."""

    def test_route_by_intent_greeting(self):
        """Test routing to greeting agent."""
        state = {"intent": INTENT_GREETING}

        result = route_by_intent(state)

        assert result == "greeting_agent"

    def test_route_by_intent_summarization(self):
        """Test routing to summarization agent."""
        state = {"intent": INTENT_SUMMARIZATION}

        result = route_by_intent(state)

        assert result == "summarization_agent"

    def test_route_by_intent_company_qa(self):
        """Test routing to company QA agent."""
        state = {"intent": INTENT_COMPANY_QA}

        result = route_by_intent(state)

        assert result == "company_qa_agent"

    def test_route_by_intent_fallback(self):
        """Test routing to fallback agent."""
        state = {"intent": INTENT_FALLBACK}

        result = route_by_intent(state)

        assert result == "fallback_agent"

    def test_route_by_intent_error_handling(self):
        """Test that errors route to error handler."""
        state = {"intent": INTENT_GREETING, "error": "Something went wrong"}

        result = route_by_intent(state)

        assert result == "error_handler"

    def test_route_by_intent_missing_intent_defaults_to_fallback(self):
        """Test that missing intent defaults to fallback routing."""
        state = {}

        result = route_by_intent(state)

        assert result == "fallback_agent"

    def test_route_by_intent_none_intent_defaults_to_fallback(self):
        """Test that None intent defaults to fallback routing."""
        state = {"intent": None}

        result = route_by_intent(state)

        assert result == "fallback_agent"

    def test_route_by_intent_invalid_intent_defaults_to_fallback(self):
        """Test that invalid intent defaults to fallback routing."""
        state = {"intent": "INVALID_INTENT"}

        result = route_by_intent(state)

        assert result == "fallback_agent"

    def test_route_by_intent_company_qa_routes_to_retrieval_first(self):
        """Test that COMPANY_QA intent routes to retrieval node first (via graph builder, not this function)."""
        # This is implicitly tested in graph builder, but we validate the routing decision
        state = {"intent": INTENT_COMPANY_QA}

        result = route_by_intent(state)

        # The route_by_intent function returns the agent node
        # The graph builder handles routing to retrieve_node before company_qa_agent
        assert result == "company_qa_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
