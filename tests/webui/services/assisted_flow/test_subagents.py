"""Tests for assisted flow subagent definitions."""

import pytest


class TestExplorerSubagent:
    """Test Explorer subagent definition."""

    def test_explorer_definition_exists(self) -> None:
        """Explorer subagent is defined."""
        from webui.services.assisted_flow.subagents import EXPLORER_AGENT

        assert EXPLORER_AGENT is not None
        assert EXPLORER_AGENT.description
        assert EXPLORER_AGENT.prompt
        assert EXPLORER_AGENT.model == "haiku"

    def test_explorer_has_description(self) -> None:
        """Explorer has a meaningful description."""
        from webui.services.assisted_flow.subagents import EXPLORER_AGENT

        assert "analyze" in EXPLORER_AGENT.description.lower() or "source" in EXPLORER_AGENT.description.lower()


class TestValidatorSubagent:
    """Test Validator subagent definition."""

    def test_validator_definition_exists(self) -> None:
        """Validator subagent is defined."""
        from webui.services.assisted_flow.subagents import VALIDATOR_AGENT

        assert VALIDATOR_AGENT is not None
        assert VALIDATOR_AGENT.description
        assert VALIDATOR_AGENT.prompt
        assert VALIDATOR_AGENT.model == "haiku"

    def test_validator_has_description(self) -> None:
        """Validator has a meaningful description."""
        from webui.services.assisted_flow.subagents import VALIDATOR_AGENT

        assert "validate" in VALIDATOR_AGENT.description.lower() or "pipeline" in VALIDATOR_AGENT.description.lower()


class TestSubagentDictionary:
    """Test subagent dictionary for SDK options."""

    def test_get_subagents_returns_dict(self) -> None:
        """get_subagents returns a dictionary of agents."""
        from webui.services.assisted_flow.subagents import get_subagents

        agents = get_subagents()

        assert isinstance(agents, dict)
        assert "explorer" in agents
        assert "validator" in agents
