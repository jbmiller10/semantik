"""Subagent definitions for assisted flow.

This module defines the Explorer and Validator subagents that can be
spawned by the main pipeline configuration assistant.
"""

from __future__ import annotations

from claude_agent_sdk import AgentDefinition

from webui.services.assisted_flow.prompts import (
    EXPLORER_SUBAGENT_PROMPT,
    VALIDATOR_SUBAGENT_PROMPT,
)

# Explorer subagent: Analyzes data sources for pipeline recommendations
EXPLORER_AGENT = AgentDefinition(
    description="Analyzes data sources to recommend optimal pipeline configurations",
    prompt=EXPLORER_SUBAGENT_PROMPT,
    model="haiku",
)

# Validator subagent: Validates pipeline configurations against actual data
VALIDATOR_AGENT = AgentDefinition(
    description="Validates pipeline configurations against sample data",
    prompt=VALIDATOR_SUBAGENT_PROMPT,
    model="haiku",
)


def get_subagents() -> dict[str, AgentDefinition]:
    """Get dictionary of subagent definitions for SDK options.

    Returns:
        Dictionary mapping agent names to AgentDefinition instances
    """
    return {
        "explorer": EXPLORER_AGENT,
        "validator": VALIDATOR_AGENT,
    }
