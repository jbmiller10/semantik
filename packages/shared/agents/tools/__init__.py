"""Agent tools package.

This package contains tool-related abstractions for the agent system.

The tool registry provides a central place to register and discover
tools that agents can use during execution.

Usage:
    from shared.agents.tools import (
        AgentTool,
        ToolDefinition,
        ToolParameter,
        get_tool_registry,
    )

    # Define a tool
    class MyTool(AgentTool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="my_tool",
                description="Does something useful",
                parameters=[
                    ToolParameter("input", "string", "The input"),
                ],
            )

        async def execute(self, args, context=None):
            return {"result": args["input"].upper()}

    # Register it
    registry = get_tool_registry()
    registry.register(MyTool(), source="custom")
"""

from shared.agents.tools.base import (
    AgentTool,
    ToolDefinition,
    ToolParameter,
)
from shared.agents.tools.registry import (
    ToolRecord,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
)

__all__ = [
    # Base classes
    "AgentTool",
    "ToolDefinition",
    "ToolParameter",
    # Registry
    "ToolRecord",
    "ToolRegistry",
    "get_tool_registry",
    "reset_tool_registry",
]
