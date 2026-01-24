"""Base class for agent tools.

Tools provide specific capabilities to agents. Each tool has:
- A name and description
- A JSON schema for parameters
- An execute method that performs the action
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all agent tools.

    Tools are the actions that agents can take. They provide a structured
    interface with parameter validation and consistent error handling.

    To create a tool, subclass this and implement:
    - NAME: Unique identifier for this tool
    - DESCRIPTION: Human-readable description of what the tool does
    - PARAMETERS: JSON schema for the tool's parameters
    - execute(): The actual implementation

    Class Attributes:
        NAME: Unique identifier for this tool
        DESCRIPTION: Human-readable description for the LLM
        PARAMETERS: JSON schema for parameters (OpenAI function calling format)

    Example:
        class GetWeatherTool(BaseTool):
            NAME = "get_weather"
            DESCRIPTION = "Get current weather for a location"
            PARAMETERS = {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    }
                },
                "required": ["location"]
            }

            async def execute(self, location: str) -> dict:
                return {"temperature": 72, "conditions": "sunny"}
    """

    NAME: ClassVar[str]
    DESCRIPTION: ClassVar[str]
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, context: dict[str, Any]):
        """Initialize the tool with context.

        Args:
            context: Shared context from the agent, may include:
                - session: Database session
                - user_id: Current user ID
                - conversation: Current conversation
                - orchestrator: Reference to orchestrator (for spawn tools)
        """
        self.context = context

    def get_schema(self) -> dict[str, Any]:
        """Get the JSON schema for this tool.

        Returns a schema in the format expected by LLM tool calling APIs.

        Returns:
            Tool schema dictionary
        """
        return {
            "type": "function",
            "function": {
                "name": self.NAME,
                "description": self.DESCRIPTION,
                "parameters": self.PARAMETERS,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Tool arguments as specified in PARAMETERS

        Returns:
            Tool result (will be serialized to JSON for the LLM)

        Raises:
            ToolExecutionError: If the tool fails
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.NAME}>"
