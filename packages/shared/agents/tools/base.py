"""Agent tool abstract base class.

This module defines the AgentTool ABC and supporting dataclasses
for the tool registry system.

Tool Hierarchy:
    ToolParameter: Single parameter definition
    ToolDefinition: Complete tool specification
    AgentTool: Abstract base for tool implementations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.agents.types import AgentContext


@dataclass(frozen=True)
class ToolParameter:
    """Definition of a tool parameter.

    Maps to JSON Schema properties for SDK consumption.

    Attributes:
        name: Parameter name (must be valid identifier).
        type: JSON Schema type ("string", "integer", "number", "boolean", "array", "object").
        description: Human-readable description for the LLM.
        required: Whether the parameter is required.
        default: Default value if not provided.
        enum: Allowed values (for constrained types).
        items: Schema for array items (only for type="array").
    """

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    items: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """Complete tool definition.

    Contains all metadata needed to register and execute a tool.

    Attributes:
        name: Unique tool identifier (lowercase, underscores allowed).
        description: Human-readable description for the LLM.
        parameters: List of parameter definitions.
        category: Category for grouping (e.g., "search", "utility").
        requires_context: Whether execute() needs AgentContext.
        is_destructive: Whether the tool modifies external state.
        timeout_seconds: Maximum execution time.
    """

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    category: str = "general"
    requires_context: bool = False
    is_destructive: bool = False
    timeout_seconds: float = 30.0

    def to_json_schema(self) -> dict[str, Any]:
        """Convert parameters to JSON Schema for SDK consumption.

        Returns:
            JSON Schema object with properties and required arrays.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum is not None:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            if param.items is not None:
                prop["items"] = param.items

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


class AgentTool(ABC):
    """Abstract base class for agent-accessible tools.

    Tools are stateless functions that agents can invoke to interact
    with external systems or perform computations.

    Example:
        class SemanticSearchTool(AgentTool):
            @property
            def definition(self) -> ToolDefinition:
                return ToolDefinition(
                    name="semantic_search",
                    description="Search documents by semantic similarity",
                    parameters=[
                        ToolParameter("query", "string", "Search query"),
                        ToolParameter(
                            "top_k", "integer", "Number of results",
                            required=False, default=10
                        ),
                    ],
                    category="search",
                    requires_context=True,
                )

            async def execute(
                self,
                args: dict[str, Any],
                context: AgentContext | None = None,
            ) -> Any:
                query = args["query"]
                top_k = args.get("top_k", 10)
                return await self._search_service.search(query, top_k)
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool definition.

        Returns:
            ToolDefinition with name, description, and parameters.
        """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return self.definition.name

    @property
    def description(self) -> str:
        """Return the tool description."""
        return self.definition.description

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return JSON Schema for tool input.

        Used by SDKs to validate and describe tool parameters.
        """
        return self.definition.to_json_schema()

    @abstractmethod
    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> Any:
        """Execute the tool with given arguments.

        Args:
            args: Tool arguments matching input schema.
            context: Optional runtime context (collection, user, etc.).

        Returns:
            Tool result (will be serialized to string for LLM).

        Raises:
            ToolExecutionError: If execution fails.
        """

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate arguments against the tool schema.

        Checks for required parameters and enum constraints.

        Args:
            args: Arguments to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: list[str] = []
        definition = self.definition

        # Check required parameters
        for param in definition.parameters:
            if param.required and param.name not in args:
                errors.append(f"Missing required parameter: {param.name}")

        # Check enum constraints
        for name, value in args.items():
            matched_param: ToolParameter | None = next(
                (p for p in definition.parameters if p.name == name),
                None,
            )
            if matched_param is not None and matched_param.enum is not None and value not in matched_param.enum:
                errors.append(f"Invalid value for {name}: must be one of {matched_param.enum}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize tool for registry storage and API responses.

        Returns:
            Dictionary with tool metadata.
        """
        definition = self.definition
        return {
            "name": definition.name,
            "description": definition.description,
            "input_schema": self.input_schema,
            "category": definition.category,
            "requires_context": definition.requires_context,
            "is_destructive": definition.is_destructive,
            "timeout_seconds": definition.timeout_seconds,
        }
