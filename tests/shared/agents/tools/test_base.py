"""Tests for agent tool base classes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from shared.agents.tools.base import (
    AgentTool,
    ToolDefinition,
    ToolParameter,
)
from shared.agents.types import AgentContext

# --- Fixtures ---


class MockTool(AgentTool):
    """Concrete AgentTool implementation for testing."""

    def __init__(self, definition: ToolDefinition | None = None) -> None:
        self._definition = definition or ToolDefinition(
            name="mock_tool",
            description="A mock tool for testing",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The query string",
                    required=True,
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results to return",
                    required=False,
                    default=10,
                ),
            ],
            category="test",
        )
        self.last_args: dict[str, Any] | None = None
        self.last_context: AgentContext | None = None

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> dict[str, Any]:
        self.last_args = args
        self.last_context = context
        return {"result": f"Executed with query: {args.get('query')}"}


@pytest.fixture()
def mock_tool() -> MockTool:
    """Create a mock tool instance."""
    return MockTool()


# --- ToolParameter Tests ---


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic parameter creation."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )
        assert param.name == "query"
        assert param.type == "string"
        assert param.description == "Search query"
        assert param.required is True
        assert param.default is None
        assert param.enum is None
        assert param.items is None

    def test_frozen(self) -> None:
        """Test that ToolParameter is immutable."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )
        with pytest.raises(FrozenInstanceError):
            param.name = "modified"  # type: ignore[misc]

    def test_with_defaults(self) -> None:
        """Test parameter with default value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )
        assert param.required is False
        assert param.default == 10

    def test_with_enum(self) -> None:
        """Test parameter with enum constraint."""
        param = ToolParameter(
            name="sort_order",
            type="string",
            description="Sort order",
            enum=["asc", "desc"],
        )
        assert param.enum == ["asc", "desc"]

    def test_with_items(self) -> None:
        """Test array parameter with items schema."""
        param = ToolParameter(
            name="tags",
            type="array",
            description="Filter tags",
            items={"type": "string"},
        )
        assert param.items == {"type": "string"}


# --- ToolDefinition Tests ---


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic definition creation."""
        definition = ToolDefinition(
            name="search",
            description="Search documents",
        )
        assert definition.name == "search"
        assert definition.description == "Search documents"
        assert definition.parameters == []
        assert definition.category == "general"
        assert definition.requires_context is False
        assert definition.is_destructive is False
        assert definition.timeout_seconds == 30.0

    def test_frozen(self) -> None:
        """Test that ToolDefinition is immutable."""
        definition = ToolDefinition(
            name="search",
            description="Search documents",
        )
        with pytest.raises(FrozenInstanceError):
            definition.name = "modified"  # type: ignore[misc]

    def test_with_parameters(self) -> None:
        """Test definition with parameters."""
        params = [
            ToolParameter("query", "string", "Search query"),
            ToolParameter("limit", "integer", "Max results", required=False, default=10),
        ]
        definition = ToolDefinition(
            name="search",
            description="Search documents",
            parameters=params,
        )
        assert len(definition.parameters) == 2
        assert definition.parameters[0].name == "query"

    def test_with_metadata(self) -> None:
        """Test definition with full metadata."""
        definition = ToolDefinition(
            name="delete_document",
            description="Delete a document",
            category="admin",
            requires_context=True,
            is_destructive=True,
            timeout_seconds=60.0,
        )
        assert definition.category == "admin"
        assert definition.requires_context is True
        assert definition.is_destructive is True
        assert definition.timeout_seconds == 60.0

    def test_to_json_schema_empty(self) -> None:
        """Test JSON Schema with no parameters."""
        definition = ToolDefinition(
            name="ping",
            description="Ping the server",
        )
        schema = definition.to_json_schema()
        assert schema == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_to_json_schema_basic(self) -> None:
        """Test JSON Schema with basic parameters."""
        definition = ToolDefinition(
            name="search",
            description="Search documents",
            parameters=[
                ToolParameter("query", "string", "Search query"),
            ],
        )
        schema = definition.to_json_schema()
        assert schema == {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        }

    def test_to_json_schema_with_required_and_optional(self) -> None:
        """Test JSON Schema with required and optional parameters."""
        definition = ToolDefinition(
            name="search",
            description="Search documents",
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
                ToolParameter("limit", "integer", "Max results", required=False, default=10),
            ],
        )
        schema = definition.to_json_schema()
        assert schema["required"] == ["query"]
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert schema["properties"]["limit"]["default"] == 10

    def test_to_json_schema_with_enum(self) -> None:
        """Test JSON Schema with enum constraint."""
        definition = ToolDefinition(
            name="sort",
            description="Sort results",
            parameters=[
                ToolParameter(
                    "order",
                    "string",
                    "Sort order",
                    enum=["asc", "desc"],
                ),
            ],
        )
        schema = definition.to_json_schema()
        assert schema["properties"]["order"]["enum"] == ["asc", "desc"]

    def test_to_json_schema_with_array_items(self) -> None:
        """Test JSON Schema with array items."""
        definition = ToolDefinition(
            name="filter",
            description="Filter by tags",
            parameters=[
                ToolParameter(
                    "tags",
                    "array",
                    "Filter tags",
                    items={"type": "string"},
                ),
            ],
        )
        schema = definition.to_json_schema()
        assert schema["properties"]["tags"]["items"] == {"type": "string"}


# --- AgentTool Tests ---


class TestAgentTool:
    """Tests for AgentTool ABC."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that AgentTool cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            AgentTool()  # type: ignore[abstract]

    def test_name_property(self, mock_tool: MockTool) -> None:
        """Test name property returns definition name."""
        assert mock_tool.name == "mock_tool"

    def test_description_property(self, mock_tool: MockTool) -> None:
        """Test description property returns definition description."""
        assert mock_tool.description == "A mock tool for testing"

    def test_input_schema_property(self, mock_tool: MockTool) -> None:
        """Test input_schema property returns JSON Schema."""
        schema = mock_tool.input_schema
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    @pytest.mark.asyncio()
    async def test_execute(self, mock_tool: MockTool) -> None:
        """Test execute method."""
        result = await mock_tool.execute({"query": "test query"})
        assert result == {"result": "Executed with query: test query"}
        assert mock_tool.last_args == {"query": "test query"}

    @pytest.mark.asyncio()
    async def test_execute_with_context(self, mock_tool: MockTool) -> None:
        """Test execute method with context."""
        context = AgentContext(
            request_id="test-123",
            collection_id="col-456",
        )
        await mock_tool.execute({"query": "test"}, context=context)
        assert mock_tool.last_context is context
        assert mock_tool.last_context.request_id == "test-123"

    def test_validate_args_valid(self, mock_tool: MockTool) -> None:
        """Test validate_args with valid arguments."""
        is_valid, errors = mock_tool.validate_args({"query": "test"})
        assert is_valid is True
        assert errors == []

    def test_validate_args_with_optional(self, mock_tool: MockTool) -> None:
        """Test validate_args with optional arguments."""
        is_valid, errors = mock_tool.validate_args({"query": "test", "limit": 20})
        assert is_valid is True
        assert errors == []

    def test_validate_args_missing_required(self, mock_tool: MockTool) -> None:
        """Test validate_args with missing required parameter."""
        is_valid, errors = mock_tool.validate_args({})
        assert is_valid is False
        assert len(errors) == 1
        assert "Missing required parameter: query" in errors

    def test_validate_args_invalid_enum(self) -> None:
        """Test validate_args with invalid enum value."""
        tool = MockTool(
            ToolDefinition(
                name="sort_tool",
                description="Sort results",
                parameters=[
                    ToolParameter(
                        name="order",
                        type="string",
                        description="Sort order",
                        enum=["asc", "desc"],
                    ),
                ],
            )
        )
        is_valid, errors = tool.validate_args({"order": "invalid"})
        assert is_valid is False
        assert len(errors) == 1
        assert "must be one of" in errors[0]

    def test_validate_args_enum_valid(self) -> None:
        """Test validate_args with valid enum value."""
        tool = MockTool(
            ToolDefinition(
                name="sort_tool",
                description="Sort results",
                parameters=[
                    ToolParameter(
                        name="order",
                        type="string",
                        description="Sort order",
                        enum=["asc", "desc"],
                    ),
                ],
            )
        )
        is_valid, errors = tool.validate_args({"order": "asc"})
        assert is_valid is True
        assert errors == []

    def test_to_dict(self, mock_tool: MockTool) -> None:
        """Test to_dict serialization."""
        data = mock_tool.to_dict()
        assert data["name"] == "mock_tool"
        assert data["description"] == "A mock tool for testing"
        assert data["category"] == "test"
        assert data["requires_context"] is False
        assert data["is_destructive"] is False
        assert data["timeout_seconds"] == 30.0
        assert "input_schema" in data
        assert data["input_schema"]["type"] == "object"

    def test_to_dict_with_metadata(self) -> None:
        """Test to_dict with full metadata."""
        tool = MockTool(
            ToolDefinition(
                name="admin_tool",
                description="Admin operation",
                category="admin",
                requires_context=True,
                is_destructive=True,
                timeout_seconds=120.0,
            )
        )
        data = tool.to_dict()
        assert data["category"] == "admin"
        assert data["requires_context"] is True
        assert data["is_destructive"] is True
        assert data["timeout_seconds"] == 120.0
