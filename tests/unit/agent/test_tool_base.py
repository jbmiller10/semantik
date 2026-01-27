"""Unit tests for BaseTool class."""

from typing import Any, ClassVar

import pytest

from webui.services.agent.tools.base import BaseTool


class TestTool(BaseTool):
    """Concrete test tool implementation."""

    NAME: ClassVar[str] = "test_tool"
    DESCRIPTION: ClassVar[str] = "A test tool for unit testing"
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to echo",
            },
            "count": {
                "type": "integer",
                "description": "Number of times to repeat",
            },
        },
        "required": ["message"],
    }

    async def execute(self, message: str, count: int = 1) -> dict[str, Any]:
        """Echo the message count times."""
        return {"result": message * count}


class FailingTool(BaseTool):
    """Tool that always fails for testing error handling."""

    NAME: ClassVar[str] = "failing_tool"
    DESCRIPTION: ClassVar[str] = "A tool that always fails"

    async def execute(self, **kwargs: Any) -> Any:
        """Always raise an exception."""
        raise RuntimeError("Tool execution failed")


class MinimalTool(BaseTool):
    """Minimal tool with default PARAMETERS."""

    NAME: ClassVar[str] = "minimal_tool"
    DESCRIPTION: ClassVar[str] = "A minimal test tool"

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Return empty result."""
        return {}


class TestBaseToolSchema:
    """Tests for BaseTool.get_schema()."""

    def test_get_schema_returns_openai_format(self):
        """Test that get_schema returns OpenAI function calling format."""
        context = {"user_id": 1}
        tool = TestTool(context)
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool for unit testing"
        assert schema["function"]["parameters"] == TestTool.PARAMETERS

    def test_get_schema_with_custom_parameters(self):
        """Test schema with custom PARAMETERS."""
        context = {}
        tool = TestTool(context)
        schema = tool.get_schema()

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "message" in params["properties"]
        assert "count" in params["properties"]
        assert "message" in params["required"]

    def test_get_schema_with_default_parameters(self):
        """Test schema with default PARAMETERS."""
        context = {}
        tool = MinimalTool(context)
        schema = tool.get_schema()

        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert params["properties"] == {}
        assert params["required"] == []


class TestBaseToolContext:
    """Tests for context handling."""

    def test_context_stored_correctly(self):
        """Test that context is stored as instance attribute."""
        context = {
            "user_id": 42,
            "session": "mock_session",
            "conversation_id": "test-conv-id",
        }
        tool = TestTool(context)

        assert tool.context == context
        assert tool.context["user_id"] == 42

    def test_context_can_be_empty(self):
        """Test that empty context is valid."""
        tool = TestTool({})
        assert tool.context == {}


class TestBaseToolRepr:
    """Tests for BaseTool.__repr__()."""

    def test_repr_format(self):
        """Test __repr__ returns expected format."""
        tool = TestTool({})
        repr_str = repr(tool)

        assert "TestTool" in repr_str
        assert "name=test_tool" in repr_str

    def test_repr_with_different_tools(self):
        """Test __repr__ works for different tool types."""
        minimal = MinimalTool({})
        failing = FailingTool({})

        assert "MinimalTool" in repr(minimal)
        assert "minimal_tool" in repr(minimal)
        assert "FailingTool" in repr(failing)
        assert "failing_tool" in repr(failing)


class TestBaseToolExecute:
    """Tests for tool execution."""

    @pytest.mark.asyncio()
    async def test_execute_success(self):
        """Test successful tool execution."""
        tool = TestTool({"user_id": 1})
        result = await tool.execute(message="hello", count=3)

        assert result == {"result": "hellohellohello"}

    @pytest.mark.asyncio()
    async def test_execute_with_default_args(self):
        """Test execution with default arguments."""
        tool = TestTool({})
        result = await tool.execute(message="test")

        assert result == {"result": "test"}

    @pytest.mark.asyncio()
    async def test_execute_failure(self):
        """Test that tool execution errors propagate."""
        tool = FailingTool({})

        with pytest.raises(RuntimeError, match="Tool execution failed"):
            await tool.execute()

    @pytest.mark.asyncio()
    async def test_minimal_tool_execute(self):
        """Test minimal tool execution returns empty dict."""
        tool = MinimalTool({})
        result = await tool.execute()

        assert result == {}
