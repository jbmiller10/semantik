"""Tests for agent tool registry."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from shared.agents.exceptions import (
    ToolDisabledError,
    ToolExecutionError,
    ToolNotFoundError,
)
from shared.agents.tools.base import AgentTool, ToolDefinition, ToolParameter
from shared.agents.tools.registry import (
    ToolRecord,
    ToolRegistry,
    get_tool_registry,
    reset_tool_registry,
)
from shared.agents.types import AgentContext

# --- Fixtures ---


class MockTool(AgentTool):
    """Concrete AgentTool implementation for testing."""

    def __init__(
        self,
        name: str = "mock_tool",
        category: str = "test",
        timeout: float = 30.0,
        execute_result: Any = None,
        execute_error: Exception | None = None,
        execute_delay: float = 0.0,
    ) -> None:
        self._name = name
        self._category = category
        self._timeout = timeout
        self._execute_result = execute_result or {"status": "ok"}
        self._execute_error = execute_error
        self._execute_delay = execute_delay

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=f"Mock tool: {self._name}",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The query string",
                    required=True,
                ),
            ],
            category=self._category,
            timeout_seconds=self._timeout,
        )

    async def execute(
        self,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> Any:
        if self._execute_delay > 0:
            await asyncio.sleep(self._execute_delay)
        if self._execute_error:
            raise self._execute_error
        return self._execute_result


@pytest.fixture()
def registry() -> ToolRegistry:
    """Create a fresh registry for each test."""
    reset_tool_registry()
    return get_tool_registry()


@pytest.fixture()
def mock_tool() -> MockTool:
    """Create a mock tool."""
    return MockTool()


@pytest.fixture()
def search_tool() -> MockTool:
    """Create a search category tool."""
    return MockTool(name="search_tool", category="search")


@pytest.fixture()
def admin_tool() -> MockTool:
    """Create an admin category tool."""
    return MockTool(name="admin_tool", category="admin")


# --- Registration Tests ---


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test successful tool registration."""
        result = registry.register(mock_tool, source="test")
        assert result is True
        assert registry.has_tool("mock_tool")

    def test_register_duplicate_returns_false(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that duplicate registration returns False."""
        registry.register(mock_tool)
        result = registry.register(mock_tool)
        assert result is False

    def test_register_with_metadata(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test registration with metadata."""
        metadata = {"version": "1.0", "author": "test"}
        registry.register(mock_tool, metadata=metadata)

        record = registry.get_record("mock_tool")
        assert record is not None
        assert record.metadata == metadata

    def test_register_disabled(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test registration as disabled."""
        registry.register(mock_tool, enabled=False)

        assert registry.has_tool("mock_tool") is False
        record = registry.get_record("mock_tool")
        assert record is not None
        assert record.enabled is False

    def test_unregister_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test tool unregistration."""
        registry.register(mock_tool)
        result = registry.unregister("mock_tool")

        assert result is True
        assert registry.has_tool("mock_tool") is False

    def test_unregister_nonexistent_returns_false(self, registry: ToolRegistry) -> None:
        """Test unregistering nonexistent tool returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_unregister_removes_from_category_index(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that unregister removes tool from category index."""
        registry.register(mock_tool)
        registry.unregister("mock_tool")

        tools = registry.get_by_category("test")
        assert len(tools) == 0


# --- Lookup Tests ---


class TestToolLookup:
    """Tests for tool lookup."""

    def test_get_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test getting a tool by name."""
        registry.register(mock_tool)
        tool = registry.get("mock_tool")

        assert tool is not None
        assert tool.name == "mock_tool"

    def test_get_nonexistent_returns_none(self, registry: ToolRegistry) -> None:
        """Test that getting nonexistent tool returns None."""
        tool = registry.get("nonexistent")
        assert tool is None

    def test_get_disabled_tool_returns_none(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that getting disabled tool returns None."""
        registry.register(mock_tool, enabled=False)
        tool = registry.get("mock_tool")
        assert tool is None

    def test_get_record_returns_disabled(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that get_record returns disabled tools."""
        registry.register(mock_tool, enabled=False)
        record = registry.get_record("mock_tool")

        assert record is not None
        assert record.enabled is False

    def test_has_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test has_tool for existing tool."""
        registry.register(mock_tool)
        assert registry.has_tool("mock_tool") is True
        assert registry.has_tool("nonexistent") is False

    def test_get_by_category(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
        admin_tool: MockTool,
    ) -> None:
        """Test getting tools by category."""
        registry.register(mock_tool)
        registry.register(search_tool)
        registry.register(admin_tool)

        test_tools = registry.get_by_category("test")
        search_tools = registry.get_by_category("search")

        assert len(test_tools) == 1
        assert test_tools[0].name == "mock_tool"
        assert len(search_tools) == 1
        assert search_tools[0].name == "search_tool"

    def test_get_by_category_excludes_disabled(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that get_by_category excludes disabled tools."""
        registry.register(mock_tool, enabled=False)
        tools = registry.get_by_category("test")
        assert len(tools) == 0

    def test_get_by_names(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
    ) -> None:
        """Test getting multiple tools by name."""
        registry.register(mock_tool)
        registry.register(search_tool)

        tools = registry.get_by_names(["mock_tool", "search_tool"])
        assert len(tools) == 2

    def test_get_by_names_filters_missing(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that get_by_names filters out missing tools."""
        registry.register(mock_tool)

        tools = registry.get_by_names(["mock_tool", "nonexistent"])
        assert len(tools) == 1
        assert tools[0].name == "mock_tool"

    def test_get_by_names_filters_disabled(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
    ) -> None:
        """Test that get_by_names filters out disabled tools."""
        registry.register(mock_tool)
        registry.register(search_tool, enabled=False)

        tools = registry.get_by_names(["mock_tool", "search_tool"])
        assert len(tools) == 1


# --- Listing Tests ---


class TestToolListing:
    """Tests for tool listing."""

    def test_list_all(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
    ) -> None:
        """Test listing all tools."""
        registry.register(mock_tool, source="builtin")
        registry.register(search_tool, source="plugin")

        records = registry.list_all()
        assert len(records) == 2

    def test_list_all_includes_disabled(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that list_all includes disabled tools."""
        registry.register(mock_tool, enabled=False)
        records = registry.list_all()
        assert len(records) == 1
        assert records[0].enabled is False

    def test_list_all_filter_by_source(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
    ) -> None:
        """Test filtering by source."""
        registry.register(mock_tool, source="builtin")
        registry.register(search_tool, source="plugin")

        builtin = registry.list_all(source="builtin")
        plugin = registry.list_all(source="plugin")

        assert len(builtin) == 1
        assert builtin[0].source == "builtin"
        assert len(plugin) == 1
        assert plugin[0].source == "plugin"

    def test_list_all_filter_by_category(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
    ) -> None:
        """Test filtering by category."""
        registry.register(mock_tool)
        registry.register(search_tool)

        test_records = registry.list_all(category="test")
        search_records = registry.list_all(category="search")

        assert len(test_records) == 1
        assert len(search_records) == 1


# --- Management Tests ---


class TestToolManagement:
    """Tests for tool management."""

    def test_set_enabled(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test enabling and disabling tools."""
        registry.register(mock_tool)

        registry.set_enabled("mock_tool", False)
        assert registry.has_tool("mock_tool") is False

        registry.set_enabled("mock_tool", True)
        assert registry.has_tool("mock_tool") is True

    def test_set_enabled_nonexistent_returns_false(self, registry: ToolRegistry) -> None:
        """Test set_enabled on nonexistent tool returns False."""
        result = registry.set_enabled("nonexistent", True)
        assert result is False

    def test_clear(
        self,
        registry: ToolRegistry,
        mock_tool: MockTool,
        search_tool: MockTool,
    ) -> None:
        """Test clearing all tools."""
        registry.register(mock_tool)
        registry.register(search_tool)

        registry.clear()

        assert registry.has_tool("mock_tool") is False
        assert registry.has_tool("search_tool") is False
        assert len(registry.list_all()) == 0


# --- Execution Tests ---


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.mark.asyncio()
    async def test_execute_tool(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test executing a tool."""
        registry.register(mock_tool)
        result = await registry.execute("mock_tool", {"query": "test"})
        assert result == {"status": "ok"}

    @pytest.mark.asyncio()
    async def test_execute_with_context(self, registry: ToolRegistry) -> None:
        """Test executing a tool with context."""
        tool = MockTool()
        registry.register(tool)

        context = AgentContext(request_id="test-123")
        await registry.execute("mock_tool", {"query": "test"}, context=context)

    @pytest.mark.asyncio()
    async def test_execute_nonexistent_raises(self, registry: ToolRegistry) -> None:
        """Test that executing nonexistent tool raises ToolNotFoundError."""
        with pytest.raises(ToolNotFoundError) as exc_info:
            await registry.execute("nonexistent", {})

        assert exc_info.value.tool_name == "nonexistent"
        assert "Tool not found" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_execute_disabled_tool_raises(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that executing disabled tool raises ToolDisabledError."""
        registry.register(mock_tool, enabled=False)

        with pytest.raises(ToolDisabledError) as exc_info:
            await registry.execute("mock_tool", {"query": "test"})

        assert exc_info.value.tool_name == "mock_tool"

    @pytest.mark.asyncio()
    async def test_execute_invalid_args_raises(self, registry: ToolRegistry, mock_tool: MockTool) -> None:
        """Test that executing with invalid args raises ToolExecutionError."""
        registry.register(mock_tool)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute("mock_tool", {})  # Missing required 'query'

        assert exc_info.value.tool_name == "mock_tool"
        assert "validation_failed" in str(exc_info.value.details.get("cause", ""))

    @pytest.mark.asyncio()
    async def test_execute_timeout_raises(self, registry: ToolRegistry) -> None:
        """Test that execution timeout raises ToolExecutionError."""
        slow_tool = MockTool(
            name="slow_tool",
            timeout=0.1,  # 100ms timeout
            execute_delay=0.5,  # 500ms delay
        )
        registry.register(slow_tool)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute("slow_tool", {"query": "test"})

        assert exc_info.value.tool_name == "slow_tool"
        assert exc_info.value.details.get("cause") == "timeout"

    @pytest.mark.asyncio()
    async def test_execute_error_wrapped(self, registry: ToolRegistry) -> None:
        """Test that execution errors are wrapped in ToolExecutionError."""
        error_tool = MockTool(
            name="error_tool",
            execute_error=ValueError("Something went wrong"),
        )
        registry.register(error_tool)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute("error_tool", {"query": "test"})

        assert exc_info.value.tool_name == "error_tool"
        assert "Something went wrong" in str(exc_info.value.details.get("cause", ""))

    @pytest.mark.asyncio()
    async def test_execute_tool_execution_error_not_wrapped(self, registry: ToolRegistry) -> None:
        """Test that ToolExecutionError from tool is not double-wrapped."""
        original_error = ToolExecutionError(
            "Custom error",
            tool_name="error_tool",
        )
        error_tool = MockTool(
            name="error_tool",
            execute_error=original_error,
        )
        registry.register(error_tool)

        with pytest.raises(ToolExecutionError) as exc_info:
            await registry.execute("error_tool", {"query": "test"})

        assert exc_info.value is original_error


# --- Singleton Tests ---


class TestSingleton:
    """Tests for singleton behavior."""

    def test_get_tool_registry_returns_same_instance(self) -> None:
        """Test that get_tool_registry returns the same instance."""
        reset_tool_registry()
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()
        assert registry1 is registry2

    def test_reset_tool_registry_creates_new_instance(self) -> None:
        """Test that reset_tool_registry creates a new instance."""
        registry1 = get_tool_registry()
        reset_tool_registry()
        registry2 = get_tool_registry()
        assert registry1 is not registry2

    def test_reset_clears_tools(self) -> None:
        """Test that reset clears registered tools."""
        registry = get_tool_registry()
        registry.register(MockTool())
        assert registry.has_tool("mock_tool")

        reset_tool_registry()
        registry = get_tool_registry()
        assert registry.has_tool("mock_tool") is False


# --- Thread Safety Tests ---


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_registration(self, registry: ToolRegistry) -> None:
        """Test concurrent tool registration."""
        results: list[bool] = []
        errors: list[Exception] = []

        def register_tool(index: int) -> None:
            try:
                tool = MockTool(name=f"tool_{index}")
                result = registry.register(tool)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_tool, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)
        assert len(registry.list_all()) == 10

    def test_concurrent_lookup(self, registry: ToolRegistry) -> None:
        """Test concurrent tool lookup."""
        # Register some tools first
        for i in range(5):
            registry.register(MockTool(name=f"tool_{i}"))

        results: list[AgentTool | None] = []
        errors: list[Exception] = []

        def lookup_tool(name: str) -> None:
            try:
                for _ in range(100):
                    tool = registry.get(name)
                    if tool:
                        results.append(tool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=lookup_tool, args=(f"tool_{i % 5}",)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000  # 10 threads * 100 lookups each


# --- ToolRecord Tests ---


class TestToolRecord:
    """Tests for ToolRecord dataclass."""

    def test_creation(self, mock_tool: MockTool) -> None:
        """Test ToolRecord creation."""
        record = ToolRecord(
            tool=mock_tool,
            source="builtin",
            enabled=True,
            metadata={"version": "1.0"},
        )

        assert record.tool is mock_tool
        assert record.source == "builtin"
        assert record.enabled is True
        assert record.metadata == {"version": "1.0"}

    def test_defaults(self, mock_tool: MockTool) -> None:
        """Test ToolRecord default values."""
        record = ToolRecord(tool=mock_tool, source="custom")

        assert record.enabled is True
        assert record.metadata == {}

    def test_mutable(self, mock_tool: MockTool) -> None:
        """Test that ToolRecord is mutable (not frozen)."""
        record = ToolRecord(tool=mock_tool, source="custom")
        record.enabled = False
        assert record.enabled is False
