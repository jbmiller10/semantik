"""Central tool registry for agent-accessible tools.

This module provides a thread-safe singleton registry for managing
agent-accessible tools.

Usage:
    registry = get_tool_registry()

    # Register a tool
    registry.register(SemanticSearchTool(), source="builtin")

    # Look up tools
    tool = registry.get("semantic_search")

    # Get tools by category
    search_tools = registry.get_by_category("search")

    # Execute a tool
    result = await registry.execute("semantic_search", {"query": "hello"})
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.agents.exceptions import (
    ToolDisabledError,
    ToolExecutionError,
    ToolNotFoundError,
)

if TYPE_CHECKING:
    from shared.agents.tools.base import AgentTool
    from shared.agents.types import AgentContext


@dataclass
class ToolRecord:
    """Registry entry for a tool.

    Attributes:
        tool: The tool instance.
        source: Where the tool came from ("builtin", "plugin", "custom").
        enabled: Whether the tool is currently available.
        metadata: Additional metadata about the tool.
    """

    tool: AgentTool
    source: str
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Central registry for agent-accessible tools.

    Thread-safe singleton that manages tool registration,
    lookup, and lifecycle.

    Features:
        - Thread-safe registration and lookup
        - Category-based indexing for efficient queries
        - Enable/disable without unregistering
        - Direct tool execution with timeout handling
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: dict[str, ToolRecord] = {}
        self._lock = threading.RLock()
        self._categories: dict[str, set[str]] = {}

    def register(
        self,
        tool: AgentTool,
        *,
        source: str = "custom",
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Register a tool.

        Args:
            tool: The tool to register.
            source: Where the tool came from ("builtin", "plugin", "custom").
            enabled: Whether the tool is initially enabled.
            metadata: Additional metadata about the tool.

        Returns:
            True if registered, False if tool with same name already exists.
        """
        with self._lock:
            if tool.name in self._tools:
                return False

            record = ToolRecord(
                tool=tool,
                source=source,
                enabled=enabled,
                metadata=metadata or {},
            )
            self._tools[tool.name] = record

            # Index by category
            category = tool.definition.category
            if category not in self._categories:
                self._categories[category] = set()
            self._categories[category].add(tool.name)

            return True

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry.

        Args:
            name: The name of the tool to remove.

        Returns:
            True if removed, False if tool was not found.
        """
        with self._lock:
            if name not in self._tools:
                return False

            tool = self._tools[name].tool
            del self._tools[name]

            # Remove from category index
            category = tool.definition.category
            if category in self._categories:
                self._categories[category].discard(name)
                if not self._categories[category]:
                    del self._categories[category]

            return True

    def get(self, name: str) -> AgentTool | None:
        """Look up a tool by name.

        Only returns tools that are enabled.

        Args:
            name: The name of the tool to look up.

        Returns:
            The tool if found and enabled, None otherwise.
        """
        with self._lock:
            record = self._tools.get(name)
            if record and record.enabled:
                return record.tool
            return None

    def get_record(self, name: str) -> ToolRecord | None:
        """Get the full tool record.

        Returns the record even if the tool is disabled,
        useful for admin operations.

        Args:
            name: The name of the tool.

        Returns:
            The tool record if found, None otherwise.
        """
        with self._lock:
            return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists and is enabled.

        Args:
            name: The name of the tool.

        Returns:
            True if the tool exists and is enabled.
        """
        return self.get(name) is not None

    def get_by_category(self, category: str) -> list[AgentTool]:
        """Get all enabled tools in a category.

        Args:
            category: The category to filter by.

        Returns:
            List of enabled tools in the category.
        """
        with self._lock:
            names = self._categories.get(category, set())
            return [self._tools[name].tool for name in names if name in self._tools and self._tools[name].enabled]

    def get_by_names(self, names: list[str]) -> list[AgentTool]:
        """Get multiple tools by name.

        Filters out tools that don't exist or are disabled.

        Args:
            names: List of tool names.

        Returns:
            List of enabled tools found.
        """
        with self._lock:
            return [self._tools[name].tool for name in names if name in self._tools and self._tools[name].enabled]

    def list_all(
        self,
        *,
        source: str | None = None,
        category: str | None = None,
    ) -> list[ToolRecord]:
        """List all tool records with optional filtering.

        Args:
            source: Filter by source ("builtin", "plugin", "custom").
            category: Filter by category.

        Returns:
            List of matching tool records (including disabled).
        """
        with self._lock:
            records = list(self._tools.values())

            if source is not None:
                records = [r for r in records if r.source == source]

            if category is not None:
                records = [r for r in records if r.tool.definition.category == category]

            return records

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a tool.

        Args:
            name: The name of the tool.
            enabled: Whether to enable or disable.

        Returns:
            True if the tool was found, False otherwise.
        """
        with self._lock:
            record = self._tools.get(name)
            if record is None:
                return False
            record.enabled = enabled
            return True

    def clear(self) -> None:
        """Clear all tools from the registry.

        Useful for testing.
        """
        with self._lock:
            self._tools.clear()
            self._categories.clear()

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: AgentContext | None = None,
    ) -> Any:
        """Execute a tool by name.

        Validates arguments, handles timeouts, and wraps errors.

        Args:
            name: The name of the tool to execute.
            args: Arguments to pass to the tool.
            context: Optional runtime context.

        Returns:
            The tool's result.

        Raises:
            ToolNotFoundError: If the tool doesn't exist.
            ToolDisabledError: If the tool is disabled.
            ToolExecutionError: If validation or execution fails.
        """
        # Get tool record (including disabled for better error messages)
        with self._lock:
            record = self._tools.get(name)

        if record is None:
            raise ToolNotFoundError(
                f"Tool not found: {name}",
                tool_name=name,
            )

        if not record.enabled:
            raise ToolDisabledError(
                f"Tool is disabled: {name}",
                tool_name=name,
            )

        tool = record.tool

        # Validate arguments
        is_valid, errors = tool.validate_args(args)
        if not is_valid:
            raise ToolExecutionError(
                f"Invalid arguments for {name}: {'; '.join(errors)}",
                tool_name=name,
                cause="validation_failed",
                validation_errors=errors,
            )

        # Execute with timeout
        timeout = tool.definition.timeout_seconds
        try:
            return await asyncio.wait_for(
                tool.execute(args, context),
                timeout=timeout,
            )
        except TimeoutError:
            raise ToolExecutionError(
                f"Tool execution timed out after {timeout}s: {name}",
                tool_name=name,
                cause="timeout",
                timeout_seconds=timeout,
            ) from None
        except ToolExecutionError:
            # Re-raise tool errors as-is
            raise
        except Exception as e:
            # Wrap other exceptions
            raise ToolExecutionError(
                f"Tool execution failed: {name}: {e!s}",
                tool_name=name,
                cause=str(e),
            ) from e


# Singleton instance
_tool_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry.

    Creates the registry on first access (lazy initialization).

    Returns:
        The global ToolRegistry instance.
    """
    global _tool_registry
    if _tool_registry is None:
        with _registry_lock:
            if _tool_registry is None:
                _tool_registry = ToolRegistry()
    return _tool_registry


def reset_tool_registry() -> None:
    """Reset the global tool registry.

    Creates a new empty registry. Useful for test isolation.
    """
    global _tool_registry
    with _registry_lock:
        _tool_registry = ToolRegistry()
