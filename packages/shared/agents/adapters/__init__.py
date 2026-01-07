"""SDK adapters for agent execution.

This package contains adapters that translate between Semantik's
unified interface and SDK-specific implementations.

Available Adapters:
    - ClaudeAgentAdapter: Adapter for Claude Agent SDK

Example:
    >>> from shared.agents.adapters import ClaudeAgentAdapter
    >>> adapter = ClaudeAgentAdapter({"model": "claude-sonnet-4-20250514"})
    >>> await adapter.initialize()
    >>> async for msg in adapter.execute("Hello!"):
    ...     print(msg.content)
"""

from shared.agents.adapters.claude import ClaudeAgentAdapter

__all__ = [
    "ClaudeAgentAdapter",
]
