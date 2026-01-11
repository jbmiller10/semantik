"""Claude Agent Plugin.

This module provides the ClaudeAgentPlugin, a built-in agent plugin that wraps
the Claude Agent SDK for LLM-powered capabilities in Semantik.

The plugin supports:
- Conversational assistants
- Tool-using agents (agentic search)
- Query enhancement (HyDE, expansion)
- Result synthesis and summarization

Example usage:

    plugin = ClaudeAgentPlugin({"model": "claude-sonnet-4-20250514"})
    await plugin.initialize()

    async for message in plugin.execute("Summarize these documents"):
        print(message.content)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from shared.agents.adapters.claude import ClaudeAgentAdapter
from shared.agents.types import (
    AgentCapabilities,
    AgentMessage,
    AgentUseCase,
)
from shared.plugins.manifest import PluginManifest
from shared.plugins.types.agent import AgentPlugin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shared.agents.types import AgentContext

logger = logging.getLogger(__name__)


class ClaudeAgentPlugin(AgentPlugin):
    """Built-in Claude agent plugin.

    This plugin wraps the Claude Agent SDK and provides LLM-powered
    capabilities for Semantik, including:

    - Conversational assistant interface
    - Agentic search with multi-step retrieval
    - Tool use with semantic search integration
    - Query enhancement (HyDE, expansion)
    - Result summarization and synthesis

    Class Variables:
        PLUGIN_ID: Unique identifier "claude-agent".
        PLUGIN_VERSION: Semantic version of the plugin.
        METADATA: Display metadata for UI.

    Attributes:
        _adapter: The underlying ClaudeAgentAdapter instance.
    """

    PLUGIN_ID: ClassVar[str] = "claude-agent"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Claude Agent",
        "description": "LLM agent powered by Claude for conversational AI, "
        "agentic search, and intelligent document processing",
        "author": "Semantik",
        "license": "Apache-2.0",
        "homepage": "https://github.com/semantik/semantik",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the Claude agent plugin.

        Args:
            config: Plugin configuration. Supported fields:
                - model: Model to use (default: claude-sonnet-4-20250514)
                - system_prompt: Default system prompt
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum output tokens
                - max_turns: Maximum agentic turns
                - permission_mode: SDK permission mode
                - allowed_tools: Whitelisted tool names
                - timeout_seconds: Execution timeout
        """
        super().__init__(config)
        self._adapter = ClaudeAgentAdapter(config)

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest with metadata.

        Returns:
            PluginManifest for plugin discovery and UI rendering.
        """
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            author=cls.METADATA.get("author"),
            license=cls.METADATA.get("license"),
            homepage=cls.METADATA.get("homepage"),
            capabilities=cls.get_capabilities().to_dict(),
        )

    @classmethod
    def get_capabilities(cls) -> AgentCapabilities:
        """Return Claude-specific capabilities.

        Delegates to the adapter for capability declaration.

        Returns:
            AgentCapabilities with supported features.
        """
        return ClaudeAgentAdapter.get_capabilities()

    @classmethod
    def supported_use_cases(cls) -> list[AgentUseCase]:
        """Return use cases this agent is suitable for.

        Claude supports a wide range of use cases:
        - ASSISTANT: Conversational interface
        - AGENTIC_SEARCH: Multi-step retrieval with reasoning
        - TOOL_USE: General tool-using agent
        - REASONING: Chain-of-thought reasoning
        - HYDE: Hypothetical document generation
        - QUERY_EXPANSION: Generate alternative queries
        - SUMMARIZATION: Content summarization
        - ANSWER_SYNTHESIS: RAG answer generation

        Returns:
            List of supported AgentUseCase values.
        """
        return [
            AgentUseCase.ASSISTANT,
            AgentUseCase.AGENTIC_SEARCH,
            AgentUseCase.TOOL_USE,
            AgentUseCase.REASONING,
            AgentUseCase.HYDE,
            AgentUseCase.QUERY_EXPANSION,
            AgentUseCase.SUMMARIZATION,
            AgentUseCase.ANSWER_SYNTHESIS,
        ]

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the plugin and underlying adapter.

        Args:
            config: Optional configuration override.

        Raises:
            AgentInitializationError: If SDK is not available.
        """
        await super().initialize(config)

        # Update adapter config if new config provided
        if config:
            self._adapter = ClaudeAgentAdapter({**self._config, **config})

        await self._adapter.initialize()
        logger.info("ClaudeAgentPlugin initialized")

    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        await self._adapter.cleanup()
        logger.info("ClaudeAgentPlugin cleaned up")

    async def execute(
        self,
        prompt: str,
        *,
        context: AgentContext | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stream: bool = True,
    ) -> AsyncIterator[AgentMessage]:
        """Execute the agent with the given prompt.

        Delegates to the underlying ClaudeAgentAdapter.

        Args:
            prompt: User message or task description.
            context: Runtime context (collection, retrieved content, etc.).
            system_prompt: Override the default system prompt.
            tools: Tool names to enable (from central registry).
            model: Model override.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            session_id: Resume an existing session by ID.
            stream: Whether to stream partial responses.

        Yields:
            AgentMessage objects representing the response stream.

        Raises:
            AgentInitializationError: If initialization fails.
            AgentExecutionError: If execution fails.
            AgentInterruptedError: If execution is interrupted.
        """
        # Resolve tools from registry if names provided
        tool_instances = None
        if tools:
            tool_instances = self._resolve_tools(tools)

        async for message in self._adapter.execute(
            prompt,
            context=context,
            system_prompt=system_prompt,
            tools=tool_instances,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id,
            stream=stream,
        ):
            yield message

    async def interrupt(self) -> None:
        """Interrupt the current execution.

        Delegates to the underlying adapter.
        """
        await self._adapter.interrupt()

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        """Check if the Claude SDK is available.

        Args:
            config: Optional configuration for config-dependent checks.

        Returns:
            True if SDK is available and ready.
        """
        try:
            from claude_agent_sdk import query  # noqa: F401

            return True
        except ImportError:
            logger.warning("Claude Agent SDK not installed")
            return False

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for plugin configuration.

        Extends the base schema with Claude-specific fields.

        Returns:
            JSON Schema dict.
        """
        base_schema = super().get_config_schema() or {}

        # Extend with Claude-specific fields
        properties = base_schema.get("properties", {})
        properties.update(
            {
                "model": {
                    "type": "string",
                    "enum": list(ClaudeAgentAdapter.SUPPORTED_MODELS.keys()),
                    "default": ClaudeAgentAdapter.DEFAULT_MODEL,
                    "description": "Claude model to use for inference",
                },
                "max_turns": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum number of agentic turns",
                },
                "permission_mode": {
                    "type": "string",
                    "enum": ["default", "bypassPermissions"],
                    "default": "default",
                    "description": "SDK permission mode for tool use",
                },
            }
        )

        return {
            **base_schema,
            "properties": properties,
        }

    def _resolve_tools(self, tool_names: list[str]) -> list[Any] | None:
        """Resolve tool names to tool instances from the registry.

        Args:
            tool_names: List of tool names to resolve.

        Returns:
            List of tool instances, or None if registry not available.
        """
        try:
            from shared.agents.tools.registry import get_tool_registry

            registry = get_tool_registry()
            return registry.get_by_names(tool_names)  # type: ignore[no-any-return]
        except ImportError:
            # Tool registry not yet implemented (Phase 4)
            logger.warning("Tool registry not available, skipping tool resolution")
            return None
