"""Agent plugin base class for LLM-powered capabilities.

This module defines the AgentPlugin ABC, which provides a unified interface
for LLM agent implementations in the Semantik plugin system.

Agents support various use cases including:
- Conversational assistants
- Tool-using agents
- Query enhancement (HyDE, expansion)
- Result synthesis
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from shared.plugins.base import SemanticPlugin

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shared.agents.types import (
        AgentCapabilities,
        AgentContext,
        AgentMessage,
        AgentUseCase,
    )
    from shared.plugins.manifest import PluginManifest


class AgentPlugin(SemanticPlugin, ABC):
    """Base class for LLM agent plugins.

    Agents provide LLM-powered capabilities including:
    - Conversational assistants
    - Tool-using agents
    - Query enhancement (HyDE, expansion)
    - Result synthesis

    Class Variables:
        PLUGIN_TYPE: Always "agent" for agent plugins.
        PLUGIN_ID: Unique identifier for the plugin (must be set by subclass).
        PLUGIN_VERSION: Semantic version of the plugin (must be set by subclass).

    Example implementation:

        class ClaudeAgentPlugin(AgentPlugin):
            PLUGIN_ID = "claude-agent"
            PLUGIN_VERSION = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Claude Agent",
                    description="LLM agent powered by Claude",
                )

            @classmethod
            def get_capabilities(cls) -> AgentCapabilities:
                return AgentCapabilities(
                    supports_streaming=True,
                    supports_tools=True,
                    supported_models=("claude-sonnet-4-20250514",),
                )

            @classmethod
            def supported_use_cases(cls) -> list[AgentUseCase]:
                return [AgentUseCase.ASSISTANT, AgentUseCase.TOOL_USE]

            async def execute(
                self,
                prompt: str,
                *,
                context: AgentContext | None = None,
                **kwargs,
            ) -> AsyncIterator[AgentMessage]:
                async for message in self._adapter.execute(prompt, context=context, **kwargs):
                    yield message
    """

    PLUGIN_TYPE: ClassVar[str] = "agent"

    # Subclasses must define these
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the agent plugin.

        Args:
            config: Plugin configuration dictionary. May include:
                - model: Model to use for inference
                - system_prompt: Base system prompt
                - temperature: Sampling temperature (0.0-2.0)
                - max_tokens: Maximum output tokens
                - allowed_tools: Whitelisted tool names
                - timeout_seconds: Execution timeout
        """
        super().__init__(config)
        # Subclasses may set self._adapter to an AgentAdapter instance
        self._adapter: Any = None

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest with metadata.

        The manifest provides discovery information for the plugin system
        and UI rendering.

        Returns:
            PluginManifest with id, type, version, display_name, description,
            and capabilities dictionary.
        """

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> AgentCapabilities:
        """Return capabilities of this agent implementation.

        Capabilities enable:
        - Feature detection at runtime
        - Use case matching for agent selection
        - UI adaptation based on supported features

        Returns:
            AgentCapabilities declaring what this agent supports.
        """

    @classmethod
    @abstractmethod
    def supported_use_cases(cls) -> list[AgentUseCase]:
        """Return use cases this agent is suitable for.

        Enables intelligent agent selection based on task requirements.
        The system can automatically select the best agent for a given
        use case (e.g., HyDE, query expansion, summarization).

        Returns:
            List of AgentUseCase values this agent can handle.
        """

    @abstractmethod
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

        This is the core execution method that all agent plugins must implement.
        The method yields AgentMessage objects representing the response stream.

        Args:
            prompt: User message or task description.
            context: Runtime context including collection, search results, etc.
            system_prompt: Override the default system prompt.
            tools: Tool names to enable (from central registry).
            model: Model override (must be in supported_models if specified).
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            session_id: Resume an existing session by ID.
            stream: Whether to stream partial responses.

        Yields:
            AgentMessage objects. Messages may include:
            - Partial text (is_partial=True)
            - Tool calls (type=MessageType.TOOL_USE)
            - Final responses (type=MessageType.FINAL)
            - Errors (type=MessageType.ERROR)

        Raises:
            AgentExecutionError: If execution fails.
            AgentTimeoutError: If execution times out.
            ToolExecutionError: If tool execution fails.
        """

    async def execute_batch(
        self,
        prompts: list[str],
        *,
        context: AgentContext | None = None,
        **kwargs: Any,
    ) -> list[list[AgentMessage]]:
        """Execute multiple prompts.

        Default implementation processes prompts sequentially.
        Override for optimized batch or parallel processing.

        Args:
            prompts: List of prompts to execute.
            context: Shared runtime context for all executions.
            **kwargs: Additional arguments passed to execute().

        Returns:
            List of message lists, one per prompt.
        """
        results: list[list[AgentMessage]] = []
        for prompt in prompts:
            # Type ignore: mypy can't infer that abstract async generator is iterable
            messages = [msg async for msg in self.execute(prompt, context=context, **kwargs)]  # type: ignore[attr-defined]
            results.append(messages)
        return results

    async def interrupt(self) -> None:
        """Interrupt the current execution.

        Safe to call when:
        - Execution is in progress (will stop it)
        - No execution is running (no-op)
        - Called multiple times (idempotent)

        This method should return quickly without blocking.
        """
        if self._adapter is not None and hasattr(self._adapter, "interrupt"):
            await self._adapter.interrupt()

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        """Return JSON Schema for plugin configuration.

        Provides a standard schema for agent-specific configuration fields.
        Subclasses may override to extend with additional fields.

        Returns:
            JSON Schema dict or None if no configuration is needed.
        """
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model to use for inference",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Base system prompt for the agent",
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 2.0,
                    "default": 0.7,
                    "description": "Sampling temperature",
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 32000,
                    "default": 4096,
                    "description": "Maximum output tokens",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tool names this agent can use",
                },
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 600,
                    "default": 120,
                    "description": "Execution timeout in seconds",
                },
            },
            "additionalProperties": True,
        }

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        """Check if the agent is operational.

        Override to verify:
        - SDK availability
        - API connectivity
        - Model access
        - Required dependencies

        Args:
            config: Optional configuration for config-dependent checks.

        Returns:
            True if agent is healthy and ready for use.
        """
        return True

    def validate_model(self, model: str) -> bool:
        """Check if a model is supported by this agent.

        Args:
            model: Model identifier to validate.

        Returns:
            True if the model is supported or if no restrictions exist.
        """
        capabilities = self.get_capabilities()
        if not capabilities.supported_models:
            return True  # No restrictions
        return model in capabilities.supported_models

    def validate_tools(self, tools: list[str]) -> tuple[bool, list[str]]:
        """Validate that requested tools exist in the registry.

        Args:
            tools: List of tool names to validate.

        Returns:
            Tuple of (is_valid, list of invalid tool names).
            is_valid is True if all tools exist.
        """
        # Import here to avoid circular imports
        try:
            from shared.agents.tools.registry import get_tool_registry

            registry = get_tool_registry()
            invalid = [t for t in tools if not registry.has_tool(t)]
            return len(invalid) == 0, invalid
        except ImportError:
            # Registry not yet implemented (Phase 4)
            return True, []
