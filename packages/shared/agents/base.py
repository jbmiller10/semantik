"""Agent adapter abstract base class.

This module defines the AgentAdapter ABC, which provides an SDK-agnostic
interface for LLM agent execution. Concrete implementations wrap specific
SDKs (e.g., Claude Agent SDK, OpenAI Agents SDK) and translate between
SDK-specific formats and the unified AgentMessage type.

Example implementation:

    class MyAdapter(AgentAdapter):
        @classmethod
        def get_capabilities(cls) -> AgentCapabilities:
            return AgentCapabilities(supports_streaming=True)

        async def execute(self, prompt: str, **kwargs) -> AsyncIterator[AgentMessage]:
            async for response in my_sdk.query(prompt):
                yield self._translate(response)
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shared.agents.types import (
        AgentCapabilities,
        AgentContext,
        AgentMessage,
    )


class AgentAdapter(ABC):
    """SDK-agnostic interface for LLM agent execution.

    This abstract base class defines the contract that all SDK adapters
    must implement. It provides:

    - Lifecycle management (initialize, cleanup)
    - Core execution via async generator
    - Session management (interrupt, fork)
    - Capability declaration

    Subclasses should:
    1. Implement all abstract methods
    2. Handle SDK import lazily in initialize()
    3. Translate SDK messages to AgentMessage in execute()
    4. Support interruption via _interrupt_event

    Attributes:
        _config: Adapter configuration dictionary.
        _is_initialized: Whether initialize() has been called.
        _interrupt_event: Event to signal execution interruption.
        _current_session_id: Active SDK session ID (if any).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the adapter with configuration.

        Args:
            config: Adapter configuration. May include model selection,
                timeouts, API keys, etc. Specific fields depend on the
                SDK being wrapped.
        """
        self._config = config or {}
        self._is_initialized = False
        self._interrupt_event: asyncio.Event | None = None
        self._current_session_id: str | None = None

    @property
    def is_initialized(self) -> bool:
        """Whether the adapter has been initialized."""
        return self._is_initialized

    @property
    def current_session_id(self) -> str | None:
        """The current SDK session ID, if any."""
        return self._current_session_id

    async def initialize(self) -> None:
        """Initialize the adapter for use.

        This method should:
        - Verify SDK is available (lazy import)
        - Validate configuration
        - Create interrupt event
        - Set up any SDK-specific resources

        Safe to call multiple times (idempotent).

        Raises:
            AgentInitializationError: If initialization fails.
        """
        if self._is_initialized:
            return

        self._interrupt_event = asyncio.Event()
        self._is_initialized = True

    async def cleanup(self) -> None:
        """Clean up adapter resources.

        This method should:
        - Release any SDK resources
        - Clear session state
        - Reset initialization flag

        Safe to call multiple times (idempotent).
        """
        self._is_initialized = False
        self._interrupt_event = None
        self._current_session_id = None

    @classmethod
    @abstractmethod
    def get_capabilities(cls) -> AgentCapabilities:
        """Return capabilities of this adapter implementation.

        Capabilities inform the system what features the adapter supports,
        enabling graceful degradation and feature detection.

        Returns:
            AgentCapabilities declaring supported features.
        """

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        *,
        context: AgentContext | None = None,
        system_prompt: str | None = None,
        tools: list[Any] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stream: bool = True,
    ) -> AsyncIterator[AgentMessage]:
        """Execute the agent with the given prompt.

        This is the core execution method. Implementations should:
        1. Auto-initialize if needed
        2. Build SDK options from parameters
        3. Inject context into prompt
        4. Call SDK and yield translated messages
        5. Respect interrupt event

        Args:
            prompt: User message or task description.
            context: Runtime context (collection, retrieved content, etc.).
            system_prompt: Override the default system prompt.
            tools: Tool instances to make available.
            model: Model override (must be supported).
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            session_id: Resume an existing SDK session.
            stream: Whether to stream partial responses.

        Yields:
            AgentMessage objects representing the response stream.

        Raises:
            AgentInitializationError: If initialization fails.
            AgentExecutionError: If execution fails.
            AgentTimeoutError: If execution times out.
            AgentInterruptedError: If execution is interrupted.
        """
        # Required for abstract async generator
        yield
        raise NotImplementedError

    async def interrupt(self) -> None:
        """Interrupt the current execution.

        Sets the interrupt event, which should be checked during execution.
        Safe to call when:
        - Execution is in progress (will stop it)
        - No execution is running (no-op)
        - Called multiple times (idempotent)

        This method returns quickly without blocking.
        """
        if self._interrupt_event is not None:
            self._interrupt_event.set()

    async def fork_session(self, session_id: str) -> str:
        """Fork a session to create a branch point.

        Creates a new session that shares history with the original
        up to the fork point. Useful for exploring alternative
        conversation paths.

        Args:
            session_id: ID of the session to fork.

        Returns:
            ID of the new forked session.

        Raises:
            NotImplementedError: If forking is not supported.
            SessionNotFoundError: If the session doesn't exist.
        """
        raise NotImplementedError("Session forking not supported by this adapter")

    def validate_model(self, model: str) -> bool:
        """Check if a model is supported by this adapter.

        Args:
            model: Model identifier to validate.

        Returns:
            True if the model is supported, False otherwise.
        """
        capabilities = self.get_capabilities()
        if not capabilities.supported_models:
            return True  # No restrictions
        return model in capabilities.supported_models
