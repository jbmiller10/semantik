"""Tests for AgentAdapter abstract base class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from shared.agents.base import AgentAdapter
from shared.agents.types import (
    AgentCapabilities,
    AgentMessage,
    MessageRole,
    MessageType,
)


class TestAgentAdapterABC:
    """Tests for AgentAdapter abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        """Test that AgentAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AgentAdapter()  # type: ignore[abstract]


class ConcreteTestAdapter(AgentAdapter):
    """Concrete implementation for testing."""

    @classmethod
    def get_capabilities(cls) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supported_models=("test-model-1", "test-model-2"),
            default_model="test-model-1",
        )

    async def execute(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[AgentMessage]:
        """Simple test implementation."""
        yield AgentMessage(
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=f"Response to: {prompt}",
        )


class TestConcreteAdapter:
    """Tests for a concrete AgentAdapter implementation."""

    @pytest.fixture()
    def adapter(self) -> ConcreteTestAdapter:
        """Create a test adapter instance."""
        return ConcreteTestAdapter()

    def test_can_instantiate(self, adapter: ConcreteTestAdapter) -> None:
        """Test that concrete adapter can be instantiated."""
        assert adapter is not None
        assert isinstance(adapter, AgentAdapter)

    def test_config_defaults_to_empty_dict(self, adapter: ConcreteTestAdapter) -> None:
        """Test that config defaults to empty dict."""
        assert adapter._config == {}

    def test_init_with_config(self) -> None:
        """Test initialization with configuration."""
        config = {"model": "custom-model", "temperature": 0.5}
        adapter = ConcreteTestAdapter(config)
        assert adapter._config == config

    def test_not_initialized_by_default(self, adapter: ConcreteTestAdapter) -> None:
        """Test that adapter is not initialized by default."""
        assert adapter.is_initialized is False
        assert adapter._interrupt_event is None

    @pytest.mark.asyncio()
    async def test_initialize_sets_flag(self, adapter: ConcreteTestAdapter) -> None:
        """Test that initialize sets the initialized flag."""
        await adapter.initialize()

        assert adapter.is_initialized is True
        assert adapter._interrupt_event is not None

    @pytest.mark.asyncio()
    async def test_initialize_is_idempotent(self, adapter: ConcreteTestAdapter) -> None:
        """Test that initialize can be called multiple times."""
        await adapter.initialize()
        event1 = adapter._interrupt_event

        await adapter.initialize()
        event2 = adapter._interrupt_event

        # Should be the same event (not recreated)
        assert event1 is event2

    @pytest.mark.asyncio()
    async def test_cleanup_resets_state(self, adapter: ConcreteTestAdapter) -> None:
        """Test that cleanup resets adapter state."""
        await adapter.initialize()
        adapter._current_session_id = "test-session"

        await adapter.cleanup()

        assert adapter.is_initialized is False
        assert adapter._interrupt_event is None
        assert adapter._current_session_id is None

    def test_get_capabilities(self, adapter: ConcreteTestAdapter) -> None:
        """Test get_capabilities returns correct capabilities."""
        caps = adapter.get_capabilities()

        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert "test-model-1" in caps.supported_models
        assert caps.default_model == "test-model-1"

    @pytest.mark.asyncio()
    async def test_execute_yields_message(self, adapter: ConcreteTestAdapter) -> None:
        """Test execute yields AgentMessage."""
        messages = []
        async for msg in adapter.execute("Hello"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].type == MessageType.TEXT
        assert "Hello" in messages[0].content

    @pytest.mark.asyncio()
    async def test_interrupt_sets_event(self, adapter: ConcreteTestAdapter) -> None:
        """Test interrupt sets the interrupt event."""
        await adapter.initialize()

        await adapter.interrupt()

        assert adapter._interrupt_event is not None
        assert adapter._interrupt_event.is_set()

    @pytest.mark.asyncio()
    async def test_interrupt_when_not_initialized(self, adapter: ConcreteTestAdapter) -> None:
        """Test interrupt is safe when not initialized."""
        # Should not raise
        await adapter.interrupt()

    @pytest.mark.asyncio()
    async def test_fork_session_not_implemented(self, adapter: ConcreteTestAdapter) -> None:
        """Test fork_session raises NotImplementedError by default."""
        with pytest.raises(NotImplementedError, match="Session forking not supported"):
            await adapter.fork_session("test-session")

    def test_validate_model_valid(self, adapter: ConcreteTestAdapter) -> None:
        """Test validate_model returns True for valid model."""
        assert adapter.validate_model("test-model-1") is True
        assert adapter.validate_model("test-model-2") is True

    def test_validate_model_invalid(self, adapter: ConcreteTestAdapter) -> None:
        """Test validate_model returns False for invalid model."""
        assert adapter.validate_model("unknown-model") is False

    def test_current_session_id_property(self, adapter: ConcreteTestAdapter) -> None:
        """Test current_session_id property."""
        assert adapter.current_session_id is None

        adapter._current_session_id = "test-session"
        assert adapter.current_session_id == "test-session"


class UnrestrictedTestAdapter(ConcreteTestAdapter):
    """Adapter with no model restrictions."""

    @classmethod
    def get_capabilities(cls) -> AgentCapabilities:
        return AgentCapabilities(
            supported_models=(),  # Empty = no restrictions
        )


class TestAdapterWithNoModelRestrictions:
    """Tests for adapter with no model restrictions."""

    def test_validate_model_accepts_any(self) -> None:
        """Test validate_model accepts any model when no restrictions."""
        adapter = UnrestrictedTestAdapter()
        assert adapter.validate_model("any-model") is True
        assert adapter.validate_model("another-model") is True
