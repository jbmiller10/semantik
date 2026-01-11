"""Tests for ClaudeAgentPlugin."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.agents.types import (
    AgentCapabilities,
    AgentUseCase,
    MessageRole,
)
from shared.plugins.manifest import PluginManifest
from shared.plugins.types.agent import AgentPlugin


class AsyncIteratorMock:
    """Mock async iterator for testing."""

    def __init__(self, items: list[Any]):
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.fixture(autouse=True)
def mock_claude_sdk():
    """Mock the claude_agent_sdk module."""
    # Create a mock module
    mock_sdk = ModuleType("claude_agent_sdk")
    mock_sdk.query = MagicMock()  # Will be configured per test
    mock_sdk.ClaudeAgentOptions = MagicMock()

    # Insert into sys.modules
    sys.modules["claude_agent_sdk"] = mock_sdk

    yield mock_sdk

    # Cleanup
    del sys.modules["claude_agent_sdk"]


class TestClaudeAgentPluginImport:
    """Test plugin can be imported with mocked SDK."""

    def test_import_plugin(self, mock_claude_sdk: ModuleType) -> None:
        """Test that ClaudeAgentPlugin can be imported."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        assert ClaudeAgentPlugin is not None

    def test_inherits_from_agent_plugin(self, mock_claude_sdk: ModuleType) -> None:
        """Test that ClaudeAgentPlugin inherits from AgentPlugin."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        assert issubclass(ClaudeAgentPlugin, AgentPlugin)


class TestClaudeAgentPluginMetadata:
    """Tests for plugin metadata."""

    def test_plugin_id(self, mock_claude_sdk: ModuleType) -> None:
        """Test PLUGIN_ID is correct."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        assert ClaudeAgentPlugin.PLUGIN_ID == "claude-agent"

    def test_plugin_version(self, mock_claude_sdk: ModuleType) -> None:
        """Test PLUGIN_VERSION is correct."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        assert ClaudeAgentPlugin.PLUGIN_VERSION == "1.0.0"

    def test_plugin_type(self, mock_claude_sdk: ModuleType) -> None:
        """Test PLUGIN_TYPE is 'agent'."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        assert ClaudeAgentPlugin.PLUGIN_TYPE == "agent"


class TestClaudeAgentPluginManifest:
    """Tests for plugin manifest."""

    def test_get_manifest(self, mock_claude_sdk: ModuleType) -> None:
        """Test get_manifest returns valid manifest."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        manifest = ClaudeAgentPlugin.get_manifest()

        assert isinstance(manifest, PluginManifest)
        assert manifest.id == "claude-agent"
        assert manifest.type == "agent"
        assert manifest.version == "1.0.0"
        assert manifest.display_name == "Claude Agent"
        assert "LLM agent" in manifest.description
        assert manifest.author == "Semantik"
        assert manifest.license == "Apache-2.0"

    def test_manifest_includes_capabilities(self, mock_claude_sdk: ModuleType) -> None:
        """Test manifest includes capabilities dict."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        manifest = ClaudeAgentPlugin.get_manifest()

        assert manifest.capabilities is not None
        assert manifest.capabilities["supports_streaming"] is True
        assert manifest.capabilities["supports_tools"] is True
        assert "claude-sonnet-4-20250514" in manifest.capabilities["supported_models"]


class TestClaudeAgentPluginCapabilities:
    """Tests for plugin capabilities."""

    def test_get_capabilities(self, mock_claude_sdk: ModuleType) -> None:
        """Test get_capabilities returns Claude capabilities."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        caps = ClaudeAgentPlugin.get_capabilities()

        assert isinstance(caps, AgentCapabilities)
        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_extended_thinking is True
        assert caps.supports_sessions is True
        assert caps.max_context_tokens == 200000


class TestClaudeAgentPluginUseCases:
    """Tests for supported use cases."""

    def test_supported_use_cases(self, mock_claude_sdk: ModuleType) -> None:
        """Test supported_use_cases returns expected use cases."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        use_cases = ClaudeAgentPlugin.supported_use_cases()

        assert AgentUseCase.ASSISTANT in use_cases
        assert AgentUseCase.AGENTIC_SEARCH in use_cases
        assert AgentUseCase.TOOL_USE in use_cases
        assert AgentUseCase.REASONING in use_cases
        assert AgentUseCase.HYDE in use_cases
        assert AgentUseCase.QUERY_EXPANSION in use_cases
        assert AgentUseCase.SUMMARIZATION in use_cases
        assert AgentUseCase.ANSWER_SYNTHESIS in use_cases
        assert len(use_cases) == 8


class TestClaudeAgentPluginInstantiation:
    """Tests for plugin instantiation."""

    def test_can_instantiate(self, mock_claude_sdk: ModuleType) -> None:
        """Test plugin can be instantiated."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()
        assert plugin is not None

    def test_instantiate_with_config(self, mock_claude_sdk: ModuleType) -> None:
        """Test plugin can be instantiated with config."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        config = {"model": "claude-opus-4-20250514", "temperature": 0.5}
        plugin = ClaudeAgentPlugin(config)

        assert plugin._config == config

    def test_adapter_created(self, mock_claude_sdk: ModuleType) -> None:
        """Test adapter is created on instantiation."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()

        assert plugin._adapter is not None


class TestClaudeAgentPluginLifecycle:
    """Tests for plugin lifecycle."""

    @pytest.mark.asyncio()
    async def test_initialize(self, mock_claude_sdk: ModuleType) -> None:
        """Test initialize initializes adapter."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()
        await plugin.initialize()

        assert plugin.is_initialized is True
        assert plugin._adapter.is_initialized is True

    @pytest.mark.asyncio()
    async def test_cleanup(self, mock_claude_sdk: ModuleType) -> None:
        """Test cleanup cleans up adapter."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()
        await plugin.initialize()
        await plugin.cleanup()

        assert plugin._adapter.is_initialized is False


class TestClaudeAgentPluginExecution:
    """Tests for plugin execution."""

    @pytest.fixture()
    def mock_sdk_message(self) -> MagicMock:
        """Create a mock SDK message."""
        msg = MagicMock()
        msg.subtype = "success"
        msg.content = "Hello from Claude!"
        msg.session_id = "session-123"
        msg.is_partial = False
        # Explicitly set to None to avoid MagicMock truthy behavior
        msg.tool_use = None
        msg.tool_result = None
        msg.thinking = None
        msg.usage = MagicMock()
        msg.usage.input_tokens = 10
        msg.usage.output_tokens = 20
        msg.usage.cache_read_input_tokens = 0
        msg.usage.cache_creation_input_tokens = 0
        msg.usage.reasoning_tokens = 0
        return msg

    @pytest.mark.asyncio()
    async def test_execute_delegates_to_adapter(
        self,
        mock_claude_sdk: ModuleType,
        mock_sdk_message: MagicMock,
    ) -> None:
        """Test execute delegates to adapter."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        mock_claude_sdk.query.return_value = AsyncIteratorMock([mock_sdk_message])

        plugin = ClaudeAgentPlugin()
        await plugin.initialize()

        messages = []
        async for msg in plugin.execute("Hello"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].content == "Hello from Claude!"

    @pytest.mark.asyncio()
    async def test_execute_with_model_override(
        self,
        mock_claude_sdk: ModuleType,
        mock_sdk_message: MagicMock,
    ) -> None:
        """Test execute with model override."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        mock_claude_sdk.query.return_value = AsyncIteratorMock([mock_sdk_message])

        plugin = ClaudeAgentPlugin()
        await plugin.initialize()

        messages = []
        async for msg in plugin.execute("Hello", model="claude-opus-4-20250514"):
            messages.append(msg)

        # Verify options were built with the model
        call_kwargs = mock_claude_sdk.ClaudeAgentOptions.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio()
    async def test_interrupt_delegates_to_adapter(
        self,
        mock_claude_sdk: ModuleType,
    ) -> None:
        """Test interrupt delegates to adapter."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()
        await plugin.initialize()

        await plugin.interrupt()

        assert plugin._adapter._interrupt_event.is_set()


class TestClaudeAgentPluginHealthCheck:
    """Tests for health check."""

    @pytest.mark.asyncio()
    async def test_health_check_success(self, mock_claude_sdk: ModuleType) -> None:
        """Test health_check returns True when SDK available."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        result = await ClaudeAgentPlugin.health_check()
        assert result is True

    @pytest.mark.asyncio()
    @pytest.mark.skip(reason="SDK unavailability is hard to test with module mocking")
    async def test_health_check_sdk_missing(self) -> None:
        """Test health_check returns False when SDK missing.

        Note: This test is skipped because Python's module caching makes it
        difficult to reliably test ImportError scenarios with autouse fixtures.
        The functionality is tested manually.
        """


class TestClaudeAgentPluginConfigSchema:
    """Tests for config schema."""

    def test_get_config_schema(self, mock_claude_sdk: ModuleType) -> None:
        """Test get_config_schema returns extended schema."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        schema = ClaudeAgentPlugin.get_config_schema()

        assert schema is not None
        assert schema["type"] == "object"

        # Check standard agent fields
        assert "model" in schema["properties"]
        assert "temperature" in schema["properties"]
        assert "max_tokens" in schema["properties"]

        # Check Claude-specific fields
        assert "max_turns" in schema["properties"]
        assert "permission_mode" in schema["properties"]

        # Check model enum
        model_prop = schema["properties"]["model"]
        assert "claude-sonnet-4-20250514" in model_prop["enum"]
        assert "claude-opus-4-20250514" in model_prop["enum"]


class TestClaudeAgentPluginModelValidation:
    """Tests for model validation."""

    def test_validate_model_valid(self, mock_claude_sdk: ModuleType) -> None:
        """Test validate_model for valid model."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()
        assert plugin.validate_model("claude-sonnet-4-20250514") is True
        assert plugin.validate_model("claude-opus-4-20250514") is True

    def test_validate_model_invalid(self, mock_claude_sdk: ModuleType) -> None:
        """Test validate_model for invalid model."""
        from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin

        plugin = ClaudeAgentPlugin()
        assert plugin.validate_model("gpt-4") is False
        assert plugin.validate_model("unknown") is False
