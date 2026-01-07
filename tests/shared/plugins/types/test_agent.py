"""Tests for AgentPlugin base class."""

from __future__ import annotations

from typing import Any, AsyncIterator, ClassVar

import pytest

from shared.agents.types import (
    AgentCapabilities,
    AgentContext,
    AgentMessage,
    AgentUseCase,
    MessageRole,
    MessageType,
)
from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest
from shared.plugins.types.agent import AgentPlugin


class TestAgentPluginABC:
    """Tests for AgentPlugin abstract base class."""

    def test_cannot_instantiate_abc(self) -> None:
        """Test that AgentPlugin cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AgentPlugin()  # type: ignore[abstract]

    def test_plugin_type_is_agent(self) -> None:
        """Test that PLUGIN_TYPE is 'agent'."""
        assert AgentPlugin.PLUGIN_TYPE == "agent"

    def test_inherits_from_semantic_plugin(self) -> None:
        """Test that AgentPlugin inherits from SemanticPlugin."""
        assert issubclass(AgentPlugin, SemanticPlugin)


class ConcreteTestAgent(AgentPlugin):
    """Concrete implementation for testing."""

    PLUGIN_ID: ClassVar[str] = "test-agent"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Test Agent",
        "description": "A test agent plugin",
        "author": "Test Author",
        "license": "MIT",
    }

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            author=cls.METADATA.get("author"),
            license=cls.METADATA.get("license"),
            capabilities=cls.get_capabilities().to_dict(),
        )

    @classmethod
    def get_capabilities(cls) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_sessions=True,
            max_context_tokens=100000,
            supported_models=("test-model-1", "test-model-2"),
            default_model="test-model-1",
        )

    @classmethod
    def supported_use_cases(cls) -> list[AgentUseCase]:
        return [
            AgentUseCase.ASSISTANT,
            AgentUseCase.TOOL_USE,
            AgentUseCase.QUERY_EXPANSION,
        ]

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
        """Simple test implementation."""
        yield AgentMessage(
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content=f"Response to: {prompt}",
        )


class TestConcreteAgentPlugin:
    """Tests for a concrete AgentPlugin implementation."""

    @pytest.fixture
    def agent(self) -> ConcreteTestAgent:
        """Create a test agent instance."""
        return ConcreteTestAgent()

    def test_can_instantiate(self, agent: ConcreteTestAgent) -> None:
        """Test that concrete agent can be instantiated."""
        assert agent is not None
        assert isinstance(agent, AgentPlugin)

    def test_plugin_id(self, agent: ConcreteTestAgent) -> None:
        """Test PLUGIN_ID is set correctly."""
        assert agent.PLUGIN_ID == "test-agent"

    def test_plugin_version(self, agent: ConcreteTestAgent) -> None:
        """Test PLUGIN_VERSION is set correctly."""
        assert agent.PLUGIN_VERSION == "1.0.0"

    def test_plugin_type(self, agent: ConcreteTestAgent) -> None:
        """Test PLUGIN_TYPE is 'agent'."""
        assert agent.PLUGIN_TYPE == "agent"

    def test_get_manifest(self, agent: ConcreteTestAgent) -> None:
        """Test get_manifest returns valid manifest."""
        manifest = agent.get_manifest()

        assert manifest.id == "test-agent"
        assert manifest.type == "agent"
        assert manifest.version == "1.0.0"
        assert manifest.display_name == "Test Agent"
        assert manifest.description == "A test agent plugin"
        assert manifest.author == "Test Author"
        assert manifest.license == "MIT"

    def test_get_capabilities(self, agent: ConcreteTestAgent) -> None:
        """Test get_capabilities returns valid capabilities."""
        caps = agent.get_capabilities()

        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_sessions is True
        assert caps.max_context_tokens == 100000
        assert "test-model-1" in caps.supported_models
        assert caps.default_model == "test-model-1"

    def test_supported_use_cases(self, agent: ConcreteTestAgent) -> None:
        """Test supported_use_cases returns expected use cases."""
        use_cases = agent.supported_use_cases()

        assert AgentUseCase.ASSISTANT in use_cases
        assert AgentUseCase.TOOL_USE in use_cases
        assert AgentUseCase.QUERY_EXPANSION in use_cases
        assert len(use_cases) == 3

    @pytest.mark.asyncio
    async def test_execute_yields_message(self, agent: ConcreteTestAgent) -> None:
        """Test execute yields AgentMessage."""
        messages = []
        async for msg in agent.execute("Hello"):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].type == MessageType.TEXT
        assert "Hello" in messages[0].content

    @pytest.mark.asyncio
    async def test_execute_batch_default(self, agent: ConcreteTestAgent) -> None:
        """Test execute_batch default implementation."""
        prompts = ["Hello", "World"]
        results = await agent.execute_batch(prompts)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert "Hello" in results[0][0].content
        assert "World" in results[1][0].content

    @pytest.mark.asyncio
    async def test_interrupt_no_adapter(self, agent: ConcreteTestAgent) -> None:
        """Test interrupt works when no adapter is set."""
        # Should not raise
        await agent.interrupt()

    def test_validate_model_valid(self, agent: ConcreteTestAgent) -> None:
        """Test validate_model returns True for valid model."""
        assert agent.validate_model("test-model-1") is True
        assert agent.validate_model("test-model-2") is True

    def test_validate_model_invalid(self, agent: ConcreteTestAgent) -> None:
        """Test validate_model returns False for invalid model."""
        assert agent.validate_model("unknown-model") is False

    def test_get_config_schema(self, agent: ConcreteTestAgent) -> None:
        """Test get_config_schema returns valid schema."""
        schema = agent.get_config_schema()

        assert schema is not None
        assert schema["type"] == "object"
        assert "model" in schema["properties"]
        assert "system_prompt" in schema["properties"]
        assert "temperature" in schema["properties"]
        assert "max_tokens" in schema["properties"]
        assert "allowed_tools" in schema["properties"]
        assert "timeout_seconds" in schema["properties"]

    @pytest.mark.asyncio
    async def test_health_check_default(self, agent: ConcreteTestAgent) -> None:
        """Test health_check default returns True."""
        result = await agent.health_check()
        assert result is True


class TestAgentPluginWithConfig:
    """Tests for AgentPlugin configuration handling."""

    def test_init_with_config(self) -> None:
        """Test initialization with configuration."""
        config = {
            "model": "custom-model",
            "temperature": 0.5,
            "max_tokens": 2048,
        }
        agent = ConcreteTestAgent(config=config)

        assert agent.config == config
        assert agent._config["model"] == "custom-model"
        assert agent._config["temperature"] == 0.5

    def test_init_without_config(self) -> None:
        """Test initialization without configuration."""
        agent = ConcreteTestAgent()
        assert agent.config == {}

    @pytest.mark.asyncio
    async def test_initialize_merges_config(self) -> None:
        """Test initialize merges with constructor config."""
        agent = ConcreteTestAgent(config={"model": "initial-model"})
        await agent.initialize(config={"temperature": 0.8})

        # Constructor config takes precedence
        assert agent._config["model"] == "initial-model"
        assert agent._config["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_is_initialized_flag(self) -> None:
        """Test is_initialized flag is set after initialize."""
        agent = ConcreteTestAgent()
        assert agent.is_initialized is False

        await agent.initialize()
        assert agent.is_initialized is True

        await agent.cleanup()
        assert agent.is_initialized is False


class TestAgentPluginCapabilitiesSerialization:
    """Tests for capabilities serialization in manifest."""

    def test_manifest_includes_capabilities(self) -> None:
        """Test that manifest includes serialized capabilities."""
        agent = ConcreteTestAgent()
        manifest = agent.get_manifest()

        caps = manifest.capabilities
        assert caps is not None
        assert caps["supports_streaming"] is True
        assert caps["supports_tools"] is True
        assert caps["max_context_tokens"] == 100000
        assert caps["default_model"] == "test-model-1"


class TestAgentPluginWithNoModelRestrictions:
    """Tests for agent with no model restrictions."""

    class UnrestrictedAgent(ConcreteTestAgent):
        """Agent with no model restrictions."""

        @classmethod
        def get_capabilities(cls) -> AgentCapabilities:
            return AgentCapabilities(
                supported_models=(),  # Empty = no restrictions
            )

    def test_validate_model_no_restrictions(self) -> None:
        """Test validate_model accepts any model when no restrictions."""
        agent = self.UnrestrictedAgent()
        assert agent.validate_model("any-model") is True
        assert agent.validate_model("another-model") is True


class TestAgentPluginToolValidation:
    """Tests for tool validation."""

    def test_validate_tools_returns_tuple(self) -> None:
        """Test validate_tools returns a tuple of (is_valid, invalid_tools)."""
        agent = ConcreteTestAgent()
        # Since tool registry is not yet implemented (Phase 4),
        # it should return (True, []) as a fallback
        is_valid, invalid = agent.validate_tools(["some-tool"])
        assert isinstance(is_valid, bool)
        assert isinstance(invalid, list)
