"""Unit tests for the AgentOrchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from webui.services.agent.exceptions import ConversationNotActiveError
from webui.services.agent.message_store import ConversationMessage
from webui.services.agent.models import AgentConversation, ConversationStatus
from webui.services.agent.orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    ToolCall,
    ToolResult,
)


@pytest.fixture()
def mock_conversation():
    """Create a mock conversation."""
    conv = MagicMock(spec=AgentConversation)
    conv.id = "test-conv-123"
    conv.user_id = 42
    conv.source_id = 1
    conv.status = ConversationStatus.ACTIVE
    conv.current_pipeline = None
    conv.source_analysis = None
    conv.summary = None
    conv.uncertainties = []
    conv.inline_source_config = None
    return conv


@pytest.fixture()
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture()
def mock_llm_factory():
    """Create a mock LLM factory."""
    return MagicMock()


@pytest.fixture()
def mock_message_store():
    """Create a mock message store."""
    store = MagicMock()
    store.has_messages = AsyncMock(return_value=False)
    store.get_messages = AsyncMock(return_value=[])
    store.append_message = AsyncMock()
    store.set_messages = AsyncMock()
    return store


@pytest.fixture()
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.__aenter__ = AsyncMock(return_value=provider)
    provider.__aexit__ = AsyncMock(return_value=None)
    return provider


@pytest.fixture()
def orchestrator(mock_conversation, mock_session, mock_llm_factory, mock_message_store):
    """Create an orchestrator instance with mocked dependencies."""
    # We need to patch the tool classes to avoid imports
    with patch("webui.services.agent.orchestrator.ORCHESTRATOR_TOOL_CLASSES", []):
        return AgentOrchestrator(
            conversation=mock_conversation,
            session=mock_session,
            llm_factory=mock_llm_factory,
            message_store=mock_message_store,
        )


class TestAgentOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_initializes_with_dependencies(self, mock_conversation, mock_session, mock_llm_factory, mock_message_store):
        """Orchestrator initializes correctly with all dependencies."""
        with patch("webui.services.agent.orchestrator.ORCHESTRATOR_TOOL_CLASSES", []):
            orch = AgentOrchestrator(
                conversation=mock_conversation,
                session=mock_session,
                llm_factory=mock_llm_factory,
                message_store=mock_message_store,
            )

        assert orch.conversation == mock_conversation
        assert orch.session == mock_session
        assert orch.llm_factory == mock_llm_factory
        assert orch.message_store == mock_message_store

    def test_builds_tool_context(self, orchestrator, mock_conversation, mock_session):
        """Tool context includes required keys."""
        context = orchestrator._build_tool_context()

        assert context["session"] == mock_session
        assert context["user_id"] == mock_conversation.user_id
        assert context["conversation"] == mock_conversation
        assert context["orchestrator"] == orchestrator


class TestHandleMessage:
    """Tests for the handle_message method."""

    @pytest.mark.asyncio()
    async def test_raises_when_conversation_not_active(
        self, mock_conversation, mock_session, mock_llm_factory, mock_message_store
    ):
        """Raises ConversationNotActiveError for non-active conversation."""
        mock_conversation.status = ConversationStatus.APPLIED

        with patch("webui.services.agent.orchestrator.ORCHESTRATOR_TOOL_CLASSES", []):
            orch = AgentOrchestrator(
                conversation=mock_conversation,
                session=mock_session,
                llm_factory=mock_llm_factory,
                message_store=mock_message_store,
            )

        with pytest.raises(ConversationNotActiveError) as exc_info:
            await orch.handle_message("test message")

        assert exc_info.value.conversation_id == "test-conv-123"
        assert exc_info.value.status == "applied"

    @pytest.mark.asyncio()
    async def test_appends_user_message_to_store(
        self, orchestrator, mock_message_store, mock_llm_factory, mock_llm_provider
    ):
        """User message is appended to message store."""
        mock_response = MagicMock()
        mock_response.content = "Hello, how can I help?"
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        await orchestrator.handle_message("User question")

        # Verify message was appended
        assert mock_message_store.append_message.call_count >= 1
        first_call_args = mock_message_store.append_message.call_args_list[0]
        assert first_call_args[0][0] == "test-conv-123"
        assert first_call_args[0][1].role == "user"
        assert first_call_args[0][1].content == "User question"

    @pytest.mark.asyncio()
    async def test_returns_agent_response(
        self, orchestrator, mock_message_store, mock_llm_factory, mock_llm_provider
    ):  # noqa: ARG002
        """Returns AgentResponse with content."""
        mock_response = MagicMock()
        mock_response.content = "Here is my response."
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        result = await orchestrator.handle_message("Hello")

        assert isinstance(result, AgentResponse)
        assert result.content == "Here is my response."
        assert result.pipeline_updated is False
        assert result.uncertainties_added == []

    @pytest.mark.asyncio()
    async def test_recovers_from_summary_when_no_messages(
        self, orchestrator, mock_message_store, mock_llm_factory, mock_llm_provider, mock_conversation
    ):  # noqa: ARG002
        """Recovers conversation from summary when Redis messages expired."""
        mock_conversation.summary = "User wanted semantic chunking."
        mock_message_store.has_messages = AsyncMock(return_value=False)

        mock_response = MagicMock()
        mock_response.content = "I recall you wanted semantic chunking. How can I continue?"
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        await orchestrator.handle_message("Continue please")

        # Should have set recovery messages
        assert mock_message_store.set_messages.call_count == 1


class TestParseToolCalls:
    """Tests for tool call parsing."""

    def test_parses_single_tool_call(self, orchestrator):
        """Parses a single tool call from response."""
        response = """Let me check the plugins.

```tool
{"name": "list_plugins", "arguments": {"type": "embedding"}}
```

I'll analyze the available options."""

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "list_plugins"
        assert tool_calls[0].arguments == {"type": "embedding"}

    def test_parses_multiple_tool_calls(self, orchestrator):
        """Parses multiple tool calls from response."""
        response = """I'll gather information:

```tool
{"name": "list_plugins", "arguments": {}}
```

And also:

```tool
{"name": "list_templates", "arguments": {}}
```

Let me analyze these."""

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "list_plugins"
        assert tool_calls[1].name == "list_templates"

    def test_returns_empty_for_no_tool_calls(self, orchestrator):
        """Returns empty list when no tool calls present."""
        response = "I understand. Let me explain the options available."

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 0

    def test_handles_malformed_json(self, orchestrator):
        """Ignores malformed JSON in tool blocks."""
        response = """Here's a call:

```tool
{"name": "list_plugins", "arguments": {invalid json}}
```

And a valid one:

```tool
{"name": "list_templates", "arguments": {}}
```
"""

        tool_calls = orchestrator._parse_tool_calls(response)

        # Only the valid one should be parsed
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "list_templates"


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_formats_user_messages(self, orchestrator):
        """User messages are prefixed correctly."""
        messages = [
            ConversationMessage.create("user", "Hello"),
            ConversationMessage.create("assistant", "Hi there!"),
        ]

        prompt = orchestrator._build_prompt(messages)

        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt

    def test_formats_tool_messages(self, orchestrator):
        """Tool messages include tool name."""
        messages = [
            ConversationMessage.create("tool", '{"plugins": []}', metadata={"tool_name": "list_plugins"}),
        ]

        prompt = orchestrator._build_prompt(messages)

        assert "Tool (list_plugins):" in prompt

    def test_formats_subagent_messages(self, orchestrator):
        """Subagent messages include type."""
        messages = [
            ConversationMessage.create(
                "subagent",
                "Analysis complete.",
                metadata={"subagent_type": "source_analyzer"},
            ),
        ]

        prompt = orchestrator._build_prompt(messages)

        assert "SubAgent (source_analyzer):" in prompt


class TestExecuteTools:
    """Tests for tool execution."""

    @pytest.mark.asyncio()
    async def test_executes_known_tool(self, orchestrator):
        """Executes tool when it exists."""
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value={"plugins": ["plugin1"]})
        orchestrator.tools["list_plugins"] = mock_tool

        tool_calls = [ToolCall(id="call_0", name="list_plugins", arguments={"type": "embedding"})]

        results = await orchestrator._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].data == {"plugins": ["plugin1"]}
        mock_tool.execute.assert_called_once_with(type="embedding")

    @pytest.mark.asyncio()
    async def test_returns_error_for_unknown_tool(self, orchestrator):
        """Returns error result for unknown tool."""
        tool_calls = [ToolCall(id="call_0", name="unknown_tool", arguments={})]

        results = await orchestrator._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert "Unknown tool" in results[0].error

    @pytest.mark.asyncio()
    async def test_handles_tool_exception(self, orchestrator):
        """Handles exceptions from tool execution."""
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(side_effect=Exception("Tool failed"))
        orchestrator.tools["failing_tool"] = mock_tool

        tool_calls = [ToolCall(id="call_0", name="failing_tool", arguments={})]

        results = await orchestrator._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert "Internal error" in results[0].error


class TestToolCallDataclass:
    """Tests for ToolCall dataclass."""

    def test_creates_tool_call(self):
        """ToolCall can be created with required fields."""
        tc = ToolCall(id="call_1", name="test_tool", arguments={"param": "value"})

        assert tc.id == "call_1"
        assert tc.name == "test_tool"
        assert tc.arguments == {"param": "value"}


class TestToolResultDataclass:
    """Tests for ToolResult dataclass."""

    def test_creates_success_result(self):
        """ToolResult can represent success."""
        result = ToolResult(
            tool_call_id="call_1",
            name="test_tool",
            success=True,
            data={"result": "success"},
        )

        assert result.success is True
        assert result.data == {"result": "success"}
        assert result.error is None

    def test_creates_error_result(self):
        """ToolResult can represent error."""
        result = ToolResult(
            tool_call_id="call_1",
            name="test_tool",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"


class TestAgentResponseDataclass:
    """Tests for AgentResponse dataclass."""

    def test_creates_agent_response(self):
        """AgentResponse can be created with all fields."""
        response = AgentResponse(
            content="Here is my response",
            tool_calls_made=[{"name": "list_plugins"}],
            pipeline_updated=True,
            uncertainties_added=[{"severity": "blocking", "message": "Issue found"}],
        )

        assert response.content == "Here is my response"
        assert response.pipeline_updated is True
        assert len(response.tool_calls_made) == 1
        assert len(response.uncertainties_added) == 1


class TestSystemPrompt:
    """Tests for system prompt generation."""

    def test_system_prompt_includes_tool_descriptions(self, orchestrator):
        """System prompt includes formatted tool descriptions."""
        # Add a mock tool
        mock_tool = MagicMock()
        mock_tool.NAME = "test_tool"
        mock_tool.DESCRIPTION = "A test tool for testing"
        mock_tool.PARAMETERS = {
            "type": "object",
            "properties": {"param1": {"type": "string", "description": "First param"}},
            "required": ["param1"],
        }
        orchestrator.tools["test_tool"] = mock_tool

        prompt = orchestrator._build_system_prompt()

        assert "test_tool" in prompt
        assert "A test tool for testing" in prompt
        assert "param1" in prompt

    def test_system_prompt_has_required_sections(self, orchestrator):
        """System prompt contains required instruction sections."""
        prompt = orchestrator._build_system_prompt()

        assert "Pipeline Builder Agent" in prompt
        assert "```tool" in prompt  # Tool call format
        assert "Guidelines" in prompt


class TestConversationRecovery:
    """Tests for conversation recovery from summary."""

    @pytest.mark.asyncio()
    async def test_loads_messages_from_redis(self, orchestrator, mock_message_store):
        """Loads existing messages from Redis."""
        existing_messages = [
            ConversationMessage.create("user", "Hello"),
            ConversationMessage.create("assistant", "Hi there!"),
        ]
        mock_message_store.has_messages = AsyncMock(return_value=True)
        mock_message_store.get_messages = AsyncMock(return_value=existing_messages)

        messages = await orchestrator._load_or_recover_messages()

        assert len(messages) == 2
        assert messages[0].content == "Hello"

    @pytest.mark.asyncio()
    async def test_recovers_from_summary(self, orchestrator, mock_message_store, mock_conversation):
        """Creates recovery message from summary when no Redis messages."""
        mock_message_store.has_messages = AsyncMock(return_value=False)
        mock_conversation.summary = "User configured semantic chunking with 512 tokens."

        messages = await orchestrator._load_or_recover_messages()

        assert len(messages) == 1
        assert "Previous conversation summary" in messages[0].content
        assert "512 tokens" in messages[0].content

    @pytest.mark.asyncio()
    async def test_returns_empty_for_new_conversation(self, orchestrator, mock_message_store, mock_conversation):
        """Returns empty list for new conversation."""
        mock_message_store.has_messages = AsyncMock(return_value=False)
        mock_conversation.summary = None

        messages = await orchestrator._load_or_recover_messages()

        assert messages == []


class TestMaxTurnsLimit:
    """Tests for max turns handling."""

    @pytest.mark.asyncio()
    async def test_respects_max_turns(
        self, orchestrator, mock_message_store, mock_llm_factory, mock_llm_provider
    ):  # noqa: ARG002
        """Conversation loop respects MAX_TURNS limit."""
        # Make LLM always return tool calls so it never completes naturally
        mock_response = MagicMock()
        mock_response.content = """```tool
{"name": "list_plugins", "arguments": {}}
```"""
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        # Add a mock tool that returns something
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value={"plugins": []})
        orchestrator.tools["list_plugins"] = mock_tool

        # Set a low max turns for testing
        original_max = AgentOrchestrator.MAX_TURNS
        AgentOrchestrator.MAX_TURNS = 3

        try:
            result = await orchestrator.handle_message("Test")
            assert "maximum number of steps" in result.content
        finally:
            AgentOrchestrator.MAX_TURNS = original_max

    @pytest.mark.asyncio()
    async def test_streaming_max_turns_yields_done_event_with_flag(
        self, orchestrator, mock_message_store, mock_llm_factory, mock_llm_provider
    ):  # noqa: ARG002
        """Streaming yields done event with max_turns_reached flag when limit hit."""
        # Force max turns by always returning tool calls
        mock_response = MagicMock()
        mock_response.content = '```tool\n{"name": "list_plugins", "arguments": {}}\n```'
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value={"plugins": []})
        orchestrator.tools["list_plugins"] = mock_tool

        original_max = AgentOrchestrator.MAX_TURNS
        AgentOrchestrator.MAX_TURNS = 2
        try:
            events = []
            async for event in orchestrator.handle_message_streaming("Test"):
                events.append(event)

            done_events = [e for e in events if e.event.value == "done"]
            assert len(done_events) == 1
            assert done_events[0].data.get("max_turns_reached") is True
        finally:
            AgentOrchestrator.MAX_TURNS = original_max


class TestStreamingEvents:
    """Tests for SSE streaming message handling."""

    @pytest.mark.asyncio()
    async def test_yields_content_and_done_events(
        self, orchestrator, mock_llm_factory, mock_llm_provider, mock_message_store
    ):  # noqa: ARG002
        """Streaming yields content and done events for a simple response."""
        mock_response = MagicMock()
        mock_response.content = "Hello, how can I help you?"
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        events = []
        async for event in orchestrator.handle_message_streaming("Test"):
            events.append(event)

        # Should have at least content and done events
        event_types = [e.event.value for e in events]
        assert "content" in event_types
        assert "done" in event_types

        # Done should be last
        assert event_types[-1] == "done"

    @pytest.mark.asyncio()
    async def test_yields_tool_call_events(
        self, orchestrator, mock_llm_factory, mock_llm_provider, mock_message_store
    ):  # noqa: ARG002
        """Streaming yields tool_call_start and tool_call_end events."""
        # First response has tool call, second is final response
        tool_response = MagicMock()
        tool_response.content = """Let me check the plugins.
```tool
{"name": "list_plugins", "arguments": {"type": "embedding"}}
```"""
        final_response = MagicMock()
        final_response.content = "Here are the plugins."

        mock_llm_provider.generate = AsyncMock(side_effect=[tool_response, final_response])
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        # Add mock tool
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value={"plugins": []})
        orchestrator.tools["list_plugins"] = mock_tool

        events = []
        async for event in orchestrator.handle_message_streaming("List plugins"):
            events.append(event)

        event_types = [e.event.value for e in events]
        assert "tool_call_start" in event_types
        assert "tool_call_end" in event_types

    @pytest.mark.asyncio()
    async def test_yields_error_event_on_inactive_conversation(self, orchestrator):
        """Streaming yields error event for inactive conversation."""
        orchestrator.conversation.status = ConversationStatus.ABANDONED

        events = []
        async for event in orchestrator.handle_message_streaming("Test"):
            events.append(event)

        assert len(events) == 1
        assert events[0].event.value == "error"
        assert "not active" in events[0].data["error"]

    @pytest.mark.asyncio()
    async def test_yields_pipeline_update_event(
        self, orchestrator, mock_llm_factory, mock_llm_provider, mock_message_store
    ):  # noqa: ARG002
        """Streaming yields pipeline_update event when pipeline changes."""
        # Response with build_pipeline tool call
        tool_response = MagicMock()
        tool_response.content = """```tool
{"name": "build_pipeline", "arguments": {"nodes": [], "edges": []}}
```"""
        final_response = MagicMock()
        final_response.content = "Pipeline built."

        mock_llm_provider.generate = AsyncMock(side_effect=[tool_response, final_response])
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        # Mock build_pipeline tool to return success
        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value={"success": True, "pipeline": {}})
        orchestrator.tools["build_pipeline"] = mock_tool

        events = []
        async for event in orchestrator.handle_message_streaming("Build pipeline"):
            events.append(event)

        event_types = [e.event.value for e in events]
        assert "pipeline_update" in event_types

    @pytest.mark.asyncio()
    async def test_tool_call_events_include_metadata(
        self, orchestrator, mock_llm_factory, mock_llm_provider, mock_message_store
    ):  # noqa: ARG002
        """Tool call events include tool name and call ID."""
        tool_response = MagicMock()
        tool_response.content = """```tool
{"name": "list_plugins", "arguments": {"type": "embedding"}}
```"""
        final_response = MagicMock()
        final_response.content = "Done."

        mock_llm_provider.generate = AsyncMock(side_effect=[tool_response, final_response])
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        mock_tool = MagicMock()
        mock_tool.execute = AsyncMock(return_value={"plugins": []})
        orchestrator.tools["list_plugins"] = mock_tool

        events = []
        async for event in orchestrator.handle_message_streaming("Test"):
            events.append(event)

        # Find tool_call_start event
        start_events = [e for e in events if e.event.value == "tool_call_start"]
        assert len(start_events) >= 1
        assert start_events[0].data["tool"] == "list_plugins"
        assert "call_id" in start_events[0].data

        # Find tool_call_end event
        end_events = [e for e in events if e.event.value == "tool_call_end"]
        assert len(end_events) >= 1
        assert end_events[0].data["tool"] == "list_plugins"
        assert end_events[0].data["success"] is True

    @pytest.mark.asyncio()
    async def test_done_event_includes_metadata(
        self, orchestrator, mock_llm_factory, mock_llm_provider, mock_message_store
    ):  # noqa: ARG002
        """Done event includes pipeline_updated and tool_calls metadata."""
        mock_response = MagicMock()
        mock_response.content = "Hello!"
        mock_llm_provider.generate = AsyncMock(return_value=mock_response)
        mock_llm_factory.create_provider_for_tier = AsyncMock(return_value=mock_llm_provider)

        events = []
        async for event in orchestrator.handle_message_streaming("Test"):
            events.append(event)

        done_events = [e for e in events if e.event.value == "done"]
        assert len(done_events) == 1
        done_data = done_events[0].data
        assert "pipeline_updated" in done_data
        assert "tool_calls" in done_data
        assert "uncertainties_added" in done_data
