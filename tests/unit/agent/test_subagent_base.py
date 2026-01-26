"""Unit tests for SubAgent base class and related dataclasses."""

from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.llm.exceptions import LLMRateLimitError, LLMTimeoutError
from webui.services.agent.subagents.base import (
    Message,
    SubAgent,
    SubAgentResult,
    ToolCall,
    ToolResult,
    Uncertainty,
)
from webui.services.agent.tools.base import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing SubAgent."""

    NAME: ClassVar[str] = "mock_tool"
    DESCRIPTION: ClassVar[str] = "A mock tool"
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "input": {"type": "string"},
        },
        "required": ["input"],
    }

    async def execute(self, input: str) -> dict[str, Any]:
        return {"output": f"processed: {input}"}


class FailingMockTool(BaseTool):
    """Mock tool that fails."""

    NAME: ClassVar[str] = "failing_tool"
    DESCRIPTION: ClassVar[str] = "A tool that fails"

    async def execute(self, **kwargs: Any) -> Any:
        raise RuntimeError("Tool failed intentionally")


class TestSubAgent(SubAgent):
    """Concrete test subagent implementation."""

    AGENT_ID: ClassVar[str] = "test_agent"
    SYSTEM_PROMPT: ClassVar[str] = "You are a test agent."
    TOOLS: ClassVar[list[type[BaseTool]]] = [MockTool]
    MAX_TURNS: ClassVar[int] = 5
    TIMEOUT_SECONDS: ClassVar[int] = 10

    def _build_initial_message(self) -> Message:
        return Message(
            role="user",
            content=f"Process this context: {self.context}",
        )

    def _extract_result(self, response: Message) -> SubAgentResult:
        return SubAgentResult(
            success=True,
            data={"response": response.content},
            summary="Test completed successfully",
        )


class TestUncertaintyDataclass:
    """Tests for Uncertainty dataclass."""

    def test_uncertainty_creation(self):
        """Test creating an uncertainty."""
        uncertainty = Uncertainty(
            severity="blocking",
            message="Critical issue found",
            context={"file": "test.txt"},
        )

        assert uncertainty.severity == "blocking"
        assert uncertainty.message == "Critical issue found"
        assert uncertainty.context == {"file": "test.txt"}

    def test_uncertainty_without_context(self):
        """Test uncertainty without context."""
        uncertainty = Uncertainty(
            severity="info",
            message="Informational note",
        )

        assert uncertainty.context is None

    def test_uncertainty_is_frozen(self):
        """Test that uncertainty is immutable."""
        uncertainty = Uncertainty(severity="notable", message="Note")

        with pytest.raises(AttributeError):
            uncertainty.severity = "blocking"  # type: ignore[misc]


class TestSubAgentResultDataclass:
    """Tests for SubAgentResult dataclass."""

    def test_result_creation(self):
        """Test creating a result."""
        result = SubAgentResult(
            success=True,
            data={"key": "value"},
            summary="All done",
        )

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.summary == "All done"
        assert result.uncertainties == []

    def test_result_with_uncertainties(self):
        """Test result with uncertainties."""
        uncertainties = [
            Uncertainty(severity="notable", message="Issue 1"),
            Uncertainty(severity="info", message="Issue 2"),
        ]
        result = SubAgentResult(
            success=True,
            data={},
            uncertainties=uncertainties,
        )

        assert len(result.uncertainties) == 2

    def test_result_default_values(self):
        """Test result default values."""
        result = SubAgentResult(success=False, data={})

        assert result.uncertainties == []
        assert result.summary == ""


class TestMessageDataclass:
    """Tests for Message dataclass."""

    def test_simple_message(self):
        """Test creating a simple message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls is None
        assert msg.tool_results is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        tool_calls = [
            ToolCall(id="tc1", name="mock_tool", arguments={"input": "test"}),
        ]
        msg = Message(role="assistant", content="", tool_calls=tool_calls)

        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "mock_tool"

    def test_message_with_tool_results(self):
        """Test message with tool results."""
        results = [
            ToolResult(tool_call_id="tc1", name="mock_tool", success=True, data="result"),
        ]
        msg = Message(role="tool", content="", tool_results=results)

        assert msg.tool_results is not None
        assert len(msg.tool_results) == 1
        assert msg.tool_results[0].success is True


class TestToolCallDataclass:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            id="call-123",
            name="mock_tool",
            arguments={"input": "test data"},
        )

        assert call.id == "call-123"
        assert call.name == "mock_tool"
        assert call.arguments == {"input": "test data"}


class TestToolResultDataclass:
    """Tests for ToolResult dataclass."""

    def test_successful_result(self):
        """Test successful tool result."""
        result = ToolResult(
            tool_call_id="call-123",
            name="mock_tool",
            success=True,
            data={"output": "result"},
        )

        assert result.success is True
        assert result.data == {"output": "result"}
        assert result.error is None

    def test_failed_result(self):
        """Test failed tool result."""
        result = ToolResult(
            tool_call_id="call-123",
            name="mock_tool",
            success=False,
            error="Something went wrong",
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None


class TestSubAgentInit:
    """Tests for SubAgent initialization."""

    def test_subagent_init(self):
        """Test subagent initialization."""
        mock_llm = AsyncMock()
        context = {"user_id": 1, "data": "test"}

        agent = TestSubAgent(mock_llm, context)

        assert agent.llm is mock_llm
        assert agent.context == context
        assert agent.messages == []
        assert "mock_tool" in agent.tools

    def test_tools_initialized(self):
        """Test that tools are properly initialized."""
        mock_llm = AsyncMock()
        context = {"key": "value"}

        agent = TestSubAgent(mock_llm, context)

        assert len(agent.tools) == 1
        tool = agent.tools["mock_tool"]
        assert isinstance(tool, MockTool)
        assert tool.context == context


class TestSubAgentGetToolSchemas:
    """Tests for SubAgent.get_tool_schemas()."""

    def test_get_tool_schemas(self):
        """Test getting tool schemas."""
        mock_llm = AsyncMock()
        agent = TestSubAgent(mock_llm, {})

        schemas = agent.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "mock_tool"


class TestSubAgentBuildLLMMessages:
    """Tests for SubAgent._build_llm_messages()."""

    def test_build_simple_messages(self):
        """Test building simple messages."""
        mock_llm = AsyncMock()
        agent = TestSubAgent(mock_llm, {})

        # Add some messages
        agent.messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        llm_messages = agent._build_llm_messages()

        assert len(llm_messages) == 2
        assert llm_messages[0]["role"] == "user"
        assert llm_messages[0]["content"] == "Hello"
        assert llm_messages[1]["role"] == "assistant"

    def test_build_messages_with_tool_results(self):
        """Test building messages with tool results."""
        mock_llm = AsyncMock()
        agent = TestSubAgent(mock_llm, {})

        tool_results = [
            ToolResult(tool_call_id="tc1", name="mock_tool", success=True, data="result data"),
            ToolResult(tool_call_id="tc2", name="mock_tool", success=False, error="failed"),
        ]
        agent.messages = [
            Message(role="user", content="Start"),
            Message(role="tool", content="", tool_results=tool_results),
        ]

        llm_messages = agent._build_llm_messages()

        # User message + 2 tool results
        assert len(llm_messages) == 3
        assert llm_messages[1]["role"] == "tool"
        assert llm_messages[1]["tool_call_id"] == "tc1"
        assert llm_messages[1]["content"] == "result data"
        assert llm_messages[2]["tool_call_id"] == "tc2"
        assert llm_messages[2]["content"] == "failed"


class TestSubAgentExecuteTools:
    """Tests for SubAgent._execute_tools()."""

    @pytest.mark.asyncio()
    async def test_execute_tools_success(self):
        """Test successful tool execution."""
        mock_llm = AsyncMock()
        agent = TestSubAgent(mock_llm, {})

        tool_calls = [
            ToolCall(id="tc1", name="mock_tool", arguments={"input": "test"}),
        ]

        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].data == {"output": "processed: test"}
        assert results[0].tool_call_id == "tc1"

    @pytest.mark.asyncio()
    async def test_execute_unknown_tool(self):
        """Test execution of unknown tool."""
        mock_llm = AsyncMock()
        agent = TestSubAgent(mock_llm, {})

        tool_calls = [
            ToolCall(id="tc1", name="unknown_tool", arguments={}),
        ]

        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert "Unknown tool" in results[0].error

    @pytest.mark.asyncio()
    async def test_execute_failing_tool(self):
        """Test execution when tool raises exception."""
        mock_llm = AsyncMock()

        # Create agent with failing tool
        class AgentWithFailingTool(TestSubAgent):
            TOOLS: ClassVar[list[type[BaseTool]]] = [FailingMockTool]

        agent = AgentWithFailingTool(mock_llm, {})

        tool_calls = [
            ToolCall(id="tc1", name="failing_tool", arguments={}),
        ]

        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0].success is False
        assert "Tool failed intentionally" in results[0].error


class TestSubAgentRun:
    """Tests for SubAgent.run()."""

    @pytest.mark.asyncio()
    async def test_run_timeout_handling(self):
        """Test that timeout is handled correctly."""

        class SlowAgent(TestSubAgent):
            TIMEOUT_SECONDS: ClassVar[int] = 0  # Immediate timeout

            async def _run_loop(self) -> SubAgentResult:
                import asyncio

                await asyncio.sleep(1)  # Will timeout
                return SubAgentResult(success=True, data={})

        mock_llm = AsyncMock()
        agent = SlowAgent(mock_llm, {})

        result = await agent.run()

        assert result.success is False
        assert "Timed out" in result.summary

    @pytest.mark.asyncio()
    async def test_run_exception_handling(self):
        """Test that exceptions are handled correctly."""

        class ErrorAgent(TestSubAgent):
            async def _run_loop(self) -> SubAgentResult:
                raise ValueError("Unexpected error")

        mock_llm = AsyncMock()
        agent = ErrorAgent(mock_llm, {})

        result = await agent.run()

        assert result.success is False
        assert "Unexpected error" in result.summary


class TestSubAgentRunLoop:
    """Tests for SubAgent._run_loop()."""

    @pytest.mark.asyncio()
    async def test_run_loop_completes_without_tool_calls(self):
        """Test that run loop completes when LLM returns no tool calls."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=MagicMock(content="Final response"))

        agent = TestSubAgent(mock_llm, {"data": "test"})

        result = await agent._run_loop()

        assert result.success is True
        assert result.data["response"] == "Final response"

    @pytest.mark.asyncio()
    async def test_run_loop_max_turns(self):
        """Test that run loop stops at MAX_TURNS."""

        class AlwaysCallsTools(TestSubAgent):
            MAX_TURNS: ClassVar[int] = 3

            async def _generate_response(self) -> Message:
                # Always return tool calls to keep looping
                return Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="tc", name="mock_tool", arguments={"input": "x"})],
                )

        mock_llm = AsyncMock()
        agent = AlwaysCallsTools(mock_llm, {})

        result = await agent._run_loop()

        assert result.success is False
        assert "max turns" in result.summary.lower()
        # Should have made exactly MAX_TURNS iterations
        # Each iteration adds 2 messages: assistant response and tool result
        # Plus 1 initial message
        assert len(agent.messages) == 1 + (2 * 3)  # initial + (response + tool) * MAX_TURNS


class TestSubAgentGenerateResponseRetry:
    """Retry/backoff tests for SubAgent._generate_response()."""

    @pytest.mark.asyncio()
    async def test_retries_on_rate_limit(self):
        """Rate limits are retried with backoff."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                LLMRateLimitError("openai", retry_after=0.0),
                MagicMock(content="Final response"),
            ]
        )

        agent = TestSubAgent(mock_llm, {"data": "test"})

        with patch("webui.services.agent.subagents.base.asyncio.sleep", new=AsyncMock()) as sleep_mock:
            msg = await agent._generate_response()

        assert msg.role == "assistant"
        assert msg.content == "Final response"
        assert mock_llm.generate.call_count == 2
        assert sleep_mock.await_count >= 1

    @pytest.mark.asyncio()
    async def test_retries_once_on_timeout_with_longer_timeout(self):
        """Timeouts are retried once with longer timeout."""
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                LLMTimeoutError("openai", timeout=60.0),
                MagicMock(content="Final response"),
            ]
        )

        agent = TestSubAgent(mock_llm, {"data": "test"})

        with patch("webui.services.agent.subagents.base.asyncio.sleep", new=AsyncMock()):
            msg = await agent._generate_response()

        assert msg.role == "assistant"
        assert msg.content == "Final response"
        assert mock_llm.generate.call_count == 2
        # Second call should use capped MAX_BACKGROUND_TIMEOUT (=120s)
        assert mock_llm.generate.call_args_list[1].kwargs["timeout"] == 120.0
