"""Tests for agent metrics module.

Tests the Prometheus metrics collection for agent operations, tool calls,
and session management.
"""

from __future__ import annotations

import pytest

from shared.agents.metrics import (
    AGENT_ACTIVE_SESSIONS,
    AGENT_ERRORS_TOTAL,
    AGENT_EXECUTION_DURATION,
    AGENT_EXECUTIONS_TOTAL,
    AGENT_SESSIONS_CREATED_TOTAL,
    AGENT_TOKENS_TOTAL,
    AGENT_TOOL_CALLS_TOTAL,
    AGENT_TOOL_DURATION,
    record_error,
    record_execution,
    record_session_created,
    record_tokens,
    record_tool_call,
    timed_execution,
    timed_tool_call,
    update_active_sessions,
)


class TestRecordExecution:
    """Tests for record_execution function."""

    def test_records_success_status(self) -> None:
        """Should increment execution counter with success status."""
        # Get current value
        before = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="test-agent",
            status="success",
        )._value.get()

        record_execution("test-agent", "success", 1.5)

        after = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="test-agent",
            status="success",
        )._value.get()

        assert after == before + 1

    def test_records_error_status(self) -> None:
        """Should increment execution counter with error status."""
        before = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="test-agent",
            status="error",
        )._value.get()

        record_execution("test-agent", "error", 0.5)

        after = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="test-agent",
            status="error",
        )._value.get()

        assert after == before + 1

    def test_records_duration_histogram(self) -> None:
        """Should record execution duration in histogram."""
        # Histogram sum should increase by duration
        before = AGENT_EXECUTION_DURATION.labels(agent_id="test-agent")._sum.get()

        record_execution("test-agent", "success", 2.5)

        after = AGENT_EXECUTION_DURATION.labels(agent_id="test-agent")._sum.get()

        assert after >= before + 2.5


class TestRecordTokens:
    """Tests for record_tokens function."""

    def test_records_input_tokens(self) -> None:
        """Should increment input token counter."""
        before = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent",
            direction="input",
        )._value.get()

        record_tokens("test-agent", input_tokens=100, output_tokens=0)

        after = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent",
            direction="input",
        )._value.get()

        assert after == before + 100

    def test_records_output_tokens(self) -> None:
        """Should increment output token counter."""
        before = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent",
            direction="output",
        )._value.get()

        record_tokens("test-agent", input_tokens=0, output_tokens=50)

        after = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent",
            direction="output",
        )._value.get()

        assert after == before + 50

    def test_records_both_token_types(self) -> None:
        """Should record both input and output tokens."""
        before_input = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent2",
            direction="input",
        )._value.get()
        before_output = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent2",
            direction="output",
        )._value.get()

        record_tokens("test-agent2", input_tokens=150, output_tokens=75)

        after_input = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent2",
            direction="input",
        )._value.get()
        after_output = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent2",
            direction="output",
        )._value.get()

        assert after_input == before_input + 150
        assert after_output == before_output + 75

    def test_skips_zero_tokens(self) -> None:
        """Should not increment counters when tokens are zero."""
        before_input = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent3",
            direction="input",
        )._value.get()
        before_output = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent3",
            direction="output",
        )._value.get()

        record_tokens("test-agent3", input_tokens=0, output_tokens=0)

        after_input = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent3",
            direction="input",
        )._value.get()
        after_output = AGENT_TOKENS_TOTAL.labels(
            agent_id="test-agent3",
            direction="output",
        )._value.get()

        assert after_input == before_input
        assert after_output == before_output


class TestRecordToolCall:
    """Tests for record_tool_call function."""

    def test_records_success_tool_call(self) -> None:
        """Should increment tool call counter with success status."""
        before = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="test_tool",
            status="success",
        )._value.get()

        record_tool_call("test_tool", "success", 0.1)

        after = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="test_tool",
            status="success",
        )._value.get()

        assert after == before + 1

    def test_records_error_tool_call(self) -> None:
        """Should increment tool call counter with error status."""
        before = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="test_tool",
            status="error",
        )._value.get()

        record_tool_call("test_tool", "error", 0.5)

        after = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="test_tool",
            status="error",
        )._value.get()

        assert after == before + 1

    def test_records_tool_duration(self) -> None:
        """Should record tool execution duration."""
        before = AGENT_TOOL_DURATION.labels(tool_name="test_tool")._sum.get()

        record_tool_call("test_tool", "success", 0.25)

        after = AGENT_TOOL_DURATION.labels(tool_name="test_tool")._sum.get()

        assert after >= before + 0.25


class TestRecordSessionCreated:
    """Tests for record_session_created function."""

    def test_records_session_creation(self) -> None:
        """Should increment session created counter."""
        before = AGENT_SESSIONS_CREATED_TOTAL.labels(
            agent_id="test-agent",
        )._value.get()

        record_session_created("test-agent")

        after = AGENT_SESSIONS_CREATED_TOTAL.labels(
            agent_id="test-agent",
        )._value.get()

        assert after == before + 1


class TestRecordError:
    """Tests for record_error function."""

    def test_records_execution_error(self) -> None:
        """Should increment error counter for execution errors."""
        before = AGENT_ERRORS_TOTAL.labels(
            agent_id="test-agent",
            error_type="execution",
        )._value.get()

        record_error("test-agent", "execution")

        after = AGENT_ERRORS_TOTAL.labels(
            agent_id="test-agent",
            error_type="execution",
        )._value.get()

        assert after == before + 1

    def test_records_timeout_error(self) -> None:
        """Should increment error counter for timeout errors."""
        before = AGENT_ERRORS_TOTAL.labels(
            agent_id="test-agent",
            error_type="timeout",
        )._value.get()

        record_error("test-agent", "timeout")

        after = AGENT_ERRORS_TOTAL.labels(
            agent_id="test-agent",
            error_type="timeout",
        )._value.get()

        assert after == before + 1


class TestUpdateActiveSessions:
    """Tests for update_active_sessions function."""

    def test_sets_active_session_count(self) -> None:
        """Should set the active sessions gauge."""
        update_active_sessions(5)

        value = AGENT_ACTIVE_SESSIONS._value.get()
        assert value == 5

    def test_updates_active_session_count(self) -> None:
        """Should update the active sessions gauge."""
        update_active_sessions(10)
        update_active_sessions(3)

        value = AGENT_ACTIVE_SESSIONS._value.get()
        assert value == 3


class TestTimedExecution:
    """Tests for timed_execution context manager."""

    def test_records_successful_execution(self) -> None:
        """Should record successful execution with duration."""
        before = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="success",
        )._value.get()

        with timed_execution("timed-agent") as timing:
            pass  # Simulate work

        after = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="success",
        )._value.get()

        assert after == before + 1
        assert "duration" in timing
        assert timing["duration"] >= 0

    def test_records_error_execution(self) -> None:
        """Should record error execution when exception raised."""
        before = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="error",
        )._value.get()

        with pytest.raises(ValueError, match="Test error"):
            with timed_execution("timed-agent"):
                raise ValueError("Test error")

        after = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="error",
        )._value.get()

        assert after == before + 1

    def test_records_timeout_for_timeout_error(self) -> None:
        """Should record timeout status for timeout-like exceptions."""
        before = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="timeout",
        )._value.get()

        class AgentTimeoutError(Exception):
            pass

        with pytest.raises(AgentTimeoutError):
            with timed_execution("timed-agent"):
                raise AgentTimeoutError("Timed out")

        after = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="timeout",
        )._value.get()

        assert after == before + 1

    def test_records_interrupted_for_interrupt_error(self) -> None:
        """Should record interrupted status for interrupt exceptions."""
        before = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="interrupted",
        )._value.get()

        class AgentInterruptedError(Exception):
            pass

        with pytest.raises(AgentInterruptedError):
            with timed_execution("timed-agent"):
                raise AgentInterruptedError("Interrupted")

        after = AGENT_EXECUTIONS_TOTAL.labels(
            agent_id="timed-agent",
            status="interrupted",
        )._value.get()

        assert after == before + 1


class TestTimedToolCall:
    """Tests for timed_tool_call context manager."""

    def test_records_successful_tool_call(self) -> None:
        """Should record successful tool call with duration."""
        before = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="timed-tool",
            status="success",
        )._value.get()

        with timed_tool_call("timed-tool") as timing:
            pass  # Simulate work

        after = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="timed-tool",
            status="success",
        )._value.get()

        assert after == before + 1
        assert "duration" in timing
        assert timing["duration"] >= 0

    def test_records_error_tool_call(self) -> None:
        """Should record error tool call when exception raised."""
        before = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="timed-tool",
            status="error",
        )._value.get()

        with pytest.raises(RuntimeError):
            with timed_tool_call("timed-tool"):
                raise RuntimeError("Test error")

        after = AGENT_TOOL_CALLS_TOTAL.labels(
            tool_name="timed-tool",
            status="error",
        )._value.get()

        assert after == before + 1

    def test_measures_duration(self) -> None:
        """Should measure execution duration."""
        import time

        with timed_tool_call("timed-tool") as timing:
            time.sleep(0.01)  # 10ms

        assert timing["duration"] >= 0.01
