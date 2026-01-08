"""Prometheus metrics for agent operations.

This module provides observability into agent execution, tool usage,
and session management.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

from shared.metrics.prometheus import registry

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    # Metrics
    "AGENT_EXECUTIONS_TOTAL",
    "AGENT_EXECUTION_DURATION",
    "AGENT_TOKENS_TOTAL",
    "AGENT_TOOL_CALLS_TOTAL",
    "AGENT_TOOL_DURATION",
    "AGENT_ACTIVE_SESSIONS",
    "AGENT_SESSIONS_CREATED_TOTAL",
    "AGENT_ERRORS_TOTAL",
    # Helper functions
    "record_execution",
    "record_tokens",
    "record_tool_call",
    "record_session_created",
    "record_error",
    "update_active_sessions",
    # Context managers
    "timed_execution",
    "timed_tool_call",
]


# Agent Execution Metrics
AGENT_EXECUTIONS_TOTAL = Counter(
    "semantik_agent_executions_total",
    "Total agent execution attempts",
    ["agent_id", "status"],  # status: success, error, interrupted, timeout
    registry=registry,
)

AGENT_EXECUTION_DURATION = Histogram(
    "semantik_agent_execution_duration_seconds",
    "Agent execution duration in seconds",
    ["agent_id"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    registry=registry,
)

# Token Usage Metrics
AGENT_TOKENS_TOTAL = Counter(
    "semantik_agent_tokens_total",
    "Total tokens used by agents",
    ["agent_id", "direction"],  # direction: input, output
    registry=registry,
)

# Tool Execution Metrics
AGENT_TOOL_CALLS_TOTAL = Counter(
    "semantik_agent_tool_calls_total",
    "Total tool calls by agents",
    ["tool_name", "status"],  # status: success, error
    registry=registry,
)

AGENT_TOOL_DURATION = Histogram(
    "semantik_agent_tool_duration_seconds",
    "Tool execution duration in seconds",
    ["tool_name"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    registry=registry,
)

# Session Metrics
AGENT_ACTIVE_SESSIONS = Gauge(
    "semantik_agent_sessions_active",
    "Number of currently active agent sessions",
    registry=registry,
)

AGENT_SESSIONS_CREATED_TOTAL = Counter(
    "semantik_agent_sessions_created_total",
    "Total agent sessions created",
    ["agent_id"],
    registry=registry,
)

# Error Metrics
AGENT_ERRORS_TOTAL = Counter(
    "semantik_agent_errors_total",
    "Total agent errors by type",
    ["agent_id", "error_type"],  # error_type: execution, timeout, tool, session
    registry=registry,
)


def record_execution(
    agent_id: str,
    status: str,
    duration: float,
) -> None:
    """Record an agent execution attempt.

    Args:
        agent_id: Identifier of the agent plugin
        status: Execution status (success, error, interrupted, timeout)
        duration: Time taken in seconds
    """
    AGENT_EXECUTIONS_TOTAL.labels(agent_id=agent_id, status=status).inc()
    AGENT_EXECUTION_DURATION.labels(agent_id=agent_id).observe(duration)


def record_tokens(
    agent_id: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Record token usage from an agent execution.

    Args:
        agent_id: Identifier of the agent plugin
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
    """
    if input_tokens > 0:
        AGENT_TOKENS_TOTAL.labels(agent_id=agent_id, direction="input").inc(input_tokens)
    if output_tokens > 0:
        AGENT_TOKENS_TOTAL.labels(agent_id=agent_id, direction="output").inc(output_tokens)


def record_tool_call(
    tool_name: str,
    status: str,
    duration: float,
) -> None:
    """Record a tool call by an agent.

    Args:
        tool_name: Name of the tool called
        status: Call status (success, error)
        duration: Time taken in seconds
    """
    AGENT_TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status=status).inc()
    AGENT_TOOL_DURATION.labels(tool_name=tool_name).observe(duration)


def record_session_created(agent_id: str) -> None:
    """Record a new session creation.

    Args:
        agent_id: Identifier of the agent plugin
    """
    AGENT_SESSIONS_CREATED_TOTAL.labels(agent_id=agent_id).inc()


def record_error(agent_id: str, error_type: str) -> None:
    """Record an agent error.

    Args:
        agent_id: Identifier of the agent plugin
        error_type: Type of error (execution, timeout, tool, session)
    """
    AGENT_ERRORS_TOTAL.labels(agent_id=agent_id, error_type=error_type).inc()


def update_active_sessions(count: int) -> None:
    """Update the count of active sessions.

    Args:
        count: Current number of active sessions
    """
    AGENT_ACTIVE_SESSIONS.set(count)


@contextmanager
def timed_execution(agent_id: str) -> Generator[dict[str, float], None, None]:
    """Context manager for timing agent executions.

    Yields a dict that will contain 'duration' after the context exits.
    Records the execution metric automatically on exit.

    Args:
        agent_id: Identifier of the agent plugin

    Yields:
        Dict with 'duration' key set after context exits

    Example:
        with timed_execution("claude-agent") as timing:
            # execute agent
            ...
        print(f"Took {timing['duration']} seconds")
    """
    timing: dict[str, float] = {}
    status = "success"
    start = time.perf_counter()
    try:
        yield timing
    except Exception as e:
        # Determine error type from exception
        error_name = type(e).__name__
        if "Timeout" in error_name:
            status = "timeout"
        elif "Interrupt" in error_name:
            status = "interrupted"
        else:
            status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        timing["duration"] = duration
        record_execution(agent_id, status, duration)


@contextmanager
def timed_tool_call(tool_name: str) -> Generator[dict[str, float], None, None]:
    """Context manager for timing tool calls.

    Yields a dict that will contain 'duration' after the context exits.
    Records the tool call metric automatically on exit.

    Args:
        tool_name: Name of the tool being called

    Yields:
        Dict with 'duration' key set after context exits

    Example:
        with timed_tool_call("semantic_search") as timing:
            # call tool
            result = await tool.execute(args)
        print(f"Tool took {timing['duration']} seconds")
    """
    timing: dict[str, float] = {}
    status = "success"
    start = time.perf_counter()
    try:
        yield timing
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.perf_counter() - start
        timing["duration"] = duration
        record_tool_call(tool_name, status, duration)
