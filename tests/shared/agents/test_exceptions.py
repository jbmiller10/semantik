"""Tests for agent exceptions."""

import pickle

import pytest

from shared.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentInitializationError,
    AgentInterruptedError,
    AgentTimeoutError,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
    ToolDisabledError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_agent_error_is_base(self) -> None:
        """Test AgentError inherits from Exception."""
        assert issubclass(AgentError, Exception)

    def test_execution_errors_inherit_agent_error(self) -> None:
        """Test execution-related exceptions inherit from AgentError."""
        assert issubclass(AgentInitializationError, AgentError)
        assert issubclass(AgentExecutionError, AgentError)
        assert issubclass(AgentTimeoutError, AgentError)
        assert issubclass(AgentInterruptedError, AgentError)

    def test_tool_error_inherits_agent_error(self) -> None:
        """Test ToolError inherits from AgentError."""
        assert issubclass(ToolError, AgentError)

    def test_tool_specific_errors_inherit_tool_error(self) -> None:
        """Test tool-specific exceptions inherit from ToolError."""
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(ToolDisabledError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)

    def test_session_error_inherits_agent_error(self) -> None:
        """Test SessionError inherits from AgentError."""
        assert issubclass(SessionError, AgentError)

    def test_session_specific_errors_inherit_session_error(self) -> None:
        """Test session-specific exceptions inherit from SessionError."""
        assert issubclass(SessionNotFoundError, SessionError)
        assert issubclass(SessionExpiredError, SessionError)


class TestAgentError:
    """Tests for AgentError base class."""

    def test_message(self) -> None:
        """Test error message is set."""
        err = AgentError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"

    def test_auto_error_code(self) -> None:
        """Test error code is auto-generated from class name."""
        err = AgentError("Test error")
        assert err.error_code == "AGENT_ERROR"

    def test_custom_error_code(self) -> None:
        """Test custom error code overrides auto-generation."""
        err = AgentError("Test error", error_code="CUSTOM_CODE")
        assert err.error_code == "CUSTOM_CODE"

    def test_details_from_kwargs(self) -> None:
        """Test additional kwargs are stored in details."""
        err = AgentError("Test", foo="bar", count=42)
        assert err.details["foo"] == "bar"
        assert err.details["count"] == 42

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        err = AgentError("Test error", foo="bar")
        data = err.to_dict()

        assert data["error"] == "AGENT_ERROR"
        assert data["message"] == "Test error"
        assert data["details"]["foo"] == "bar"

    def test_to_dict_no_details(self) -> None:
        """Test to_dict without details doesn't include empty dict."""
        err = AgentError("Test error")
        data = err.to_dict()

        assert "details" not in data

    def test_can_be_caught_as_agent_error(self) -> None:
        """Test AgentError can be caught."""
        with pytest.raises(AgentError, match="Test"):
            raise AgentError("Test")


class TestAgentInitializationError:
    """Tests for AgentInitializationError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = AgentInitializationError("Init failed")
        assert err.error_code == "AGENT_INITIALIZATION_ERROR"

    def test_adapter_field(self) -> None:
        """Test adapter field is set."""
        err = AgentInitializationError("Init failed", adapter="claude")
        assert err.adapter == "claude"
        assert err.details["adapter"] == "claude"

    def test_adapter_none(self) -> None:
        """Test adapter can be None."""
        err = AgentInitializationError("Init failed")
        assert err.adapter is None
        assert "adapter" not in err.details


class TestAgentExecutionError:
    """Tests for AgentExecutionError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = AgentExecutionError("Execution failed")
        assert err.error_code == "AGENT_EXECUTION_ERROR"

    def test_adapter_and_cause(self) -> None:
        """Test adapter and cause fields."""
        err = AgentExecutionError(
            "Execution failed",
            adapter="claude",
            cause="API rate limit exceeded",
        )
        assert err.adapter == "claude"
        assert err.cause == "API rate limit exceeded"
        assert err.details["adapter"] == "claude"
        assert err.details["cause"] == "API rate limit exceeded"


class TestAgentTimeoutError:
    """Tests for AgentTimeoutError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = AgentTimeoutError("Timeout")
        assert err.error_code == "AGENT_TIMEOUT_ERROR"

    def test_timeout_seconds_field(self) -> None:
        """Test timeout_seconds field."""
        err = AgentTimeoutError("Timeout after 30s", timeout_seconds=30.0)
        assert err.timeout_seconds == 30.0
        assert err.details["timeout_seconds"] == 30.0


class TestAgentInterruptedError:
    """Tests for AgentInterruptedError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = AgentInterruptedError("User cancelled")
        assert err.error_code == "AGENT_INTERRUPTED_ERROR"


class TestToolError:
    """Tests for ToolError base class."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = ToolError("Tool error")
        assert err.error_code == "TOOL_ERROR"

    def test_tool_name_field(self) -> None:
        """Test tool_name field."""
        err = ToolError("Tool error", tool_name="search")
        assert err.tool_name == "search"
        assert err.details["tool_name"] == "search"

    def test_tool_name_none(self) -> None:
        """Test tool_name can be None."""
        err = ToolError("Tool error")
        assert err.tool_name is None
        assert "tool_name" not in err.details


class TestToolNotFoundError:
    """Tests for ToolNotFoundError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = ToolNotFoundError("Tool not found", tool_name="missing")
        assert err.error_code == "TOOL_NOT_FOUND_ERROR"

    def test_inherits_tool_name(self) -> None:
        """Test tool_name is inherited from ToolError."""
        err = ToolNotFoundError("Not found", tool_name="missing")
        assert err.tool_name == "missing"


class TestToolDisabledError:
    """Tests for ToolDisabledError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = ToolDisabledError("Tool disabled", tool_name="dangerous")
        assert err.error_code == "TOOL_DISABLED_ERROR"


class TestToolExecutionError:
    """Tests for ToolExecutionError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = ToolExecutionError("Execution failed")
        assert err.error_code == "TOOL_EXECUTION_ERROR"

    def test_cause_field(self) -> None:
        """Test cause field."""
        err = ToolExecutionError(
            "Search failed",
            tool_name="search",
            cause="Invalid query syntax",
        )
        assert err.cause == "Invalid query syntax"
        assert err.details["cause"] == "Invalid query syntax"
        assert err.tool_name == "search"


class TestSessionError:
    """Tests for SessionError base class."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = SessionError("Session error")
        assert err.error_code == "SESSION_ERROR"

    def test_session_id_field(self) -> None:
        """Test session_id field."""
        err = SessionError("Session error", session_id="sess-123")
        assert err.session_id == "sess-123"
        assert err.details["session_id"] == "sess-123"

    def test_session_id_none(self) -> None:
        """Test session_id can be None."""
        err = SessionError("Session error")
        assert err.session_id is None
        assert "session_id" not in err.details


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = SessionNotFoundError("Not found", session_id="missing")
        assert err.error_code == "SESSION_NOT_FOUND_ERROR"

    def test_inherits_session_id(self) -> None:
        """Test session_id is inherited from SessionError."""
        err = SessionNotFoundError("Not found", session_id="missing")
        assert err.session_id == "missing"


class TestSessionExpiredError:
    """Tests for SessionExpiredError."""

    def test_error_code(self) -> None:
        """Test error code is generated correctly."""
        err = SessionExpiredError("Expired", session_id="old-session")
        assert err.error_code == "SESSION_EXPIRED_ERROR"


class TestExceptionPickling:
    """Tests for exception pickling (Celery compatibility)."""

    def test_agent_error_picklable(self) -> None:
        """Test AgentError can be pickled and unpickled."""
        original = AgentError("Test error", foo="bar", count=42)
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        assert str(restored) == str(original)
        assert restored.error_code == original.error_code
        assert restored.details == original.details

    def test_agent_execution_error_picklable(self) -> None:
        """Test AgentExecutionError can be pickled."""
        original = AgentExecutionError(
            "Execution failed",
            adapter="claude",
            cause="API error",
        )
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        assert restored.adapter == original.adapter
        assert restored.cause == original.cause

    def test_tool_execution_error_picklable(self) -> None:
        """Test ToolExecutionError can be pickled."""
        original = ToolExecutionError(
            "Tool failed",
            tool_name="search",
            cause="Invalid args",
        )
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        assert restored.tool_name == original.tool_name
        assert restored.cause == original.cause

    def test_session_error_picklable(self) -> None:
        """Test SessionError can be pickled."""
        original = SessionNotFoundError("Not found", session_id="sess-123")
        pickled = pickle.dumps(original)
        restored = pickle.loads(pickled)

        assert restored.session_id == original.session_id

    def test_all_exceptions_picklable(self) -> None:
        """Test all exception types can be pickled."""
        exceptions = [
            AgentError("test"),
            AgentInitializationError("test", adapter="claude"),
            AgentExecutionError("test", adapter="claude", cause="error"),
            AgentTimeoutError("test", timeout_seconds=30.0),
            AgentInterruptedError("test"),
            ToolError("test", tool_name="search"),
            ToolNotFoundError("test", tool_name="missing"),
            ToolDisabledError("test", tool_name="disabled"),
            ToolExecutionError("test", tool_name="search", cause="error"),
            SessionError("test", session_id="sess"),
            SessionNotFoundError("test", session_id="missing"),
            SessionExpiredError("test", session_id="old"),
        ]

        for exc in exceptions:
            pickled = pickle.dumps(exc)
            restored = pickle.loads(pickled)
            assert str(restored) == str(exc)
            assert restored.error_code == exc.error_code


class TestExceptionCatching:
    """Tests for exception catching behavior."""

    def test_catch_tool_errors_as_agent_error(self) -> None:
        """Test tool errors can be caught as AgentError."""
        with pytest.raises(AgentError):
            raise ToolNotFoundError("Not found")

    def test_catch_session_errors_as_agent_error(self) -> None:
        """Test session errors can be caught as AgentError."""
        with pytest.raises(AgentError):
            raise SessionExpiredError("Expired")

    def test_catch_specific_tool_error(self) -> None:
        """Test catching specific tool error type."""
        with pytest.raises(ToolNotFoundError):
            raise ToolNotFoundError("Not found")

        # But not other tool errors
        with pytest.raises(ToolDisabledError):
            raise ToolDisabledError("Disabled")

    def test_catch_tool_error_base(self) -> None:
        """Test catching ToolError catches all tool exceptions."""
        for exc_class in [ToolNotFoundError, ToolDisabledError, ToolExecutionError]:
            with pytest.raises(ToolError):
                raise exc_class("test")
