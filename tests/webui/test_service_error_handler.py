"""Tests for the centralized service error handling decorator."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
    ValidationError,
)
from webui.utils.service_error_handler import DEFAULT_EXCEPTION_MAPPINGS, ExceptionMapping, handle_service_errors


class TestHandleServiceErrorsDecorator:
    """Tests for the @handle_service_errors decorator."""

    @pytest.mark.asyncio()
    async def test_successful_execution_returns_result(self) -> None:
        """Decorator should not interfere with successful execution."""

        @handle_service_errors
        async def endpoint() -> dict[str, str]:
            return {"status": "ok"}

        result = await endpoint()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio()
    async def test_entity_not_found_returns_404(self) -> None:
        """EntityNotFoundError should be converted to 404 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise EntityNotFoundError("Collection", "abc-123")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail.lower()

    @pytest.mark.asyncio()
    async def test_entity_already_exists_returns_409(self) -> None:
        """EntityAlreadyExistsError should be converted to 409 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise EntityAlreadyExistsError("Collection", "my-collection")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 409
        assert "already exists" in exc_info.value.detail.lower()

    @pytest.mark.asyncio()
    async def test_access_denied_returns_403(self) -> None:
        """AccessDeniedError should be converted to 403 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise AccessDeniedError("user-123", "Collection", "coll-456")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 403
        # Should NOT contain sensitive info like user ID
        assert "user-123" not in exc_info.value.detail
        assert "permission" in exc_info.value.detail.lower()

    @pytest.mark.asyncio()
    async def test_validation_error_returns_400(self) -> None:
        """ValidationError should be converted to 400 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise ValidationError("Invalid chunk size", field="chunk_size")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio()
    async def test_invalid_state_error_returns_409(self) -> None:
        """InvalidStateError should be converted to 409 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise InvalidStateError(
                "Cannot delete while processing",
                current_state="processing",
                allowed_states=["ready", "error"],
            )

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 409
        # Should include state info
        assert "processing" in exc_info.value.detail.lower()

    @pytest.mark.asyncio()
    async def test_value_error_returns_400(self) -> None:
        """ValueError should be converted to 400 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise ValueError("Invalid parameter value")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio()
    async def test_file_not_found_returns_404(self) -> None:
        """FileNotFoundError should be converted to 404 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise FileNotFoundError("/path/to/missing/file.txt")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 404
        # Path should be sanitized
        assert "/path/to/missing" not in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_permission_error_returns_403(self) -> None:
        """OS PermissionError should be converted to 403 HTTPException."""

        @handle_service_errors
        async def endpoint() -> None:
            raise PermissionError("Access denied to /secret/file")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 403
        # Path should be sanitized
        assert "/secret" not in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_http_exception_passes_through(self) -> None:
        """HTTPException should pass through unchanged."""

        @handle_service_errors
        async def endpoint() -> None:
            raise HTTPException(status_code=418, detail="I'm a teapot")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 418
        assert exc_info.value.detail == "I'm a teapot"

    @pytest.mark.asyncio()
    async def test_unhandled_exception_raises_original(self) -> None:
        """Unhandled exceptions should be re-raised for global handler."""

        class CustomUnhandledError(Exception):
            pass

        @handle_service_errors
        async def endpoint() -> None:
            raise CustomUnhandledError("Something custom went wrong")

        with pytest.raises(CustomUnhandledError):
            await endpoint()

    @pytest.mark.asyncio()
    async def test_excluded_exception_passes_through(self) -> None:
        """Excluded exception types should pass through without handling."""

        @handle_service_errors(exclude_exceptions={ValueError})
        async def endpoint() -> None:
            raise ValueError("Custom handling needed")

        with pytest.raises(ValueError, match="Custom handling needed"):
            await endpoint()

    @pytest.mark.asyncio()
    async def test_extra_mappings_take_precedence(self) -> None:
        """Extra mappings should be checked before default mappings."""
        custom_mapping = ExceptionMapping(
            exception_type=ValueError,
            status_code=422,  # Different from default 400
            sanitize=False,
        )

        @handle_service_errors(extra_mappings=[custom_mapping])
        async def endpoint() -> None:
            raise ValueError("Custom message")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 422
        assert "Custom message" in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_decorator_preserves_function_metadata(self) -> None:
        """Decorator should preserve function name and docstring."""

        @handle_service_errors
        async def my_endpoint_name() -> dict[str, str]:
            """My docstring."""
            return {}

        assert my_endpoint_name.__name__ == "my_endpoint_name"
        assert my_endpoint_name.__doc__ == """My docstring."""

    @pytest.mark.asyncio()
    async def test_decorator_works_without_parentheses(self) -> None:
        """Decorator should work when used without parentheses."""

        @handle_service_errors
        async def endpoint() -> dict[str, str]:
            return {"result": "value"}

        result = await endpoint()
        assert result == {"result": "value"}

    @pytest.mark.asyncio()
    async def test_decorator_works_with_empty_parentheses(self) -> None:
        """Decorator should work when used with empty parentheses."""

        @handle_service_errors()
        async def endpoint() -> dict[str, str]:
            return {"result": "value"}

        result = await endpoint()
        assert result == {"result": "value"}

    @pytest.mark.asyncio()
    async def test_logs_exception_with_correlation_id(self) -> None:
        """Decorator should log exceptions with correlation ID."""
        with (
            patch("webui.utils.service_error_handler.logger") as mock_logger,
            patch("webui.utils.service_error_handler.get_correlation_id", return_value="test-correlation-id"),
        ):

            @handle_service_errors
            async def endpoint() -> None:
                raise EntityNotFoundError("Document", "doc-123")

            with pytest.raises(HTTPException):
                await endpoint()

            # Verify logger was called with correlation ID
            mock_logger.warning.assert_called_once()
            call_kwargs = mock_logger.warning.call_args
            assert call_kwargs[1]["extra"]["correlation_id"] == "test-correlation-id"

    @pytest.mark.asyncio()
    async def test_handles_async_function_correctly(self) -> None:
        """Decorator should correctly handle async functions."""
        import asyncio

        @handle_service_errors
        async def async_endpoint() -> str:
            await asyncio.sleep(0.001)  # Simulate async operation
            return "async result"

        result = await async_endpoint()
        assert result == "async result"

    @pytest.mark.asyncio()
    async def test_exception_chaining_preserved(self) -> None:
        """HTTPException should have original exception as __cause__."""

        @handle_service_errors
        async def endpoint() -> None:
            raise EntityNotFoundError("Collection", "test-id")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, EntityNotFoundError)


class TestExceptionMapping:
    """Tests for ExceptionMapping configuration."""

    def test_default_mappings_cover_common_exceptions(self) -> None:
        """Default mappings should cover all common service exceptions."""
        exception_types = {m.exception_type for m in DEFAULT_EXCEPTION_MAPPINGS}

        assert EntityNotFoundError in exception_types
        assert EntityAlreadyExistsError in exception_types
        assert AccessDeniedError in exception_types
        assert ValidationError in exception_types
        assert InvalidStateError in exception_types
        assert ValueError in exception_types
        assert FileNotFoundError in exception_types
        assert PermissionError in exception_types

    def test_mapping_order_is_specific_to_general(self) -> None:
        """Mappings should be ordered from most specific to least specific."""
        # EntityNotFoundError should come before generic exceptions
        exception_types = [m.exception_type for m in DEFAULT_EXCEPTION_MAPPINGS]

        entity_idx = exception_types.index(EntityNotFoundError)
        value_idx = exception_types.index(ValueError)

        # Specific database exceptions should come before generic Python exceptions
        assert entity_idx < value_idx

    def test_all_mappings_have_appropriate_status_codes(self) -> None:
        """All mappings should have appropriate HTTP status codes."""
        for mapping in DEFAULT_EXCEPTION_MAPPINGS:
            assert 400 <= mapping.status_code < 600

            # Verify specific expected codes
            if mapping.exception_type == EntityNotFoundError:
                assert mapping.status_code == 404
            elif mapping.exception_type == EntityAlreadyExistsError:
                assert mapping.status_code == 409
            elif mapping.exception_type == AccessDeniedError:
                assert mapping.status_code == 403
            elif mapping.exception_type == InvalidStateError:
                assert mapping.status_code == 409


class TestSanitization:
    """Tests for error message sanitization."""

    @pytest.mark.asyncio()
    async def test_user_id_not_in_error_message(self) -> None:
        """User IDs should not appear in error messages."""

        @handle_service_errors
        async def endpoint() -> None:
            raise AccessDeniedError("sensitive-user-123", "Collection", "coll-id")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert "sensitive-user-123" not in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_entity_id_not_in_generic_error(self) -> None:
        """Entity IDs should use generic messages for security-sensitive exceptions."""

        @handle_service_errors
        async def endpoint() -> None:
            raise AccessDeniedError("user", "Collection", "coll-abc-sensitive-id")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        # AccessDeniedError is fully sanitized
        assert "coll-abc-sensitive-id" not in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_file_paths_sanitized(self) -> None:
        """File paths should be sanitized in error messages."""

        @handle_service_errors
        async def endpoint() -> None:
            raise FileNotFoundError("/home/user/secrets/password.txt")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert "/home/user/secrets" not in exc_info.value.detail
        assert "password.txt" not in exc_info.value.detail

    @pytest.mark.asyncio()
    async def test_entity_type_preserved_in_not_found(self) -> None:
        """Entity type (but not ID) should be preserved in not-found errors."""

        @handle_service_errors
        async def endpoint() -> None:
            raise EntityNotFoundError("Collection", "secret-uuid")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        # Entity type is useful for the client
        assert "Collection" in exc_info.value.detail
        # But UUID should not be exposed
        assert "secret-uuid" not in exc_info.value.detail


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    @pytest.mark.asyncio()
    async def test_exception_with_none_attributes(self) -> None:
        """Handler should work with exceptions that have None attributes."""

        @handle_service_errors
        async def endpoint() -> None:
            exc = InvalidStateError("Invalid state")
            exc.current_state = None
            exc.allowed_states = None
            raise exc

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        assert exc_info.value.status_code == 409

    @pytest.mark.asyncio()
    async def test_function_with_args_and_kwargs(self) -> None:
        """Decorator should work with functions that have args and kwargs."""

        @handle_service_errors
        async def endpoint(arg1: str, arg2: int, *, kwarg1: bool = False) -> dict[str, Any]:
            return {"arg1": arg1, "arg2": arg2, "kwarg1": kwarg1}

        result = await endpoint("test", 42, kwarg1=True)
        assert result == {"arg1": "test", "arg2": 42, "kwarg1": True}

    @pytest.mark.asyncio()
    async def test_inheritance_matching(self) -> None:
        """Mappings should match exception subclasses."""

        class CustomNotFoundError(EntityNotFoundError):
            pass

        @handle_service_errors
        async def endpoint() -> None:
            raise CustomNotFoundError("CustomEntity", "custom-id")

        with pytest.raises(HTTPException) as exc_info:
            await endpoint()

        # Should match EntityNotFoundError mapping
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio()
    async def test_log_unhandled_false_suppresses_logging(self) -> None:
        """log_unhandled=False should suppress logging for unhandled exceptions."""

        class UnhandledOperationError(Exception):
            pass

        with patch("webui.utils.service_error_handler.logger") as mock_logger:

            @handle_service_errors(log_unhandled=False)
            async def endpoint() -> None:
                raise UnhandledOperationError("Not logged")

            with pytest.raises(UnhandledOperationError):
                await endpoint()

            # Exception should not be logged
            mock_logger.exception.assert_not_called()
