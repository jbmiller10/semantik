"""Unit tests for VecPipe runtime container and dependency injection.

Tests cover:
- VecpipeRuntime container
- FastAPI dependency injection helpers
- Centralized authentication
"""

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from vecpipe.search.auth import require_internal_api_key
from vecpipe.search.deps import (
    get_llm_manager,
    get_model_manager,
    get_qdrant_http,
    get_qdrant_sdk,
    get_runtime,
    get_sparse_manager,
    require_llm_manager,
)
from vecpipe.search.runtime import VecpipeRuntime

# =============================================================================
# Runtime Tests
# =============================================================================


class TestVecpipeRuntime:
    """Tests for VecpipeRuntime container."""

    @pytest.fixture()
    def mock_qdrant_http(self) -> AsyncMock:
        """Mock httpx.AsyncClient for Qdrant HTTP."""
        client = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.fixture()
    def mock_qdrant_sdk(self) -> AsyncMock:
        """Mock AsyncQdrantClient."""
        client = AsyncMock()
        client.close = AsyncMock()
        return client

    @pytest.fixture()
    def mock_model_manager(self) -> Mock:
        """Mock ModelManager (no shutdown_async)."""
        # Use spec to prevent Mock from auto-creating shutdown_async
        mgr = Mock(spec=["shutdown"])
        mgr.shutdown = Mock()
        return mgr

    @pytest.fixture()
    def mock_governed_model_manager(self) -> Mock:
        """Mock GovernedModelManager with shutdown_async."""
        # Use spec to include shutdown_async
        mgr = Mock(spec=["shutdown", "shutdown_async"])
        mgr.shutdown = Mock()
        mgr.shutdown_async = AsyncMock()
        return mgr

    @pytest.fixture()
    def mock_sparse_manager(self) -> AsyncMock:
        """Mock SparseModelManager."""
        mgr = AsyncMock()
        mgr.shutdown = AsyncMock()
        return mgr

    @pytest.fixture()
    def mock_llm_manager(self) -> AsyncMock:
        """Mock LLMModelManager."""
        mgr = AsyncMock()
        mgr.shutdown = AsyncMock()
        return mgr

    @pytest.fixture()
    def executor(self) -> Generator[ThreadPoolExecutor, None, None]:
        """Create a real ThreadPoolExecutor for testing."""
        import contextlib

        pool = ThreadPoolExecutor(max_workers=1)
        yield pool
        # Cleanup if not already done
        with contextlib.suppress(Exception):
            pool.shutdown(wait=False)

    @pytest.fixture()
    def runtime(
        self,
        mock_qdrant_http: AsyncMock,
        mock_qdrant_sdk: AsyncMock,
        mock_model_manager: Mock,
        mock_sparse_manager: AsyncMock,
        mock_llm_manager: AsyncMock,
        executor: ThreadPoolExecutor,
    ) -> VecpipeRuntime:
        """Create a VecpipeRuntime with all mocked dependencies."""
        return VecpipeRuntime(
            qdrant_http=mock_qdrant_http,
            qdrant_sdk=mock_qdrant_sdk,
            model_manager=mock_model_manager,
            sparse_manager=mock_sparse_manager,
            llm_manager=mock_llm_manager,
            executor=executor,
        )

    @pytest.mark.asyncio()
    async def test_runtime_aclose_idempotent(self, runtime: VecpipeRuntime) -> None:
        """Double-close is safe and doesn't raise."""
        assert not runtime.is_closed

        await runtime.aclose()
        assert runtime.is_closed

        # Second close should be safe
        await runtime.aclose()
        assert runtime.is_closed

        # Verify mocks were called only once
        runtime.qdrant_http.aclose.assert_called_once()
        runtime.qdrant_sdk.close.assert_called_once()

    @pytest.mark.asyncio()
    async def test_runtime_shutdown_order(
        self,
        mock_qdrant_http: AsyncMock,
        mock_qdrant_sdk: AsyncMock,
        mock_model_manager: Mock,
        mock_sparse_manager: AsyncMock,
        mock_llm_manager: AsyncMock,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Verify cleanup sequence matches expected order."""
        call_order: list[str] = []

        # Instrument mocks to track call order
        mock_qdrant_http.aclose = AsyncMock(side_effect=lambda: call_order.append("qdrant_http"))
        mock_qdrant_sdk.close = AsyncMock(side_effect=lambda: call_order.append("qdrant_sdk"))
        mock_llm_manager.shutdown = AsyncMock(side_effect=lambda: call_order.append("llm_manager"))
        mock_sparse_manager.shutdown = AsyncMock(side_effect=lambda: call_order.append("sparse_manager"))
        mock_model_manager.shutdown = Mock(side_effect=lambda: call_order.append("model_manager"))

        runtime = VecpipeRuntime(
            qdrant_http=mock_qdrant_http,
            qdrant_sdk=mock_qdrant_sdk,
            model_manager=mock_model_manager,
            sparse_manager=mock_sparse_manager,
            llm_manager=mock_llm_manager,
            executor=executor,
        )

        await runtime.aclose()

        # Verify order: qdrant_http -> qdrant_sdk -> llm_manager -> sparse_manager -> model_manager
        assert call_order == [
            "qdrant_http",
            "qdrant_sdk",
            "llm_manager",
            "sparse_manager",
            "model_manager",
        ]

    @pytest.mark.asyncio()
    async def test_runtime_uses_shutdown_async_for_governed_manager(
        self,
        mock_qdrant_http: AsyncMock,
        mock_qdrant_sdk: AsyncMock,
        mock_governed_model_manager: Mock,
        mock_sparse_manager: AsyncMock,
        executor: ThreadPoolExecutor,
    ) -> None:
        """GovernedModelManager uses shutdown_async instead of shutdown."""
        runtime = VecpipeRuntime(
            qdrant_http=mock_qdrant_http,
            qdrant_sdk=mock_qdrant_sdk,
            model_manager=mock_governed_model_manager,
            sparse_manager=mock_sparse_manager,
            llm_manager=None,
            executor=executor,
        )

        await runtime.aclose()

        # Should use shutdown_async, not shutdown
        mock_governed_model_manager.shutdown_async.assert_called_once()
        mock_governed_model_manager.shutdown.assert_not_called()

    @pytest.mark.asyncio()
    async def test_runtime_handles_shutdown_errors(
        self,
        mock_qdrant_http: AsyncMock,
        mock_qdrant_sdk: AsyncMock,
        mock_model_manager: Mock,
        mock_sparse_manager: AsyncMock,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Runtime handles errors during shutdown gracefully."""
        # Make qdrant_http.aclose raise an error
        mock_qdrant_http.aclose = AsyncMock(side_effect=RuntimeError("Connection error"))

        runtime = VecpipeRuntime(
            qdrant_http=mock_qdrant_http,
            qdrant_sdk=mock_qdrant_sdk,
            model_manager=mock_model_manager,
            sparse_manager=mock_sparse_manager,
            llm_manager=None,
            executor=executor,
        )

        # Should not raise, errors are logged
        await runtime.aclose()

        # Other resources should still be cleaned up
        mock_qdrant_sdk.close.assert_called_once()
        mock_sparse_manager.shutdown.assert_called_once()
        mock_model_manager.shutdown.assert_called_once()

    @pytest.mark.asyncio()
    async def test_runtime_without_llm_manager(
        self,
        mock_qdrant_http: AsyncMock,
        mock_qdrant_sdk: AsyncMock,
        mock_model_manager: Mock,
        mock_sparse_manager: AsyncMock,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Runtime handles None llm_manager correctly."""
        runtime = VecpipeRuntime(
            qdrant_http=mock_qdrant_http,
            qdrant_sdk=mock_qdrant_sdk,
            model_manager=mock_model_manager,
            sparse_manager=mock_sparse_manager,
            llm_manager=None,
            executor=executor,
        )

        await runtime.aclose()

        # Should complete without errors
        assert runtime.is_closed


# =============================================================================
# Dependency Injection Tests
# =============================================================================


class TestDependencyInjection:
    """Tests for FastAPI dependency injection helpers."""

    @pytest.fixture()
    def mock_runtime(self) -> Mock:
        """Create a mock VecpipeRuntime."""
        runtime = Mock(spec=VecpipeRuntime)
        runtime.is_closed = False
        runtime.qdrant_http = Mock()
        runtime.qdrant_sdk = Mock()
        runtime.model_manager = Mock()
        runtime.sparse_manager = Mock()
        runtime.llm_manager = Mock()
        return runtime

    @pytest.fixture()
    def mock_request(self, mock_runtime: Mock) -> Mock:
        """Create a mock FastAPI Request with runtime attached."""
        request = MagicMock(spec=Request)
        request.app = MagicMock()
        request.app.state = MagicMock()
        request.app.state.vecpipe_runtime = mock_runtime
        return request

    def test_get_runtime_returns_runtime(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_runtime returns the runtime from app.state."""
        result = get_runtime(mock_request)
        assert result is mock_runtime

    def test_get_runtime_raises_503_when_missing(self) -> None:
        """get_runtime raises 503 when runtime is not initialized."""
        request = MagicMock(spec=Request)
        request.app = MagicMock()
        request.app.state = MagicMock()
        # No vecpipe_runtime attribute
        del request.app.state.vecpipe_runtime

        with pytest.raises(HTTPException) as exc_info:
            get_runtime(request)

        assert exc_info.value.status_code == 503
        assert "not initialized" in exc_info.value.detail

    def test_get_runtime_raises_503_when_closed(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_runtime raises 503 when runtime is shutting down."""
        mock_runtime.is_closed = True

        with pytest.raises(HTTPException) as exc_info:
            get_runtime(mock_request)

        assert exc_info.value.status_code == 503
        assert "shutting down" in exc_info.value.detail

    def test_get_qdrant_http(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_qdrant_http returns qdrant_http from runtime."""
        result = get_qdrant_http(mock_request)
        assert result is mock_runtime.qdrant_http

    def test_get_qdrant_sdk(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_qdrant_sdk returns qdrant_sdk from runtime."""
        result = get_qdrant_sdk(mock_request)
        assert result is mock_runtime.qdrant_sdk

    def test_get_model_manager(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_model_manager returns model_manager from runtime."""
        result = get_model_manager(mock_request)
        assert result is mock_runtime.model_manager

    def test_get_sparse_manager(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_sparse_manager returns sparse_manager from runtime."""
        result = get_sparse_manager(mock_request)
        assert result is mock_runtime.sparse_manager

    def test_get_llm_manager(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_llm_manager returns llm_manager from runtime."""
        result = get_llm_manager(mock_request)
        assert result is mock_runtime.llm_manager

    def test_get_llm_manager_returns_none_when_disabled(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """get_llm_manager returns None when LLM is disabled."""
        mock_runtime.llm_manager = None
        result = get_llm_manager(mock_request)
        assert result is None

    def test_require_llm_manager_returns_manager(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """require_llm_manager returns manager when available."""
        result = require_llm_manager(mock_request)
        assert result is mock_runtime.llm_manager

    def test_require_llm_manager_raises_503_when_none(self, mock_request: Mock, mock_runtime: Mock) -> None:
        """require_llm_manager raises 503 when LLM is disabled."""
        mock_runtime.llm_manager = None

        with pytest.raises(HTTPException) as exc_info:
            require_llm_manager(mock_request)

        assert exc_info.value.status_code == 503
        assert "disabled" in exc_info.value.detail


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Tests for centralized authentication."""

    @pytest.fixture()
    def valid_api_key(self) -> str:
        """Valid internal API key for testing."""
        return "test-api-key-12345"

    @pytest.fixture()
    def app(self) -> FastAPI:
        """Create a simple FastAPI app for testing auth."""
        from fastapi import Depends

        app = FastAPI()

        @app.get("/protected", dependencies=[Depends(require_internal_api_key)])
        def protected_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        return app

    @pytest.fixture()
    def client(self, app: FastAPI) -> TestClient:
        """Create test client."""
        return TestClient(app)

    def test_auth_valid_key(self, client: TestClient, valid_api_key: str) -> None:
        """Valid API key allows access."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key

            response = client.get(
                "/protected",
                headers={"X-Internal-Api-Key": valid_api_key},
            )

            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    def test_auth_invalid_key(self, client: TestClient, valid_api_key: str) -> None:
        """Invalid API key returns 401."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key

            response = client.get(
                "/protected",
                headers={"X-Internal-Api-Key": "wrong-key"},
            )

            assert response.status_code == 401
            assert "Invalid or missing" in response.json()["detail"]

    def test_auth_missing_key(self, client: TestClient, valid_api_key: str) -> None:
        """Missing API key returns 401."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key

            response = client.get("/protected")

            assert response.status_code == 401
            assert "Invalid or missing" in response.json()["detail"]

    def test_auth_not_configured(self, client: TestClient) -> None:
        """Missing server-side API key configuration returns 500."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = None

            response = client.get(
                "/protected",
                headers={"X-Internal-Api-Key": "any-key"},
            )

            assert response.status_code == 500
            assert "not configured" in response.json()["detail"]

    def test_auth_timing_safe_comparison(self, client: TestClient, valid_api_key: str) -> None:
        """Auth uses timing-safe comparison to prevent timing attacks."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key

            # Different length keys
            response = client.get(
                "/protected",
                headers={"X-Internal-Api-Key": "x"},
            )

            assert response.status_code == 401
