"""Unit tests for LLM API endpoints.

Tests cover:
- /llm/generate endpoint
- /llm/models endpoint
- /llm/models/load endpoint
- /llm/health endpoint
- Error handling (503, 507, 500)
- Authentication requirements
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vecpipe.search.llm_api import router

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def mock_llm_manager() -> Mock:
    """Mock LLMModelManager."""
    manager = Mock()
    manager._models = {"Qwen/Qwen2.5-1.5B-Instruct:int8": ("mock_model", "mock_tokenizer")}
    manager._governor = Mock()

    # Mock generate method
    async def mock_generate(**_kwargs: Any) -> dict[str, Any]:
        return {
            "content": "Generated response",
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }

    manager.generate = AsyncMock(side_effect=mock_generate)

    # Mock _ensure_model_loaded
    async def mock_ensure_loaded(_model_name: str, _quantization: str) -> tuple[Mock, Mock]:
        return Mock(), Mock()

    manager._ensure_model_loaded = AsyncMock(side_effect=mock_ensure_loaded)

    return manager


@pytest.fixture()
def app_with_llm_manager(mock_llm_manager: Mock) -> FastAPI:
    """Create FastAPI app with mocked LLM manager."""
    app = FastAPI()
    app.include_router(router)
    app.state.vecpipe_runtime = Mock(is_closed=False, llm_manager=mock_llm_manager)

    return app


@pytest.fixture()
def app_without_llm_manager() -> FastAPI:
    """Create FastAPI app without LLM manager (disabled)."""
    app = FastAPI()
    app.include_router(router)
    app.state.vecpipe_runtime = Mock(is_closed=False, llm_manager=None)

    return app


@pytest.fixture()
def client_with_manager(app_with_llm_manager: FastAPI) -> TestClient:
    """Test client with LLM manager available."""
    return TestClient(app_with_llm_manager)


@pytest.fixture()
def client_without_manager(app_without_llm_manager: FastAPI) -> TestClient:
    """Test client without LLM manager."""
    return TestClient(app_without_llm_manager)


@pytest.fixture()
def valid_api_key() -> str:
    """Valid internal API key for testing."""
    return "test-api-key-12345"


@pytest.fixture()
def auth_headers(valid_api_key: str) -> dict[str, str]:
    """Headers with valid API key."""
    return {"X-Internal-Api-Key": valid_api_key}


# =============================================================================
# Test Classes
# =============================================================================


class TestLLMGenerateEndpoint:
    """Tests for POST /llm/generate endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, valid_api_key: str) -> Generator[None, None, None]:
        """Patch settings to use test API key."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key
            yield

    def test_generate_success(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test successful generation."""
        response = client_with_manager.post(
            "/llm/generate",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompts": ["What is 2+2?"],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "contents" in data
        assert "prompt_tokens" in data
        assert "completion_tokens" in data
        assert data["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_generate_batch(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test batch generation with multiple prompts."""
        response = client_with_manager.post(
            "/llm/generate",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompts": ["Question 1?", "Question 2?"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["contents"]) == 2
        assert len(data["prompt_tokens"]) == 2
        assert len(data["completion_tokens"]) == 2

    def test_generate_without_auth(self, client_with_manager: TestClient) -> None:
        """Test generate requires authentication."""
        response = client_with_manager.post(
            "/llm/generate",
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompts": ["Test prompt"],
            },
        )
        assert response.status_code == 401

    def test_generate_invalid_auth(self, client_with_manager: TestClient) -> None:
        """Test generate rejects invalid auth."""
        response = client_with_manager.post(
            "/llm/generate",
            headers={"X-Internal-Api-Key": "wrong-key"},
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompts": ["Test prompt"],
            },
        )
        assert response.status_code == 401

    def test_generate_llm_disabled(self, client_without_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test generate returns 503 when LLM is disabled."""
        response = client_without_manager.post(
            "/llm/generate",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompts": ["Test prompt"],
            },
        )
        assert response.status_code == 503

    def test_generate_oom_error(
        self, client_with_manager: TestClient, auth_headers: dict[str, str], mock_llm_manager: Mock
    ) -> None:
        """Test generate returns 507 on OOM."""
        mock_llm_manager.generate.side_effect = RuntimeError("Insufficient GPU memory")
        response = client_with_manager.post(
            "/llm/generate",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "prompts": ["Test prompt"],
            },
        )
        assert response.status_code == 507

    def test_generate_invalid_quantization(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test generate rejects unknown quantization values."""
        response = client_with_manager.post(
            "/llm/generate",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "quantization": "float32",
                "prompts": ["Test prompt"],
            },
        )
        assert response.status_code == 422


class TestLLMModelsEndpoint:
    """Tests for GET /llm/models endpoint."""

    def test_list_models(self, client_with_manager: TestClient) -> None:
        """Test listing available models."""
        response = client_with_manager.get("/llm/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        # Should have local models from registry
        model_ids = [m["id"] for m in data["models"]]
        assert "Qwen/Qwen2.5-1.5B-Instruct" in model_ids

    def test_list_models_has_memory_info(self, client_with_manager: TestClient) -> None:
        """Test models include memory requirements."""
        response = client_with_manager.get("/llm/models")
        data = response.json()
        # Find a model with memory_mb
        for model in data["models"]:
            if model["id"] == "Qwen/Qwen2.5-1.5B-Instruct":
                assert "memory_mb" in model
                assert "int8" in model["memory_mb"]
                break

    def test_list_models_no_auth_required(self, client_with_manager: TestClient) -> None:
        """Test models endpoint doesn't require auth."""
        response = client_with_manager.get("/llm/models")
        assert response.status_code == 200


class TestLLMPreloadEndpoint:
    """Tests for POST /llm/models/load endpoint."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, valid_api_key: str) -> Generator[None, None, None]:
        """Patch settings to use test API key."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key
            yield

    def test_preload_success(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test successful model preload."""
        response = client_with_manager.post(
            "/llm/models/load",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "quantization": "int8",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loaded"
        assert data["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_preload_without_auth(self, client_with_manager: TestClient) -> None:
        """Test preload requires authentication."""
        response = client_with_manager.post(
            "/llm/models/load",
            json={"model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
        )
        assert response.status_code == 401

    def test_preload_oom(
        self, client_with_manager: TestClient, auth_headers: dict[str, str], mock_llm_manager: Mock
    ) -> None:
        """Test preload returns 507 on OOM."""
        mock_llm_manager._ensure_model_loaded.side_effect = RuntimeError("Insufficient GPU memory")
        response = client_with_manager.post(
            "/llm/models/load",
            headers=auth_headers,
            json={"model_name": "Qwen/Qwen2.5-7B-Instruct"},
        )
        assert response.status_code == 507

    def test_preload_invalid_quantization(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test preload rejects unknown quantization values."""
        response = client_with_manager.post(
            "/llm/models/load",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "quantization": "float32",
            },
        )
        assert response.status_code == 422


class TestLLMHealthEndpoint:
    """Tests for GET /llm/health endpoint."""

    def test_health_with_manager(self, client_with_manager: TestClient) -> None:
        """Test health check with LLM manager enabled."""
        response = client_with_manager.get("/llm/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "loaded_models" in data
        assert "governor_enabled" in data

    def test_health_without_manager(self, client_without_manager: TestClient) -> None:
        """Test health check with LLM manager disabled."""
        response = client_without_manager.get("/llm/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "disabled"
        assert data["loaded_models"] == []
        assert data["governor_enabled"] is False


class TestLLMStreamingPlaceholders:
    """Tests for streaming placeholder endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_api_key(self, valid_api_key: str) -> Generator[None, None, None]:
        """Patch settings to use test API key."""
        with patch("vecpipe.search.auth.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = valid_api_key
            yield

    def test_stream_not_implemented(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test streaming returns 501."""
        response = client_with_manager.post(
            "/llm/generate/stream",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "prompts": ["Test"],
            },
        )
        assert response.status_code == 501

    def test_cancel_not_implemented(self, client_with_manager: TestClient, auth_headers: dict[str, str]) -> None:
        """Test cancel returns 501."""
        response = client_with_manager.post(
            "/llm/requests/12345678-1234-1234-1234-123456789012/cancel",
            headers=auth_headers,
        )
        assert response.status_code == 501
