"""Live E2E tests for local LLM feature requiring running stack.

These tests require:
- Running VecPipe service with ENABLE_LOCAL_LLM=true
- GPU with sufficient VRAM (or USE_MOCK_EMBEDDINGS=true for testing without GPU)

Tests are skipped if VecPipe is not available.
"""

from __future__ import annotations

import os

import pytest
import requests


def vecpipe_available() -> bool:
    """Check if VecPipe service is available."""
    vecpipe_url = os.getenv("VECPIPE_URL", "http://localhost:8000")
    try:
        response = requests.get(f"{vecpipe_url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_internal_api_key() -> str | None:
    """Get internal API key from environment or generate."""
    api_key = os.getenv("INTERNAL_API_KEY")
    if api_key:
        return api_key

    # Try to generate from settings
    try:
        from shared.config import settings
        from shared.config.internal_api_key import ensure_internal_api_key

        return ensure_internal_api_key(settings)
    except Exception:
        return None


VECPIPE_URL = os.getenv("VECPIPE_URL", "http://localhost:8000")
SKIP_REASON = "VecPipe service not available - run with docker compose up"


@pytest.mark.e2e()
@pytest.mark.skipif(not vecpipe_available(), reason=SKIP_REASON)
class TestLocalLLMLiveGeneration:
    """Live E2E tests requiring running VecPipe with GPU."""

    @pytest.fixture()
    def api_key(self) -> str:
        """Get internal API key for VecPipe requests."""
        key = get_internal_api_key()
        if key is None:
            pytest.skip("INTERNAL_API_KEY not available")
        return key

    @pytest.fixture()
    def auth_headers(self, api_key: str) -> dict[str, str]:
        """Auth headers for VecPipe requests."""
        return {"X-Internal-Api-Key": api_key}

    def test_live_llm_health_check(self) -> None:
        """Verify LLM health endpoint is accessible."""
        response = requests.get(f"{VECPIPE_URL}/llm/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        # Status is either "ok" or "disabled"
        assert data["status"] in ["ok", "disabled"]

    def test_live_list_models(self) -> None:
        """Verify models endpoint returns local models."""
        response = requests.get(f"{VECPIPE_URL}/llm/models", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        # Should have at least the curated Qwen models
        model_ids = [m["id"] for m in data["models"]]
        assert len(model_ids) > 0

    def test_live_models_have_memory_info(self) -> None:
        """Verify local models include memory requirements."""
        response = requests.get(f"{VECPIPE_URL}/llm/models", timeout=10)
        data = response.json()

        # Find a model and check memory_mb
        for model in data["models"]:
            if "Qwen" in model["id"]:
                assert "memory_mb" in model
                assert "int8" in model["memory_mb"]
                assert "int4" in model["memory_mb"]
                break

    def test_live_generate_requires_auth(self) -> None:
        """Verify generate endpoint requires authentication."""
        response = requests.post(
            f"{VECPIPE_URL}/llm/generate",
            json={
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompts": ["Test"],
            },
            timeout=10,
        )
        assert response.status_code == 401

    def test_live_preload_requires_auth(self) -> None:
        """Verify preload endpoint requires authentication."""
        response = requests.post(
            f"{VECPIPE_URL}/llm/models/load",
            json={
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "quantization": "int8",
            },
            timeout=10,
        )
        assert response.status_code == 401


@pytest.mark.e2e()
@pytest.mark.skipif(not vecpipe_available(), reason=SKIP_REASON)
class TestLocalLLMLiveAuthenticated:
    """Live E2E tests that require authentication."""

    @pytest.fixture()
    def api_key(self) -> str:
        """Get internal API key for VecPipe requests."""
        key = get_internal_api_key()
        if key is None:
            pytest.skip("INTERNAL_API_KEY not available")
        return key

    @pytest.fixture()
    def auth_headers(self, api_key: str) -> dict[str, str]:
        """Auth headers for VecPipe requests."""
        return {"X-Internal-Api-Key": api_key}

    def test_live_generate_with_auth(self, auth_headers: dict[str, str]) -> None:
        """Test generation with valid authentication (may timeout on first load)."""
        response = requests.post(
            f"{VECPIPE_URL}/llm/generate",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "quantization": "int8",
                "prompts": ["What is 2+2?"],
                "max_tokens": 50,
            },
            timeout=300,  # Long timeout for first model load
        )

        # May fail with 507 (OOM) if no GPU, but should not be 401/403
        assert response.status_code in [200, 507, 503]

        if response.status_code == 200:
            data = response.json()
            assert "contents" in data
            assert len(data["contents"]) == 1
            assert "prompt_tokens" in data
            assert "completion_tokens" in data

    def test_live_health_shows_loaded_models(self) -> None:
        """Health check shows list of loaded models."""
        response = requests.get(f"{VECPIPE_URL}/llm/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "loaded_models" in data
        assert isinstance(data["loaded_models"], list)
        assert "governor_enabled" in data


@pytest.mark.e2e()
@pytest.mark.skipif(not vecpipe_available(), reason=SKIP_REASON)
class TestLocalLLMLiveMemoryStats:
    """Live E2E tests for memory statistics."""

    def test_live_memory_stats_endpoint(self) -> None:
        """Verify memory stats endpoint is accessible."""
        response = requests.get(f"{VECPIPE_URL}/memory/stats", timeout=10)
        # May fail if memory governor not enabled, but endpoint should exist
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            # Should have memory-related fields
            assert "pressure_level" in data or "status" in data

    def test_live_memory_models_endpoint(self) -> None:
        """Verify memory models endpoint is accessible."""
        response = requests.get(f"{VECPIPE_URL}/memory/models", timeout=10)
        # May fail if memory governor not enabled
        assert response.status_code in [200, 503]


@pytest.mark.e2e()
@pytest.mark.skipif(not vecpipe_available(), reason=SKIP_REASON)
class TestLocalLLMLiveStreamingPlaceholders:
    """Live tests for streaming placeholder endpoints."""

    @pytest.fixture()
    def api_key(self) -> str:
        """Get internal API key."""
        key = get_internal_api_key()
        if key is None:
            pytest.skip("INTERNAL_API_KEY not available")
        return key

    @pytest.fixture()
    def auth_headers(self, api_key: str) -> dict[str, str]:
        """Auth headers."""
        return {"X-Internal-Api-Key": api_key}

    def test_live_streaming_returns_501(self, auth_headers: dict[str, str]) -> None:
        """Streaming endpoint returns 501 Not Implemented."""
        response = requests.post(
            f"{VECPIPE_URL}/llm/generate/stream",
            headers=auth_headers,
            json={
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "prompts": ["Test"],
            },
            timeout=10,
        )
        assert response.status_code == 501

    def test_live_cancel_returns_501(self, auth_headers: dict[str, str]) -> None:
        """Cancel endpoint returns 501 Not Implemented."""
        response = requests.post(
            f"{VECPIPE_URL}/llm/requests/12345678-1234-1234-1234-123456789012/cancel",
            headers=auth_headers,
            timeout=10,
        )
        assert response.status_code == 501
