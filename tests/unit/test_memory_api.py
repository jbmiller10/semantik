"""Comprehensive unit tests for memory_api.py endpoints.

Tests cover:
- All 9 memory API endpoints
- Both GovernedModelManager and basic ModelManager scenarios
- Success paths, error paths, and edge cases
- Response model validation
"""

import time
from collections.abc import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vecpipe.search.memory_api import router

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def mock_governed_model_manager() -> Mock:
    """Mock GovernedModelManager with governor attribute."""
    manager = Mock()

    # Governor mock
    governor = Mock()
    governor.get_memory_stats = Mock(
        return_value={
            "cuda_available": True,
            "total_mb": 16000,
            "free_mb": 8000,
            "used_mb": 8000,
            "used_percent": 50.0,
            "allocated_mb": 4000,
            "reserved_mb": 16000,
            "budget_total_mb": 16000,
            "budget_usable_mb": 14400,
            "cpu_budget_total_mb": 32000,
            "cpu_budget_usable_mb": 16000,
            "cpu_used_mb": 2000,
            "models_loaded": 1,
            "models_offloaded": 1,
            "pressure_level": "LOW",
            "total_evictions": 5,
            "total_offloads": 3,
            "total_restorations": 2,
            "total_unloads": 2,
        }
    )
    governor.get_loaded_models = Mock(
        return_value=[
            {
                "model_name": "test-embedding",
                "model_type": "embedding",
                "quantization": "float16",
                "location": "gpu",
                "memory_mb": 2000,
                "idle_seconds": 30.5,
                "use_count": 10,
            }
        ]
    )
    governor.get_eviction_history = Mock(
        return_value=[
            {
                "model_name": "old-model",
                "model_type": "embedding",
                "quantization": "int8",
                "reason": "memory_pressure",
                "action": "offloaded",
                "memory_freed_mb": 1000,
                "timestamp": time.time() - 3600,
            }
        ]
    )

    manager._governor = governor
    manager.current_model_key = "test-embedding_float16"
    manager.current_reranker_key = None
    manager.unload_model_async = AsyncMock()
    manager.unload_reranker = Mock()
    manager.preload_models = AsyncMock(return_value={"embedding:test:float16": True})

    return manager


@pytest.fixture()
def mock_basic_model_manager() -> Mock:
    """Mock basic ModelManager without governor attribute."""
    manager = Mock(
        spec=[
            "current_model_key",
            "current_reranker_key",
            "unload_model",
            "unload_reranker",
            "last_used",
            "last_reranker_used",
        ]
    )
    manager.current_model_key = "test-model_float32"
    manager.current_reranker_key = None
    manager.last_used = time.time() - 60
    manager.last_reranker_used = time.time() - 120
    manager.unload_model = Mock()
    manager.unload_reranker = Mock()
    return manager


@pytest.fixture()
def mock_offloader() -> Mock:
    """Mock CPU offloader."""
    offloader = Mock()
    offloader.get_offloaded_models = Mock(return_value=["embedding:offloaded-model:float16"])
    offloader.get_offload_info = Mock(
        return_value={
            "original_device": "cuda:0",
            "offload_time": time.time() - 300,
            "seconds_offloaded": 300.0,
        }
    )
    return offloader


@pytest.fixture()
def app_with_router() -> FastAPI:
    """Create FastAPI app with memory router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def test_client_governed(
    app_with_router: FastAPI,
    mock_governed_model_manager: Mock,
    mock_offloader: Mock,
) -> Generator[TestClient, None, None]:
    """Test client with GovernedModelManager."""
    app_with_router.state.vecpipe_runtime = Mock(is_closed=False, model_manager=mock_governed_model_manager)

    with (patch("vecpipe.search.memory_api.get_offloader", return_value=mock_offloader),):
        yield TestClient(app_with_router)


@pytest.fixture()
def test_client_basic(
    app_with_router: FastAPI,
    mock_basic_model_manager: Mock,
    mock_offloader: Mock,
) -> Generator[TestClient, None, None]:
    """Test client with basic ModelManager (no governor)."""
    app_with_router.state.vecpipe_runtime = Mock(is_closed=False, model_manager=mock_basic_model_manager)

    # Patch torch at the module level since it's imported inside the function
    with (
        patch("vecpipe.search.memory_api.get_offloader", return_value=mock_offloader),
        patch.dict("sys.modules", {"torch": Mock()}),
    ):
        import sys

        mock_torch = sys.modules["torch"]
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (8 * 1024**3, 16 * 1024**3)
        yield TestClient(app_with_router)


@pytest.fixture()
def test_client_no_manager(
    app_with_router: FastAPI,
    mock_offloader: Mock,
) -> Generator[TestClient, None, None]:
    """Test client with no model manager."""
    app_with_router.state.vecpipe_runtime = Mock(is_closed=False, model_manager=None)

    with (patch("vecpipe.search.memory_api.get_offloader", return_value=mock_offloader),):
        yield TestClient(app_with_router)


# =============================================================================
# GET /memory/stats Tests
# =============================================================================


class TestMemoryStatsEndpoint:
    """Tests for GET /memory/stats endpoint."""

    def test_stats_with_governed_manager(self, test_client_governed: TestClient) -> None:
        """Test stats endpoint returns governor memory stats."""
        response = test_client_governed.get("/memory/stats")

        assert response.status_code == 200
        result = response.json()

        assert result["cuda_available"] is True
        assert result["total_mb"] == 16000
        assert result["free_mb"] == 8000
        assert result["pressure_level"] == "LOW"
        assert result["models_loaded"] == 1
        assert result["models_offloaded"] == 1

    def test_stats_with_basic_manager(self, test_client_basic: TestClient) -> None:
        """Test stats endpoint falls back to basic torch stats."""
        response = test_client_basic.get("/memory/stats")

        assert response.status_code == 200
        result = response.json()

        assert result["cuda_available"] is True
        assert result["total_mb"] == 16384  # 16GB
        assert result["free_mb"] == 8192  # 8GB
        assert result["pressure_level"] == "UNKNOWN"

    def test_stats_no_model_manager(self, test_client_no_manager: TestClient) -> None:
        """Test stats endpoint when model manager not initialized."""
        response = test_client_no_manager.get("/memory/stats")

        assert response.status_code == 503
        result = response.json()
        assert result["detail"] == "Model manager not initialized. Service may be starting up."


# =============================================================================
# GET /memory/models Tests
# =============================================================================


class TestLoadedModelsEndpoint:
    """Tests for GET /memory/models endpoint."""

    def test_models_with_governed_manager(self, test_client_governed: TestClient) -> None:
        """Test models endpoint returns governor tracked models."""
        response = test_client_governed.get("/memory/models")

        assert response.status_code == 200
        result = response.json()

        assert len(result) == 1
        assert result[0]["model_name"] == "test-embedding"
        assert result[0]["model_type"] == "embedding"
        assert result[0]["location"] == "gpu"
        assert result[0]["memory_mb"] == 2000

    def test_models_with_basic_manager(self, test_client_basic: TestClient) -> None:
        """Test models endpoint falls back to basic model info."""
        response = test_client_basic.get("/memory/models")

        assert response.status_code == 200
        result = response.json()

        assert len(result) == 1
        assert result[0]["model_type"] == "embedding"
        assert result[0]["location"] == "gpu"

    def test_models_no_model_manager(self, test_client_no_manager: TestClient) -> None:
        """Test 503 when model manager not initialized."""
        response = test_client_no_manager.get("/memory/models")

        assert response.status_code == 503
        assert "not initialized" in response.json()["detail"]


# =============================================================================
# GET /memory/evictions Tests
# =============================================================================


class TestEvictionHistoryEndpoint:
    """Tests for GET /memory/evictions endpoint."""

    def test_evictions_with_governed_manager(self, test_client_governed: TestClient) -> None:
        """Test evictions endpoint returns history."""
        response = test_client_governed.get("/memory/evictions")

        assert response.status_code == 200
        result = response.json()

        assert len(result) == 1
        assert result[0]["model_name"] == "old-model"
        assert result[0]["action"] == "offloaded"
        assert result[0]["reason"] == "memory_pressure"

    def test_evictions_with_basic_manager_returns_empty(self, test_client_basic: TestClient) -> None:
        """Test evictions returns empty list for non-governed manager."""
        response = test_client_basic.get("/memory/evictions")

        assert response.status_code == 200
        assert response.json() == []

    def test_evictions_no_model_manager(self, test_client_no_manager: TestClient) -> None:
        """Test 503 when model manager not initialized."""
        response = test_client_no_manager.get("/memory/evictions")

        assert response.status_code == 503


# =============================================================================
# GET /memory/fragmentation Tests
# =============================================================================


class TestFragmentationEndpoint:
    """Tests for GET /memory/fragmentation endpoint."""

    def test_fragmentation_cuda_available(self, test_client_governed: TestClient) -> None:
        """Test fragmentation analysis with CUDA."""
        with patch("vecpipe.search.memory_api.get_cuda_memory_fragmentation") as mock_frag:
            mock_frag.return_value = {
                "cuda_available": True,
                "allocated_mb": 4000,
                "reserved_mb": 6000,
                "fragmentation_mb": 2000,
                "fragmentation_percent": 33.3,
                "num_alloc_retries": 2,
                "num_ooms": 0,
            }

            response = test_client_governed.get("/memory/fragmentation")

            assert response.status_code == 200
            result = response.json()

            assert result["cuda_available"] is True
            assert result["fragmentation_mb"] == 2000
            assert result["fragmentation_percent"] == 33.3

    def test_fragmentation_no_cuda(self, test_client_governed: TestClient) -> None:
        """Test fragmentation when CUDA not available."""
        with patch("vecpipe.search.memory_api.get_cuda_memory_fragmentation") as mock_frag:
            mock_frag.return_value = {"cuda_available": False}

            response = test_client_governed.get("/memory/fragmentation")

            assert response.status_code == 200
            assert response.json()["cuda_available"] is False


# =============================================================================
# POST /memory/defragment Tests
# =============================================================================


class TestDefragmentEndpoint:
    """Tests for POST /memory/defragment endpoint."""

    def test_defragment_success(self, test_client_governed: TestClient) -> None:
        """Test defragmentation trigger."""
        with patch("vecpipe.search.memory_api.defragment_cuda_memory") as mock_defrag:
            response = test_client_governed.post("/memory/defragment")

            assert response.status_code == 200
            assert response.json()["status"] == "defragmentation_triggered"
            mock_defrag.assert_called_once()


# =============================================================================
# POST /memory/evict/{model_type} Tests
# =============================================================================


class TestEvictModelEndpoint:
    """Tests for POST /memory/evict/{model_type} endpoint."""

    def test_evict_embedding_success(self, test_client_governed: TestClient, mock_governed_model_manager: Mock) -> None:
        """Test evicting embedding model."""

        # After eviction, current_model_key should be None
        async def mock_unload() -> None:
            mock_governed_model_manager.current_model_key = None

        mock_governed_model_manager.unload_model_async = AsyncMock(side_effect=mock_unload)

        response = test_client_governed.post("/memory/evict/embedding")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "evicted"
        assert result["model_type"] == "embedding"

    def test_evict_embedding_no_model_loaded(
        self, test_client_governed: TestClient, mock_governed_model_manager: Mock
    ) -> None:
        """Test evict when no embedding model loaded."""
        mock_governed_model_manager.current_model_key = None

        response = test_client_governed.post("/memory/evict/embedding")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "no_action"

    def test_evict_reranker_success(self, test_client_governed: TestClient, mock_governed_model_manager: Mock) -> None:
        """Test evicting reranker model."""
        mock_governed_model_manager.current_reranker_key = "test-reranker_float16"

        def mock_unload() -> None:
            mock_governed_model_manager.current_reranker_key = None

        mock_governed_model_manager.unload_reranker = Mock(side_effect=mock_unload)

        response = test_client_governed.post("/memory/evict/reranker")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "evicted"

    def test_evict_reranker_no_model_loaded(
        self, test_client_governed: TestClient, mock_governed_model_manager: Mock
    ) -> None:
        """Test evict when no reranker loaded."""
        mock_governed_model_manager.current_reranker_key = None

        response = test_client_governed.post("/memory/evict/reranker")

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "no_action"

    def test_evict_invalid_model_type(self, test_client_governed: TestClient) -> None:
        """Test evict with invalid model type."""
        response = test_client_governed.post("/memory/evict/invalid")

        assert response.status_code == 400
        assert "Invalid model_type" in response.json()["detail"]

    def test_evict_failure_model_still_loaded(
        self, test_client_governed: TestClient, mock_governed_model_manager: Mock
    ) -> None:
        """Test 500 error when eviction fails (model still loaded)."""
        # unload_model_async doesn't clear the key (simulates failure)
        mock_governed_model_manager.unload_model_async = AsyncMock()

        response = test_client_governed.post("/memory/evict/embedding")

        assert response.status_code == 500
        assert "Eviction failed" in response.json()["detail"]

    def test_evict_no_model_manager(self, test_client_no_manager: TestClient) -> None:
        """Test 503 when model manager not available."""
        response = test_client_no_manager.post("/memory/evict/embedding")

        assert response.status_code == 503


# =============================================================================
# POST /memory/preload Tests
# =============================================================================


class TestPreloadEndpoint:
    """Tests for POST /memory/preload endpoint."""

    def test_preload_success(self, test_client_governed: TestClient) -> None:
        """Test preloading models."""
        response = test_client_governed.post(
            "/memory/preload",
            json={"models": [{"name": "test-model", "model_type": "embedding", "quantization": "float16"}]},
        )

        assert response.status_code == 200
        result = response.json()
        assert "results" in result

    def test_preload_invalid_model_type(self, test_client_governed: TestClient) -> None:
        """Test preload with invalid model type."""
        response = test_client_governed.post(
            "/memory/preload",
            json={"models": [{"name": "test", "model_type": "INVALID", "quantization": "float16"}]},
        )

        assert response.status_code == 422  # Validation error

    def test_preload_not_supported_basic_manager(self, test_client_basic: TestClient) -> None:
        """Test preload returns 501 for non-governed manager."""
        response = test_client_basic.post(
            "/memory/preload",
            json={"models": [{"name": "test", "model_type": "embedding", "quantization": "float16"}]},
        )

        assert response.status_code == 501
        assert "not supported" in response.json()["detail"]

    def test_preload_no_model_manager(self, test_client_no_manager: TestClient) -> None:
        """Test 503 when model manager not available."""
        response = test_client_no_manager.post(
            "/memory/preload",
            json={"models": [{"name": "test", "model_type": "embedding", "quantization": "float16"}]},
        )

        assert response.status_code == 503

    def test_preload_model_type_case_insensitive(self, test_client_governed: TestClient) -> None:
        """Test PreloadModelSpec normalizes model_type to lowercase."""
        response = test_client_governed.post(
            "/memory/preload",
            json={"models": [{"name": "test", "model_type": "EMBEDDING", "quantization": "float16"}]},
        )

        # Should succeed - case insensitive
        assert response.status_code == 200


# =============================================================================
# GET /memory/offloaded Tests
# =============================================================================


class TestOffloadedModelsEndpoint:
    """Tests for GET /memory/offloaded endpoint."""

    def test_offloaded_models_success(self, test_client_governed: TestClient) -> None:
        """Test getting list of offloaded models."""
        response = test_client_governed.get("/memory/offloaded")

        assert response.status_code == 200
        result = response.json()

        assert len(result) == 1
        assert result[0]["model_key"] == "embedding:offloaded-model:float16"
        assert "original_device" in result[0]
        assert "seconds_offloaded" in result[0]

    def test_offloaded_models_empty(self, test_client_governed: TestClient, mock_offloader: Mock) -> None:
        """Test when no models are offloaded."""
        mock_offloader.get_offloaded_models.return_value = []

        response = test_client_governed.get("/memory/offloaded")

        assert response.status_code == 200
        assert response.json() == []


# =============================================================================
# GET /memory/health Tests
# =============================================================================


class TestMemoryHealthEndpoint:
    """Tests for GET /memory/health endpoint."""

    def test_health_low_pressure(self, test_client_governed: TestClient) -> None:
        """Test health check with low memory pressure."""
        response = test_client_governed.get("/memory/health")

        assert response.status_code == 200
        result = response.json()

        assert result["healthy"] is True
        assert result["pressure"] == "LOW"
        assert "normal" in result["message"].lower()

    def test_health_high_pressure(self, test_client_governed: TestClient, mock_governed_model_manager: Mock) -> None:
        """Test health check with high memory pressure."""
        mock_governed_model_manager._governor.get_memory_stats.return_value = {
            "cuda_available": True,
            "pressure_level": "HIGH",
            "used_percent": 85.0,
        }

        response = test_client_governed.get("/memory/health")

        assert response.status_code == 200
        result = response.json()

        assert result["healthy"] is True
        assert result["pressure"] == "HIGH"
        assert "eviction active" in result["message"].lower()

    def test_health_critical_pressure(
        self, test_client_governed: TestClient, mock_governed_model_manager: Mock
    ) -> None:
        """Test health check with critical memory pressure."""
        mock_governed_model_manager._governor.get_memory_stats.return_value = {
            "cuda_available": True,
            "pressure_level": "CRITICAL",
            "used_percent": 95.0,
        }

        response = test_client_governed.get("/memory/health")

        assert response.status_code == 200
        result = response.json()

        assert result["healthy"] is False
        assert result["pressure"] == "CRITICAL"
        assert "OOM risk" in result["message"]

    def test_health_cpu_mode(self, test_client_governed: TestClient, mock_governed_model_manager: Mock) -> None:
        """Test health check in CPU-only mode."""
        mock_governed_model_manager._governor.get_memory_stats.return_value = {
            "cuda_available": False,
        }

        response = test_client_governed.get("/memory/health")

        assert response.status_code == 200
        result = response.json()

        assert result["healthy"] is True
        assert result["mode"] == "cpu"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_basic_manager_with_reranker(
        self, app_with_router: FastAPI, mock_basic_model_manager: Mock, mock_offloader: Mock
    ) -> None:
        """Test basic manager with both embedding and reranker loaded."""
        mock_basic_model_manager.current_reranker_key = "test-reranker_int8"
        app_with_router.state.vecpipe_runtime = Mock(is_closed=False, model_manager=mock_basic_model_manager)

        with (patch("vecpipe.search.memory_api.get_offloader", return_value=mock_offloader),):
            client = TestClient(app_with_router)
            response = client.get("/memory/models")

        assert response.status_code == 200
        result = response.json()
        assert len(result) == 2

        model_types = {m["model_type"] for m in result}
        assert model_types == {"embedding", "reranker"}

    def test_offload_info_returns_none(self, test_client_governed: TestClient, mock_offloader: Mock) -> None:
        """Test handling when offload info returns None for a key."""
        mock_offloader.get_offload_info.return_value = None

        response = test_client_governed.get("/memory/offloaded")

        assert response.status_code == 200
        # Should still return something, just with empty extra fields
        result = response.json()
        assert len(result) == 1
        assert result[0]["model_key"] == "embedding:offloaded-model:float16"

    def test_evict_embedding_with_sync_unload(self, app_with_router: FastAPI, mock_offloader: Mock) -> None:
        """Test evict embedding when only sync unload_model available."""
        manager = Mock(spec=["current_model_key", "current_reranker_key", "unload_model", "unload_reranker"])
        manager.current_model_key = "test-model_float32"
        manager.current_reranker_key = None

        def mock_unload() -> None:
            manager.current_model_key = None

        manager.unload_model = Mock(side_effect=mock_unload)
        app_with_router.state.vecpipe_runtime = Mock(is_closed=False, model_manager=manager)

        with (patch("vecpipe.search.memory_api.get_offloader", return_value=mock_offloader),):
            client = TestClient(app_with_router)
            response = client.post("/memory/evict/embedding")

        assert response.status_code == 200
        assert response.json()["status"] == "evicted"
        manager.unload_model.assert_called_once()

    def test_basic_manager_no_cuda(self, app_with_router: FastAPI, mock_offloader: Mock) -> None:
        """Test stats with basic manager when CUDA not available."""
        manager = Mock(spec=["current_model_key", "current_reranker_key"])
        manager.current_model_key = "test-model_float32"
        manager.current_reranker_key = None
        app_with_router.state.vecpipe_runtime = Mock(is_closed=False, model_manager=manager)

        with (
            patch("vecpipe.search.memory_api.get_offloader", return_value=mock_offloader),
            patch.dict("sys.modules", {"torch": Mock()}),
        ):
            import sys

            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = False
            client = TestClient(app_with_router)
            response = client.get("/memory/stats")

        assert response.status_code == 200
        result = response.json()
        # Response model adds default values, but cuda_available should be False
        assert result["cuda_available"] is False
