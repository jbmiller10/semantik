"""Unit tests for sparse_router.py endpoints."""

from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.plugins.types.sparse_indexer import SparseQueryVector, SparseVector
from vecpipe.search.sparse_router import router


def make_client(sparse_manager: Mock | None = None) -> TestClient:
    """Create test client with patched auth and runtime."""
    app = FastAPI()
    app.include_router(router)

    # Set up mock runtime with sparse_manager
    runtime = Mock(is_closed=False)
    runtime.sparse_manager = sparse_manager or Mock()
    app.state.vecpipe_runtime = runtime

    client = TestClient(app)
    client.headers.update({"X-Internal-Api-Key": "test-internal-key"})
    return client


class TestEncodeDocuments:
    """Tests for POST /sparse/encode endpoint."""

    def test_encode_documents_success(self) -> None:
        """Test successful document encoding."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_documents = AsyncMock(
            return_value=[
                SparseVector(
                    chunk_id="chunk-1",
                    indices=(1, 5, 10),
                    values=(0.5, 0.8, 0.3),
                ),
                SparseVector(
                    chunk_id="chunk-2",
                    indices=(2, 7),
                    values=(0.4, 0.6),
                ),
            ]
        )

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello world", "foo bar"],
                    "chunk_ids": ["chunk-1", "chunk-2"],
                    "plugin_id": "bm25-local",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["plugin_id"] == "bm25-local"
        assert body["document_count"] == 2
        assert len(body["vectors"]) == 2
        assert body["vectors"][0]["chunk_id"] == "chunk-1"
        assert body["vectors"][0]["indices"] == [1, 5, 10]
        assert body["vectors"][0]["values"] == [0.5, 0.8, 0.3]
        assert "encoding_time_ms" in body
        mock_sparse_manager.encode_documents.assert_awaited_once()

    def test_encode_documents_validates_length_mismatch(self) -> None:
        """Test that texts and chunk_ids must have same length."""
        mock_sparse_manager = AsyncMock()

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello", "world", "foo"],
                    "chunk_ids": ["chunk-1", "chunk-2"],  # Only 2 vs 3 texts
                    "plugin_id": "bm25-local",
                },
            )

        assert resp.status_code == 400
        assert "texts (3) and chunk_ids (2) must have same length" in resp.json()["detail"]
        mock_sparse_manager.encode_documents.assert_not_awaited()

    def test_encode_documents_returns_404_when_plugin_not_found(self) -> None:
        """Test 404 when sparse plugin is not found."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_documents = AsyncMock(side_effect=ValueError("Plugin 'unknown-plugin' not found"))

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello"],
                    "chunk_ids": ["chunk-1"],
                    "plugin_id": "unknown-plugin",
                },
            )

        assert resp.status_code == 404
        assert "Plugin 'unknown-plugin' not found" in resp.json()["detail"]

    def test_encode_documents_returns_507_on_memory_error(self) -> None:
        """Test 507 when insufficient GPU memory."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_documents = AsyncMock(
            side_effect=RuntimeError("Cannot allocate memory: only 100MB free, need 500MB")
        )

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello"],
                    "chunk_ids": ["chunk-1"],
                    "plugin_id": "splade-local",
                },
            )

        assert resp.status_code == 507
        assert "Cannot allocate memory" in resp.json()["detail"]

    def test_encode_documents_returns_500_on_runtime_error(self) -> None:
        """Test 500 for other runtime errors."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_documents = AsyncMock(side_effect=RuntimeError("Model failed to initialize"))

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello"],
                    "chunk_ids": ["chunk-1"],
                    "plugin_id": "splade-local",
                },
            )

        assert resp.status_code == 500
        assert "Model failed to initialize" in resp.json()["detail"]

    def test_encode_documents_requires_auth(self) -> None:
        """Test that endpoint requires internal API key."""
        mock_sparse_manager = AsyncMock()

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            # Remove auth header
            client.headers.pop("X-Internal-Api-Key")
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello"],
                    "chunk_ids": ["chunk-1"],
                    "plugin_id": "bm25-local",
                },
            )

        assert resp.status_code == 401

    def test_encode_documents_with_model_config(self) -> None:
        """Test encoding with custom model configuration."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_documents = AsyncMock(
            return_value=[
                SparseVector(
                    chunk_id="chunk-1",
                    indices=(1,),
                    values=(0.5,),
                )
            ]
        )

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/encode",
                json={
                    "texts": ["hello"],
                    "chunk_ids": ["chunk-1"],
                    "plugin_id": "splade-local",
                    "model_config_data": {"batch_size": 16, "quantization": "int8"},
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_sparse_manager.encode_documents.call_args.kwargs
        assert call_kwargs["config"] == {"batch_size": 16, "quantization": "int8"}


class TestEncodeQuery:
    """Tests for POST /sparse/query endpoint."""

    def test_encode_query_success(self) -> None:
        """Test successful query encoding."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_query = AsyncMock(
            return_value=SparseQueryVector(
                indices=(1, 5, 10, 20),
                values=(0.9, 0.7, 0.5, 0.3),
            )
        )

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/query",
                json={
                    "query": "what is machine learning",
                    "plugin_id": "bm25-local",
                },
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["indices"] == [1, 5, 10, 20]
        assert body["values"] == [0.9, 0.7, 0.5, 0.3]
        assert "encoding_time_ms" in body
        mock_sparse_manager.encode_query.assert_awaited_once_with(
            plugin_id="bm25-local",
            query="what is machine learning",
            config=None,
        )

    def test_encode_query_returns_404_when_plugin_not_found(self) -> None:
        """Test 404 when sparse plugin is not found."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_query = AsyncMock(side_effect=ValueError("Plugin 'nonexistent' not found"))

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/query",
                json={
                    "query": "test query",
                    "plugin_id": "nonexistent",
                },
            )

        assert resp.status_code == 404
        assert "Plugin 'nonexistent' not found" in resp.json()["detail"]

    def test_encode_query_returns_507_on_memory_error(self) -> None:
        """Test 507 when insufficient GPU memory."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_query = AsyncMock(
            side_effect=RuntimeError("Cannot allocate memory for SPLADE model")
        )

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/query",
                json={
                    "query": "test query",
                    "plugin_id": "splade-local",
                },
            )

        assert resp.status_code == 507
        assert "Cannot allocate memory" in resp.json()["detail"]

    def test_encode_query_returns_500_on_runtime_error(self) -> None:
        """Test 500 for other runtime errors."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_query = AsyncMock(side_effect=RuntimeError("Tokenizer initialization failed"))

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/query",
                json={
                    "query": "test query",
                    "plugin_id": "splade-local",
                },
            )

        assert resp.status_code == 500
        assert "Tokenizer initialization failed" in resp.json()["detail"]

    def test_encode_query_requires_auth(self) -> None:
        """Test that endpoint requires internal API key."""
        mock_sparse_manager = AsyncMock()

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            client.headers.pop("X-Internal-Api-Key")
            resp = client.post(
                "/sparse/query",
                json={
                    "query": "test",
                    "plugin_id": "bm25-local",
                },
            )

        assert resp.status_code == 401

    def test_encode_query_with_model_config(self) -> None:
        """Test query encoding with custom model configuration."""
        mock_sparse_manager = AsyncMock()
        mock_sparse_manager.encode_query = AsyncMock(return_value=SparseQueryVector(indices=(1,), values=(0.5,)))

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.post(
                "/sparse/query",
                json={
                    "query": "test",
                    "plugin_id": "splade-local",
                    "model_config_data": {"max_length": 256},
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_sparse_manager.encode_query.call_args.kwargs
        assert call_kwargs["config"] == {"max_length": 256}


class TestListSparsePlugins:
    """Tests for GET /sparse/plugins endpoint."""

    def test_list_sparse_plugins_returns_available_plugins(self) -> None:
        """Test listing available sparse plugins."""
        mock_manifest_bm25 = Mock()
        mock_manifest_bm25.display_name = "BM25 Local"
        mock_manifest_bm25.description = "Local BM25 sparse indexer"

        mock_manifest_splade = Mock()
        mock_manifest_splade.display_name = "SPLADE Local"
        mock_manifest_splade.description = "Neural SPLADE sparse indexer"

        class MockBM25Plugin:
            SPARSE_TYPE = "bm25"

            @staticmethod
            def get_manifest():
                return mock_manifest_bm25

        class MockSPLADEPlugin:
            SPARSE_TYPE = "splade"

            @staticmethod
            def get_manifest():
                return mock_manifest_splade

        mock_record_bm25 = Mock()
        mock_record_bm25.plugin_id = "bm25-local"
        mock_record_bm25.plugin_type = "sparse_indexer"
        mock_record_bm25.plugin_class = MockBM25Plugin

        mock_record_splade = Mock()
        mock_record_splade.plugin_id = "splade-local"
        mock_record_splade.plugin_type = "sparse_indexer"
        mock_record_splade.plugin_class = MockSPLADEPlugin

        with (
            patch("vecpipe.search.sparse_router.load_plugins"),
            patch(
                "vecpipe.search.sparse_router.plugin_registry.get_all",
                return_value=[mock_record_bm25, mock_record_splade],
            ),
        ):
            client = make_client()
            resp = client.get("/sparse/plugins")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["plugins"]) == 2

        bm25_plugin = next(p for p in body["plugins"] if p["plugin_id"] == "bm25-local")
        assert bm25_plugin["display_name"] == "BM25 Local"
        assert bm25_plugin["description"] == "Local BM25 sparse indexer"
        assert bm25_plugin["sparse_type"] == "bm25"
        assert bm25_plugin["requires_gpu"] is False

        splade_plugin = next(p for p in body["plugins"] if p["plugin_id"] == "splade-local")
        assert splade_plugin["display_name"] == "SPLADE Local"
        assert splade_plugin["sparse_type"] == "splade"
        assert splade_plugin["requires_gpu"] is True

    def test_list_sparse_plugins_with_no_manifest(self) -> None:
        """Test plugin listing when plugin has no manifest."""

        class MockPluginNoManifest:
            SPARSE_TYPE = "bm25"

        mock_record = Mock()
        mock_record.plugin_id = "custom-bm25"
        mock_record.plugin_type = "sparse_indexer"
        mock_record.plugin_class = MockPluginNoManifest

        with (
            patch("vecpipe.search.sparse_router.load_plugins"),
            patch("vecpipe.search.sparse_router.plugin_registry.get_all", return_value=[mock_record]),
        ):
            client = make_client()
            resp = client.get("/sparse/plugins")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["plugins"]) == 1
        # Falls back to plugin_id for display_name
        assert body["plugins"][0]["display_name"] == "custom-bm25"
        assert body["plugins"][0]["description"] == ""

    def test_list_sparse_plugins_empty(self) -> None:
        """Test listing when no sparse plugins are registered."""
        with (
            patch("vecpipe.search.sparse_router.load_plugins"),
            patch("vecpipe.search.sparse_router.plugin_registry.get_all", return_value=[]),
        ):
            client = make_client()
            resp = client.get("/sparse/plugins")

        assert resp.status_code == 200
        assert resp.json()["plugins"] == []

    def test_list_sparse_plugins_does_not_require_auth(self) -> None:
        """Test that plugin listing is public (no auth required)."""
        with (
            patch("vecpipe.search.sparse_router.load_plugins"),
            patch("vecpipe.search.sparse_router.plugin_registry.get_all", return_value=[]),
        ):
            client = make_client()
            client.headers.pop("X-Internal-Api-Key")  # Remove auth
            resp = client.get("/sparse/plugins")

        assert resp.status_code == 200


class TestSparseStatus:
    """Tests for GET /sparse/status endpoint."""

    def test_sparse_status_returns_loaded_plugins(self) -> None:
        """Test status endpoint returns loaded plugins info."""
        mock_sparse_manager = Mock()
        mock_sparse_manager.get_loaded_plugins = Mock(return_value=["bm25-local", "splade-local"])

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.get("/sparse/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["loaded_plugins"] == ["bm25-local", "splade-local"]
        mock_sparse_manager.get_loaded_plugins.assert_called_once()

    def test_sparse_status_empty_plugins(self) -> None:
        """Test status when no plugins are loaded."""
        mock_sparse_manager = Mock()
        mock_sparse_manager.get_loaded_plugins = Mock(return_value=[])

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            client = make_client(sparse_manager=mock_sparse_manager)
            resp = client.get("/sparse/status")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["loaded_plugins"] == []

    def test_sparse_status_does_not_require_auth(self) -> None:
        """Test that status endpoint is public (no auth required)."""
        mock_sparse_manager = Mock()
        mock_sparse_manager.get_loaded_plugins = Mock(return_value=[])

        client = make_client(sparse_manager=mock_sparse_manager)
        client.headers.pop("X-Internal-Api-Key")  # Remove auth
        resp = client.get("/sparse/status")

        # Status endpoint doesn't have require_internal_api_key dependency
        assert resp.status_code == 200


class TestRuntimeDependency:
    """Tests for runtime dependency handling."""

    def test_endpoints_fail_when_runtime_not_initialized(self) -> None:
        """Test that endpoints fail gracefully when runtime is missing."""
        app = FastAPI()
        app.include_router(router)
        # Don't set app.state.vecpipe_runtime

        client = TestClient(app)
        client.headers.update({"X-Internal-Api-Key": "test-internal-key"})

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            resp = client.get("/sparse/status")

        assert resp.status_code == 503
        assert "runtime not initialized" in resp.json()["detail"].lower()

    def test_endpoints_fail_when_runtime_is_closed(self) -> None:
        """Test that endpoints fail when runtime is shutting down."""
        app = FastAPI()
        app.include_router(router)
        runtime = Mock(is_closed=True)
        app.state.vecpipe_runtime = runtime

        client = TestClient(app)
        client.headers.update({"X-Internal-Api-Key": "test-internal-key"})

        with patch("vecpipe.search.auth.settings.INTERNAL_API_KEY", "test-internal-key"):
            resp = client.get("/sparse/status")

        assert resp.status_code == 503
        assert "shutting down" in resp.json()["detail"].lower()
