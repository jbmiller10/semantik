"""Unit tests for user preferences API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database.repositories.user_preferences_repository import UserPreferencesRepository
from webui.api.v2.user_preferences import _get_preferences_repo
from webui.auth import get_current_user
from webui.main import app


@pytest.fixture()
def mock_user_preferences():
    """Create a mock UserPreferences object with default values."""
    prefs = MagicMock()
    prefs.id = 1
    prefs.user_id = 1
    # Search preferences
    prefs.search_top_k = 10
    prefs.search_mode = "dense"
    prefs.search_use_reranker = False
    prefs.search_rrf_k = 60
    prefs.search_similarity_threshold = None
    # HyDE settings
    prefs.search_use_hyde = False
    prefs.search_hyde_quality_tier = "low"
    prefs.search_hyde_timeout_seconds = 10
    # Collection defaults
    prefs.default_embedding_model = None
    prefs.default_quantization = "float16"
    prefs.default_chunking_strategy = "recursive"
    prefs.default_chunk_size = 1024
    prefs.default_chunk_overlap = 200
    prefs.default_enable_sparse = False
    prefs.default_sparse_type = "bm25"
    prefs.default_enable_hybrid = False
    # Interface preferences
    prefs.data_refresh_interval_ms = 30000
    prefs.visualization_sample_limit = 200000
    prefs.animation_enabled = True
    # Timestamps
    prefs.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    prefs.updated_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
    return prefs


@pytest_asyncio.fixture
async def mock_preferences_repo(mock_user_preferences):
    """Create a mock user preferences repository."""
    repo = MagicMock(spec=UserPreferencesRepository)
    repo.get_by_user_id = AsyncMock(return_value=mock_user_preferences)
    repo.get_or_create = AsyncMock(return_value=mock_user_preferences)
    repo.update = AsyncMock(return_value=mock_user_preferences)
    repo.reset_search = AsyncMock(return_value=mock_user_preferences)
    repo.reset_collection_defaults = AsyncMock(return_value=mock_user_preferences)
    repo.reset_interface = AsyncMock(return_value=mock_user_preferences)
    return repo


@pytest_asyncio.fixture
async def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest_asyncio.fixture
async def preferences_api_client(mock_preferences_repo, mock_db_session):
    """Provide an AsyncClient with user preferences dependencies mocked."""
    mock_user = {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_user

    async def override_get_preferences_repo() -> UserPreferencesRepository:
        return mock_preferences_repo

    from shared.database import get_db

    async def override_get_db():
        return mock_db_session

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[_get_preferences_repo] = override_get_preferences_repo
    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, mock_preferences_repo

    app.dependency_overrides.clear()


class TestGetPreferences:
    """Tests for GET /api/v2/preferences endpoint."""

    @pytest.mark.asyncio()
    async def test_get_preferences_returns_defaults(self, preferences_api_client):
        """Test that GET returns preferences with default values."""
        client, mock_repo = preferences_api_client

        response = await client.get("/api/v2/preferences")

        assert response.status_code == 200
        data = response.json()

        # Verify search preferences
        assert data["search"]["top_k"] == 10
        assert data["search"]["mode"] == "dense"
        assert data["search"]["use_reranker"] is False
        assert data["search"]["rrf_k"] == 60
        assert data["search"]["similarity_threshold"] is None

        # Verify collection defaults
        assert data["collection_defaults"]["embedding_model"] is None
        assert data["collection_defaults"]["quantization"] == "float16"
        assert data["collection_defaults"]["chunking_strategy"] == "recursive"
        assert data["collection_defaults"]["chunk_size"] == 1024
        assert data["collection_defaults"]["chunk_overlap"] == 200
        assert data["collection_defaults"]["enable_sparse"] is False
        assert data["collection_defaults"]["sparse_type"] == "bm25"
        assert data["collection_defaults"]["enable_hybrid"] is False

        # Verify interface preferences
        assert data["interface"]["data_refresh_interval_ms"] == 30000
        assert data["interface"]["visualization_sample_limit"] == 200000
        assert data["interface"]["animation_enabled"] is True

        # Verify timestamps
        assert "created_at" in data
        assert "updated_at" in data

        mock_repo.get_or_create.assert_called_once_with(1)

    @pytest.mark.asyncio()
    async def test_get_preferences_requires_auth(self, api_client_unauthenticated):
        """Test that GET requires authentication."""
        from shared.config import settings

        original_disable_auth = settings.DISABLE_AUTH
        settings.DISABLE_AUTH = False
        try:
            response = await api_client_unauthenticated.get("/api/v2/preferences")
            assert response.status_code == 401
        finally:
            settings.DISABLE_AUTH = original_disable_auth


class TestUpdatePreferences:
    """Tests for PUT /api/v2/preferences endpoint."""

    @pytest.mark.asyncio()
    async def test_put_preferences_updates_search(self, preferences_api_client):
        """Test that PUT updates search preferences."""
        client, mock_repo = preferences_api_client

        response = await client.put(
            "/api/v2/preferences",
            json={
                "search": {
                    "top_k": 20,
                    "mode": "hybrid",
                    "use_reranker": True,
                    "rrf_k": 80,
                    "similarity_threshold": 0.5,
                }
            },
        )

        assert response.status_code == 200
        mock_repo.update.assert_called_once()
        call_kwargs = mock_repo.update.call_args[1]
        assert call_kwargs["search_top_k"] == 20
        assert call_kwargs["search_mode"] == "hybrid"
        assert call_kwargs["search_use_reranker"] is True
        assert call_kwargs["search_rrf_k"] == 80
        assert call_kwargs["search_similarity_threshold"] == 0.5

    @pytest.mark.asyncio()
    async def test_put_preferences_updates_collection_defaults(self, preferences_api_client):
        """Test that PUT updates collection defaults."""
        client, mock_repo = preferences_api_client

        response = await client.put(
            "/api/v2/preferences",
            json={
                "collection_defaults": {
                    "embedding_model": "nomic-embed-text",
                    "quantization": "float32",
                    "chunking_strategy": "markdown",
                    "chunk_size": 512,
                    "chunk_overlap": 100,
                    "enable_sparse": True,
                    "sparse_type": "splade",
                    "enable_hybrid": True,
                }
            },
        )

        assert response.status_code == 200
        mock_repo.update.assert_called_once()
        call_kwargs = mock_repo.update.call_args[1]
        assert call_kwargs["default_embedding_model"] == "nomic-embed-text"
        assert call_kwargs["default_quantization"] == "float32"
        assert call_kwargs["default_chunking_strategy"] == "markdown"
        assert call_kwargs["default_chunk_size"] == 512
        assert call_kwargs["default_chunk_overlap"] == 100
        assert call_kwargs["default_enable_sparse"] is True
        assert call_kwargs["default_sparse_type"] == "splade"
        assert call_kwargs["default_enable_hybrid"] is True

    @pytest.mark.asyncio()
    async def test_put_preferences_validates_top_k_range(self, preferences_api_client):
        """Test that top_k is validated to be between 1 and 250."""
        client, _ = preferences_api_client

        # Below minimum
        response = await client.put(
            "/api/v2/preferences",
            json={"search": {"top_k": 0, "mode": "dense", "use_reranker": False, "rrf_k": 60}},
        )
        assert response.status_code == 422

        # Above maximum
        response = await client.put(
            "/api/v2/preferences",
            json={"search": {"top_k": 300, "mode": "dense", "use_reranker": False, "rrf_k": 60}},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio()
    async def test_put_preferences_validates_chunk_size_range(self, preferences_api_client):
        """Test that chunk_size is validated to be between 256 and 4096."""
        client, _ = preferences_api_client

        # Below minimum
        response = await client.put(
            "/api/v2/preferences",
            json={
                "collection_defaults": {
                    "chunk_size": 100,
                    "embedding_model": None,
                    "quantization": "float16",
                    "chunking_strategy": "recursive",
                    "chunk_overlap": 200,
                    "enable_sparse": False,
                    "sparse_type": "bm25",
                    "enable_hybrid": False,
                }
            },
        )
        assert response.status_code == 422

        # Above maximum
        response = await client.put(
            "/api/v2/preferences",
            json={
                "collection_defaults": {
                    "chunk_size": 10000,
                    "embedding_model": None,
                    "quantization": "float16",
                    "chunking_strategy": "recursive",
                    "chunk_overlap": 200,
                    "enable_sparse": False,
                    "sparse_type": "bm25",
                    "enable_hybrid": False,
                }
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio()
    async def test_put_preferences_hybrid_requires_sparse(self, preferences_api_client):
        """Test that enable_hybrid=true requires enable_sparse=true."""
        client, _ = preferences_api_client

        response = await client.put(
            "/api/v2/preferences",
            json={
                "collection_defaults": {
                    "embedding_model": None,
                    "quantization": "float16",
                    "chunking_strategy": "recursive",
                    "chunk_size": 1024,
                    "chunk_overlap": 200,
                    "enable_sparse": False,
                    "sparse_type": "bm25",
                    "enable_hybrid": True,
                }
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio()
    async def test_put_preferences_requires_auth(self, api_client_unauthenticated):
        """Test that PUT requires authentication."""
        from shared.config import settings

        original_disable_auth = settings.DISABLE_AUTH
        settings.DISABLE_AUTH = False
        try:
            response = await api_client_unauthenticated.put(
                "/api/v2/preferences",
                json={"search": {"top_k": 20}},
            )
            assert response.status_code == 401
        finally:
            settings.DISABLE_AUTH = original_disable_auth


class TestResetSearchPreferences:
    """Tests for POST /api/v2/preferences/reset/search endpoint."""

    @pytest.mark.asyncio()
    async def test_reset_search_preferences(self, preferences_api_client):
        """Test that reset search preferences works."""
        client, mock_repo = preferences_api_client

        response = await client.post("/api/v2/preferences/reset/search")

        assert response.status_code == 200
        mock_repo.reset_search.assert_called_once_with(1)

    @pytest.mark.asyncio()
    async def test_reset_search_requires_auth(self, api_client_unauthenticated):
        """Test that reset search requires authentication."""
        from shared.config import settings

        original_disable_auth = settings.DISABLE_AUTH
        settings.DISABLE_AUTH = False
        try:
            response = await api_client_unauthenticated.post("/api/v2/preferences/reset/search")
            assert response.status_code == 401
        finally:
            settings.DISABLE_AUTH = original_disable_auth


class TestResetCollectionDefaults:
    """Tests for POST /api/v2/preferences/reset/collection-defaults endpoint."""

    @pytest.mark.asyncio()
    async def test_reset_collection_defaults(self, preferences_api_client):
        """Test that reset collection defaults works."""
        client, mock_repo = preferences_api_client

        response = await client.post("/api/v2/preferences/reset/collection-defaults")

        assert response.status_code == 200
        mock_repo.reset_collection_defaults.assert_called_once_with(1)

    @pytest.mark.asyncio()
    async def test_reset_collection_defaults_requires_auth(self, api_client_unauthenticated):
        """Test that reset collection defaults requires authentication."""
        from shared.config import settings

        original_disable_auth = settings.DISABLE_AUTH
        settings.DISABLE_AUTH = False
        try:
            response = await api_client_unauthenticated.post("/api/v2/preferences/reset/collection-defaults")
            assert response.status_code == 401
        finally:
            settings.DISABLE_AUTH = original_disable_auth


class TestResetInterfacePreferences:
    """Tests for POST /api/v2/preferences/reset/interface endpoint."""

    @pytest.mark.asyncio()
    async def test_reset_interface_preferences(self, preferences_api_client):
        """Test that reset interface preferences works."""
        client, mock_repo = preferences_api_client

        response = await client.post("/api/v2/preferences/reset/interface")

        assert response.status_code == 200
        mock_repo.reset_interface.assert_called_once_with(1)

    @pytest.mark.asyncio()
    async def test_reset_interface_requires_auth(self, api_client_unauthenticated):
        """Test that reset interface requires authentication."""
        from shared.config import settings

        original_disable_auth = settings.DISABLE_AUTH
        settings.DISABLE_AUTH = False
        try:
            response = await api_client_unauthenticated.post("/api/v2/preferences/reset/interface")
            assert response.status_code == 401
        finally:
            settings.DISABLE_AUTH = original_disable_auth
