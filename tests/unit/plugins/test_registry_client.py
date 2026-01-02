"""Tests for plugin registry client with caching."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

from shared.plugins.registry_client import (
    BUNDLED_REGISTRY_PATH,
    DEFAULT_REGISTRY_URL,
    PluginRegistry,
    RegistryCache,
    RegistryPlugin,
    _fetch_remote_registry,
    fetch_registry,
    get_cached_registry,
    get_registry_metadata,
    get_registry_source,
    get_registry_url,
    invalidate_registry_cache,
    list_available_plugins,
    load_bundled_registry,
)


class TestRegistryPluginModel:
    """Tests for RegistryPlugin Pydantic model."""

    def test_minimal_plugin(self) -> None:
        """Should create plugin with required fields only."""
        plugin = RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="test",
            repository="https://github.com/test/test",
            pypi="test-plugin",
        )
        assert plugin.id == "test-plugin"
        assert plugin.verified is False  # default
        assert plugin.min_semantik_version is None  # default
        assert plugin.tags == []  # default

    def test_full_plugin(self) -> None:
        """Should create plugin with all fields."""
        plugin = RegistryPlugin(
            id="test-plugin",
            type="embedding",
            name="Test Plugin",
            description="A test plugin",
            author="test",
            repository="https://github.com/test/test",
            pypi="test-plugin",
            verified=True,
            min_semantik_version="2.0.0",
            tags=["api", "cloud"],
        )
        assert plugin.verified is True
        assert plugin.min_semantik_version == "2.0.0"
        assert plugin.tags == ["api", "cloud"]


class TestPluginRegistryModel:
    """Tests for PluginRegistry Pydantic model."""

    def test_empty_registry(self) -> None:
        """Should create registry with no plugins."""
        registry = PluginRegistry(
            registry_version="1.0",
            last_updated="2026-01-01T00:00:00Z",
        )
        assert registry.plugins == []

    def test_registry_with_plugins(self) -> None:
        """Should create registry with plugins."""
        registry = PluginRegistry(
            registry_version="1.0",
            last_updated="2026-01-01T00:00:00Z",
            plugins=[
                RegistryPlugin(
                    id="test-plugin",
                    type="embedding",
                    name="Test",
                    description="Test",
                    author="test",
                    repository="https://github.com/test/test",
                    pypi="test",
                )
            ],
        )
        assert len(registry.plugins) == 1
        assert registry.plugins[0].id == "test-plugin"


class TestRegistryCache:
    """Tests for RegistryCache dataclass."""

    def test_empty_cache_is_invalid(self) -> None:
        """Empty cache should be invalid."""
        cache = RegistryCache()
        assert cache.is_valid() is False

    def test_cache_with_no_timestamp_is_invalid(self) -> None:
        """Cache with registry but no timestamp should be invalid."""
        cache = RegistryCache(registry=PluginRegistry(registry_version="1.0", last_updated="2026-01-01T00:00:00Z"))
        assert cache.is_valid() is False

    def test_fresh_cache_is_valid(self) -> None:
        """Recently fetched cache should be valid."""
        cache = RegistryCache(
            registry=PluginRegistry(registry_version="1.0", last_updated="2026-01-01T00:00:00Z"),
            fetched_at=datetime.now(UTC),
        )
        assert cache.is_valid() is True

    def test_expired_cache_is_invalid(self) -> None:
        """Cache older than duration should be invalid."""
        cache = RegistryCache(
            registry=PluginRegistry(registry_version="1.0", last_updated="2026-01-01T00:00:00Z"),
            fetched_at=datetime.now(UTC) - timedelta(hours=2),
            cache_duration=timedelta(hours=1),
        )
        assert cache.is_valid() is False

    def test_invalidate_clears_cache(self) -> None:
        """Invalidate should clear all cache data."""
        cache = RegistryCache(
            registry=PluginRegistry(registry_version="1.0", last_updated="2026-01-01T00:00:00Z"),
            fetched_at=datetime.now(UTC),
            source="remote",
        )
        assert cache.is_valid() is True

        cache.invalidate()

        assert cache.registry is None
        assert cache.fetched_at is None
        assert cache.source is None
        assert cache.is_valid() is False


class TestGetRegistryUrl:
    """Tests for get_registry_url function."""

    def test_returns_default_url(self) -> None:
        """Should return default URL when env not set."""
        with patch.dict(os.environ, {}, clear=True):
            url = get_registry_url()
            assert url == DEFAULT_REGISTRY_URL

    def test_returns_env_url(self) -> None:
        """Should return URL from environment variable."""
        custom_url = "https://example.com/registry.yaml"
        with patch.dict(os.environ, {"SEMANTIK_PLUGIN_REGISTRY_URL": custom_url}):
            url = get_registry_url()
            assert url == custom_url


class TestLoadBundledRegistry:
    """Tests for load_bundled_registry function."""

    def test_bundled_registry_exists(self) -> None:
        """Bundled registry file should exist."""
        assert BUNDLED_REGISTRY_PATH.exists()

    def test_loads_bundled_registry(self) -> None:
        """Should load bundled registry successfully."""
        registry = load_bundled_registry()
        assert isinstance(registry, PluginRegistry)
        assert registry.registry_version is not None
        assert isinstance(registry.plugins, list)

    def test_bundled_registry_has_plugins(self) -> None:
        """Bundled registry should contain plugins."""
        registry = load_bundled_registry()
        assert len(registry.plugins) > 0

    def test_bundled_registry_plugin_types(self) -> None:
        """Bundled registry should have valid plugin types."""
        registry = load_bundled_registry()
        valid_types = {"embedding", "chunking", "connector", "reranker", "extractor"}
        for plugin in registry.plugins:
            assert plugin.type in valid_types


class TestFetchRemoteRegistry:
    """Tests for _fetch_remote_registry function."""

    @pytest.mark.asyncio()
    async def test_successful_fetch(self) -> None:
        """Should fetch and parse registry from URL."""
        mock_yaml = """
registry_version: "1.0"
last_updated: "2026-01-01T00:00:00Z"
plugins:
  - id: test-plugin
    type: embedding
    name: Test Plugin
    description: A test
    author: test
    repository: https://github.com/test/test
    pypi: test
"""
        mock_response = MagicMock()
        mock_response.text = mock_yaml
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            registry = await _fetch_remote_registry("https://example.com/registry.yaml")

        assert registry is not None
        assert registry.registry_version == "1.0"
        assert len(registry.plugins) == 1
        assert registry.plugins[0].id == "test-plugin"

    @pytest.mark.asyncio()
    async def test_http_error_returns_none(self) -> None:
        """Should return None on HTTP error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_response = MagicMock()
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not Found",
                request=MagicMock(),
                response=MagicMock(status_code=404),
            )
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            registry = await _fetch_remote_registry("https://example.com/registry.yaml")

        assert registry is None

    @pytest.mark.asyncio()
    async def test_request_error_returns_none(self) -> None:
        """Should return None on request error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.RequestError("Connection failed")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            registry = await _fetch_remote_registry("https://example.com/registry.yaml")

        assert registry is None

    @pytest.mark.asyncio()
    async def test_invalid_yaml_returns_none(self) -> None:
        """Should return None on invalid YAML."""
        mock_response = MagicMock()
        mock_response.text = "not: valid: yaml: {{"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            registry = await _fetch_remote_registry("https://example.com/registry.yaml")

        assert registry is None

    @pytest.mark.asyncio()
    async def test_invalid_schema_returns_none(self) -> None:
        """Should return None on schema validation error."""
        mock_response = MagicMock()
        mock_response.text = yaml.dump({"invalid": "schema"})
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            registry = await _fetch_remote_registry("https://example.com/registry.yaml")

        assert registry is None


class TestFetchRegistry:
    """Tests for fetch_registry function."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Clear cache before each test."""
        invalidate_registry_cache()

    @pytest.mark.asyncio()
    async def test_returns_cached_registry(self) -> None:
        """Should return cached registry when valid."""
        # First fetch populates cache (will use bundled fallback)
        registry1 = await fetch_registry()

        # Second fetch should use cache (no remote call)
        with patch("shared.plugins.registry_client._fetch_remote_registry") as mock_fetch:
            registry2 = await fetch_registry()

        mock_fetch.assert_not_called()
        assert registry1.registry_version == registry2.registry_version

    @pytest.mark.asyncio()
    async def test_force_refresh_bypasses_cache(self) -> None:
        """Should bypass cache when force_refresh=True."""
        # First fetch
        await fetch_registry()

        # Force refresh should call remote
        with patch("shared.plugins.registry_client._fetch_remote_registry") as mock_fetch:
            mock_fetch.return_value = None  # Will fall back to bundled
            await fetch_registry(force_refresh=True)

        mock_fetch.assert_called_once()

    @pytest.mark.asyncio()
    async def test_falls_back_to_bundled(self) -> None:
        """Should fall back to bundled when remote fails."""
        with patch("shared.plugins.registry_client._fetch_remote_registry") as mock_fetch:
            mock_fetch.return_value = None  # Remote failed

            registry = await fetch_registry()

        assert registry is not None
        assert get_registry_source() == "bundled"

    @pytest.mark.asyncio()
    async def test_uses_remote_when_available(self) -> None:
        """Should use remote registry when available."""
        remote_registry = PluginRegistry(
            registry_version="2.0",
            last_updated="2026-01-01T00:00:00Z",
            plugins=[],
        )

        with patch("shared.plugins.registry_client._fetch_remote_registry") as mock_fetch:
            mock_fetch.return_value = remote_registry

            registry = await fetch_registry(force_refresh=True)

        assert registry.registry_version == "2.0"
        assert get_registry_source() == "remote"


class TestCacheHelpers:
    """Tests for cache helper functions."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Clear cache before each test."""
        invalidate_registry_cache()

    @pytest.mark.asyncio()
    async def test_get_cached_registry_empty(self) -> None:
        """Should return None when cache is empty."""
        assert get_cached_registry() is None

    @pytest.mark.asyncio()
    async def test_get_cached_registry_after_fetch(self) -> None:
        """Should return registry after fetch."""
        await fetch_registry()
        cached = get_cached_registry()
        assert cached is not None

    @pytest.mark.asyncio()
    async def test_get_registry_source_empty(self) -> None:
        """Should return None when cache is empty."""
        assert get_registry_source() is None

    @pytest.mark.asyncio()
    async def test_get_registry_source_after_fetch(self) -> None:
        """Should return source after fetch."""
        await fetch_registry()
        source = get_registry_source()
        assert source in ("remote", "bundled")

    @pytest.mark.asyncio()
    async def test_invalidate_clears_cache(self) -> None:
        """Invalidate should clear the cache."""
        await fetch_registry()
        assert get_cached_registry() is not None

        invalidate_registry_cache()

        assert get_cached_registry() is None
        assert get_registry_source() is None


class TestListAvailablePlugins:
    """Tests for list_available_plugins function."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Clear cache before each test."""
        invalidate_registry_cache()

    @pytest.mark.asyncio()
    async def test_list_all_plugins(self) -> None:
        """Should list all plugins from registry."""
        plugins = await list_available_plugins()
        assert isinstance(plugins, list)
        assert len(plugins) > 0

    @pytest.mark.asyncio()
    async def test_filter_by_type(self) -> None:
        """Should filter plugins by type."""
        plugins = await list_available_plugins(plugin_type="embedding")
        assert all(p.type == "embedding" for p in plugins)

    @pytest.mark.asyncio()
    async def test_filter_by_nonexistent_type(self) -> None:
        """Should return empty list for unknown type."""
        plugins = await list_available_plugins(plugin_type="nonexistent")
        assert plugins == []

    @pytest.mark.asyncio()
    async def test_filter_verified_only(self) -> None:
        """Should filter to verified plugins only."""
        plugins = await list_available_plugins(verified_only=True)
        assert all(p.verified for p in plugins)

    @pytest.mark.asyncio()
    async def test_combined_filters(self) -> None:
        """Should apply multiple filters."""
        plugins = await list_available_plugins(plugin_type="embedding", verified_only=True)
        assert all(p.type == "embedding" and p.verified for p in plugins)


class TestGetRegistryMetadata:
    """Tests for get_registry_metadata function."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Clear cache before each test."""
        invalidate_registry_cache()

    @pytest.mark.asyncio()
    async def test_returns_metadata(self) -> None:
        """Should return registry metadata."""
        metadata = await get_registry_metadata()

        assert "registry_version" in metadata
        assert "last_updated" in metadata
        assert "source" in metadata

    @pytest.mark.asyncio()
    async def test_metadata_has_correct_types(self) -> None:
        """Metadata values should have correct types."""
        metadata = await get_registry_metadata()

        assert isinstance(metadata["registry_version"], str)
        assert isinstance(metadata["last_updated"], str)
        assert metadata["source"] in ("remote", "bundled")
