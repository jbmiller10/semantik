"""Tests for the embedding provider registry."""

from __future__ import annotations

from shared.embedding.plugin_base import EmbeddingProviderDefinition
from shared.embedding.provider_registry import (
    _PROVIDERS,
    get_provider_definition,
    get_provider_metadata,
    get_providers_by_type,
    get_registered_provider_ids,
    is_provider_registered,
    list_provider_definitions,
    list_provider_metadata,
    list_provider_metadata_list,
    register_provider_definition,
    unregister_provider_definition,
)


class TestProviderRegistration:
    """Tests for provider registration and unregistration."""

    def test_register_provider_definition(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test registering a provider definition."""
        register_provider_definition(dummy_definition)

        assert dummy_definition.api_id in _PROVIDERS
        assert _PROVIDERS[dummy_definition.api_id] == dummy_definition

    def test_register_multiple_providers(
        self,
        empty_registry: None,
        dummy_definition: EmbeddingProviderDefinition,
        another_dummy_definition: EmbeddingProviderDefinition,
    ) -> None:
        """Test registering multiple provider definitions."""
        register_provider_definition(dummy_definition)
        register_provider_definition(another_dummy_definition)

        assert len(_PROVIDERS) == 2
        assert dummy_definition.api_id in _PROVIDERS
        assert another_dummy_definition.api_id in _PROVIDERS

    def test_unregister_provider_definition(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test unregistering a provider definition."""
        register_provider_definition(dummy_definition)
        assert dummy_definition.api_id in _PROVIDERS

        unregister_provider_definition(dummy_definition.api_id)
        assert dummy_definition.api_id not in _PROVIDERS

    def test_unregister_nonexistent_provider(self, empty_registry: None) -> None:
        """Test unregistering a provider that doesn't exist (should be no-op)."""
        # Should not raise an exception
        unregister_provider_definition("nonexistent")
        assert "nonexistent" not in _PROVIDERS


class TestProviderLookup:
    """Tests for provider lookup methods."""

    def test_list_provider_definitions(
        self,
        empty_registry: None,
        dummy_definition: EmbeddingProviderDefinition,
        another_dummy_definition: EmbeddingProviderDefinition,
    ) -> None:
        """Test listing all provider definitions."""
        register_provider_definition(dummy_definition)
        register_provider_definition(another_dummy_definition)

        definitions = list_provider_definitions()

        assert len(definitions) == 2
        assert dummy_definition in definitions
        assert another_dummy_definition in definitions

    def test_list_provider_definitions_empty(self, empty_registry: None) -> None:
        """Test listing provider definitions when registry is empty."""
        definitions = list_provider_definitions()
        assert len(definitions) == 0

    def test_get_provider_definition_by_api_id(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test getting a provider definition by API ID."""
        register_provider_definition(dummy_definition)

        result = get_provider_definition(dummy_definition.api_id)
        assert result == dummy_definition

    def test_get_provider_definition_by_internal_id(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test getting a provider definition by internal ID."""
        register_provider_definition(dummy_definition)

        result = get_provider_definition(dummy_definition.internal_id)
        assert result == dummy_definition

    def test_get_provider_definition_not_found(self, empty_registry: None) -> None:
        """Test getting a provider definition that doesn't exist."""
        result = get_provider_definition("nonexistent")
        assert result is None

    def test_is_provider_registered_true(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test checking if a provider is registered (positive case)."""
        register_provider_definition(dummy_definition)

        assert is_provider_registered(dummy_definition.api_id) is True
        assert is_provider_registered(dummy_definition.internal_id) is True

    def test_is_provider_registered_false(self, empty_registry: None) -> None:
        """Test checking if a provider is registered (negative case)."""
        assert is_provider_registered("nonexistent") is False

    def test_get_registered_provider_ids(
        self,
        empty_registry: None,
        dummy_definition: EmbeddingProviderDefinition,
        another_dummy_definition: EmbeddingProviderDefinition,
    ) -> None:
        """Test getting all registered provider IDs."""
        register_provider_definition(dummy_definition)
        register_provider_definition(another_dummy_definition)

        ids = get_registered_provider_ids()

        assert dummy_definition.api_id in ids
        assert another_dummy_definition.api_id in ids


class TestProviderMetadata:
    """Tests for provider metadata methods."""

    def test_list_provider_metadata(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test listing provider metadata."""
        register_provider_definition(dummy_definition)

        metadata = list_provider_metadata()

        assert len(metadata) == 1
        assert metadata[0]["id"] == dummy_definition.api_id
        assert metadata[0]["name"] == dummy_definition.display_name

    def test_list_provider_metadata_list(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test listing provider metadata as a mutable list."""
        register_provider_definition(dummy_definition)

        metadata = list_provider_metadata_list()

        assert isinstance(metadata, list)
        assert len(metadata) == 1
        # Verify it's a copy
        original_name = metadata[0]["name"]
        metadata[0]["name"] = "modified"
        fresh_metadata = list_provider_metadata_list()
        assert fresh_metadata[0]["name"] == original_name

    def test_get_provider_metadata(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test getting metadata for a specific provider."""
        register_provider_definition(dummy_definition)

        metadata = get_provider_metadata(dummy_definition.api_id)

        assert metadata is not None
        assert metadata["id"] == dummy_definition.api_id
        assert metadata["internal_id"] == dummy_definition.internal_id
        assert metadata["name"] == dummy_definition.display_name
        assert metadata["description"] == dummy_definition.description
        assert metadata["provider_type"] == dummy_definition.provider_type

    def test_get_provider_metadata_not_found(self, empty_registry: None) -> None:
        """Test getting metadata for a nonexistent provider."""
        result = get_provider_metadata("nonexistent")
        assert result is None


class TestProvidersByType:
    """Tests for filtering providers by type."""

    def test_get_providers_by_type(
        self,
        empty_registry: None,
        dummy_definition: EmbeddingProviderDefinition,
        another_dummy_definition: EmbeddingProviderDefinition,
    ) -> None:
        """Test getting providers filtered by type."""
        register_provider_definition(dummy_definition)
        register_provider_definition(another_dummy_definition)

        local_providers = get_providers_by_type("local")
        remote_providers = get_providers_by_type("remote")

        assert len(local_providers) == 1
        assert local_providers[0] == dummy_definition
        assert len(remote_providers) == 1
        assert remote_providers[0] == another_dummy_definition

    def test_get_providers_by_type_no_match(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test getting providers by type when none match."""
        register_provider_definition(dummy_definition)

        hybrid_providers = get_providers_by_type("hybrid")
        assert len(hybrid_providers) == 0


class TestCacheInvalidation:
    """Tests for LRU cache invalidation."""

    def test_cache_invalidation_on_register(self, empty_registry: None) -> None:
        """Test that caches are invalidated when registering a provider."""
        # First call populates cache
        _ = list_provider_definitions()

        # Register a new provider
        definition = EmbeddingProviderDefinition(
            api_id="new_provider",
            internal_id="new_provider",
            display_name="New Provider",
            description="A new provider",
            provider_type="local",
        )
        register_provider_definition(definition)

        # Cache should be invalidated (misses will increase on next call)
        _ = list_provider_definitions()
        cache_info_after = list_provider_definitions.cache_info()

        # The cache should have been cleared, so we should see a miss
        assert cache_info_after.hits == 0  # Fresh cache after clear

    def test_cache_invalidation_on_unregister(
        self, empty_registry: None, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test that caches are invalidated when unregistering a provider."""
        register_provider_definition(dummy_definition)

        # First call populates cache
        _ = list_provider_definitions()

        # Unregister the provider
        unregister_provider_definition(dummy_definition.api_id)

        # Cache should be invalidated - verify by checking the definitions
        definitions = list_provider_definitions()
        assert len(definitions) == 0


class TestDefinitionToMetadataDict:
    """Tests for the EmbeddingProviderDefinition.to_metadata_dict method."""

    def test_to_metadata_dict_basic(
        self, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test converting a definition to metadata dict."""
        metadata = dummy_definition.to_metadata_dict()

        assert metadata["id"] == dummy_definition.api_id
        assert metadata["internal_id"] == dummy_definition.internal_id
        assert metadata["name"] == dummy_definition.display_name
        assert metadata["description"] == dummy_definition.description
        assert metadata["provider_type"] == dummy_definition.provider_type
        assert metadata["supports_quantization"] == dummy_definition.supports_quantization
        assert metadata["supports_instruction"] == dummy_definition.supports_instruction
        assert metadata["supports_batch_processing"] == dummy_definition.supports_batch_processing
        assert metadata["is_plugin"] == dummy_definition.is_plugin

    def test_to_metadata_dict_supported_models(
        self, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test that supported_models is converted to list."""
        metadata = dummy_definition.to_metadata_dict()

        assert isinstance(metadata["supported_models"], list)
        assert "dummy-model" in metadata["supported_models"]

    def test_to_metadata_dict_defensive_copy(
        self, dummy_definition: EmbeddingProviderDefinition
    ) -> None:
        """Test that default_config is a defensive copy."""
        metadata = dummy_definition.to_metadata_dict()

        # Modify the returned config
        metadata["default_config"]["new_key"] = "new_value"

        # Get a fresh copy
        fresh_metadata = dummy_definition.to_metadata_dict()

        # Should not have the modification
        assert "new_key" not in fresh_metadata["default_config"]
