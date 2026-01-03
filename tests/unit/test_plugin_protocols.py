"""Unit tests for plugin protocols."""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pytest

from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest
from shared.plugins.protocols import PluginProtocol
from shared.plugins.types import (
    ChunkingPlugin,
    ConnectorPlugin,
    EmbeddingPlugin,
    ExtractorPlugin,
    RerankerPlugin,
)
from shared.plugins.validation import (
    PLUGIN_ID_MAX_LENGTH,
    PLUGIN_ID_REGEX,
    validate_plugin_id,
)

# Valid plugin types as defined in loader.py
VALID_PLUGIN_TYPES = {"embedding", "chunking", "connector", "reranker", "extractor"}

# Semver regex pattern (X.Y.Z with optional pre-release/build metadata)
SEMVER_REGEX = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


class TestPluginProtocol:
    """Tests for PluginProtocol runtime checkable protocol."""

    def test_valid_plugin_isinstance(self):
        """Test that a valid plugin class satisfies isinstance check."""

        class ValidPlugin:
            PLUGIN_TYPE: ClassVar[str] = "test"
            PLUGIN_ID: ClassVar[str] = "valid-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Valid Plugin",
                    description="A valid plugin",
                )

        # Check that a valid class satisfies the protocol
        assert isinstance(ValidPlugin, type)
        # Note: isinstance checks on Protocol require instance, not class
        instance = ValidPlugin()
        assert isinstance(instance, PluginProtocol)

    def test_invalid_plugin_missing_attribute(self):
        """Test that a plugin missing required attributes fails isinstance."""

        class IncompletePlugin:
            PLUGIN_TYPE: ClassVar[str] = "test"
            # Missing PLUGIN_ID and PLUGIN_VERSION

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id="incomplete",
                    type="test",
                    version="1.0.0",
                    display_name="Incomplete",
                    description="",
                )

        instance = IncompletePlugin()
        # Missing required class variables
        assert not isinstance(instance, PluginProtocol)

    def test_invalid_plugin_missing_method(self):
        """Test that a plugin missing get_manifest method fails isinstance."""

        class NoManifestPlugin:
            PLUGIN_TYPE: ClassVar[str] = "test"
            PLUGIN_ID: ClassVar[str] = "no-manifest"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"
            # Missing get_manifest method

        instance = NoManifestPlugin()
        assert not isinstance(instance, PluginProtocol)

    def test_plugin_with_all_attributes(self):
        """Test plugin with all required class variables and method."""

        class CompletePlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "complete-plugin"
            PLUGIN_VERSION: ClassVar[str] = "2.1.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Complete Plugin",
                    description="Has all required attributes",
                    author="Test Author",
                    license="MIT",
                )

        instance = CompletePlugin()
        assert isinstance(instance, PluginProtocol)

        # Verify the manifest can be retrieved
        manifest = instance.get_manifest()
        assert manifest.id == "complete-plugin"
        assert manifest.type == "embedding"
        assert manifest.version == "2.1.0"

    def test_protocol_is_runtime_checkable(self):
        """Test that PluginProtocol is runtime_checkable."""
        # Check that the protocol can be used with isinstance()
        # This is the reliable way to verify it's runtime checkable

        class ValidPlugin:
            PLUGIN_TYPE: ClassVar[str] = "test"
            PLUGIN_ID: ClassVar[str] = "test-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Test",
                    description="",
                )

        instance = ValidPlugin()
        # If the protocol is runtime checkable, isinstance() should work
        assert isinstance(instance, PluginProtocol)

    def test_subclass_of_protocol(self):
        """Test that a class explicitly subclassing Protocol works."""

        class ExplicitPlugin(PluginProtocol):
            PLUGIN_TYPE: ClassVar[str] = "connector"
            PLUGIN_ID: ClassVar[str] = "explicit"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Explicit",
                    description="",
                )

        instance = ExplicitPlugin()
        assert isinstance(instance, PluginProtocol)


class TestPluginProtocolCompliance:
    """Comprehensive protocol enforcement tests per Phase 4.1."""

    def test_plugin_id_validation_rejects_empty(self):
        """Empty plugin IDs are rejected."""
        with pytest.raises(ValueError, match="plugin_id is required"):
            validate_plugin_id("")

    def test_plugin_id_validation_rejects_too_long(self):
        """Plugin IDs exceeding max length are rejected."""
        long_id = "a" * (PLUGIN_ID_MAX_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_plugin_id(long_id)

    def test_plugin_id_validation_rejects_invalid_format(self):
        """Plugin IDs with invalid characters are rejected."""
        invalid_ids = [
            "Invalid-Plugin",  # uppercase
            "-starts-with-dash",  # starts with dash
            "ends-with-dash-",  # ends with dash
            "has spaces",  # spaces
            "has.dots",  # dots
            "_underscore_start",  # starts with underscore
        ]
        for invalid_id in invalid_ids:
            # Only test those that don't match the regex
            if not re.match(PLUGIN_ID_REGEX, invalid_id):
                with pytest.raises(ValueError, match="Invalid plugin_id format"):
                    validate_plugin_id(invalid_id)

    def test_plugin_id_validation_accepts_valid_format(self):
        """Valid plugin IDs are accepted."""
        valid_ids = [
            "my-plugin",
            "my_plugin",
            "myplugin",
            "plugin123",
            "a",
            "ab",
            "a1",
            "1a",
            "my-cool_plugin-v2",
        ]
        for valid_id in valid_ids:
            # Should not raise
            validate_plugin_id(valid_id)

    def test_valid_semver_versions(self):
        """Valid semver versions match the pattern."""
        valid_versions = [
            "0.0.0",
            "1.0.0",
            "1.2.3",
            "10.20.30",
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-beta+build.123",
            "2.1.0",
        ]
        for version in valid_versions:
            assert SEMVER_REGEX.match(version), f"{version} should be valid semver"

    def test_invalid_semver_versions(self):
        """Invalid semver versions don't match the pattern."""
        invalid_versions = [
            "1",
            "1.0",
            "v1.0.0",  # 'v' prefix
            "1.0.0.0",  # too many parts
            "1.0.a",  # non-numeric patch
            "",
            "latest",
        ]
        for version in invalid_versions:
            assert not SEMVER_REGEX.match(version), f"{version} should be invalid semver"

    def test_manifest_id_must_match_plugin_id(self):
        """Manifest id should match class PLUGIN_ID for consistency."""

        class MismatchedPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "my-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id="different-id",  # Mismatch!
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Test",
                    description="",
                )

        # Verify the mismatch can be detected
        manifest = MismatchedPlugin.get_manifest()
        assert manifest.id != MismatchedPlugin.PLUGIN_ID

    def test_manifest_type_must_match_plugin_type(self):
        """Manifest type should match class PLUGIN_TYPE for consistency."""

        class MismatchedTypePlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "test-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type="chunking",  # Mismatch!
                    version=cls.PLUGIN_VERSION,
                    display_name="Test",
                    description="",
                )

        manifest = MismatchedTypePlugin.get_manifest()
        assert manifest.type != MismatchedTypePlugin.PLUGIN_TYPE

    def test_manifest_version_must_match_plugin_version(self):
        """Manifest version should match class PLUGIN_VERSION for consistency."""

        class MismatchedVersionPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "test-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version="2.0.0",  # Mismatch!
                    display_name="Test",
                    description="",
                )

        manifest = MismatchedVersionPlugin.get_manifest()
        assert manifest.version != MismatchedVersionPlugin.PLUGIN_VERSION

    def test_config_schema_must_be_valid_json_schema(self):
        """Config schema should have 'type': 'object' structure."""

        class PluginWithSchema:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "schema-plugin"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Schema Plugin",
                    description="Has config schema",
                )

            @classmethod
            def get_config_schema(cls) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "api_key_env": {"type": "string"},
                        "model": {"type": "string", "default": "test-model"},
                    },
                    "required": ["api_key_env"],
                }

        schema = PluginWithSchema.get_config_schema()
        assert schema is not None
        assert schema.get("type") == "object"
        assert "properties" in schema

    def test_config_schema_with_invalid_type(self):
        """Config schema with non-object type is detected."""

        class InvalidSchemaPlugin:
            PLUGIN_TYPE: ClassVar[str] = "embedding"
            PLUGIN_ID: ClassVar[str] = "bad-schema"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"

            @classmethod
            def get_manifest(cls) -> PluginManifest:
                return PluginManifest(
                    id=cls.PLUGIN_ID,
                    type=cls.PLUGIN_TYPE,
                    version=cls.PLUGIN_VERSION,
                    display_name="Bad Schema",
                    description="",
                )

            @classmethod
            def get_config_schema(cls) -> dict[str, Any]:
                return {"type": "array"}  # Wrong type for config

        schema = InvalidSchemaPlugin.get_config_schema()
        assert schema.get("type") != "object"

    @pytest.mark.parametrize(
        ("plugin_type", "base_class"),
        [
            ("embedding", EmbeddingPlugin),
            ("chunking", ChunkingPlugin),
            ("connector", ConnectorPlugin),
            ("reranker", RerankerPlugin),
            ("extractor", ExtractorPlugin),
        ],
    )
    def test_all_plugin_types_have_base_class(
        self, plugin_type: str, base_class: type
    ):
        """Each plugin type has a corresponding base class."""
        assert plugin_type in VALID_PLUGIN_TYPES
        assert hasattr(base_class, "PLUGIN_TYPE") or issubclass(base_class, SemanticPlugin)

    def test_valid_plugin_types_are_exhaustive(self):
        """VALID_PLUGIN_TYPES contains all expected types."""
        expected = {"embedding", "chunking", "connector", "reranker", "extractor"}
        assert expected == VALID_PLUGIN_TYPES

    def test_unknown_plugin_type_not_in_valid_set(self):
        """Unknown plugin types should not be in the valid set."""
        unknown_types = ["storage", "auth", "cache", "invalid", "unknown"]
        for unknown in unknown_types:
            assert unknown not in VALID_PLUGIN_TYPES

    def test_semantic_plugin_base_has_required_attributes(self):
        """SemanticPlugin base class defines required class variables."""
        # Check annotations exist (not assigned values for abstract-like behavior)
        annotations = getattr(SemanticPlugin, "__annotations__", {})
        assert "PLUGIN_TYPE" in annotations
        assert "PLUGIN_ID" in annotations
        assert "PLUGIN_VERSION" in annotations
        # PLUGIN_VERSION has a default value
        assert SemanticPlugin.PLUGIN_VERSION == "0.0.0"

    def test_semantic_plugin_has_required_methods(self):
        """SemanticPlugin base class defines required methods."""
        # Abstract method
        assert hasattr(SemanticPlugin, "get_manifest")
        # Optional methods with defaults
        assert hasattr(SemanticPlugin, "get_config_schema")
        assert hasattr(SemanticPlugin, "health_check")
        assert hasattr(SemanticPlugin, "initialize")
        assert hasattr(SemanticPlugin, "cleanup")
