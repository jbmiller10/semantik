"""Unit tests for plugin protocols."""

from __future__ import annotations

from typing import ClassVar

from shared.plugins.manifest import PluginManifest
from shared.plugins.protocols import PluginProtocol


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

        # The protocol should have the __protocol_attrs__ attribute
        # indicating it's runtime checkable
        assert hasattr(PluginProtocol, "__protocol_attrs__")

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
