"""Tests verifying built-in base classes satisfy their protocol interfaces.

These tests ensure that the built-in plugin base classes implement all
requirements defined by their corresponding protocol interfaces, enabling
the loader to use Protocol-based validation in Phase 5.

Note: Python's @runtime_checkable protocols don't support issubclass() when
they have non-method members (like ClassVar attributes). Therefore, we verify
structural compliance by checking all requirements directly rather than using
issubclass() with protocols.
"""

from __future__ import annotations

from shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from shared.connectors.base import BaseConnector
from shared.embedding.plugin_base import BaseEmbeddingPlugin
from shared.plugins.types.agent import AgentPlugin
from shared.plugins.types.extractor import ExtractorPlugin
from shared.plugins.types.reranker import RerankerPlugin


class TestConnectorProtocolCompliance:
    """Verify BaseConnector satisfies ConnectorProtocol."""

    def test_has_required_class_vars(self) -> None:
        """BaseConnector must have all protocol-required class variables."""
        assert hasattr(BaseConnector, "PLUGIN_ID")
        assert hasattr(BaseConnector, "PLUGIN_TYPE")
        assert hasattr(BaseConnector, "PLUGIN_VERSION")
        assert hasattr(BaseConnector, "METADATA")

    def test_class_var_types(self) -> None:
        """Class variables must have correct types."""
        assert isinstance(BaseConnector.PLUGIN_TYPE, str)
        assert BaseConnector.PLUGIN_TYPE == "connector"

    def test_has_get_manifest(self) -> None:
        """BaseConnector must have get_manifest classmethod."""
        assert hasattr(BaseConnector, "get_manifest")
        assert callable(BaseConnector.get_manifest)

    def test_has_required_methods(self) -> None:
        """BaseConnector must have all protocol-required methods."""
        assert hasattr(BaseConnector, "authenticate")
        assert hasattr(BaseConnector, "load_documents")
        assert hasattr(BaseConnector, "get_config_fields")
        assert hasattr(BaseConnector, "get_secret_fields")

    def test_satisfies_protocol_requirements(self) -> None:
        """BaseConnector must satisfy all ConnectorProtocol requirements.

        Note: issubclass() doesn't work with protocols that have ClassVar members,
        so we verify structural compliance by checking all requirements directly.
        """
        # Verify all protocol requirements are present
        # Class variables
        assert hasattr(BaseConnector, "PLUGIN_ID")
        assert hasattr(BaseConnector, "PLUGIN_TYPE")
        assert hasattr(BaseConnector, "PLUGIN_VERSION")
        assert hasattr(BaseConnector, "METADATA")
        # Methods
        assert callable(getattr(BaseConnector, "authenticate", None))
        assert callable(getattr(BaseConnector, "load_documents", None))
        assert callable(getattr(BaseConnector, "get_config_fields", None))
        assert callable(getattr(BaseConnector, "get_secret_fields", None))
        assert callable(getattr(BaseConnector, "get_manifest", None))


class TestChunkingProtocolCompliance:
    """Verify ChunkingStrategy satisfies ChunkingProtocol."""

    def test_has_required_class_vars(self) -> None:
        """ChunkingStrategy must have all protocol-required class variables."""
        assert hasattr(ChunkingStrategy, "PLUGIN_ID")
        assert hasattr(ChunkingStrategy, "PLUGIN_TYPE")
        assert hasattr(ChunkingStrategy, "PLUGIN_VERSION")

    def test_class_var_types(self) -> None:
        """Class variables must have correct types."""
        assert isinstance(ChunkingStrategy.PLUGIN_TYPE, str)
        assert ChunkingStrategy.PLUGIN_TYPE == "chunking"

    def test_has_required_methods(self) -> None:
        """ChunkingStrategy must have all protocol-required methods."""
        assert hasattr(ChunkingStrategy, "chunk")
        assert hasattr(ChunkingStrategy, "validate_content")
        assert hasattr(ChunkingStrategy, "estimate_chunks")
        assert hasattr(ChunkingStrategy, "get_manifest")

    def test_get_manifest_returns_dict(self) -> None:
        """get_manifest should return a dictionary."""
        manifest = ChunkingStrategy.get_manifest()
        assert isinstance(manifest, dict)
        assert "id" in manifest
        assert "type" in manifest
        assert "version" in manifest
        assert manifest["type"] == "chunking"

    def test_satisfies_protocol_requirements(self) -> None:
        """ChunkingStrategy must satisfy all ChunkingProtocol requirements.

        Note: issubclass() doesn't work with protocols that have ClassVar members,
        so we verify structural compliance by checking all requirements directly.
        """
        # Class variables
        assert hasattr(ChunkingStrategy, "PLUGIN_ID")
        assert hasattr(ChunkingStrategy, "PLUGIN_TYPE")
        assert hasattr(ChunkingStrategy, "PLUGIN_VERSION")
        # Methods
        assert callable(getattr(ChunkingStrategy, "chunk", None))
        assert callable(getattr(ChunkingStrategy, "validate_content", None))
        assert callable(getattr(ChunkingStrategy, "estimate_chunks", None))
        assert callable(getattr(ChunkingStrategy, "get_manifest", None))


class TestRerankerProtocolCompliance:
    """Verify RerankerPlugin satisfies RerankerProtocol."""

    def test_has_plugin_type(self) -> None:
        """RerankerPlugin must have PLUGIN_TYPE set to 'reranker'."""
        assert hasattr(RerankerPlugin, "PLUGIN_TYPE")
        assert isinstance(RerankerPlugin.PLUGIN_TYPE, str)
        assert RerankerPlugin.PLUGIN_TYPE == "reranker"

    def test_inherits_plugin_class_vars(self) -> None:
        """RerankerPlugin inherits PLUGIN_ID and PLUGIN_VERSION from SemanticPlugin."""
        # These are inherited from SemanticPlugin base class
        # They will be set by concrete subclasses
        from shared.plugins.base import SemanticPlugin

        assert issubclass(RerankerPlugin, SemanticPlugin)

    def test_has_required_methods(self) -> None:
        """RerankerPlugin must have all protocol-required methods."""
        assert hasattr(RerankerPlugin, "rerank")
        assert hasattr(RerankerPlugin, "get_capabilities")
        assert hasattr(RerankerPlugin, "get_manifest")

    def test_satisfies_protocol_requirements(self) -> None:
        """RerankerPlugin must satisfy all RerankerProtocol requirements.

        Note: issubclass() doesn't work with protocols that have ClassVar members,
        so we verify structural compliance by checking all requirements directly.
        """
        # Methods
        assert callable(getattr(RerankerPlugin, "rerank", None))
        assert callable(getattr(RerankerPlugin, "get_capabilities", None))
        assert callable(getattr(RerankerPlugin, "get_manifest", None))


class TestExtractorProtocolCompliance:
    """Verify ExtractorPlugin satisfies ExtractorProtocol."""

    def test_has_plugin_type(self) -> None:
        """ExtractorPlugin must have PLUGIN_TYPE set to 'extractor'."""
        assert hasattr(ExtractorPlugin, "PLUGIN_TYPE")
        assert isinstance(ExtractorPlugin.PLUGIN_TYPE, str)
        assert ExtractorPlugin.PLUGIN_TYPE == "extractor"

    def test_inherits_plugin_class_vars(self) -> None:
        """ExtractorPlugin inherits PLUGIN_ID and PLUGIN_VERSION from SemanticPlugin."""
        # These are inherited from SemanticPlugin base class
        # They will be set by concrete subclasses
        from shared.plugins.base import SemanticPlugin

        assert issubclass(ExtractorPlugin, SemanticPlugin)

    def test_has_required_methods(self) -> None:
        """ExtractorPlugin must have all protocol-required methods."""
        assert hasattr(ExtractorPlugin, "extract")
        assert hasattr(ExtractorPlugin, "supported_extractions")
        assert hasattr(ExtractorPlugin, "get_manifest")

    def test_satisfies_protocol_requirements(self) -> None:
        """ExtractorPlugin must satisfy all ExtractorProtocol requirements.

        Note: issubclass() doesn't work with protocols that have ClassVar members,
        so we verify structural compliance by checking all requirements directly.
        """
        # Methods
        assert callable(getattr(ExtractorPlugin, "extract", None))
        assert callable(getattr(ExtractorPlugin, "supported_extractions", None))
        assert callable(getattr(ExtractorPlugin, "get_manifest", None))


class TestEmbeddingProtocolCompliance:
    """Verify BaseEmbeddingPlugin satisfies EmbeddingProtocol."""

    def test_has_required_class_vars(self) -> None:
        """BaseEmbeddingPlugin must have all protocol-required class variables."""
        assert hasattr(BaseEmbeddingPlugin, "PLUGIN_TYPE")
        assert hasattr(BaseEmbeddingPlugin, "PLUGIN_VERSION")
        assert hasattr(BaseEmbeddingPlugin, "INTERNAL_NAME")
        assert hasattr(BaseEmbeddingPlugin, "API_ID")
        assert hasattr(BaseEmbeddingPlugin, "PROVIDER_TYPE")
        assert hasattr(BaseEmbeddingPlugin, "METADATA")

    def test_class_var_types(self) -> None:
        """Class variables must have correct types."""
        assert isinstance(BaseEmbeddingPlugin.PLUGIN_TYPE, str)
        assert BaseEmbeddingPlugin.PLUGIN_TYPE == "embedding"

    def test_has_required_methods(self) -> None:
        """BaseEmbeddingPlugin must have all protocol-required methods."""
        assert hasattr(BaseEmbeddingPlugin, "embed_texts")
        assert hasattr(BaseEmbeddingPlugin, "get_definition")
        assert hasattr(BaseEmbeddingPlugin, "supports_model")
        assert hasattr(BaseEmbeddingPlugin, "get_manifest")

    def test_satisfies_protocol_requirements(self) -> None:
        """BaseEmbeddingPlugin must satisfy all EmbeddingProtocol requirements.

        Note: issubclass() doesn't work with protocols that have ClassVar members,
        so we verify structural compliance by checking all requirements directly.
        """
        # Class variables
        assert hasattr(BaseEmbeddingPlugin, "PLUGIN_TYPE")
        assert hasattr(BaseEmbeddingPlugin, "PLUGIN_VERSION")
        assert hasattr(BaseEmbeddingPlugin, "INTERNAL_NAME")
        assert hasattr(BaseEmbeddingPlugin, "API_ID")
        assert hasattr(BaseEmbeddingPlugin, "PROVIDER_TYPE")
        assert hasattr(BaseEmbeddingPlugin, "METADATA")
        # Methods
        assert callable(getattr(BaseEmbeddingPlugin, "embed_texts", None))
        assert callable(getattr(BaseEmbeddingPlugin, "get_definition", None))
        assert callable(getattr(BaseEmbeddingPlugin, "supports_model", None))
        assert callable(getattr(BaseEmbeddingPlugin, "get_manifest", None))


class TestAgentProtocolCompliance:
    """Verify AgentPlugin satisfies AgentProtocol."""

    def test_has_plugin_type(self) -> None:
        """AgentPlugin must have PLUGIN_TYPE set to 'agent'."""
        assert hasattr(AgentPlugin, "PLUGIN_TYPE")
        assert isinstance(AgentPlugin.PLUGIN_TYPE, str)
        assert AgentPlugin.PLUGIN_TYPE == "agent"

    def test_inherits_plugin_class_vars(self) -> None:
        """AgentPlugin inherits PLUGIN_ID and PLUGIN_VERSION from SemanticPlugin."""
        # These are inherited from SemanticPlugin base class
        # They will be set by concrete subclasses
        from shared.plugins.base import SemanticPlugin

        assert issubclass(AgentPlugin, SemanticPlugin)

    def test_has_required_methods(self) -> None:
        """AgentPlugin must have all protocol-required methods."""
        assert hasattr(AgentPlugin, "execute")
        assert hasattr(AgentPlugin, "get_capabilities")
        assert hasattr(AgentPlugin, "supported_use_cases")
        assert hasattr(AgentPlugin, "get_manifest")

    def test_satisfies_protocol_requirements(self) -> None:
        """AgentPlugin must satisfy all AgentProtocol requirements.

        Note: issubclass() doesn't work with protocols that have ClassVar members,
        so we verify structural compliance by checking all requirements directly.
        """
        # Methods
        assert callable(getattr(AgentPlugin, "execute", None))
        assert callable(getattr(AgentPlugin, "get_capabilities", None))
        assert callable(getattr(AgentPlugin, "supported_use_cases", None))
        assert callable(getattr(AgentPlugin, "get_manifest", None))


class TestProtocolVersionConsistency:
    """Verify protocol version is correctly exposed."""

    def test_protocol_version_is_defined(self) -> None:
        """PROTOCOL_VERSION must be defined in protocols module."""
        from shared.plugins.protocols import PROTOCOL_VERSION

        assert PROTOCOL_VERSION is not None
        assert isinstance(PROTOCOL_VERSION, str)

    def test_protocol_version_is_semver(self) -> None:
        """PROTOCOL_VERSION must follow semantic versioning format."""
        from shared.plugins.protocols import PROTOCOL_VERSION

        parts = PROTOCOL_VERSION.split(".")
        assert len(parts) == 3
        assert all(part.isdigit() for part in parts)
