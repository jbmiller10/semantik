"""Tests for plugin protocol TypedDict definitions and constants."""



class TestProtocolConstants:
    """Tests for protocol constants (frozensets)."""

    def test_message_roles_is_frozenset(self):
        """Verify MESSAGE_ROLES is an immutable frozenset."""
        from shared.plugins.typed_dicts import MESSAGE_ROLES

        assert isinstance(MESSAGE_ROLES, frozenset)

    def test_message_roles_values(self):
        """Verify MESSAGE_ROLES contains expected values."""
        from shared.plugins.typed_dicts import MESSAGE_ROLES

        expected = {"user", "assistant", "system", "tool_call", "tool_result", "error"}
        assert expected == MESSAGE_ROLES

    def test_message_types_is_frozenset(self):
        """Verify MESSAGE_TYPES is an immutable frozenset."""
        from shared.plugins.typed_dicts import MESSAGE_TYPES

        assert isinstance(MESSAGE_TYPES, frozenset)

    def test_message_types_values(self):
        """Verify MESSAGE_TYPES contains expected values."""
        from shared.plugins.typed_dicts import MESSAGE_TYPES

        expected = {
            "text",
            "thinking",
            "tool_use",
            "tool_output",
            "partial",
            "final",
            "error",
            "metadata",
        }
        assert expected == MESSAGE_TYPES

    def test_embedding_modes_is_frozenset(self):
        """Verify EMBEDDING_MODES is an immutable frozenset."""
        from shared.plugins.typed_dicts import EMBEDDING_MODES

        assert isinstance(EMBEDDING_MODES, frozenset)
        assert {"query", "document"} == EMBEDDING_MODES

    def test_extraction_types_is_frozenset(self):
        """Verify EXTRACTION_TYPES is an immutable frozenset."""
        from shared.plugins.typed_dicts import EXTRACTION_TYPES

        assert isinstance(EXTRACTION_TYPES, frozenset)
        expected = {
            "entities",
            "keywords",
            "language",
            "topics",
            "sentiment",
            "summary",
            "custom",
        }
        assert expected == EXTRACTION_TYPES

    def test_agent_use_cases_is_frozenset(self):
        """Verify AGENT_USE_CASES is an immutable frozenset."""
        from shared.plugins.typed_dicts import AGENT_USE_CASES

        assert isinstance(AGENT_USE_CASES, frozenset)
        expected = {
            "hyde",
            "query_expansion",
            "query_understanding",
            "summarization",
            "reranking",
            "answer_synthesis",
            "tool_use",
            "agentic_search",
            "reasoning",
            "assistant",
            "code_generation",
            "data_analysis",
        }
        assert expected == AGENT_USE_CASES

    def test_protocol_version_exists(self):
        """Verify PROTOCOL_VERSION is defined."""
        from shared.plugins.typed_dicts import PROTOCOL_VERSION

        assert PROTOCOL_VERSION == "1.0.0"


class TestTypedDictsImportable:
    """Tests that all TypedDicts can be imported."""

    def test_common_dtos_importable(self):
        """Verify common DTOs are importable."""
        from shared.plugins.typed_dicts import PluginManifestDict

        assert PluginManifestDict is not None

    def test_connector_dtos_importable(self):
        """Verify connector DTOs are importable."""
        from shared.plugins.typed_dicts import IngestedDocumentDict

        assert IngestedDocumentDict is not None

    def test_chunking_dtos_importable(self):
        """Verify chunking DTOs are importable."""
        from shared.plugins.typed_dicts import (
            ChunkConfigDict,
            ChunkDict,
            ChunkMetadataDict,
        )

        assert ChunkDict is not None
        assert ChunkMetadataDict is not None
        assert ChunkConfigDict is not None

    def test_reranker_dtos_importable(self):
        """Verify reranker DTOs are importable."""
        from shared.plugins.typed_dicts import (
            RerankerCapabilitiesDict,
            RerankResultDict,
        )

        assert RerankResultDict is not None
        assert RerankerCapabilitiesDict is not None

    def test_extractor_dtos_importable(self):
        """Verify extractor DTOs are importable."""
        from shared.plugins.typed_dicts import EntityDict, ExtractionResultDict

        assert EntityDict is not None
        assert ExtractionResultDict is not None

    def test_agent_dtos_importable(self):
        """Verify agent DTOs are importable."""
        from shared.plugins.typed_dicts import (
            AgentCapabilitiesDict,
            AgentContextDict,
            AgentMessageDict,
            TokenUsageDict,
        )

        assert TokenUsageDict is not None
        assert AgentMessageDict is not None
        assert AgentCapabilitiesDict is not None
        assert AgentContextDict is not None

    def test_embedding_dtos_importable(self):
        """Verify embedding DTOs are importable."""
        from shared.plugins.typed_dicts import EmbeddingProviderDefinitionDict

        assert EmbeddingProviderDefinitionDict is not None


class TestTypedDictStructure:
    """Tests that TypedDicts have expected structure."""

    def test_ingested_document_required_keys(self):
        """Verify IngestedDocumentDict has required keys."""
        from shared.plugins.typed_dicts import IngestedDocumentDict

        required_keys = IngestedDocumentDict.__required_keys__
        assert "content" in required_keys
        assert "unique_id" in required_keys
        assert "source_type" in required_keys
        assert "metadata" in required_keys
        assert "content_hash" in required_keys

    def test_ingested_document_has_file_path_field(self):
        """Verify IngestedDocumentDict has file_path field."""
        from shared.plugins.typed_dicts import IngestedDocumentDict

        # file_path is defined with NotRequired, so it's in annotations but not required
        annotations = IngestedDocumentDict.__annotations__
        assert "file_path" in annotations
        assert "file_path" not in IngestedDocumentDict.__required_keys__

    def test_agent_message_required_keys(self):
        """Verify AgentMessageDict has required keys."""
        from shared.plugins.typed_dicts import AgentMessageDict

        required_keys = AgentMessageDict.__required_keys__
        assert "id" in required_keys
        assert "role" in required_keys
        assert "type" in required_keys
        assert "content" in required_keys
        assert "timestamp" in required_keys

    def test_chunk_dict_required_keys(self):
        """Verify ChunkDict has required keys."""
        from shared.plugins.typed_dicts import ChunkDict

        required_keys = ChunkDict.__required_keys__
        assert "content" in required_keys
        assert "metadata" in required_keys

    def test_rerank_result_required_keys(self):
        """Verify RerankResultDict has required keys."""
        from shared.plugins.typed_dicts import RerankResultDict

        required_keys = RerankResultDict.__required_keys__
        assert "index" in required_keys
        assert "score" in required_keys

    def test_entity_dict_required_keys(self):
        """Verify EntityDict has required keys."""
        from shared.plugins.typed_dicts import EntityDict

        required_keys = EntityDict.__required_keys__
        assert "text" in required_keys
        assert "type" in required_keys
