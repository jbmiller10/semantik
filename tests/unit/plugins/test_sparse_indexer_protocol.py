"""Tests for SparseIndexerProtocol compliance."""

from __future__ import annotations

from typing import Any, ClassVar

from shared.plugins.protocols import PROTOCOL_BY_TYPE, SparseIndexerProtocol
from shared.plugins.typed_dicts import (
    SPARSE_TYPES,
    SparseIndexerCapabilitiesDict,
    SparseQueryVectorDict,
    SparseSearchResultDict,
    SparseVectorDict,
)


class TestSparseTypesConstant:
    """Tests for SPARSE_TYPES protocol constant."""

    def test_sparse_types_is_frozenset(self) -> None:
        """SPARSE_TYPES should be immutable."""
        assert isinstance(SPARSE_TYPES, frozenset)

    def test_sparse_types_contains_bm25(self) -> None:
        """SPARSE_TYPES should include 'bm25'."""
        assert "bm25" in SPARSE_TYPES

    def test_sparse_types_contains_splade(self) -> None:
        """SPARSE_TYPES should include 'splade'."""
        assert "splade" in SPARSE_TYPES

    def test_sparse_types_has_exactly_two_values(self) -> None:
        """SPARSE_TYPES should have exactly bm25 and splade."""
        assert len(SPARSE_TYPES) == 2


class TestSparseVectorDict:
    """Tests for SparseVectorDict TypedDict."""

    def test_required_keys(self) -> None:
        """Verify required keys."""
        required = SparseVectorDict.__required_keys__
        assert "indices" in required
        assert "values" in required
        assert "chunk_id" in required

    def test_optional_keys(self) -> None:
        """Verify optional keys."""
        required = SparseVectorDict.__required_keys__
        assert "metadata" not in required
        assert "metadata" in SparseVectorDict.__annotations__

    def test_can_create_valid_dict(self) -> None:
        """Verify we can create a valid SparseVectorDict."""
        vec: SparseVectorDict = {
            "indices": [1, 5, 42],
            "values": [2.3, 1.8, 3.1],
            "chunk_id": "chunk-123",
        }
        assert vec["indices"] == [1, 5, 42]
        assert vec["chunk_id"] == "chunk-123"


class TestSparseQueryVectorDict:
    """Tests for SparseQueryVectorDict TypedDict."""

    def test_required_keys(self) -> None:
        """Verify required keys."""
        required = SparseQueryVectorDict.__required_keys__
        assert "indices" in required
        assert "values" in required

    def test_has_only_two_fields(self) -> None:
        """Query vectors should be minimal."""
        assert len(SparseQueryVectorDict.__annotations__) == 2

    def test_can_create_valid_dict(self) -> None:
        """Verify we can create a valid SparseQueryVectorDict."""
        vec: SparseQueryVectorDict = {
            "indices": [1, 42],
            "values": [1.0, 2.5],
        }
        assert vec["indices"] == [1, 42]


class TestSparseSearchResultDict:
    """Tests for SparseSearchResultDict TypedDict."""

    def test_required_keys(self) -> None:
        """Verify required keys."""
        required = SparseSearchResultDict.__required_keys__
        assert "chunk_id" in required
        assert "score" in required

    def test_optional_keys(self) -> None:
        """Verify optional keys."""
        required = SparseSearchResultDict.__required_keys__
        annotations = SparseSearchResultDict.__annotations__
        assert "matched_terms" not in required
        assert "matched_terms" in annotations
        assert "sparse_vector" not in required
        assert "sparse_vector" in annotations
        assert "payload" not in required
        assert "payload" in annotations

    def test_can_create_valid_dict(self) -> None:
        """Verify we can create a valid SparseSearchResultDict."""
        result: SparseSearchResultDict = {
            "chunk_id": "chunk-123",
            "score": 0.95,
        }
        assert result["chunk_id"] == "chunk-123"
        assert result["score"] == 0.95


class TestSparseIndexerCapabilitiesDict:
    """Tests for SparseIndexerCapabilitiesDict TypedDict."""

    def test_all_fields_optional(self) -> None:
        """Capabilities dict uses total=False, all optional."""
        # With total=False, __required_keys__ should be empty
        assert len(SparseIndexerCapabilitiesDict.__required_keys__) == 0

    def test_has_sparse_type_field(self) -> None:
        """Must have sparse_type field."""
        assert "sparse_type" in SparseIndexerCapabilitiesDict.__annotations__

    def test_has_key_capability_fields(self) -> None:
        """Must have key capability fields."""
        annotations = SparseIndexerCapabilitiesDict.__annotations__
        assert "max_tokens" in annotations
        assert "supports_batching" in annotations
        assert "requires_corpus_stats" in annotations
        assert "idf_storage" in annotations
        assert "max_terms_per_vector" in annotations
        assert "vocabulary_size" in annotations
        assert "vocabulary_handling" in annotations
        assert "max_batch_size" in annotations
        assert "supports_filters" in annotations
        assert "supported_languages" in annotations

    def test_can_create_valid_dict(self) -> None:
        """Verify we can create a valid SparseIndexerCapabilitiesDict."""
        caps: SparseIndexerCapabilitiesDict = {
            "sparse_type": "bm25",
            "supports_batching": True,
            "requires_corpus_stats": True,
        }
        assert caps["sparse_type"] == "bm25"


class TestSparseIndexerProtocolInMapping:
    """Tests for SparseIndexerProtocol registration."""

    def test_sparse_indexer_in_protocol_by_type(self) -> None:
        """SparseIndexerProtocol should be in PROTOCOL_BY_TYPE."""
        assert "sparse_indexer" in PROTOCOL_BY_TYPE
        assert PROTOCOL_BY_TYPE["sparse_indexer"] is SparseIndexerProtocol


class TestSparseIndexerProtocolCompliance:
    """Tests for protocol structural compliance."""

    def test_valid_external_plugin_satisfies_protocol(self) -> None:
        """A valid external plugin should satisfy the protocol."""

        class ValidExternalPlugin:
            PLUGIN_ID: ClassVar[str] = "test-sparse"
            PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"
            SPARSE_TYPE: ClassVar[str] = "bm25"

            async def encode_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [{"indices": [1], "values": [1.0], "chunk_id": doc["chunk_id"]} for doc in documents]

            async def encode_query(self, query: str) -> dict[str, Any]:
                return {"indices": [1], "values": [1.0]}

            async def remove_documents(self, chunk_ids: list[str]) -> None:
                pass

            @classmethod
            def get_capabilities(cls) -> dict[str, Any]:
                return {"sparse_type": "bm25", "supports_batching": True}

            @classmethod
            def get_manifest(cls) -> dict[str, Any]:
                return {
                    "id": cls.PLUGIN_ID,
                    "type": cls.PLUGIN_TYPE,
                    "version": cls.PLUGIN_VERSION,
                }

        # Verify instance satisfies protocol
        instance = ValidExternalPlugin()
        assert isinstance(instance, SparseIndexerProtocol)

    def test_missing_sparse_type_fails_protocol(self) -> None:
        """Plugin missing SPARSE_TYPE should not satisfy protocol."""

        class MissingSparseTypePlugin:
            PLUGIN_ID: ClassVar[str] = "test"
            PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"
            # Missing SPARSE_TYPE

            async def encode_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return []

            async def encode_query(self, query: str) -> dict[str, Any]:
                return {"indices": [], "values": []}

            async def remove_documents(self, chunk_ids: list[str]) -> None:
                pass

            @classmethod
            def get_capabilities(cls) -> dict[str, Any]:
                return {}

            @classmethod
            def get_manifest(cls) -> dict[str, Any]:
                return {}

        instance = MissingSparseTypePlugin()
        assert not isinstance(instance, SparseIndexerProtocol)

    def test_missing_method_fails_protocol(self) -> None:
        """Plugin missing required method should not satisfy protocol."""

        class MissingMethodPlugin:
            PLUGIN_ID: ClassVar[str] = "test"
            PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"
            SPARSE_TYPE: ClassVar[str] = "bm25"

            async def encode_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return []

            # Missing encode_query, remove_documents

            @classmethod
            def get_manifest(cls) -> dict[str, Any]:
                return {}

        instance = MissingMethodPlugin()
        assert not isinstance(instance, SparseIndexerProtocol)

    def test_splade_type_plugin_satisfies_protocol(self) -> None:
        """A SPLADE-type plugin should satisfy the protocol."""

        class SPLADEPlugin:
            PLUGIN_ID: ClassVar[str] = "test-splade"
            PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"
            PLUGIN_VERSION: ClassVar[str] = "1.0.0"
            SPARSE_TYPE: ClassVar[str] = "splade"

            async def encode_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
                return [{"indices": [100, 200], "values": [0.5, 0.8], "chunk_id": doc["chunk_id"]} for doc in documents]

            async def encode_query(self, query: str) -> dict[str, Any]:
                return {"indices": [100], "values": [0.9]}

            async def remove_documents(self, chunk_ids: list[str]) -> None:
                # SPLADE is stateless, no-op
                pass

            @classmethod
            def get_capabilities(cls) -> dict[str, Any]:
                return {
                    "sparse_type": "splade",
                    "supports_batching": True,
                    "requires_corpus_stats": False,
                }

            @classmethod
            def get_manifest(cls) -> dict[str, Any]:
                return {
                    "id": cls.PLUGIN_ID,
                    "type": cls.PLUGIN_TYPE,
                    "version": cls.PLUGIN_VERSION,
                }

        instance = SPLADEPlugin()
        assert isinstance(instance, SparseIndexerProtocol)


class TestSparseIndexerProtocolClassVars:
    """Tests that protocol declares correct ClassVar attributes."""

    def test_protocol_requires_plugin_id(self) -> None:
        """Protocol should require PLUGIN_ID."""
        # Check that PLUGIN_ID is declared in the protocol
        assert hasattr(SparseIndexerProtocol, "__protocol_attrs__")
        attrs = SparseIndexerProtocol.__protocol_attrs__
        assert "PLUGIN_ID" in attrs or "PLUGIN_ID" in getattr(SparseIndexerProtocol, "__annotations__", {})

    def test_protocol_requires_plugin_type(self) -> None:
        """Protocol should require PLUGIN_TYPE."""
        attrs = SparseIndexerProtocol.__protocol_attrs__
        assert "PLUGIN_TYPE" in attrs or "PLUGIN_TYPE" in getattr(SparseIndexerProtocol, "__annotations__", {})

    def test_protocol_requires_plugin_version(self) -> None:
        """Protocol should require PLUGIN_VERSION."""
        attrs = SparseIndexerProtocol.__protocol_attrs__
        assert "PLUGIN_VERSION" in attrs or "PLUGIN_VERSION" in getattr(SparseIndexerProtocol, "__annotations__", {})

    def test_protocol_requires_sparse_type(self) -> None:
        """Protocol should require SPARSE_TYPE."""
        attrs = SparseIndexerProtocol.__protocol_attrs__
        assert "SPARSE_TYPE" in attrs or "SPARSE_TYPE" in getattr(SparseIndexerProtocol, "__annotations__", {})
