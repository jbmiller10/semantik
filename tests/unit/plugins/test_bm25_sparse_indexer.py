"""Tests for BM25SparseIndexerPlugin."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from shared.plugins.builtins.bm25_sparse_indexer import (
    DEFAULT_B,
    DEFAULT_K1,
    ENGLISH_STOPWORDS,
    BM25SparseIndexerPlugin,
)
from shared.plugins.protocols import SparseIndexerProtocol
from shared.plugins.types.sparse_indexer import SparseIndexerCapabilities, SparseQueryVector, SparseVector


class TestBM25PluginAttributes:
    """Tests for plugin class attributes."""

    def test_plugin_id(self) -> None:
        """Plugin should have correct PLUGIN_ID."""
        assert BM25SparseIndexerPlugin.PLUGIN_ID == "bm25-local"

    def test_plugin_type(self) -> None:
        """Plugin should have correct PLUGIN_TYPE."""
        assert BM25SparseIndexerPlugin.PLUGIN_TYPE == "sparse_indexer"

    def test_plugin_version(self) -> None:
        """Plugin should have a version string."""
        assert BM25SparseIndexerPlugin.PLUGIN_VERSION == "1.0.0"

    def test_sparse_type(self) -> None:
        """Plugin should have SPARSE_TYPE = 'bm25'."""
        assert BM25SparseIndexerPlugin.SPARSE_TYPE == "bm25"

    def test_metadata_has_required_keys(self) -> None:
        """Plugin METADATA should have display_name and description."""
        assert "display_name" in BM25SparseIndexerPlugin.METADATA
        assert "description" in BM25SparseIndexerPlugin.METADATA
        assert "author" in BM25SparseIndexerPlugin.METADATA


class TestBM25PluginCapabilities:
    """Tests for get_capabilities() method."""

    def test_get_capabilities_returns_correct_type(self) -> None:
        """get_capabilities should return SparseIndexerCapabilities."""
        caps = BM25SparseIndexerPlugin.get_capabilities()
        assert isinstance(caps, SparseIndexerCapabilities)

    def test_capabilities_sparse_type(self) -> None:
        """Capabilities should report sparse_type='bm25'."""
        caps = BM25SparseIndexerPlugin.get_capabilities()
        assert caps.sparse_type == "bm25"

    def test_capabilities_requires_corpus_stats(self) -> None:
        """BM25 requires corpus statistics for IDF."""
        caps = BM25SparseIndexerPlugin.get_capabilities()
        assert caps.requires_corpus_stats is True

    def test_capabilities_supports_batching(self) -> None:
        """BM25 supports batch encoding."""
        caps = BM25SparseIndexerPlugin.get_capabilities()
        assert caps.supports_batching is True

    def test_capabilities_idf_storage(self) -> None:
        """BM25 uses file-based IDF storage."""
        caps = BM25SparseIndexerPlugin.get_capabilities()
        assert caps.idf_storage == "file"


class TestBM25PluginManifest:
    """Tests for get_manifest() method."""

    def test_get_manifest_returns_plugin_manifest(self) -> None:
        """get_manifest should return a PluginManifest."""
        from shared.plugins.manifest import PluginManifest

        manifest = BM25SparseIndexerPlugin.get_manifest()
        assert isinstance(manifest, PluginManifest)

    def test_manifest_has_correct_id(self) -> None:
        """Manifest should have correct id."""
        manifest = BM25SparseIndexerPlugin.get_manifest()
        assert manifest.id == "bm25-local"

    def test_manifest_has_correct_type(self) -> None:
        """Manifest should have correct type."""
        manifest = BM25SparseIndexerPlugin.get_manifest()
        assert manifest.type == "sparse_indexer"

    def test_manifest_has_capabilities(self) -> None:
        """Manifest should include capabilities."""
        manifest = BM25SparseIndexerPlugin.get_manifest()
        assert manifest.capabilities is not None
        assert "sparse_type" in manifest.capabilities


class TestBM25PluginConfigSchema:
    """Tests for get_config_schema() method."""

    def test_get_config_schema_returns_json_schema(self) -> None:
        """get_config_schema should return a valid JSON Schema."""
        schema = BM25SparseIndexerPlugin.get_config_schema()
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_config_schema_has_k1_property(self) -> None:
        """Config schema should include k1 parameter."""
        schema = BM25SparseIndexerPlugin.get_config_schema()
        assert "k1" in schema["properties"]
        assert schema["properties"]["k1"]["type"] == "number"

    def test_config_schema_has_b_property(self) -> None:
        """Config schema should include b parameter."""
        schema = BM25SparseIndexerPlugin.get_config_schema()
        assert "b" in schema["properties"]
        assert schema["properties"]["b"]["type"] == "number"

    def test_config_schema_has_lowercase_property(self) -> None:
        """Config schema should include lowercase option."""
        schema = BM25SparseIndexerPlugin.get_config_schema()
        assert "lowercase" in schema["properties"]
        assert schema["properties"]["lowercase"]["type"] == "boolean"


class TestBM25PluginHealthCheck:
    """Tests for health_check() method."""

    @pytest.mark.asyncio()
    async def test_health_check_returns_true(self) -> None:
        """BM25 has no external dependencies, always healthy."""
        result = await BM25SparseIndexerPlugin.health_check()
        assert result is True

    @pytest.mark.asyncio()
    async def test_health_check_with_config(self) -> None:
        """Health check should accept config parameter."""
        result = await BM25SparseIndexerPlugin.health_check({"k1": 2.0})
        assert result is True


class TestBM25PluginInitialization:
    """Tests for plugin initialization."""

    def test_default_config(self) -> None:
        """Plugin should use default BM25 parameters."""
        plugin = BM25SparseIndexerPlugin()
        assert plugin._k1 == DEFAULT_K1
        assert plugin._b == DEFAULT_B

    def test_config_override_in_constructor(self) -> None:
        """Plugin should accept config in constructor."""
        plugin = BM25SparseIndexerPlugin({"k1": 2.0, "b": 0.5})
        assert plugin._k1 == 2.0
        assert plugin._b == 0.5

    @pytest.mark.asyncio()
    async def test_initialize_updates_config(self) -> None:
        """initialize() should update config parameters."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.initialize({"k1": 2.5, "b": 0.8})
        assert plugin._k1 == 2.5
        assert plugin._b == 0.8

    @pytest.mark.asyncio()
    async def test_initialize_sets_initialized_flag(self) -> None:
        """initialize() should set _initialized flag."""
        plugin = BM25SparseIndexerPlugin()
        assert not plugin._initialized
        await plugin.initialize()
        assert plugin._initialized


class TestBM25Tokenization:
    """Tests for text tokenization."""

    def test_simple_tokenization(self) -> None:
        """Basic tokenization should split on whitespace."""
        plugin = BM25SparseIndexerPlugin()
        tokens = plugin._tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_lowercase_tokenization(self) -> None:
        """Tokenization should lowercase by default."""
        plugin = BM25SparseIndexerPlugin()
        tokens = plugin._tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens
        assert "Hello" not in tokens

    def test_disable_lowercase(self) -> None:
        """Tokenization can preserve case."""
        plugin = BM25SparseIndexerPlugin({"lowercase": False})
        tokens = plugin._tokenize("Hello World")
        assert "Hello" in tokens
        assert "World" in tokens

    def test_stopword_removal(self) -> None:
        """Tokenization should remove stopwords by default."""
        plugin = BM25SparseIndexerPlugin()
        tokens = plugin._tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_disable_stopword_removal(self) -> None:
        """Stopword removal can be disabled."""
        plugin = BM25SparseIndexerPlugin({"remove_stopwords": False})
        tokens = plugin._tokenize("the quick brown fox")
        assert "the" in tokens

    def test_min_token_length(self) -> None:
        """Tokens shorter than min_token_length are removed."""
        plugin = BM25SparseIndexerPlugin({"min_token_length": 3})
        tokens = plugin._tokenize("I am a developer")
        assert "developer" in tokens
        # "I", "am", "a" are too short (< 3 chars)
        assert "am" not in tokens

    def test_empty_string(self) -> None:
        """Empty string returns empty list."""
        plugin = BM25SparseIndexerPlugin()
        tokens = plugin._tokenize("")
        assert tokens == []

    def test_punctuation_handling(self) -> None:
        """Punctuation should be stripped."""
        plugin = BM25SparseIndexerPlugin()
        tokens = plugin._tokenize("hello, world! how are you?")
        assert "hello" in tokens
        assert "world" in tokens
        # "how", "are", "you" might be stopwords or short

    def test_hyphenated_words(self) -> None:
        """Hyphenated words should be kept together."""
        plugin = BM25SparseIndexerPlugin()
        tokens = plugin._tokenize("self-hosted semantic search")
        # Hyphenated words are kept together
        assert "self-hosted" in tokens


class TestBM25EncodeDocuments:
    """Tests for encode_documents() method."""

    @pytest.mark.asyncio()
    async def test_encode_single_document(self) -> None:
        """Encoding a single document should return one SparseVector."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "hello world example", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        assert len(result) == 1
        assert isinstance(result[0], SparseVector)
        assert result[0].chunk_id == "chunk-1"

    @pytest.mark.asyncio()
    async def test_encode_batch_documents(self) -> None:
        """Encoding multiple documents should return one vector per document."""
        plugin = BM25SparseIndexerPlugin()
        documents = [
            {"content": "first document", "chunk_id": "chunk-1"},
            {"content": "second document", "chunk_id": "chunk-2"},
            {"content": "third document", "chunk_id": "chunk-3"},
        ]

        result = await plugin.encode_documents(documents)

        assert len(result) == 3
        assert result[0].chunk_id == "chunk-1"
        assert result[1].chunk_id == "chunk-2"
        assert result[2].chunk_id == "chunk-3"

    @pytest.mark.asyncio()
    async def test_encode_empty_document(self) -> None:
        """Empty document should return empty sparse vector."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        assert len(result) == 1
        assert result[0].indices == ()
        assert result[0].values == ()

    @pytest.mark.asyncio()
    async def test_encode_document_with_metadata(self) -> None:
        """Document metadata should be preserved in result."""
        plugin = BM25SparseIndexerPlugin()
        documents = [
            {
                "content": "test document",
                "chunk_id": "chunk-1",
                "metadata": {"source": "test"},
            }
        ]

        result = await plugin.encode_documents(documents)

        assert result[0].metadata == {"source": "test"}

    @pytest.mark.asyncio()
    async def test_sparse_vector_indices_sorted(self) -> None:
        """Sparse vector indices should be sorted ascending."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "zebra apple banana cherry", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        indices = list(result[0].indices)
        assert indices == sorted(indices)

    @pytest.mark.asyncio()
    async def test_sparse_vector_values_positive(self) -> None:
        """BM25 values should be positive."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "example test document", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        for value in result[0].values:
            assert value > 0

    @pytest.mark.asyncio()
    async def test_encode_updates_idf_stats(self) -> None:
        """Encoding documents should update IDF statistics."""
        plugin = BM25SparseIndexerPlugin()
        assert plugin._document_count == 0

        documents = [
            {"content": "hello world", "chunk_id": "chunk-1"},
            {"content": "hello python", "chunk_id": "chunk-2"},
        ]
        await plugin.encode_documents(documents)

        assert plugin._document_count == 2
        assert "hello" in plugin._term_doc_freqs
        assert plugin._term_doc_freqs["hello"] == 2  # Appears in both docs

    @pytest.mark.asyncio()
    async def test_empty_document_list(self) -> None:
        """Empty document list should return empty list."""
        plugin = BM25SparseIndexerPlugin()
        result = await plugin.encode_documents([])
        assert result == []


class TestBM25EncodeQuery:
    """Tests for encode_query() method."""

    @pytest.mark.asyncio()
    async def test_encode_query_returns_sparse_query_vector(self) -> None:
        """encode_query should return SparseQueryVector."""
        plugin = BM25SparseIndexerPlugin()
        # First encode some documents to build vocabulary
        await plugin.encode_documents([{"content": "hello world python", "chunk_id": "chunk-1"}])

        result = await plugin.encode_query("hello world")

        assert isinstance(result, SparseQueryVector)

    @pytest.mark.asyncio()
    async def test_encode_query_empty_string(self) -> None:
        """Empty query should return empty sparse vector."""
        plugin = BM25SparseIndexerPlugin()
        result = await plugin.encode_query("")

        assert result.indices == ()
        assert result.values == ()

    @pytest.mark.asyncio()
    async def test_encode_query_unknown_terms(self) -> None:
        """Query with unknown terms returns empty vector."""
        plugin = BM25SparseIndexerPlugin()
        # No documents encoded yet, so all terms are unknown
        result = await plugin.encode_query("hello world")

        assert result.indices == ()
        assert result.values == ()

    @pytest.mark.asyncio()
    async def test_encode_query_known_terms(self) -> None:
        """Query with known terms returns non-empty vector."""
        plugin = BM25SparseIndexerPlugin()
        # Build vocabulary
        await plugin.encode_documents(
            [
                {"content": "hello world", "chunk_id": "chunk-1"},
                {"content": "python programming", "chunk_id": "chunk-2"},
            ]
        )

        result = await plugin.encode_query("hello python")

        assert len(result.indices) > 0
        assert len(result.values) > 0

    @pytest.mark.asyncio()
    async def test_encode_query_indices_sorted(self) -> None:
        """Query vector indices should be sorted."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents(
            [
                {"content": "zebra apple banana", "chunk_id": "chunk-1"},
            ]
        )

        result = await plugin.encode_query("zebra apple banana")

        indices = list(result.indices)
        assert indices == sorted(indices)


class TestBM25RemoveDocuments:
    """Tests for remove_documents() method."""

    @pytest.mark.asyncio()
    async def test_remove_documents_updates_count(self) -> None:
        """Removing documents should decrement document count."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents(
            [
                {"content": "first document", "chunk_id": "chunk-1"},
                {"content": "second document", "chunk_id": "chunk-2"},
            ]
        )
        assert plugin._document_count == 2

        await plugin.remove_documents(["chunk-1"])

        assert plugin._document_count == 1

    @pytest.mark.asyncio()
    async def test_remove_documents_updates_term_freqs(self) -> None:
        """Removing documents should update term frequencies."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents(
            [
                {"content": "unique term", "chunk_id": "chunk-1"},
                {"content": "another document", "chunk_id": "chunk-2"},
            ]
        )
        assert "unique" in plugin._term_doc_freqs

        await plugin.remove_documents(["chunk-1"])

        # "unique" only appeared in chunk-1, should be gone
        assert "unique" not in plugin._term_doc_freqs

    @pytest.mark.asyncio()
    async def test_remove_documents_unknown_chunk(self) -> None:
        """Removing unknown chunk should not raise error."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents([{"content": "test document", "chunk_id": "chunk-1"}])

        # Should not raise
        await plugin.remove_documents(["unknown-chunk"])

        # Original should still exist
        assert plugin._document_count == 1

    @pytest.mark.asyncio()
    async def test_remove_all_documents(self) -> None:
        """Removing all documents should reset stats."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents(
            [
                {"content": "document one", "chunk_id": "chunk-1"},
                {"content": "document two", "chunk_id": "chunk-2"},
            ]
        )

        await plugin.remove_documents(["chunk-1", "chunk-2"])

        assert plugin._document_count == 0
        assert plugin._avg_doc_length == 0.0


class TestBM25IDFComputation:
    """Tests for IDF computation."""

    @pytest.mark.asyncio()
    async def test_idf_common_term_lower_value(self) -> None:
        """Common terms should have lower IDF."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents(
            [
                {"content": "hello world", "chunk_id": "chunk-1"},
                {"content": "hello python", "chunk_id": "chunk-2"},
                {"content": "hello java", "chunk_id": "chunk-3"},
            ]
        )

        # "hello" appears in all docs, "world" only in one
        idf_hello = plugin._compute_idf("hello")
        idf_world = plugin._compute_idf("world")

        assert idf_world > idf_hello

    @pytest.mark.asyncio()
    async def test_idf_unknown_term_zero(self) -> None:
        """Unknown term should have IDF of 0."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents([{"content": "hello world", "chunk_id": "chunk-1"}])

        idf = plugin._compute_idf("unknown")
        assert idf == 0.0

    @pytest.mark.asyncio()
    async def test_idf_formula_correct(self) -> None:
        """Verify BM25 IDF formula: log((N - n + 0.5) / (n + 0.5) + 1)."""
        plugin = BM25SparseIndexerPlugin()
        await plugin.encode_documents(
            [
                {"content": "apple banana", "chunk_id": "chunk-1"},
                {"content": "apple cherry", "chunk_id": "chunk-2"},
                {"content": "banana cherry", "chunk_id": "chunk-3"},
            ]
        )

        # N = 3, n("apple") = 2
        n = 3
        doc_freq = 2
        expected_idf = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)

        actual_idf = plugin._compute_idf("apple")
        assert abs(actual_idf - expected_idf) < 0.0001


class TestBM25VectorComputation:
    """Tests for BM25 sparse vector computation."""

    @pytest.mark.asyncio()
    async def test_term_frequency_affects_weight(self) -> None:
        """Higher term frequency should increase weight (with saturation)."""
        plugin = BM25SparseIndexerPlugin()
        # Use distinct documents to get proper IDF
        await plugin.encode_documents(
            [
                {"content": "apple apple apple banana", "chunk_id": "chunk-1"},
                {"content": "cherry date", "chunk_id": "chunk-2"},  # Different doc for IDF
            ]
        )

        result_multi = await plugin.encode_documents([{"content": "apple apple apple", "chunk_id": "chunk-3"}])
        result_single = await plugin.encode_documents([{"content": "apple", "chunk_id": "chunk-4"}])

        # Find weight for "apple" in each
        def get_apple_weight(sparse_vec: SparseVector) -> float:
            apple_id = plugin._term_to_id.get("apple")
            if apple_id is None:
                return 0.0
            for idx, val in zip(sparse_vec.indices, sparse_vec.values, strict=False):
                if idx == apple_id:
                    return val
            return 0.0

        weight_multi = get_apple_weight(result_multi[0])
        weight_single = get_apple_weight(result_single[0])

        # Weight should be higher for more occurrences (but with saturation)
        assert weight_multi > weight_single

    @pytest.mark.asyncio()
    async def test_length_normalization(self) -> None:
        """Longer documents should have lower per-term weights (with b > 0)."""
        plugin = BM25SparseIndexerPlugin({"b": 0.75})  # Default normalization
        await plugin.encode_documents(
            [
                {"content": "apple", "chunk_id": "chunk-1"},
                {"content": "apple banana cherry date elderberry fig grape", "chunk_id": "chunk-2"},
            ]
        )

        # Re-encode to get fresh vectors with updated stats
        result_short = await plugin.encode_documents([{"content": "apple", "chunk_id": "chunk-3"}])
        result_long = await plugin.encode_documents(
            [{"content": "apple banana cherry date elderberry fig grape", "chunk_id": "chunk-4"}]
        )

        def get_apple_weight(sparse_vec: SparseVector) -> float:
            apple_id = plugin._term_to_id.get("apple")
            if apple_id is None:
                return 0.0
            for idx, val in zip(sparse_vec.indices, sparse_vec.values, strict=False):
                if idx == apple_id:
                    return val
            return 0.0

        weight_short = get_apple_weight(result_short[0])
        weight_long = get_apple_weight(result_long[0])

        # The term "apple" should have relatively higher weight in shorter doc
        # (normalized by document length)
        assert weight_short > weight_long


class TestBM25IDFPersistence:
    """Tests for IDF statistics persistence."""

    @pytest.mark.asyncio()
    async def test_save_load_idf_stats(self, tmp_path: Path) -> None:
        """IDF stats should persist to file and load correctly."""
        idf_file = tmp_path / "sparse_indexes" / "test_collection" / "idf_stats.json"

        # Create and populate plugin
        plugin1 = BM25SparseIndexerPlugin()
        plugin1._idf_path = idf_file
        await plugin1.encode_documents(
            [
                {"content": "hello world", "chunk_id": "chunk-1"},
                {"content": "hello python", "chunk_id": "chunk-2"},
            ]
        )

        # Save
        await plugin1._save_idf_stats()
        assert idf_file.exists()

        # Create new plugin and load
        plugin2 = BM25SparseIndexerPlugin()
        plugin2._idf_path = idf_file
        await plugin2._load_idf_stats()

        # Verify state was loaded
        assert plugin2._document_count == 2
        assert "hello" in plugin2._term_doc_freqs
        assert plugin2._term_doc_freqs["hello"] == 2

    @pytest.mark.asyncio()
    async def test_cleanup_saves_dirty_stats(self, tmp_path: Path) -> None:
        """cleanup() should save IDF stats if modified."""
        idf_file = tmp_path / "sparse_indexes" / "test_collection" / "idf_stats.json"

        plugin = BM25SparseIndexerPlugin()
        plugin._idf_path = idf_file
        await plugin.encode_documents([{"content": "test document", "chunk_id": "chunk-1"}])

        assert plugin._dirty  # Should be marked dirty

        await plugin.cleanup()

        assert idf_file.exists()
        assert not plugin._dirty

    @pytest.mark.asyncio()
    async def test_idf_version_increments(self, tmp_path: Path) -> None:
        """IDF version should increment on each save."""
        idf_file = tmp_path / "sparse_indexes" / "test_collection" / "idf_stats.json"

        plugin = BM25SparseIndexerPlugin()
        plugin._idf_path = idf_file
        assert plugin._idf_version == 0

        await plugin.encode_documents([{"content": "test", "chunk_id": "chunk-1"}])
        await plugin._save_idf_stats()

        # Check file contains version 1
        data = json.loads(idf_file.read_text())
        assert data["version"] == 1
        assert plugin._idf_version == 1

        await plugin.encode_documents([{"content": "another", "chunk_id": "chunk-2"}])
        await plugin._save_idf_stats()

        data = json.loads(idf_file.read_text())
        assert data["version"] == 2


class TestBM25ProtocolCompliance:
    """Tests for SparseIndexerProtocol compliance."""

    def test_satisfies_sparse_indexer_protocol(self) -> None:
        """BM25SparseIndexerPlugin should satisfy SparseIndexerProtocol."""
        plugin = BM25SparseIndexerPlugin()
        assert isinstance(plugin, SparseIndexerProtocol)

    def test_has_required_class_vars(self) -> None:
        """Plugin should have all required class variables."""
        assert hasattr(BM25SparseIndexerPlugin, "PLUGIN_ID")
        assert hasattr(BM25SparseIndexerPlugin, "PLUGIN_TYPE")
        assert hasattr(BM25SparseIndexerPlugin, "PLUGIN_VERSION")
        assert hasattr(BM25SparseIndexerPlugin, "SPARSE_TYPE")

    def test_has_required_methods(self) -> None:
        """Plugin should have all required methods."""
        assert callable(getattr(BM25SparseIndexerPlugin, "encode_documents", None))
        assert callable(getattr(BM25SparseIndexerPlugin, "encode_query", None))
        assert callable(getattr(BM25SparseIndexerPlugin, "remove_documents", None))
        assert callable(getattr(BM25SparseIndexerPlugin, "get_capabilities", None))
        assert callable(getattr(BM25SparseIndexerPlugin, "get_manifest", None))


class TestBM25EdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio()
    async def test_document_with_only_stopwords(self) -> None:
        """Document with only stopwords should return empty vector."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "the a an", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        # All tokens are stopwords, so should be empty
        assert len(result[0].indices) == 0

    @pytest.mark.asyncio()
    async def test_document_with_numbers(self) -> None:
        """Documents with numbers should handle them correctly."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "version 123 release 456", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        # Should have some indices (version, release, numbers)
        assert len(result[0].indices) >= 2

    @pytest.mark.asyncio()
    async def test_unicode_content(self) -> None:
        """Unicode content should be handled correctly."""
        plugin = BM25SparseIndexerPlugin()
        documents = [{"content": "caf\u00e9 na\u00efve r\u00e9sum\u00e9", "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        # Should not raise and should have indices
        assert len(result[0].indices) >= 1

    @pytest.mark.asyncio()
    async def test_reindex_same_chunk(self) -> None:
        """Re-encoding same chunk_id should update, not duplicate."""
        plugin = BM25SparseIndexerPlugin()

        # First encoding
        await plugin.encode_documents([{"content": "hello world", "chunk_id": "chunk-1"}])
        assert plugin._document_count == 1

        # Re-encode same chunk with different content
        await plugin.encode_documents([{"content": "goodbye world", "chunk_id": "chunk-1"}])

        # Should still be 1 document
        assert plugin._document_count == 1
        # "goodbye" should now be in vocabulary
        assert "goodbye" in plugin._term_to_id

    @pytest.mark.asyncio()
    async def test_very_long_document(self) -> None:
        """Long documents should be handled correctly."""
        plugin = BM25SparseIndexerPlugin()
        long_content = " ".join([f"word{i}" for i in range(1000)])
        documents = [{"content": long_content, "chunk_id": "chunk-1"}]

        result = await plugin.encode_documents(documents)

        # Should have many indices
        assert len(result[0].indices) > 100

    @pytest.mark.asyncio()
    async def test_get_sparse_collection_name(self) -> None:
        """get_sparse_collection_name should format correctly."""
        plugin = BM25SparseIndexerPlugin()
        name = plugin.get_sparse_collection_name("my_collection")
        assert name == "my_collection_sparse_bm25"


class TestBM25Stopwords:
    """Tests for stopwords handling."""

    def test_english_stopwords_loaded(self) -> None:
        """ENGLISH_STOPWORDS should be a non-empty frozenset."""
        assert isinstance(ENGLISH_STOPWORDS, frozenset)
        assert len(ENGLISH_STOPWORDS) > 50

    def test_common_stopwords_included(self) -> None:
        """Common English stopwords should be in the set."""
        assert "the" in ENGLISH_STOPWORDS
        assert "a" in ENGLISH_STOPWORDS
        assert "is" in ENGLISH_STOPWORDS
        assert "and" in ENGLISH_STOPWORDS
        assert "or" in ENGLISH_STOPWORDS


class TestBM25IDFFileLock:
    """Tests for the IDF file lock context manager."""

    def test_idf_file_lock_no_path_is_noop(self) -> None:
        plugin = BM25SparseIndexerPlugin()
        plugin._idf_path = None

        with plugin._idf_file_lock():
            assert True

    def test_idf_file_lock_acquires_and_releases(self, tmp_path: Path, monkeypatch) -> None:
        import fcntl

        plugin = BM25SparseIndexerPlugin()
        plugin._idf_path = tmp_path / "sparse_indexes" / "collection" / "idf_stats.json"

        calls: list[int] = []

        def fake_flock(_fd: int, flags: int) -> None:
            calls.append(flags)

        monkeypatch.setattr(fcntl, "flock", fake_flock)

        with plugin._idf_file_lock(timeout=0.1):
            assert plugin._idf_path.with_suffix(".lock").exists()

        assert any(flags & fcntl.LOCK_EX for flags in calls)
        assert any(flags & fcntl.LOCK_UN for flags in calls)

    def test_idf_file_lock_times_out(self, tmp_path: Path, monkeypatch) -> None:
        import fcntl
        import time as py_time

        plugin = BM25SparseIndexerPlugin()
        plugin._idf_path = tmp_path / "sparse_indexes" / "collection" / "idf_stats.json"

        def fake_flock(_fd: int, _flags: int) -> None:
            raise OSError("busy")

        monkeypatch.setattr(fcntl, "flock", fake_flock)
        monkeypatch.setattr(py_time, "sleep", lambda *_args, **_kwargs: None)

        # Make time jump forward immediately so we exceed the timeout.
        # Use a list with index to handle any number of calls (coverage may add extra calls)
        call_count = [0]

        def fake_time() -> float:
            call_count[0] += 1
            # First call returns 0.0 (start time), all subsequent calls return 999.0 (way past timeout)
            return 0.0 if call_count[0] == 1 else 999.0

        monkeypatch.setattr(py_time, "time", fake_time)

        with pytest.raises(TimeoutError):
            with plugin._idf_file_lock(timeout=0.01):
                pass

    @pytest.mark.asyncio()
    async def test_save_idf_stats_timeout_is_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        from contextlib import contextmanager

        plugin = BM25SparseIndexerPlugin()
        plugin._idf_path = tmp_path / "sparse_indexes" / "collection" / "idf_stats.json"
        await plugin.encode_documents([{"content": "hello world", "chunk_id": "chunk-1"}])
        assert plugin._dirty is True

        @contextmanager
        def _raise_timeout(*_args, **_kwargs):
            raise TimeoutError("lock timeout")
            yield  # pragma: no cover

        monkeypatch.setattr(plugin, "_idf_file_lock", _raise_timeout)

        await plugin._save_idf_stats()
        assert plugin._dirty is True
