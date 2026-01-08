"""Tests for DTO adapters between TypedDicts and dataclasses.

These tests verify that:
1. Round-trip conversion preserves all data
2. Validation catches missing required keys
3. Validation catches invalid enum strings
4. Validation catches malformed content hashes
"""

from datetime import UTC, datetime

import pytest


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_required_keys_passes_when_all_present(self):
        """Verify _validate_required_keys passes with all keys present."""
        from shared.plugins.dto_adapters import _validate_required_keys

        d = {"a": 1, "b": 2, "c": 3}
        _validate_required_keys(d, {"a", "b"}, "TestDict")  # No exception

    def test_validate_required_keys_raises_when_missing(self):
        """Verify _validate_required_keys raises with missing keys."""
        from shared.plugins.dto_adapters import ValidationError, _validate_required_keys

        d = {"a": 1}
        with pytest.raises(ValidationError, match="missing required keys"):
            _validate_required_keys(d, {"a", "b", "c"}, "TestDict")

    def test_validate_string_enum_passes_valid_value(self):
        """Verify _validate_string_enum passes with valid value."""
        from shared.plugins.dto_adapters import _validate_string_enum

        _validate_string_enum("user", frozenset({"user", "assistant"}), "role")  # No exception

    def test_validate_string_enum_raises_invalid_value(self):
        """Verify _validate_string_enum raises with invalid value."""
        from shared.plugins.dto_adapters import ValidationError, _validate_string_enum

        with pytest.raises(ValidationError, match="Invalid role"):
            _validate_string_enum("invalid", frozenset({"user", "assistant"}), "role")

    def test_validate_content_hash_passes_valid_hash(self):
        """Verify _validate_content_hash passes with valid hash."""
        from shared.plugins.dto_adapters import _validate_content_hash

        valid_hash = "a" * 64
        _validate_content_hash(valid_hash)  # No exception

    def test_validate_content_hash_raises_wrong_length(self):
        """Verify _validate_content_hash raises with wrong length."""
        from shared.plugins.dto_adapters import ValidationError, _validate_content_hash

        with pytest.raises(ValidationError, match="must be 64 characters"):
            _validate_content_hash("abc123")

    def test_validate_content_hash_raises_invalid_chars(self):
        """Verify _validate_content_hash raises with invalid characters."""
        from shared.plugins.dto_adapters import ValidationError, _validate_content_hash

        invalid_hash = "A" * 64  # Uppercase not allowed
        with pytest.raises(ValidationError, match="lowercase hexadecimal"):
            _validate_content_hash(invalid_hash)


class TestIngestedDocumentAdapter:
    """Test IngestedDocument round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.dtos.ingestion import IngestedDocument
        from shared.plugins.dto_adapters import (
            dict_to_ingested_document,
            ingested_document_to_dict,
        )

        original = IngestedDocument(
            content="Test content",
            unique_id="doc-123",
            source_type="test",
            metadata={"key": "value"},
            content_hash="a" * 64,
            file_path="/path/to/file",
        )

        dict_form = ingested_document_to_dict(original)
        restored = dict_to_ingested_document(dict_form)

        assert restored.content == original.content
        assert restored.unique_id == original.unique_id
        assert restored.source_type == original.source_type
        assert restored.metadata == original.metadata
        assert restored.content_hash == original.content_hash
        assert restored.file_path == original.file_path

    def test_round_trip_without_optional_fields(self):
        """Verify round-trip works without optional fields."""
        from shared.dtos.ingestion import IngestedDocument
        from shared.plugins.dto_adapters import (
            dict_to_ingested_document,
            ingested_document_to_dict,
        )

        original = IngestedDocument(
            content="Test content",
            unique_id="doc-123",
            source_type="test",
            metadata={},
            content_hash="b" * 64,
        )

        dict_form = ingested_document_to_dict(original)
        restored = dict_to_ingested_document(dict_form)

        assert restored.file_path is None

    def test_validation_rejects_missing_required_keys(self):
        """Verify validation catches missing required fields."""
        from shared.plugins.dto_adapters import ValidationError, validate_ingested_document_dict

        with pytest.raises(ValidationError, match="missing required keys"):
            validate_ingested_document_dict({"content": "test"})

    def test_validation_rejects_invalid_content_hash(self):
        """Verify validation catches malformed content_hash."""
        from shared.plugins.dto_adapters import ValidationError, validate_ingested_document_dict

        with pytest.raises(ValidationError, match="content_hash"):
            validate_ingested_document_dict(
                {
                    "content": "test",
                    "unique_id": "id",
                    "source_type": "test",
                    "metadata": {},
                    "content_hash": "invalid",
                }
            )

    def test_coerce_returns_dataclass_unchanged(self):
        """Verify coerce_to_ingested_document returns dataclass unchanged."""
        from shared.dtos.ingestion import IngestedDocument
        from shared.plugins.dto_adapters import coerce_to_ingested_document

        doc = IngestedDocument(
            content="Test",
            unique_id="id",
            source_type="test",
            metadata={},
            content_hash="c" * 64,
        )

        result = coerce_to_ingested_document(doc)
        assert result is doc

    def test_coerce_converts_dict(self):
        """Verify coerce_to_ingested_document converts dict."""
        from shared.dtos.ingestion import IngestedDocument
        from shared.plugins.dto_adapters import coerce_to_ingested_document

        d = {
            "content": "Test",
            "unique_id": "id",
            "source_type": "test",
            "metadata": {},
            "content_hash": "d" * 64,
        }

        result = coerce_to_ingested_document(d)
        assert isinstance(result, IngestedDocument)
        assert result.content == "Test"


class TestTokenUsageAdapter:
    """Test TokenUsage round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.agents.types import TokenUsage
        from shared.plugins.dto_adapters import dict_to_token_usage, token_usage_to_dict

        original = TokenUsage(
            input_tokens=100,
            output_tokens=200,
            cache_read_tokens=50,
            cache_write_tokens=30,
            reasoning_tokens=25,
        )

        dict_form = token_usage_to_dict(original)
        restored = dict_to_token_usage(dict_form)

        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.cache_read_tokens == original.cache_read_tokens
        assert restored.cache_write_tokens == original.cache_write_tokens
        assert restored.reasoning_tokens == original.reasoning_tokens

    def test_dict_to_token_usage_uses_defaults(self):
        """Verify missing fields default to 0."""
        from shared.plugins.dto_adapters import dict_to_token_usage

        result = dict_to_token_usage({})
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestAgentMessageAdapter:
    """Test AgentMessage round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.agents.types import AgentMessage, MessageRole, MessageType, TokenUsage
        from shared.plugins.dto_adapters import agent_message_to_dict, dict_to_agent_message

        original = AgentMessage(
            id="msg-123",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello",
            model="claude-3",
            usage=TokenUsage(input_tokens=10, output_tokens=20),
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
        )

        dict_form = agent_message_to_dict(original)
        restored = dict_to_agent_message(dict_form)

        assert restored.id == original.id
        assert restored.role == original.role
        assert restored.type == original.type
        assert restored.content == original.content
        assert restored.model == original.model
        assert restored.usage is not None
        assert restored.usage.input_tokens == 10

    def test_validation_rejects_invalid_role(self):
        """Verify validation catches invalid role string."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_agent_message

        with pytest.raises(ValidationError, match="Invalid role"):
            dict_to_agent_message(
                {
                    "id": "msg-1",
                    "role": "invalid_role",
                    "type": "text",
                    "content": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    def test_validation_rejects_invalid_type(self):
        """Verify validation catches invalid type string."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_agent_message

        with pytest.raises(ValidationError, match="Invalid type"):
            dict_to_agent_message(
                {
                    "id": "msg-1",
                    "role": "user",
                    "type": "invalid_type",
                    "content": "test",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    def test_coerce_returns_dataclass_unchanged(self):
        """Verify coerce_to_agent_message returns dataclass unchanged."""
        from shared.agents.types import AgentMessage
        from shared.plugins.dto_adapters import coerce_to_agent_message

        msg = AgentMessage(content="Hello")
        result = coerce_to_agent_message(msg)
        assert result is msg


class TestAgentCapabilitiesAdapter:
    """Test AgentCapabilities round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.agents.types import AgentCapabilities
        from shared.plugins.dto_adapters import (
            agent_capabilities_to_dict,
            dict_to_agent_capabilities,
        )

        original = AgentCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=100000,
            supported_models=("model-a", "model-b"),
            default_model="model-a",
        )

        dict_form = agent_capabilities_to_dict(original)
        restored = dict_to_agent_capabilities(dict_form)

        assert restored.supports_streaming == original.supports_streaming
        assert restored.supports_tools == original.supports_tools
        assert restored.max_context_tokens == original.max_context_tokens
        assert restored.supported_models == original.supported_models
        assert restored.default_model == original.default_model

    def test_dict_to_agent_capabilities_uses_defaults(self):
        """Verify missing fields use correct defaults."""
        from shared.plugins.dto_adapters import dict_to_agent_capabilities

        result = dict_to_agent_capabilities({})
        assert result.supports_streaming is True
        assert result.supports_tools is True
        assert result.supports_extended_thinking is False


class TestAgentContextAdapter:
    """Test AgentContext round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.agents.types import AgentContext, AgentMessage
        from shared.plugins.dto_adapters import agent_context_to_dict, dict_to_agent_context

        original = AgentContext(
            request_id="req-123",
            user_id="user-456",
            collection_id="col-789",
            conversation_history=[AgentMessage(content="Hello")],
            max_tokens=4096,
        )

        dict_form = agent_context_to_dict(original)
        restored = dict_to_agent_context(dict_form)

        assert restored.request_id == original.request_id
        assert restored.user_id == original.user_id
        assert restored.collection_id == original.collection_id
        assert restored.max_tokens == original.max_tokens
        assert len(restored.conversation_history) == 1
        assert restored.conversation_history[0].content == "Hello"

    def test_validation_rejects_missing_request_id(self):
        """Verify validation catches missing request_id."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_agent_context

        with pytest.raises(ValidationError, match="request_id"):
            dict_to_agent_context({})


class TestChunkMetadataAdapter:
    """Test ChunkMetadata round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
        from shared.plugins.dto_adapters import chunk_metadata_to_dict, dict_to_chunk_metadata

        original = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            start_offset=0,
            end_offset=100,
            token_count=25,
            strategy_name="recursive",
            semantic_score=0.85,
            hierarchy_level=1,
        )

        dict_form = chunk_metadata_to_dict(original)
        restored = dict_to_chunk_metadata(dict_form)

        assert restored.chunk_id == original.chunk_id
        assert restored.document_id == original.document_id
        assert restored.chunk_index == original.chunk_index
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset
        assert restored.token_count == original.token_count
        assert restored.strategy_name == original.strategy_name
        assert restored.semantic_score == original.semantic_score
        assert restored.hierarchy_level == original.hierarchy_level

    def test_validation_rejects_missing_required_keys(self):
        """Verify validation catches missing required fields."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_chunk_metadata

        with pytest.raises(ValidationError, match="missing required keys"):
            dict_to_chunk_metadata({"chunk_id": "chunk-1"})


class TestChunkAdapter:
    """Test Chunk entity round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify entity -> dict -> entity preserves all fields."""
        from shared.chunking.domain.entities.chunk import Chunk
        from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
        from shared.plugins.dto_adapters import chunk_to_dict, dict_to_chunk

        metadata = ChunkMetadata(
            chunk_id="chunk-1",
            document_id="doc-1",
            chunk_index=0,
            start_offset=0,
            end_offset=100,
            token_count=25,
            strategy_name="recursive",
        )
        original = Chunk(content="This is test content for the chunk.", metadata=metadata)
        original.set_embedding([0.1, 0.2, 0.3])

        dict_form = chunk_to_dict(original)
        restored = dict_to_chunk(dict_form)

        assert restored.content == original.content
        assert restored.metadata.chunk_id == original.metadata.chunk_id
        assert restored.embedding == original.embedding

    def test_validation_rejects_missing_content(self):
        """Verify validation catches missing content."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_chunk

        with pytest.raises(ValidationError, match="missing required keys"):
            dict_to_chunk({"metadata": {}})


class TestRerankResultAdapter:
    """Test RerankResult round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.plugins.dto_adapters import dict_to_rerank_result, rerank_result_to_dict
        from shared.plugins.types.reranker import RerankResult

        original = RerankResult(
            index=0,
            score=0.95,
            document="This is a relevant document.",
            metadata={"source": "test"},
        )

        dict_form = rerank_result_to_dict(original)
        restored = dict_to_rerank_result(dict_form)

        assert restored.index == original.index
        assert restored.score == original.score
        assert restored.document == original.document
        assert restored.metadata == original.metadata

    def test_validation_rejects_missing_required_keys(self):
        """Verify validation catches missing required fields."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_rerank_result

        with pytest.raises(ValidationError, match="missing required keys"):
            dict_to_rerank_result({"index": 0})


class TestRerankerCapabilitiesAdapter:
    """Test RerankerCapabilities round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.plugins.dto_adapters import (
            dict_to_reranker_capabilities,
            reranker_capabilities_to_dict,
        )
        from shared.plugins.types.reranker import RerankerCapabilities

        original = RerankerCapabilities(
            max_documents=100,
            max_query_length=512,
            max_doc_length=2048,
            supports_batching=True,
            models=["model-a", "model-b"],
        )

        dict_form = reranker_capabilities_to_dict(original)
        restored = dict_to_reranker_capabilities(dict_form)

        assert restored.max_documents == original.max_documents
        assert restored.max_query_length == original.max_query_length
        assert restored.max_doc_length == original.max_doc_length
        assert restored.supports_batching == original.supports_batching
        assert restored.models == original.models


class TestEntityAdapter:
    """Test Entity round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.plugins.dto_adapters import dict_to_entity, entity_to_dict
        from shared.plugins.types.extractor import Entity

        original = Entity(
            text="Apple Inc.",
            type="ORG",
            start=0,
            end=10,
            confidence=0.95,
            metadata={"normalized": "Apple"},
        )

        dict_form = entity_to_dict(original)
        restored = dict_to_entity(dict_form)

        assert restored.text == original.text
        assert restored.type == original.type
        assert restored.start == original.start
        assert restored.end == original.end
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata

    def test_validation_rejects_missing_required_keys(self):
        """Verify validation catches missing required fields."""
        from shared.plugins.dto_adapters import ValidationError, dict_to_entity

        with pytest.raises(ValidationError, match="missing required keys"):
            dict_to_entity({"text": "Apple"})


class TestExtractionResultAdapter:
    """Test ExtractionResult round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.plugins.dto_adapters import (
            dict_to_extraction_result,
            extraction_result_to_dict,
        )
        from shared.plugins.types.extractor import Entity, ExtractionResult

        original = ExtractionResult(
            entities=[
                Entity(text="Apple", type="ORG", start=0, end=5),
                Entity(text="Cupertino", type="LOC", start=20, end=29),
            ],
            keywords=["apple", "technology", "innovation"],
            language="en",
            language_confidence=0.99,
            topics=["technology", "business"],
            sentiment=0.5,
            summary="A company announcement.",
            custom={"domain": "tech"},
        )

        dict_form = extraction_result_to_dict(original)
        restored = dict_to_extraction_result(dict_form)

        assert len(restored.entities) == 2
        assert restored.entities[0].text == "Apple"
        assert restored.entities[1].type == "LOC"
        assert restored.keywords == original.keywords
        assert restored.language == original.language
        assert restored.topics == original.topics
        assert restored.sentiment == original.sentiment
        assert restored.summary == original.summary
        assert restored.custom == original.custom

    def test_round_trip_with_empty_result(self):
        """Verify round-trip works with empty ExtractionResult."""
        from shared.plugins.dto_adapters import (
            dict_to_extraction_result,
            extraction_result_to_dict,
        )
        from shared.plugins.types.extractor import ExtractionResult

        original = ExtractionResult()
        dict_form = extraction_result_to_dict(original)
        restored = dict_to_extraction_result(dict_form)

        assert restored.entities == []
        assert restored.keywords == []
        assert restored.language is None


class TestEmbeddingProviderDefinitionAdapter:
    """Test EmbeddingProviderDefinition round-trip conversion."""

    def test_round_trip_preserves_data(self):
        """Verify dataclass -> dict -> dataclass preserves all fields."""
        from shared.embedding.plugin_base import EmbeddingProviderDefinition
        from shared.plugins.dto_adapters import (
            dict_to_embedding_provider_definition,
            embedding_provider_definition_to_dict,
        )

        original = EmbeddingProviderDefinition(
            api_id="my-provider",
            internal_id="my_provider",
            display_name="My Provider",
            description="A test embedding provider",
            provider_type="local",
            supports_quantization=True,
            supports_instruction=True,
            supports_asymmetric=True,
            supported_models=("model-a", "model-b"),
            is_plugin=True,
        )

        dict_form = embedding_provider_definition_to_dict(original)
        restored = dict_to_embedding_provider_definition(dict_form)

        assert restored.api_id == original.api_id
        assert restored.internal_id == original.internal_id
        assert restored.display_name == original.display_name
        assert restored.description == original.description
        assert restored.provider_type == original.provider_type
        assert restored.supports_quantization == original.supports_quantization
        assert restored.supports_instruction == original.supports_instruction
        assert restored.supports_asymmetric == original.supports_asymmetric
        assert restored.supported_models == original.supported_models
        assert restored.is_plugin == original.is_plugin

    def test_validation_rejects_missing_required_keys(self):
        """Verify validation catches missing required fields."""
        from shared.plugins.dto_adapters import (
            ValidationError,
            dict_to_embedding_provider_definition,
        )

        with pytest.raises(ValidationError, match="missing required keys"):
            dict_to_embedding_provider_definition({"api_id": "test"})
