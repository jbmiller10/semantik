"""Bidirectional adapters between TypedDicts and dataclasses with validation.

This module provides conversion functions between the TypedDict schemas
(used by external plugins) and internal dataclasses (used by semantik services).

External plugins return plain dicts conforming to TypedDict schemas.
The adapter layer validates these dicts and converts them to internal
dataclasses for use by semantik services.

Protocol Version: 1.0.0
"""

from __future__ import annotations

import re
from typing import Any, cast
from uuid import uuid4

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.dtos.ingestion import IngestedDocument
from shared.embedding.plugin_base import EmbeddingProviderDefinition
from shared.plugins.typed_dicts import (
    SPARSE_TYPES,
    ChunkDict,
    ChunkMetadataDict,
    EmbeddingProviderDefinitionDict,
    EntityDict,
    ExtractionResultDict,
    IngestedDocumentDict,
    RerankerCapabilitiesDict,
    RerankResultDict,
    SparseIndexerCapabilitiesDict,
    SparseQueryVectorDict,
    SparseVectorDict,
)
from shared.plugins.types.extractor import Entity, ExtractionResult
from shared.plugins.types.reranker import RerankerCapabilities, RerankResult
from shared.plugins.types.sparse_indexer import SparseIndexerCapabilities, SparseQueryVector, SparseVector

# ============================================================================
# Validation Error
# ============================================================================


class ValidationError(ValueError):
    """Raised when TypedDict validation fails.

    This exception is raised when external plugin output doesn't conform
    to the expected TypedDict schema.
    """


# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_required_keys(d: dict[str, Any], required: set[str], name: str) -> None:
    """Raise ValidationError if required keys are missing.

    Args:
        d: Dictionary to validate.
        required: Set of required key names.
        name: Name of the TypedDict for error messages.

    Raises:
        ValidationError: If any required keys are missing.
    """
    missing = required - set(d.keys())
    if missing:
        raise ValidationError(f"{name} missing required keys: {sorted(missing)}")


def _validate_string_enum(value: str, valid_values: frozenset[str], field_name: str) -> None:
    """Raise ValidationError if value is not in valid set.

    Args:
        value: String value to validate.
        valid_values: Set of valid string values.
        field_name: Name of the field for error messages.

    Raises:
        ValidationError: If value is not in valid_values.
    """
    if value not in valid_values:
        raise ValidationError(f"Invalid {field_name}: '{value}'. Must be one of: {sorted(valid_values)}")


def _validate_content_hash(hash_value: str) -> None:
    """Validate content_hash is 64-char lowercase hex SHA-256.

    Args:
        hash_value: Hash string to validate.

    Raises:
        ValidationError: If hash is not valid format.
    """
    if len(hash_value) != 64:
        raise ValidationError(f"content_hash must be 64 characters, got {len(hash_value)}")
    if not all(c in "0123456789abcdef" for c in hash_value):
        raise ValidationError("content_hash must be lowercase hexadecimal")


# ============================================================================
# IngestedDocument Adapters (Connector)
# ============================================================================


def validate_ingested_document_dict(d: dict[str, Any]) -> IngestedDocumentDict:
    """Validate and cast dict to IngestedDocumentDict.

    Args:
        d: Dictionary to validate.

    Returns:
        The validated dict cast to IngestedDocumentDict.

    Raises:
        ValidationError: If validation fails.
    """
    _validate_required_keys(
        d, {"content", "unique_id", "source_type", "metadata", "content_hash"}, "IngestedDocumentDict"
    )
    _validate_content_hash(d["content_hash"])
    return cast(IngestedDocumentDict, d)


def ingested_document_to_dict(doc: IngestedDocument) -> IngestedDocumentDict:
    """Convert IngestedDocument dataclass to TypedDict.

    Args:
        doc: IngestedDocument dataclass instance.

    Returns:
        IngestedDocumentDict representation.
    """
    result: IngestedDocumentDict = {
        "content": doc.content,
        "unique_id": doc.unique_id,
        "source_type": doc.source_type,
        "metadata": doc.metadata,
        "content_hash": doc.content_hash,
    }
    if doc.file_path is not None:
        result["file_path"] = doc.file_path
    return result


def dict_to_ingested_document(d: dict[str, Any]) -> IngestedDocument:
    """Convert TypedDict to IngestedDocument dataclass with validation.

    Args:
        d: Dictionary conforming to IngestedDocumentDict schema.

    Returns:
        IngestedDocument dataclass instance.

    Raises:
        ValidationError: If validation fails.
    """
    validated = validate_ingested_document_dict(d)
    return IngestedDocument(
        content=validated["content"],
        unique_id=validated["unique_id"],
        source_type=validated["source_type"],
        metadata=validated["metadata"],
        content_hash=validated["content_hash"],
        file_path=validated.get("file_path"),
    )


def coerce_to_ingested_document(doc: IngestedDocument | dict[str, Any]) -> IngestedDocument:
    """Accept either dataclass or dict, return dataclass.

    Args:
        doc: Either an IngestedDocument or a dict conforming to the schema.

    Returns:
        IngestedDocument dataclass instance.
    """
    if isinstance(doc, IngestedDocument):
        return doc
    return dict_to_ingested_document(doc)


# ============================================================================
# ChunkMetadata Adapters (Chunking)
# ============================================================================

_CHUNK_ID_TRAILING_INDEX_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>[0-9]+)$")


def _estimate_token_count(text: str) -> int:
    """Estimate token count for protocol chunks when plugins don't provide it.

    This is intentionally lightweight (no model-specific tokenizer dependency).
    """

    # Approximate tokens via whitespace splitting; guarantee > 0 for non-empty text.
    token_count = len(text.split())
    return max(1, token_count)


def _infer_document_id_from_chunk_id(chunk_id: str) -> str | None:
    """Infer document_id from chunk_id patterns like '<doc_id>_<index>'."""

    match = _CHUNK_ID_TRAILING_INDEX_RE.match(chunk_id)
    if match:
        return match.group("prefix")
    return None


def _infer_chunk_index_from_chunk_id(chunk_id: str) -> int | None:
    """Infer chunk_index from chunk_id patterns like '<prefix>_<index>'."""

    match = _CHUNK_ID_TRAILING_INDEX_RE.match(chunk_id)
    if match:
        try:
            return int(match.group("index"))
        except ValueError:
            return None
    return None


def _generate_chunk_id(strategy_name: str, chunk_index: int) -> str:
    safe_strategy = strategy_name.strip() or "external"
    return f"{safe_strategy}_{chunk_index:04d}_{uuid4().hex}"


def chunk_metadata_to_dict(metadata: ChunkMetadata) -> ChunkMetadataDict:
    """Convert ChunkMetadata dataclass to TypedDict.

    Args:
        metadata: ChunkMetadata dataclass instance.

    Returns:
        ChunkMetadataDict representation.
    """
    result: ChunkMetadataDict = {
        "chunk_id": metadata.chunk_id,
        "document_id": metadata.document_id,
        "chunk_index": metadata.chunk_index,
        "start_offset": metadata.start_offset,
        "end_offset": metadata.end_offset,
        "token_count": metadata.token_count,
        "strategy_name": metadata.strategy_name,
        "semantic_density": metadata.semantic_density,
        "confidence_score": metadata.confidence_score,
        "overlap_percentage": metadata.overlap_percentage,
    }
    if metadata.semantic_score is not None:
        result["semantic_score"] = metadata.semantic_score
    if metadata.hierarchy_level is not None:
        result["hierarchy_level"] = metadata.hierarchy_level
    if metadata.section_title is not None:
        result["section_title"] = metadata.section_title
    if metadata.custom_attributes:
        result["custom_attributes"] = metadata.custom_attributes
    return result


def dict_to_chunk_metadata(d: dict[str, Any]) -> ChunkMetadata:
    """Convert TypedDict to ChunkMetadata dataclass.

    Args:
        d: Dictionary conforming to ChunkMetadataDict schema.

    Returns:
        ChunkMetadata dataclass instance.

    Raises:
        ValueError: If ChunkMetadata validation fails.
    """
    # ChunkMetadataDict is defined as total=False (all fields optional). Provide
    # sensible defaults for internal ChunkMetadata requirements.
    chunk_id = d.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        strategy_name = d.get("strategy_name")
        inferred_strategy = strategy_name if isinstance(strategy_name, str) and strategy_name.strip() else "external"
        chunk_index_value = d.get("chunk_index")
        inferred_index = chunk_index_value if isinstance(chunk_index_value, int) else 0
        chunk_id = _generate_chunk_id(inferred_strategy, inferred_index)

    document_id = d.get("document_id")
    if not isinstance(document_id, str):
        inferred_doc_id = _infer_document_id_from_chunk_id(chunk_id)
        document_id = inferred_doc_id or ""

    chunk_index_raw = d.get("chunk_index")
    if isinstance(chunk_index_raw, int):
        chunk_index = chunk_index_raw
    else:
        inferred_chunk_idx = _infer_chunk_index_from_chunk_id(chunk_id)
        chunk_index = inferred_chunk_idx if inferred_chunk_idx is not None else 0

    start_offset = d.get("start_offset")
    if not isinstance(start_offset, int):
        start_offset = 0

    end_offset = d.get("end_offset")
    if not isinstance(end_offset, int):
        # Ensure the default satisfies ChunkMetadata invariant end_offset > start_offset.
        end_offset = start_offset + 1

    token_count = d.get("token_count")
    if not isinstance(token_count, int) or token_count <= 0:
        token_count = 1

    strategy_name = d.get("strategy_name")
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        strategy_name = "external"

    return ChunkMetadata(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        start_offset=start_offset,
        end_offset=end_offset,
        token_count=token_count,
        strategy_name=strategy_name,
        semantic_score=d.get("semantic_score"),
        semantic_density=d.get("semantic_density", 0.5),
        confidence_score=d.get("confidence_score", 1.0),
        overlap_percentage=d.get("overlap_percentage", 0.0),
        hierarchy_level=d.get("hierarchy_level"),
        section_title=d.get("section_title"),
        custom_attributes=d.get("custom_attributes", {}),
        created_at=d.get("created_at"),
    )


# ============================================================================
# Chunk Adapters (Chunking)
# ============================================================================


def chunk_to_dict(chunk: Chunk) -> ChunkDict:
    """Convert Chunk entity to TypedDict.

    Args:
        chunk: Chunk entity instance.

    Returns:
        ChunkDict representation.
    """
    result: ChunkDict = {
        "content": chunk.content,
        "metadata": chunk_metadata_to_dict(chunk.metadata),
    }
    if chunk.metadata.chunk_id:
        result["chunk_id"] = chunk.metadata.chunk_id
    if chunk.embedding is not None:
        result["embedding"] = chunk.embedding
    return result


def dict_to_chunk(d: dict[str, Any], min_tokens: int = 10, max_tokens: int = 10000) -> Chunk:
    """Convert TypedDict to Chunk entity.

    Args:
        d: Dictionary conforming to ChunkDict schema.
        min_tokens: Minimum tokens constraint for Chunk.
        max_tokens: Maximum tokens constraint for Chunk.

    Returns:
        Chunk entity instance.

    Raises:
        ValidationError: If required fields are missing.
        InvalidChunkError: If Chunk validation fails.
    """
    _validate_required_keys(d, {"content", "metadata"}, "ChunkDict")

    content = d["content"]
    metadata_dict = dict(d["metadata"])

    # ChunkDict supports chunk_id at top-level; prefer it if metadata omitted it.
    if "chunk_id" in d and "chunk_id" not in metadata_dict:
        metadata_dict["chunk_id"] = d["chunk_id"]

    # Fill optional protocol metadata fields that are required by internal ChunkMetadata/Chunk.
    if "token_count" not in metadata_dict:
        metadata_dict["token_count"] = _estimate_token_count(content)

    start_offset = metadata_dict.get("start_offset")
    if not isinstance(start_offset, int):
        start_offset = 0
        metadata_dict["start_offset"] = start_offset

    end_offset = metadata_dict.get("end_offset")
    if not isinstance(end_offset, int):
        # Ensure end_offset > start_offset to satisfy ChunkMetadata invariants.
        metadata_dict["end_offset"] = start_offset + max(1, len(content))

    if "chunk_index" not in metadata_dict:
        metadata_dict["chunk_index"] = 0

    if "strategy_name" not in metadata_dict:
        metadata_dict["strategy_name"] = "external"

    metadata = dict_to_chunk_metadata(metadata_dict)
    chunk = Chunk(
        content=content,
        metadata=metadata,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )

    # Set optional properties
    embedding = d.get("embedding")
    if embedding is not None:
        chunk.set_embedding(embedding)

    return chunk


# ============================================================================
# RerankResult Adapters (Reranker)
# ============================================================================


def rerank_result_to_dict(result: RerankResult) -> RerankResultDict:
    """Convert RerankResult dataclass to TypedDict.

    Args:
        result: RerankResult dataclass instance.

    Returns:
        RerankResultDict representation.
    """
    d: RerankResultDict = {
        "index": result.index,
        "score": result.score,
    }
    if result.document:
        d["text"] = result.document
    if result.metadata:
        d["metadata"] = result.metadata
    return d


def dict_to_rerank_result(d: dict[str, Any]) -> RerankResult:
    """Convert TypedDict to RerankResult dataclass.

    Args:
        d: Dictionary conforming to RerankResultDict schema.

    Returns:
        RerankResult dataclass instance.

    Raises:
        ValidationError: If required fields are missing.
    """
    _validate_required_keys(d, {"index", "score"}, "RerankResultDict")

    return RerankResult(
        index=d["index"],
        score=d["score"],
        document=d.get("text", ""),
        metadata=d.get("metadata", {}),
    )


# ============================================================================
# RerankerCapabilities Adapters (Reranker)
# ============================================================================


def reranker_capabilities_to_dict(caps: RerankerCapabilities) -> RerankerCapabilitiesDict:
    """Convert RerankerCapabilities dataclass to TypedDict.

    Args:
        caps: RerankerCapabilities dataclass instance.

    Returns:
        RerankerCapabilitiesDict representation.
    """
    return {
        "max_documents": caps.max_documents,
        "max_query_length": caps.max_query_length,
        "max_doc_length": caps.max_doc_length,
        "supports_batching": caps.supports_batching,
        "models": caps.models,
    }


def dict_to_reranker_capabilities(d: dict[str, Any]) -> RerankerCapabilities:
    """Convert TypedDict to RerankerCapabilities dataclass.

    Args:
        d: Dictionary conforming to RerankerCapabilitiesDict schema.

    Returns:
        RerankerCapabilities dataclass instance.

    Raises:
        ValidationError: If required fields are missing.
    """
    _validate_required_keys(
        d,
        {"max_documents", "max_query_length", "max_doc_length", "supports_batching"},
        "RerankerCapabilitiesDict",
    )

    return RerankerCapabilities(
        max_documents=d["max_documents"],
        max_query_length=d["max_query_length"],
        max_doc_length=d["max_doc_length"],
        supports_batching=d["supports_batching"],
        models=d.get("models", []),
    )


# ============================================================================
# Entity Adapters (Extractor)
# ============================================================================


def entity_to_dict(entity: Entity) -> EntityDict:
    """Convert Entity dataclass to TypedDict.

    Args:
        entity: Entity dataclass instance.

    Returns:
        EntityDict representation.
    """
    result: EntityDict = {
        "text": entity.text,
        "type": entity.type,
    }
    if entity.start != 0:
        result["start"] = entity.start
    if entity.end != 0:
        result["end"] = entity.end
    if entity.confidence != 1.0:
        result["confidence"] = entity.confidence
    if entity.metadata:
        result["metadata"] = entity.metadata
    return result


def dict_to_entity(d: dict[str, Any]) -> Entity:
    """Convert TypedDict to Entity dataclass.

    Args:
        d: Dictionary conforming to EntityDict schema.

    Returns:
        Entity dataclass instance.

    Raises:
        ValidationError: If required fields are missing.
    """
    _validate_required_keys(d, {"text", "type"}, "EntityDict")

    return Entity(
        text=d["text"],
        type=d["type"],
        start=d.get("start", 0),
        end=d.get("end", 0),
        confidence=d.get("confidence", 1.0),
        metadata=d.get("metadata", {}),
    )


# ============================================================================
# ExtractionResult Adapters (Extractor)
# ============================================================================


def extraction_result_to_dict(result: ExtractionResult) -> ExtractionResultDict:
    """Convert ExtractionResult dataclass to TypedDict.

    Args:
        result: ExtractionResult dataclass instance.

    Returns:
        ExtractionResultDict representation.
    """
    d: ExtractionResultDict = {}
    if result.entities:
        d["entities"] = [entity_to_dict(e) for e in result.entities]
    if result.keywords:
        d["keywords"] = result.keywords
    if result.language is not None:
        d["language"] = result.language
    if result.language_confidence is not None:
        d["language_confidence"] = result.language_confidence
    if result.topics:
        d["topics"] = result.topics
    if result.sentiment is not None:
        d["sentiment"] = result.sentiment
    if result.summary is not None:
        d["summary"] = result.summary
    if result.custom:
        d["custom"] = result.custom
    return d


def dict_to_extraction_result(d: dict[str, Any]) -> ExtractionResult:
    """Convert TypedDict to ExtractionResult dataclass.

    Args:
        d: Dictionary conforming to ExtractionResultDict schema.

    Returns:
        ExtractionResult dataclass instance.
    """
    entities = d.get("entities", [])
    if entities:
        entities = [dict_to_entity(e) for e in entities]

    return ExtractionResult(
        entities=entities,
        keywords=d.get("keywords", []),
        language=d.get("language"),
        language_confidence=d.get("language_confidence"),
        topics=d.get("topics", []),
        sentiment=d.get("sentiment"),
        summary=d.get("summary"),
        custom=d.get("custom", {}),
    )


# ============================================================================
# EmbeddingProviderDefinition Adapters (Embedding)
# ============================================================================


def embedding_provider_definition_to_dict(definition: EmbeddingProviderDefinition) -> EmbeddingProviderDefinitionDict:
    """Convert EmbeddingProviderDefinition dataclass to TypedDict.

    Args:
        definition: EmbeddingProviderDefinition dataclass instance.

    Returns:
        EmbeddingProviderDefinitionDict representation.
    """
    result: EmbeddingProviderDefinitionDict = {
        "api_id": definition.api_id,
        "internal_id": definition.internal_id,
        "display_name": definition.display_name,
        "description": definition.description,
        "provider_type": definition.provider_type,
    }
    if definition.supports_quantization:
        result["supports_quantization"] = definition.supports_quantization
    if definition.supports_instruction:
        result["supports_instruction"] = definition.supports_instruction
    if definition.supports_batch_processing:
        result["supports_batch_processing"] = definition.supports_batch_processing
    if definition.supports_asymmetric:
        result["supports_asymmetric"] = definition.supports_asymmetric
    if definition.supported_models:
        result["supported_models"] = list(definition.supported_models)
    if definition.default_config:
        result["default_config"] = definition.default_config
    if definition.performance_characteristics:
        result["performance_characteristics"] = definition.performance_characteristics
    if definition.is_plugin:
        result["is_plugin"] = definition.is_plugin
    return result


def dict_to_embedding_provider_definition(d: dict[str, Any]) -> EmbeddingProviderDefinition:
    """Convert TypedDict to EmbeddingProviderDefinition dataclass.

    Args:
        d: Dictionary conforming to EmbeddingProviderDefinitionDict schema.

    Returns:
        EmbeddingProviderDefinition dataclass instance.

    Raises:
        ValidationError: If required fields are missing.
    """
    _validate_required_keys(
        d,
        {"api_id", "internal_id", "display_name", "description", "provider_type"},
        "EmbeddingProviderDefinitionDict",
    )

    return EmbeddingProviderDefinition(
        api_id=d["api_id"],
        internal_id=d["internal_id"],
        display_name=d["display_name"],
        description=d["description"],
        provider_type=d["provider_type"],
        supports_quantization=d.get("supports_quantization", True),
        supports_instruction=d.get("supports_instruction", False),
        supports_batch_processing=d.get("supports_batch_processing", True),
        supports_asymmetric=d.get("supports_asymmetric", False),
        performance_characteristics=d.get("performance_characteristics", {}),
        supported_models=tuple(d.get("supported_models", [])),
        default_config=d.get("default_config", {}),
        is_plugin=d.get("is_plugin", False),
    )


# ============================================================================
# SparseVector Adapters (Sparse Indexer)
# ============================================================================


def validate_sparse_vector_dict(d: dict[str, Any]) -> SparseVectorDict:
    """Validate and cast dict to SparseVectorDict.

    Args:
        d: Dictionary to validate.

    Returns:
        The validated dict cast to SparseVectorDict.

    Raises:
        ValidationError: If validation fails.
    """
    _validate_required_keys(d, {"indices", "values", "chunk_id"}, "SparseVectorDict")

    indices = d["indices"]
    values = d["values"]

    if not isinstance(indices, list):
        raise ValidationError(f"indices must be a list, got {type(indices).__name__}")
    if not isinstance(values, list):
        raise ValidationError(f"values must be a list, got {type(values).__name__}")
    if len(indices) != len(values):
        raise ValidationError(f"indices and values must have same length: {len(indices)} != {len(values)}")

    return cast(SparseVectorDict, d)


def sparse_vector_to_dict(vec: SparseVector) -> SparseVectorDict:
    """Convert SparseVector dataclass to TypedDict.

    Args:
        vec: SparseVector dataclass instance.

    Returns:
        SparseVectorDict representation.
    """
    result: SparseVectorDict = {
        "indices": list(vec.indices),
        "values": list(vec.values),
        "chunk_id": vec.chunk_id,
    }
    if vec.metadata:
        result["metadata"] = vec.metadata
    return result


def dict_to_sparse_vector(d: dict[str, Any]) -> SparseVector:
    """Convert TypedDict to SparseVector dataclass with validation.

    Args:
        d: Dictionary conforming to SparseVectorDict schema.

    Returns:
        SparseVector dataclass instance.

    Raises:
        ValidationError: If validation fails.
    """
    validated = validate_sparse_vector_dict(d)
    return SparseVector(
        indices=tuple(validated["indices"]),
        values=tuple(validated["values"]),
        chunk_id=validated["chunk_id"],
        metadata=validated.get("metadata", {}),
    )


def coerce_to_sparse_vector(vec: SparseVector | dict[str, Any]) -> SparseVector:
    """Accept either dataclass or dict, return dataclass.

    Args:
        vec: Either a SparseVector or a dict conforming to the schema.

    Returns:
        SparseVector dataclass instance.
    """
    if isinstance(vec, SparseVector):
        return vec
    return dict_to_sparse_vector(vec)


# ============================================================================
# SparseQueryVector Adapters (Sparse Indexer)
# ============================================================================


def validate_sparse_query_vector_dict(d: dict[str, Any]) -> SparseQueryVectorDict:
    """Validate and cast dict to SparseQueryVectorDict.

    Args:
        d: Dictionary to validate.

    Returns:
        The validated dict cast to SparseQueryVectorDict.

    Raises:
        ValidationError: If validation fails.
    """
    _validate_required_keys(d, {"indices", "values"}, "SparseQueryVectorDict")

    indices = d["indices"]
    values = d["values"]

    if not isinstance(indices, list):
        raise ValidationError(f"indices must be a list, got {type(indices).__name__}")
    if not isinstance(values, list):
        raise ValidationError(f"values must be a list, got {type(values).__name__}")
    if len(indices) != len(values):
        raise ValidationError(f"indices and values must have same length: {len(indices)} != {len(values)}")

    return cast(SparseQueryVectorDict, d)


def sparse_query_vector_to_dict(vec: SparseQueryVector) -> SparseQueryVectorDict:
    """Convert SparseQueryVector dataclass to TypedDict.

    Args:
        vec: SparseQueryVector dataclass instance.

    Returns:
        SparseQueryVectorDict representation.
    """
    return {
        "indices": list(vec.indices),
        "values": list(vec.values),
    }


def dict_to_sparse_query_vector(d: dict[str, Any]) -> SparseQueryVector:
    """Convert TypedDict to SparseQueryVector dataclass with validation.

    Args:
        d: Dictionary conforming to SparseQueryVectorDict schema.

    Returns:
        SparseQueryVector dataclass instance.

    Raises:
        ValidationError: If validation fails.
    """
    validated = validate_sparse_query_vector_dict(d)
    return SparseQueryVector(
        indices=tuple(validated["indices"]),
        values=tuple(validated["values"]),
    )


def coerce_to_sparse_query_vector(
    vec: SparseQueryVector | dict[str, Any],
) -> SparseQueryVector:
    """Accept either dataclass or dict, return dataclass.

    Args:
        vec: Either a SparseQueryVector or a dict conforming to the schema.

    Returns:
        SparseQueryVector dataclass instance.
    """
    if isinstance(vec, SparseQueryVector):
        return vec
    return dict_to_sparse_query_vector(vec)


# ============================================================================
# SparseIndexerCapabilities Adapters (Sparse Indexer)
# ============================================================================


def sparse_indexer_capabilities_to_dict(
    caps: SparseIndexerCapabilities,
) -> SparseIndexerCapabilitiesDict:
    """Convert SparseIndexerCapabilities dataclass to TypedDict.

    Args:
        caps: SparseIndexerCapabilities dataclass instance.

    Returns:
        SparseIndexerCapabilitiesDict representation.
    """
    result: SparseIndexerCapabilitiesDict = {
        "sparse_type": caps.sparse_type,
        "max_tokens": caps.max_tokens,
        "vocabulary_handling": caps.vocabulary_handling,
        "supports_batching": caps.supports_batching,
        "max_batch_size": caps.max_batch_size,
        "requires_corpus_stats": caps.requires_corpus_stats,
        "supports_filters": caps.supports_filters,
        "idf_storage": caps.idf_storage,
    }
    if caps.max_terms_per_vector is not None:
        result["max_terms_per_vector"] = caps.max_terms_per_vector
    if caps.vocabulary_size is not None:
        result["vocabulary_size"] = caps.vocabulary_size
    if caps.supported_languages is not None:
        result["supported_languages"] = list(caps.supported_languages)
    return result


def dict_to_sparse_indexer_capabilities(d: dict[str, Any]) -> SparseIndexerCapabilities:
    """Convert TypedDict to SparseIndexerCapabilities dataclass.

    Args:
        d: Dictionary conforming to SparseIndexerCapabilitiesDict schema.

    Returns:
        SparseIndexerCapabilities dataclass instance.

    Raises:
        ValidationError: If validation fails.
    """
    _validate_required_keys(d, {"sparse_type", "max_tokens"}, "SparseIndexerCapabilitiesDict")
    _validate_string_enum(d["sparse_type"], SPARSE_TYPES, "sparse_type")

    supported_languages = d.get("supported_languages")
    if supported_languages is not None:
        supported_languages = tuple(supported_languages)

    return SparseIndexerCapabilities(
        sparse_type=d["sparse_type"],
        max_tokens=d["max_tokens"],
        vocabulary_handling=d.get("vocabulary_handling", "direct"),
        supports_batching=d.get("supports_batching", True),
        max_batch_size=d.get("max_batch_size", 64),
        requires_corpus_stats=d.get("requires_corpus_stats", False),
        max_terms_per_vector=d.get("max_terms_per_vector"),
        vocabulary_size=d.get("vocabulary_size"),
        supports_filters=d.get("supports_filters", False),
        idf_storage=d.get("idf_storage", "file"),
        supported_languages=supported_languages,
    )
