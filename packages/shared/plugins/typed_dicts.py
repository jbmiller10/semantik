"""TypedDict definitions for plugin protocol data transfer.

This module defines TypedDict schemas and protocol constants that enable
external plugins to be developed without importing anything from semantik.

External plugins can:
1. Return plain dicts that match these TypedDict schemas
2. Use the protocol constants for valid string values (instead of enums)

Protocol Version: 1.0.0

Note: This module intentionally does NOT use `from __future__ import annotations`
because it breaks TypedDict's ability to recognize NotRequired fields at runtime.
"""

from typing import Any, NotRequired, TypedDict

# Re-export protocol version for external reference
PROTOCOL_VERSION = "1.0.0"


# ============================================================================
# Protocol Constants
# ============================================================================
# Valid string values for fields that map to internal enums.
# External plugins use these strings; adapters convert to enums.

MESSAGE_ROLES = frozenset(
    {
        "user",
        "assistant",
        "system",
        "tool_call",
        "tool_result",
        "error",
    }
)
"""Valid values for AgentMessageDict.role field."""

MESSAGE_TYPES = frozenset(
    {
        "text",
        "thinking",
        "tool_use",
        "tool_output",
        "partial",
        "final",
        "error",
        "metadata",
    }
)
"""Valid values for AgentMessageDict.type field."""

EMBEDDING_MODES = frozenset({"query", "document"})
"""Valid values for embedding mode parameter."""

EXTRACTION_TYPES = frozenset(
    {
        "entities",
        "keywords",
        "language",
        "topics",
        "sentiment",
        "summary",
        "custom",
    }
)
"""Valid values for extraction_types parameter."""

AGENT_USE_CASES = frozenset(
    {
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
)
"""Valid values for AgentProtocol.supported_use_cases()."""

SPARSE_TYPES = frozenset({"bm25", "splade"})
"""Valid values for SparseIndexerCapabilitiesDict.sparse_type field."""


# ============================================================================
# Common DTOs
# ============================================================================


class PluginManifestDict(TypedDict):
    """Plugin metadata for discovery and UI.

    Returned by all plugins' get_manifest() method.
    """

    id: str
    type: str
    version: str
    display_name: str
    description: str
    author: NotRequired[str | None]
    license: NotRequired[str | None]
    homepage: NotRequired[str | None]
    requires: NotRequired[list[str | dict[str, Any]]]
    semantik_version: NotRequired[str | None]
    capabilities: NotRequired[dict[str, Any]]


# ============================================================================
# Connector DTOs
# ============================================================================


class IngestedDocumentDict(TypedDict):
    """Document returned by ConnectorProtocol.load_documents().

    Attributes:
        content: Full text content of the document.
        unique_id: Source-specific identifier (URI, path, message ID).
        source_type: Connector's PLUGIN_ID.
        metadata: Source-specific metadata dictionary.
        content_hash: SHA-256 hash as 64 lowercase hex characters.
        file_path: Local file path if applicable.
    """

    content: str
    unique_id: str
    source_type: str
    metadata: dict[str, Any]
    content_hash: str
    file_path: NotRequired[str | None]


# ============================================================================
# Chunking DTOs
# ============================================================================


class ChunkMetadataDict(TypedDict, total=False):
    """Metadata for a text chunk.

    All fields optional for flexibility.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    token_count: int
    strategy_name: str
    semantic_score: float | None
    semantic_density: float
    confidence_score: float
    overlap_percentage: float
    hierarchy_level: int | None
    section_title: str | None
    heading_hierarchy: list[str]
    custom_attributes: dict[str, Any]


class ChunkDict(TypedDict):
    """A chunk of text returned by ChunkingProtocol.chunk().

    Attributes:
        content: Chunk text content.
        metadata: Chunk metadata (positions, scores, etc.).
        chunk_id: Optional unique chunk identifier.
        embedding: Optional pre-computed embedding vector.
    """

    content: str
    metadata: ChunkMetadataDict
    chunk_id: NotRequired[str | None]
    embedding: NotRequired[list[float] | None]


class ChunkConfigDict(TypedDict, total=False):
    """Configuration for chunking operations.

    Passed to ChunkingProtocol.chunk() and estimate_chunks().
    """

    chunk_size: int
    chunk_overlap: int
    min_chunk_size: int
    max_chunk_size: int
    min_tokens: int
    max_tokens: int
    overlap_tokens: int
    separator: str
    keep_separator: bool
    preserve_structure: bool
    semantic_threshold: float
    hierarchy_levels: int


# ============================================================================
# Reranker DTOs
# ============================================================================


class RerankResultDict(TypedDict):
    """Result from RerankerProtocol.rerank().

    Attributes:
        index: Original document index in the input list.
        score: Relevance score (higher = more relevant).
        text: Document text (optional, for convenience).
        metadata: Associated metadata (optional).
    """

    index: int
    score: float
    text: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]


class RerankerCapabilitiesDict(TypedDict, total=False):
    """Reranker capability declaration.

    Returned by RerankerProtocol.get_capabilities().
    """

    max_documents: int
    max_query_length: int
    max_doc_length: int
    supports_batching: bool
    models: list[str]


# ============================================================================
# Extractor DTOs
# ============================================================================


class EntityDict(TypedDict):
    """An extracted entity.

    Attributes:
        text: Entity text as it appears in source.
        type: Entity type (PERSON, ORG, LOC, DATE, etc.).
        start: Start character offset (optional).
        end: End character offset (optional).
        confidence: Confidence score 0.0-1.0 (optional).
        metadata: Additional entity metadata (optional).
    """

    text: str
    type: str
    start: NotRequired[int]
    end: NotRequired[int]
    confidence: NotRequired[float]
    metadata: NotRequired[dict[str, Any]]


class ExtractionResultDict(TypedDict, total=False):
    """Results from ExtractorProtocol.extract().

    Keys present depend on extraction_types requested.
    """

    entities: list[EntityDict]
    keywords: list[str]
    language: str | None
    language_confidence: float | None
    topics: list[str]
    sentiment: float | None  # -1.0 to 1.0
    summary: str | None
    custom: dict[str, Any]


# ============================================================================
# Agent DTOs
# ============================================================================


class TokenUsageDict(TypedDict, total=False):
    """Token usage statistics.

    Included in AgentMessageDict for tracking costs.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int


class AgentMessageDict(TypedDict):
    """Message in agent conversation.

    Yielded by AgentProtocol.execute().

    Attributes:
        id: Unique message identifier.
        role: Message role (see MESSAGE_ROLES).
        type: Message type (see MESSAGE_TYPES).
        content: Message content.
        timestamp: ISO 8601 timestamp string.
    """

    id: str
    role: str  # See MESSAGE_ROLES
    type: str  # See MESSAGE_TYPES
    content: str
    timestamp: str  # ISO 8601 format
    tool_name: NotRequired[str | None]
    tool_call_id: NotRequired[str | None]
    tool_input: NotRequired[dict[str, Any] | None]
    tool_output: NotRequired[dict[str, Any] | None]
    model: NotRequired[str | None]
    usage: NotRequired[TokenUsageDict | None]
    cost_usd: NotRequired[float | None]
    is_partial: NotRequired[bool]
    sequence_number: NotRequired[int]
    error_code: NotRequired[str | None]
    error_details: NotRequired[dict[str, Any] | None]


class AgentCapabilitiesDict(TypedDict, total=False):
    """Agent capability declaration.

    Returned by AgentProtocol.get_capabilities().
    """

    supports_streaming: bool
    supports_tools: bool
    supports_parallel_tools: bool
    supports_sessions: bool
    supports_session_fork: bool
    supports_interruption: bool
    supports_extended_thinking: bool
    supports_thinking_budget: bool
    supports_subagents: bool
    supports_handoffs: bool
    max_context_tokens: int | None
    max_output_tokens: int | None
    supported_models: list[str]
    default_model: str | None
    max_tools: int | None


class AgentContextDict(TypedDict, total=False):
    """Runtime context for agent execution.

    Passed to AgentProtocol.execute() as context parameter.
    """

    request_id: str
    user_id: str | None
    collection_id: str | None
    collection_name: str | None
    original_query: str | None
    retrieved_chunks: list[dict[str, Any]] | None
    session_id: str | None
    conversation_history: list[AgentMessageDict] | None
    available_tools: list[str] | None
    tool_configs: dict[str, dict[str, Any]] | None
    max_tokens: int | None
    timeout_seconds: float | None
    trace_id: str | None
    parent_span_id: str | None


# ============================================================================
# Embedding DTOs
# ============================================================================


class EmbeddingProviderDefinitionDict(TypedDict):
    """Definition for an embedding provider.

    Returned by EmbeddingProtocol.get_definition().
    """

    api_id: str
    internal_id: str
    display_name: str
    description: str
    provider_type: str  # "local", "remote", "hybrid"
    supports_quantization: NotRequired[bool]
    supports_instruction: NotRequired[bool]
    supports_batch_processing: NotRequired[bool]
    supports_asymmetric: NotRequired[bool]
    supported_models: NotRequired[list[str]]
    default_config: NotRequired[dict[str, Any]]
    performance_characteristics: NotRequired[dict[str, Any]]
    is_plugin: NotRequired[bool]


# ============================================================================
# Sparse Indexer DTOs
# ============================================================================


class SparseVectorDict(TypedDict):
    """Sparse vector representation for indexing.

    For BM25: indices are term IDs, values are TF-IDF scores.
    For SPLADE: indices are token IDs, values are learned weights.

    Uses chunk_id (not document_id) to align 1:1 with dense vectors for RRF fusion.

    Attributes:
        indices: Sparse vector indices (term/token IDs).
        values: Corresponding weights/scores.
        chunk_id: Chunk identifier (aligns with dense vectors).
        metadata: Additional indexing metadata.
    """

    indices: list[int]
    values: list[float]
    chunk_id: str
    metadata: NotRequired[dict[str, Any]]


class SparseQueryVectorDict(TypedDict):
    """Sparse vector representation for query.

    Attributes:
        indices: Sparse vector indices (term/token IDs).
        values: Corresponding weights/scores.
    """

    indices: list[int]
    values: list[float]


class SparseSearchResultDict(TypedDict):
    """Search result from sparse indexing.

    Attributes:
        chunk_id: Chunk identifier (aligns with dense vectors for RRF fusion).
        score: Relevance score (higher = more relevant).
        matched_terms: Terms that matched the query (for BM25).
        sparse_vector: Original sparse vector (optional).
        payload: Chunk payload from Qdrant.
    """

    chunk_id: str
    score: float
    matched_terms: NotRequired[list[str]]
    sparse_vector: NotRequired[SparseVectorDict]
    payload: NotRequired[dict[str, Any]]


class SparseIndexerCapabilitiesDict(TypedDict, total=False):
    """Sparse indexer capability declaration.

    Attributes:
        sparse_type: Type: 'bm25' or 'splade'. See SPARSE_TYPES.
        max_tokens: Maximum tokens per document.
        max_terms_per_vector: Maximum non-zero terms in output sparse vector.
        vocabulary_size: Vocabulary size (if fixed). None = open vocabulary.
        vocabulary_handling: How vocabulary maps to sparse dimensions.
        supports_batching: Whether batch encoding is supported.
        max_batch_size: Maximum documents per batch.
        supports_filters: Whether metadata filters are supported during search.
        requires_corpus_stats: Whether indexer needs corpus statistics (e.g., BM25 IDF).
        idf_storage: IDF storage backend: 'file' or 'qdrant_point'.
        supported_languages: Supported languages (ISO 639-1 codes).
    """

    sparse_type: str
    max_tokens: int
    max_terms_per_vector: int | None
    vocabulary_size: int | None
    vocabulary_handling: str
    supports_batching: bool
    max_batch_size: int
    supports_filters: bool
    requires_corpus_stats: bool
    idf_storage: str
    supported_languages: list[str] | None
