# Plugin Protocol Reference

This document provides the complete technical reference for semantik's plugin protocol interfaces. External plugins can implement these protocols using structural typing without importing anything from semantik.

## Protocol Version

**Current Version:** 1.0.0

**Versioning Strategy:**
- **MAJOR**: Breaking changes to required methods/attributes
- **MINOR**: New optional methods/attributes added
- **PATCH**: Documentation or behavior clarifications

Backward compatibility is maintained within major versions.

---

## Base Protocol

All plugins must satisfy the base `PluginProtocol`:

```python
class PluginProtocol(Protocol):
    """Base protocol all plugins must satisfy."""

    PLUGIN_TYPE: ClassVar[str]
    """Plugin type: 'connector', 'embedding', 'chunking', 'reranker', 'extractor', 'agent', or 'sparse_indexer'."""

    PLUGIN_ID: ClassVar[str]
    """Unique plugin identifier within its type (lowercase, hyphens recommended)."""

    PLUGIN_VERSION: ClassVar[str]
    """Semantic version string (e.g., '1.0.0')."""

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery and UI."""
        ...
```

### Manifest Format

The `get_manifest()` method must return a dictionary with:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `id` | `str` | Yes | Plugin identifier (matches PLUGIN_ID) |
| `type` | `str` | Yes | Plugin type (matches PLUGIN_TYPE) |
| `version` | `str` | Yes | Version string (matches PLUGIN_VERSION) |
| `display_name` | `str` | Yes | Human-readable name |
| `description` | `str` | Yes | Plugin description |
| `author` | `str \| None` | No | Author name or organization |
| `license` | `str \| None` | No | License identifier (e.g., "MIT") |
| `homepage` | `str \| None` | No | Project homepage URL |
| `requires` | `list` | No | Plugin dependencies |
| `semantik_version` | `str \| None` | No | Required semantik version |
| `capabilities` | `dict` | No | Plugin-specific capabilities |

---

## Connector Protocol

```python
@runtime_checkable
class ConnectorProtocol(Protocol):
    """Protocol for document source connectors."""

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "connector"
    PLUGIN_VERSION: ClassVar[str]
    METADATA: ClassVar[dict[str, Any]]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration dictionary."""
        ...

    async def authenticate(self) -> bool:
        """Verify credentials and connectivity.

        Returns:
            True if authentication succeeds.
        """
        ...

    def load_documents(
        self, source_id: int | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield documents from the source.

        Args:
            source_id: Optional source identifier for filtering.

        Yields:
            IngestedDocumentDict dictionaries.
        """
        ...

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Define configuration fields for the UI.

        Returns:
            List of field definitions.
        """
        ...

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Define which fields contain secrets.

        Returns:
            List of secret field definitions.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### IngestedDocumentDict

Documents yielded by `load_documents()` must match:

```python
class IngestedDocumentDict(TypedDict):
    """Document returned by connectors."""
    content: str                          # Full text content
    unique_id: str                        # Source-specific identifier
    source_type: str                      # Connector's PLUGIN_ID
    metadata: dict[str, Any]              # Source-specific metadata
    content_hash: str                     # SHA-256 hash (64 lowercase hex chars)
    file_path: NotRequired[str | None]    # Local file path if applicable
```

### Config Field Format

Fields returned by `get_config_fields()`:

```python
{
    "name": str,           # Field identifier
    "type": str,           # "text", "password", "number", "boolean", "select"
    "label": str,          # Display label
    "description": str,    # Help text (optional)
    "required": bool,      # Whether field is required
    "placeholder": str,    # Placeholder text (optional)
    "default": Any,        # Default value (optional)
    "options": list,       # For "select" type (optional)
}
```

---

## Embedding Protocol

```python
@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding model providers."""

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "embedding"
    PLUGIN_VERSION: ClassVar[str]
    INTERNAL_NAME: ClassVar[str]    # Internal registry identifier
    API_ID: ClassVar[str]           # External API identifier
    PROVIDER_TYPE: ClassVar[str]    # "local", "remote", or "hybrid"
    METADATA: ClassVar[dict[str, Any]]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with optional configuration."""
        ...

    async def embed_texts(
        self,
        texts: list[str],
        mode: str = "document",
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings to embed.
            mode: "query" or "document" (affects asymmetric models).

        Returns:
            List of embedding vectors (list of floats).
        """
        ...

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        """Return provider definition for registration.

        Returns:
            EmbeddingProviderDefinitionDict.
        """
        ...

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this provider supports a specific model.

        Args:
            model_name: Model identifier to check.

        Returns:
            True if the model is supported.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### EmbeddingProviderDefinitionDict

```python
class EmbeddingProviderDefinitionDict(TypedDict):
    """Definition for an embedding provider."""
    api_id: str                                    # External API identifier
    internal_id: str                               # Internal registry identifier
    display_name: str                              # Human-readable name
    description: str                               # Provider description
    provider_type: str                             # "local", "remote", "hybrid"
    supports_quantization: NotRequired[bool]       # Quantization support
    supports_instruction: NotRequired[bool]        # Instruction support
    supports_batch_processing: NotRequired[bool]   # Batch processing
    supports_asymmetric: NotRequired[bool]         # Query/doc asymmetry
    supported_models: NotRequired[list[str]]       # Supported model names
    default_config: NotRequired[dict[str, Any]]    # Default configuration
    is_plugin: NotRequired[bool]                   # Mark as external plugin
```

### Embedding Modes

| Mode | Description |
|------|-------------|
| `query` | For search queries; asymmetric models may apply query prefixes |
| `document` | For document indexing; typically no prefix applied |

---

## Chunking Protocol

```python
@runtime_checkable
class ChunkingProtocol(Protocol):
    """Protocol for text chunking strategies."""

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "chunking"
    PLUGIN_VERSION: ClassVar[str]

    def chunk(
        self,
        content: str,
        config: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Split content into chunks.

        Args:
            content: Full text content to chunk.
            config: Chunking configuration.
            progress_callback: Optional progress callback (0.0-1.0).

        Returns:
            List of ChunkDict dictionaries.
        """
        ...

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate content before chunking.

        Args:
            content: Text content to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        ...

    def estimate_chunks(self, content_length: int, config: dict[str, Any]) -> int:
        """Estimate the number of chunks for given content length.

        Args:
            content_length: Length of content in characters.
            config: Chunking configuration.

        Returns:
            Estimated number of chunks.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### ChunkDict

```python
class ChunkDict(TypedDict):
    """A chunk of text produced by chunking strategies."""
    content: str                              # Chunk text content
    metadata: ChunkMetadataDict               # Chunk metadata
    chunk_id: NotRequired[str | None]         # Unique chunk identifier
    embedding: NotRequired[list[float] | None] # Pre-computed embedding
```

### ChunkMetadataDict

```python
class ChunkMetadataDict(TypedDict, total=False):
    """Metadata for a text chunk. All fields optional."""
    chunk_id: str
    document_id: str
    chunk_index: int              # Position in document
    start_offset: int             # Start character position
    end_offset: int               # End character position
    token_count: int              # Token count
    strategy_name: str            # Chunking strategy used
    semantic_score: float | None  # Semantic coherence score
    semantic_density: float       # Information density
    confidence_score: float       # Chunk quality confidence
    overlap_percentage: float     # Overlap with adjacent chunks
    hierarchy_level: int | None   # Heading level
    section_title: str | None     # Section title
    heading_hierarchy: list[str]  # Parent headings
    custom_attributes: dict[str, Any]
```

### ChunkConfigDict

Configuration passed to `chunk()`:

```python
class ChunkConfigDict(TypedDict, total=False):
    """Configuration for chunking operations."""
    chunk_size: int           # Target chunk size in characters
    chunk_overlap: int        # Overlap between chunks
    min_chunk_size: int       # Minimum chunk size
    max_chunk_size: int       # Maximum chunk size
    min_tokens: int           # Minimum tokens per chunk
    max_tokens: int           # Maximum tokens per chunk
    overlap_tokens: int       # Overlap in tokens
    separator: str            # Text separator
    keep_separator: bool      # Keep separator in chunks
    preserve_structure: bool  # Preserve document structure
    semantic_threshold: float # Semantic similarity threshold
    hierarchy_levels: int     # Heading hierarchy depth
```

---

## Reranker Protocol

```python
@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for search result rerankers."""

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "reranker"
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with optional configuration."""
        ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents for a query.

        Args:
            query: Search query string.
            documents: List of document texts to rerank.
            top_k: Optional limit on results.
            metadata: Optional metadata for each document.

        Returns:
            List of RerankResultDict dictionaries.
        """
        ...

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare reranker capabilities.

        Returns:
            RerankerCapabilitiesDict.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### RerankResultDict

```python
class RerankResultDict(TypedDict):
    """Result from reranking a document."""
    index: int                           # Original document index
    score: float                         # Relevance score (higher = better)
    text: NotRequired[str]               # Document text
    metadata: NotRequired[dict[str, Any]] # Associated metadata
```

### RerankerCapabilitiesDict

```python
class RerankerCapabilitiesDict(TypedDict, total=False):
    """Reranker capability declaration."""
    max_documents: int        # Maximum documents per request
    max_query_length: int     # Maximum query length
    max_doc_length: int       # Maximum document length
    supports_batching: bool   # Batch processing support
    models: list[str]         # Supported model names
```

---

## Extractor Protocol

```python
@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol for entity/metadata extractors."""

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "extractor"
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with optional configuration."""
        ...

    async def extract(
        self,
        text: str,
        extraction_types: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract information from text.

        Args:
            text: Input text to extract from.
            extraction_types: Types to extract (see EXTRACTION_TYPES).
            options: Additional extraction options.

        Returns:
            ExtractionResultDict.
        """
        ...

    @classmethod
    def supported_extractions(cls) -> list[str]:
        """List supported extraction types.

        Returns:
            List of EXTRACTION_TYPES strings.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### ExtractionResultDict

```python
class ExtractionResultDict(TypedDict, total=False):
    """Results from entity extraction."""
    entities: list[EntityDict]          # Extracted entities
    keywords: list[str]                  # Extracted keywords
    language: str | None                 # Detected language code
    language_confidence: float | None    # Language detection confidence
    topics: list[str]                    # Identified topics
    sentiment: float | None              # Sentiment score (-1.0 to 1.0)
    summary: str | None                  # Text summary
    custom: dict[str, Any]               # Custom extraction results
```

### EntityDict

```python
class EntityDict(TypedDict):
    """An extracted entity."""
    text: str                             # Entity text as it appears
    type: str                             # Entity type (PERSON, ORG, etc.)
    start: NotRequired[int]               # Start character offset
    end: NotRequired[int]                 # End character offset
    confidence: NotRequired[float]        # Confidence score (0.0-1.0)
    metadata: NotRequired[dict[str, Any]] # Additional entity metadata
```

### EXTRACTION_TYPES

Valid values for `extraction_types`:

| Value | Description |
|-------|-------------|
| `entities` | Named entity recognition |
| `keywords` | Keyword extraction |
| `language` | Language detection |
| `topics` | Topic identification |
| `sentiment` | Sentiment analysis |
| `summary` | Text summarization |
| `custom` | Custom extraction types |

---

## Agent Protocol

```python
@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for AI agents."""

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "agent"
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with optional configuration."""
        ...

    def execute(
        self,
        prompt: str,
        *,
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute the agent and stream responses.

        Args:
            prompt: User prompt to process.
            context: Runtime context (AgentContextDict).
            system_prompt: Optional system prompt override.
            tools: Names of available tools.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            session_id: Session ID for conversation continuity.
            stream: Whether to stream partial responses.

        Yields:
            AgentMessageDict dictionaries.
        """
        ...

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare agent capabilities.

        Returns:
            AgentCapabilitiesDict.
        """
        ...

    @classmethod
    def supported_use_cases(cls) -> list[str]:
        """List supported use cases.

        Returns:
            List of AGENT_USE_CASES strings.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### AgentMessageDict

```python
class AgentMessageDict(TypedDict):
    """Message in agent conversation."""
    id: str                                      # Unique message identifier
    role: str                                    # MESSAGE_ROLES value
    type: str                                    # MESSAGE_TYPES value
    content: str                                 # Message content
    timestamp: str                               # ISO 8601 timestamp
    tool_name: NotRequired[str | None]           # For tool_use messages
    tool_call_id: NotRequired[str | None]        # Tool call identifier
    tool_input: NotRequired[dict[str, Any] | None]   # Tool arguments
    tool_output: NotRequired[dict[str, Any] | None]  # Tool result
    model: NotRequired[str | None]               # Model used
    usage: NotRequired[TokenUsageDict | None]    # Token usage stats
    cost_usd: NotRequired[float | None]          # Cost in USD
    is_partial: NotRequired[bool]                # Streaming partial flag
    sequence_number: NotRequired[int]            # Message order
    error_code: NotRequired[str | None]          # Error code
    error_details: NotRequired[dict[str, Any] | None]  # Error details
```

### TokenUsageDict

```python
class TokenUsageDict(TypedDict, total=False):
    """Token usage statistics."""
    input_tokens: int          # Input/prompt tokens
    output_tokens: int         # Output/completion tokens
    cache_read_tokens: int     # Tokens read from cache
    cache_write_tokens: int    # Tokens written to cache
    reasoning_tokens: int      # Tokens used for reasoning
```

### AgentCapabilitiesDict

```python
class AgentCapabilitiesDict(TypedDict, total=False):
    """Agent capability declaration."""
    supports_streaming: bool          # Streaming response support
    supports_tools: bool              # Tool use support
    supports_parallel_tools: bool     # Parallel tool execution
    supports_sessions: bool           # Session/conversation support
    supports_session_fork: bool       # Session forking support
    supports_interruption: bool       # Can be interrupted
    supports_extended_thinking: bool  # Extended thinking mode
    supports_thinking_budget: bool    # Thinking token budget
    supports_subagents: bool          # Sub-agent delegation
    supports_handoffs: bool           # Agent handoffs
    max_context_tokens: int | None    # Maximum context size
    max_output_tokens: int | None     # Maximum output size
    supported_models: list[str]       # Supported model names
    default_model: str | None         # Default model
    max_tools: int | None             # Maximum tools per request
```

### AgentContextDict

```python
class AgentContextDict(TypedDict, total=False):
    """Runtime context for agent execution."""
    request_id: str                               # Request identifier
    user_id: str | None                           # User identifier
    collection_id: str | None                     # Collection identifier
    collection_name: str | None                   # Collection name
    original_query: str | None                    # Original search query
    retrieved_chunks: list[dict[str, Any]] | None # Retrieved context
    session_id: str | None                        # Session identifier
    conversation_history: list[AgentMessageDict] | None  # Prior messages
    available_tools: list[str] | None             # Available tool names
    tool_configs: dict[str, dict[str, Any]] | None # Tool configurations
    max_tokens: int | None                        # Token limit
    timeout_seconds: float | None                 # Execution timeout
    trace_id: str | None                          # Distributed trace ID
    parent_span_id: str | None                    # Parent span ID
```

### MESSAGE_ROLES

Valid values for `AgentMessageDict.role`:

| Value | Description |
|-------|-------------|
| `user` | User message |
| `assistant` | Assistant response |
| `system` | System message |
| `tool_call` | Tool invocation request |
| `tool_result` | Tool execution result |
| `error` | Error message |

### MESSAGE_TYPES

Valid values for `AgentMessageDict.type`:

| Value | Description |
|-------|-------------|
| `text` | Regular text content |
| `thinking` | Reasoning/thinking content |
| `tool_use` | Tool invocation |
| `tool_output` | Tool result |
| `partial` | Streaming partial response |
| `final` | Final complete response |
| `error` | Error content |
| `metadata` | Metadata message |

### AGENT_USE_CASES

Valid values for `supported_use_cases()`:

| Value | Description |
|-------|-------------|
| `hyde` | Hypothetical Document Embeddings |
| `query_expansion` | Query expansion/rewriting |
| `query_understanding` | Query analysis and intent |
| `summarization` | Text summarization |
| `reranking` | Result reranking |
| `answer_synthesis` | Answer generation from context |
| `tool_use` | Tool-using agent |
| `agentic_search` | Multi-step search agent |
| `reasoning` | Complex reasoning tasks |
| `assistant` | General assistant |
| `code_generation` | Code generation |
| `data_analysis` | Data analysis |

---

## Sparse Indexer Protocol

```python
@runtime_checkable
class SparseIndexerProtocol(Protocol):
    """Protocol for sparse indexing plugins (BM25, SPLADE, etc.).

    Sparse indexers generate sparse vector representations for documents.
    The plugin is responsible ONLY for vector generation - persistence to
    Qdrant is handled by vecpipe infrastructure.

    Two main types:
    - BM25: Classic term-frequency based retrieval (indices=term_ids, values=tf-idf)
    - SPLADE: Learned sparse representations (indices=token_ids, values=learned_weights)

    Return Type Convention:
    - External plugins (dict-based): Return list[dict] from encode_documents()
    - Built-in plugins (ABC-based): Return list[SparseVector] from encode_documents()

    The DTO adapter layer handles bidirectional conversion.
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]  # Must be "sparse_indexer"
    PLUGIN_VERSION: ClassVar[str]
    SPARSE_TYPE: ClassVar[str]
    """Sparse representation type: 'bm25' or 'splade'. See SPARSE_TYPES."""

    async def encode_documents(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate sparse vectors for documents.

        Args:
            documents: List of documents with keys:
                - content (str): Document/chunk text
                - chunk_id (str): Unique chunk identifier (aligns with dense vectors)
                - metadata (dict, optional): Additional metadata

        Returns:
            List of SparseVectorDict dictionaries.
        """
        ...

    async def encode_query(self, query: str) -> dict[str, Any]:
        """Generate sparse vector for a search query.

        Args:
            query: Search query text.

        Returns:
            SparseQueryVectorDict with indices and values.
        """
        ...

    async def remove_documents(self, chunk_ids: list[str]) -> None:
        """Clean up any plugin-internal state for removed chunks.

        Called when chunks are deleted from the collection.
        For stateless plugins (like SPLADE), this may be a no-op.
        For BM25, this updates IDF statistics.

        Args:
            chunk_ids: List of chunk IDs being removed.
        """
        ...

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare sparse indexer capabilities.

        Returns:
            SparseIndexerCapabilitiesDict.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata."""
        ...
```

### SparseVectorDict

```python
class SparseVectorDict(TypedDict):
    """Sparse vector representation for indexing.

    For BM25: indices are term IDs, values are TF-IDF scores.
    For SPLADE: indices are token IDs, values are learned weights.

    Uses chunk_id (not document_id) to align 1:1 with dense vectors for RRF fusion.
    """
    indices: list[int]                    # Sparse vector indices (term/token IDs)
    values: list[float]                   # Corresponding weights/scores
    chunk_id: str                         # Chunk identifier (aligns with dense vectors)
    metadata: NotRequired[dict[str, Any]] # Additional indexing metadata
```

### SparseQueryVectorDict

```python
class SparseQueryVectorDict(TypedDict):
    """Sparse vector representation for query."""
    indices: list[int]   # Sparse vector indices (term/token IDs)
    values: list[float]  # Corresponding weights/scores
```

### SparseSearchResultDict

```python
class SparseSearchResultDict(TypedDict):
    """Search result from sparse indexing."""
    chunk_id: str                                  # Chunk identifier
    score: float                                   # Relevance score (higher = more relevant)
    matched_terms: NotRequired[list[str]]          # Terms that matched (for BM25)
    sparse_vector: NotRequired[SparseVectorDict]   # Original sparse vector
    payload: NotRequired[dict[str, Any]]           # Chunk payload from Qdrant
```

### SparseIndexerCapabilitiesDict

```python
class SparseIndexerCapabilitiesDict(TypedDict, total=False):
    """Sparse indexer capability declaration."""
    sparse_type: str           # 'bm25' or 'splade'. See SPARSE_TYPES.
    max_tokens: int            # Maximum tokens per document
    max_terms_per_vector: int | None  # Maximum non-zero terms (None = no limit)
    vocabulary_size: int | None       # Vocabulary size (None = open vocabulary)
    vocabulary_handling: str          # 'direct' or 'hashed'
    supports_batching: bool           # Whether batch encoding is supported
    max_batch_size: int               # Maximum documents per batch
    supports_filters: bool            # Whether metadata filters are supported
    requires_corpus_stats: bool       # Whether indexer needs corpus statistics
    idf_storage: str                  # IDF storage backend: 'file' or 'qdrant_point'
    supported_languages: list[str] | None  # ISO 639-1 codes (None = all)
```

### SPARSE_TYPES

Valid values for `sparse_type`:

| Value | Description |
|-------|-------------|
| `bm25` | Classic BM25/TF-IDF based sparse vectors |
| `splade` | Neural learned sparse representations |

### External Plugin Example

```python
# No semantik imports required
class MyBM25Indexer:
    PLUGIN_ID = "my-bm25"
    PLUGIN_TYPE = "sparse_indexer"
    PLUGIN_VERSION = "1.0.0"
    SPARSE_TYPE = "bm25"

    async def encode_documents(self, documents):
        results = []
        for doc in documents:
            # Generate sparse vector
            indices, values = self._compute_bm25(doc["content"])
            results.append({
                "indices": indices,
                "values": values,
                "chunk_id": doc["chunk_id"],
            })
        return results

    async def encode_query(self, query):
        indices, values = self._compute_query_vector(query)
        return {"indices": indices, "values": values}

    async def remove_documents(self, chunk_ids):
        # Update IDF stats when chunks are removed
        pass

    @classmethod
    def get_capabilities(cls):
        return {
            "sparse_type": "bm25",
            "max_tokens": 8192,
            "requires_corpus_stats": True,
            "supports_batching": True,
            "max_batch_size": 64,
        }

    @classmethod
    def get_manifest(cls):
        return {
            "id": cls.PLUGIN_ID,
            "type": cls.PLUGIN_TYPE,
            "version": cls.PLUGIN_VERSION,
            "display_name": "My BM25 Indexer",
            "description": "Custom BM25 implementation",
        }
```

---

## Protocol Type Mapping

The `PROTOCOL_BY_TYPE` mapping connects plugin types to protocols:

```python
PROTOCOL_BY_TYPE: dict[str, type] = {
    "connector": ConnectorProtocol,
    "embedding": EmbeddingProtocol,
    "chunking": ChunkingProtocol,
    "reranker": RerankerProtocol,
    "extractor": ExtractorProtocol,
    "agent": AgentProtocol,
    "sparse_indexer": SparseIndexerProtocol,
}
```

---

## Validation

### Adapter Validation

When external plugins return data, semantik's adapter layer validates:

1. **Required keys present**: All required TypedDict fields must exist
2. **String enum values**: Role/type strings must be in valid sets
3. **Content hash format**: Must be 64 lowercase hexadecimal characters
4. **Type correctness**: Values must match expected types

### Validation Errors

Common validation errors and how to fix them:

| Error | Cause | Fix |
|-------|-------|-----|
| `missing required keys: {'content'}` | Required field missing | Add the missing field |
| `Invalid role: 'user_message'` | Invalid enum string | Use valid MESSAGE_ROLES value |
| `content_hash must be 64 characters` | Hash wrong length | Use `hashlib.sha256(...).hexdigest()` |
| `content_hash must be lowercase hexadecimal` | Uppercase or invalid chars | Ensure lowercase hex only |

---

## Source Files

Protocol definitions: `packages/shared/plugins/protocols.py`
TypedDict definitions: `packages/shared/plugins/typed_dicts.py`
Adapters: `packages/shared/plugins/dto_adapters.py`
Test mixins: `packages/shared/plugins/testing/contracts.py`
Sparse indexer ABC: `packages/shared/plugins/types/sparse_indexer.py`
BM25 built-in: `packages/shared/plugins/builtins/bm25_sparse_indexer.py`
SPLADE built-in: `packages/shared/plugins/builtins/splade_indexer.py`

---

## See Also

- [Creating External Plugins](external-plugins.md) - Developer guide with examples
- [Plugin Development Guide](PLUGIN_DEVELOPMENT.md) - ABC-based plugin development
- [Plugin Testing](PLUGIN_TESTING.md) - Testing infrastructure
- [Sparse Indexing Guide](SPARSE_INDEXING.md) - User guide for sparse search
