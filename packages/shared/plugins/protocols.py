"""Protocol interfaces for plugin runtime validation.

Defines structural typing protocols for all 6 plugin types, enabling external
plugins to be developed without importing anything from semantik.

External plugins can implement these interfaces using only Python standard
library types - no pip install semantik-* dependency required.

Protocol Version: 1.0.0
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable  # noqa: TCH003 - needed for @runtime_checkable
from typing import Any, ClassVar, Protocol, runtime_checkable

# Protocol version for compatibility tracking
# MAJOR: Breaking changes to required methods/attributes
# MINOR: New optional methods/attributes added
# PATCH: Documentation or behavior clarifications
PROTOCOL_VERSION = "1.0.0"


# ============================================================================
# Base Protocol
# ============================================================================


@runtime_checkable
class PluginProtocol(Protocol):
    """Base protocol all plugins must satisfy.

    Defines the minimal contract for plugin discovery and registration.
    Type-specific protocols extend this with additional requirements.
    """

    PLUGIN_TYPE: ClassVar[str]
    """Plugin type identifier (e.g., 'connector', 'embedding', 'agent')."""

    PLUGIN_ID: ClassVar[str]
    """Unique plugin identifier within its type."""

    PLUGIN_VERSION: ClassVar[str]
    """Semantic version string (e.g., '1.0.0')."""

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery and UI.

        Returns:
            Dictionary with keys: id, type, version, display_name, description.
            Optional keys: author, license, homepage, requires, capabilities.
        """
        ...


# ============================================================================
# Connector Protocol
# ============================================================================


@runtime_checkable
class ConnectorProtocol(Protocol):
    """Protocol for document source connectors.

    Connectors ingest documents from external sources (filesystems, APIs,
    databases) and yield them for indexing.

    Example external implementation:
        class MyConnector:
            PLUGIN_ID = "my-connector"
            PLUGIN_TYPE = "connector"
            PLUGIN_VERSION = "1.0.0"

            def __init__(self, config: dict[str, Any]) -> None:
                self._config = config

            async def authenticate(self) -> bool:
                return True

            async def load_documents(self, source_id: int | None = None):
                yield {
                    "content": "...",
                    "unique_id": "doc-1",
                    "source_type": "my-connector",
                    "metadata": {},
                    "content_hash": "a" * 64,  # SHA-256 hex
                }
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]
    METADATA: ClassVar[dict[str, Any]]

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize connector with configuration.

        Args:
            config: Dictionary containing connector-specific settings.
        """
        ...

    async def authenticate(self) -> bool:
        """Verify credentials and connectivity.

        Returns:
            True if authentication succeeds, False otherwise.
        """
        ...

    def load_documents(self, source_id: int | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield documents from the source.

        Args:
            source_id: Optional source identifier for filtering.

        Yields:
            Dictionaries with keys:
                - content (str): Document text content
                - unique_id (str): Source-specific identifier
                - source_type (str): Connector type identifier
                - metadata (dict): Source-specific metadata
                - content_hash (str): SHA-256 hash (64 lowercase hex chars)
                - file_path (str, optional): Local file path if applicable
        """
        ...

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        """Define configuration fields for the UI.

        Returns:
            List of field definitions with keys: name, type, required, etc.
        """
        ...

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        """Define which fields contain secrets.

        Returns:
            List of field definitions that should be encrypted/masked.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Embedding Protocol
# ============================================================================


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Protocol for embedding model providers.

    Embedding plugins convert text into vector representations for
    semantic search and similarity operations.

    Example external implementation:
        class MyEmbedding:
            PLUGIN_ID = "my-embedding"
            PLUGIN_TYPE = "embedding"
            PLUGIN_VERSION = "1.0.0"
            INTERNAL_NAME = "my_embedding"
            API_ID = "my-embedding"
            PROVIDER_TYPE = "remote"

            async def embed_texts(self, texts: list[str], mode: str = "document"):
                return [[0.1, 0.2, ...] for _ in texts]
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]
    INTERNAL_NAME: ClassVar[str]
    """Internal registry identifier."""
    API_ID: ClassVar[str]
    """External API identifier (often same as PLUGIN_ID)."""
    PROVIDER_TYPE: ClassVar[str]
    """Provider type: 'local', 'remote', or 'hybrid'."""
    METADATA: ClassVar[dict[str, Any]]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize embedding provider.

        Args:
            config: Optional provider-specific configuration.
        """
        ...

    async def embed_texts(
        self,
        texts: list[str],
        mode: str = "document",
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            mode: Embedding mode - 'query' or 'document'.
                  Query mode may apply prefixes for asymmetric models.

        Returns:
            List of embedding vectors (list of floats) for each input text.
        """
        ...

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        """Return provider definition for registration.

        Returns:
            Dictionary with keys: api_id, internal_id, display_name,
            description, provider_type, and optional capability flags.
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
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Chunking Protocol
# ============================================================================


@runtime_checkable
class ChunkingProtocol(Protocol):
    """Protocol for text chunking strategies.

    Chunking plugins split documents into smaller segments suitable
    for embedding and retrieval.

    Example external implementation:
        class MyChunking:
            PLUGIN_ID = "my-chunking"
            PLUGIN_TYPE = "chunking"
            PLUGIN_VERSION = "1.0.0"

            def chunk(self, content: str, config: dict, progress_callback=None):
                return [{"content": content[:500], "metadata": {"chunk_index": 0}}]
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
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
            config: Chunking configuration with keys like chunk_size,
                   chunk_overlap, min_chunk_size, max_chunk_size.
            progress_callback: Optional callback for progress updates (0.0-1.0).

        Returns:
            List of chunk dictionaries with keys:
                - content (str): Chunk text
                - metadata (dict): Chunk metadata (index, positions, etc.)
                - chunk_id (str, optional): Unique chunk identifier
                - embedding (list[float], optional): Pre-computed embedding
        """
        ...

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate content before chunking.

        Args:
            content: Text content to validate.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
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
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Reranker Protocol
# ============================================================================


@runtime_checkable
class RerankerProtocol(Protocol):
    """Protocol for search result rerankers.

    Reranker plugins rescore search results using more sophisticated
    models than initial retrieval.

    Example external implementation:
        class MyReranker:
            PLUGIN_ID = "my-reranker"
            PLUGIN_TYPE = "reranker"
            PLUGIN_VERSION = "1.0.0"

            async def rerank(self, query: str, documents: list[str], top_k=None):
                return [{"index": 0, "score": 0.95}]
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize reranker.

        Args:
            config: Optional reranker-specific configuration.
        """
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
            top_k: Optional limit on number of results to return.
            metadata: Optional metadata for each document.

        Returns:
            List of result dictionaries with keys:
                - index (int): Original document index
                - score (float): Relevance score
                - text (str, optional): Document text
                - metadata (dict, optional): Associated metadata
        """
        ...

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare reranker capabilities.

        Returns:
            Dictionary with keys: max_documents, max_query_length,
            max_doc_length, supports_batching, models.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Extractor Protocol
# ============================================================================


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Protocol for entity/metadata extractors.

    Extractor plugins extract structured information from text,
    such as entities, keywords, topics, and sentiment.

    Example external implementation:
        class MyExtractor:
            PLUGIN_ID = "my-extractor"
            PLUGIN_TYPE = "extractor"
            PLUGIN_VERSION = "1.0.0"

            async def extract(self, text: str, extraction_types=None):
                return {"keywords": ["example"], "entities": []}
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize extractor.

        Args:
            config: Optional extractor-specific configuration.
        """
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
            extraction_types: Types to extract. Valid values:
                'entities', 'keywords', 'topics', 'summary', 'sentiment'.
            options: Additional extraction options.

        Returns:
            Dictionary with extracted data. Keys depend on extraction_types:
                - entities (list[dict]): Extracted entities with text, type, etc.
                - keywords (list[str]): Extracted keywords
                - topics (list[str]): Identified topics
                - summary (str): Text summary
                - sentiment (dict): Sentiment analysis results
        """
        ...

    @classmethod
    def supported_extractions(cls) -> list[str]:
        """List supported extraction types.

        Returns:
            List of supported extraction type strings.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Agent Protocol
# ============================================================================


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for AI agents.

    Agent plugins provide LLM-powered capabilities for various use cases
    like search enhancement, summarization, and conversational interfaces.

    Example external implementation:
        class MyAgent:
            PLUGIN_ID = "my-agent"
            PLUGIN_TYPE = "agent"
            PLUGIN_VERSION = "1.0.0"

            async def execute(self, prompt: str, *, context=None, **kwargs):
                yield {
                    "id": "msg-1",
                    "role": "assistant",
                    "type": "text",
                    "content": f"Response to: {prompt}",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize agent.

        Args:
            config: Optional agent-specific configuration.
        """
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
            context: Runtime context with request, user, and collection info.
            system_prompt: Optional system prompt override.
            tools: Names of tools available for this execution.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            session_id: Session ID for conversation continuity.
            stream: Whether to stream partial responses.

        Yields:
            Message dictionaries with keys:
                - id (str): Unique message identifier
                - role (str): 'user', 'assistant', 'system', 'tool_call',
                             'tool_result', or 'error'
                - type (str): 'text', 'thinking', 'tool_use', 'tool_output',
                             'partial', 'final', 'error', or 'metadata'
                - content (str): Message content
                - timestamp (str): ISO 8601 timestamp
                - is_partial (bool, optional): Whether this is streaming partial
                - usage (dict, optional): Token usage statistics
                - tool_name (str, optional): For tool_use messages
                - tool_input (dict, optional): Tool call arguments
        """
        ...

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare agent capabilities.

        Returns:
            Dictionary with capability flags:
                - supports_streaming (bool)
                - supports_tools (bool)
                - supports_sessions (bool)
                - supports_extended_thinking (bool)
                - max_context_tokens (int, optional)
                - max_output_tokens (int, optional)
                - supported_models (list[str])
                - default_model (str, optional)
        """
        ...

    @classmethod
    def supported_use_cases(cls) -> list[str]:
        """List supported use cases.

        Returns:
            List of use case strings. Valid values:
                'hyde', 'query_expansion', 'query_understanding',
                'summarization', 'reranking', 'answer_synthesis',
                'tool_use', 'agentic_search', 'reasoning',
                'assistant', 'code_generation', 'data_analysis'.
        """
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Sparse Indexer Protocol
# ============================================================================


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

    Example external implementation:
        class MyBM25Indexer:
            PLUGIN_ID = "my-bm25"
            PLUGIN_TYPE = "sparse_indexer"
            PLUGIN_VERSION = "1.0.0"
            SPARSE_TYPE = "bm25"

            async def encode_documents(self, documents):
                return [{
                    "indices": [1, 5, 42],
                    "values": [2.3, 1.8, 3.1],
                    "chunk_id": "chunk-1",
                }]

            async def encode_query(self, query):
                return {"indices": [1, 42], "values": [1.0, 2.5]}
    """

    PLUGIN_ID: ClassVar[str]
    PLUGIN_TYPE: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]
    SPARSE_TYPE: ClassVar[str]
    """Sparse representation type: 'bm25' or 'splade'."""

    async def encode_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate sparse vectors for documents.

        NOTE: This method generates vectors only. Persistence to Qdrant
        is handled by vecpipe infrastructure, not the plugin.

        Args:
            documents: List of documents with keys:
                - content (str): Document/chunk text
                - chunk_id (str): Unique chunk identifier (aligns with dense vectors)
                - metadata (dict, optional): Additional metadata

        Returns:
            List of SparseVectorDict with keys:
                - indices (list[int]): Sparse vector indices
                - values (list[float]): Corresponding weights/scores
                - chunk_id (str): Chunk identifier
        """
        ...

    async def encode_query(self, query: str) -> dict[str, Any]:
        """Generate sparse vector for a search query.

        Args:
            query: Search query text

        Returns:
            SparseQueryVectorDict with keys:
                - indices (list[int]): Sparse vector indices
                - values (list[float]): Corresponding weights/scores
        """
        ...

    async def remove_documents(self, chunk_ids: list[str]) -> None:
        """Clean up any plugin-internal state for removed chunks.

        Called when chunks are deleted from the collection.
        For stateless plugins (like SPLADE), this may be a no-op.
        For BM25, this updates IDF statistics.

        Args:
            chunk_ids: List of chunk IDs being removed (aligns with dense vectors)
        """
        ...

    @classmethod
    def get_capabilities(cls) -> dict[str, Any]:
        """Declare sparse indexer capabilities."""
        ...

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        """Return plugin metadata for discovery."""
        ...


# ============================================================================
# Protocol Type Mapping
# ============================================================================

PROTOCOL_BY_TYPE: dict[str, type] = {
    "connector": ConnectorProtocol,
    "embedding": EmbeddingProtocol,
    "chunking": ChunkingProtocol,
    "reranker": RerankerProtocol,
    "extractor": ExtractorProtocol,
    "agent": AgentProtocol,
    "sparse_indexer": SparseIndexerProtocol,
}
"""Mapping from plugin type string to corresponding protocol class.

Used by the loader to validate plugins against their type-specific protocol.
"""
