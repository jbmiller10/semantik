"""
Core chunking processor service.

Handles the actual document chunking logic, strategy execution,
and fallback mechanisms.
"""

import logging
from typing import Any

from shared.chunking.domain.entities.chunk import Chunk as DomainChunk
from shared.chunking.domain.services.chunking_strategies import get_strategy
from shared.chunking.infrastructure.exceptions import ChunkingStrategyError, DocumentTooLargeError
from shared.text_processing.base_chunker import ChunkResult

logger = logging.getLogger(__name__)


class ChunkingProcessor:
    """Service responsible for core chunking operations."""

    # Default configuration for simple fallback
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MIN_TOKEN_THRESHOLD = 100
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self) -> None:
        """Initialize the chunking processor."""
        self.strategy_mapping = {
            "fixed_size": "character",
            "sliding_window": "character",
            "semantic": "semantic",
            "recursive": "recursive",
            "document_structure": "markdown",
            "markdown": "markdown",
            "hierarchical": "hierarchical",
            "hybrid": "hybrid",
        }

    async def process_document(
        self,
        content: str,
        strategy: str,
        config: dict[str, Any] | None = None,
        use_fallback: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Process a document using the specified chunking strategy.

        Args:
            content: Document content to chunk
            strategy: Chunking strategy name
            config: Strategy configuration
            use_fallback: Whether to use fallback on error

        Returns:
            List of chunks with metadata

        Raises:
            DocumentTooLargeError: If document exceeds size limit
            ChunkingStrategyError: If strategy fails and no fallback
        """
        # Validate document size
        if len(content) > self.MAX_DOCUMENT_SIZE:
            raise DocumentTooLargeError(size=len(content), max_size=self.MAX_DOCUMENT_SIZE)

        config = config or {}

        try:
            return await self._execute_strategy(content, strategy, config)
        except Exception as e:
            if use_fallback:
                logger.warning(
                    "Strategy %s failed, using fallback: %s",
                    strategy,
                    str(e),
                )
                return self._apply_simple_fallback(content, config)
            raise ChunkingStrategyError(strategy=strategy, reason=f"Chunking failed: {str(e)}") from e

    async def _execute_strategy(
        self,
        content: str,
        strategy: str,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Execute a specific chunking strategy."""
        factory_name = self._map_strategy_name(strategy)

        # Get and validate configuration.
        #
        # Legacy configs (from UI + strategy registry) are character-based:
        #   {chunk_size: 1000, chunk_overlap: 200}
        #
        # The shared ChunkConfig is token-based and most unified strategies use
        # an approximate chars_per_token conversion internally. Convert legacy
        # values to token units so defaults like overlap=200 don't violate
        # overlap_tokens < min_tokens.
        is_token_config = any(key in config for key in ("max_tokens", "min_tokens", "overlap_tokens"))
        chars_per_token = 4

        if is_token_config:
            max_tokens = int(config.get("max_tokens", self.DEFAULT_CHUNK_SIZE))
            overlap_tokens = int(config.get("overlap_tokens", self.DEFAULT_CHUNK_OVERLAP))
            min_tokens_raw = config.get("min_tokens")
        else:
            chunk_size_chars = int(config.get("chunk_size", self.DEFAULT_CHUNK_SIZE))
            chunk_overlap_chars = int(config.get("chunk_overlap", self.DEFAULT_CHUNK_OVERLAP))

            # Ceiling division to avoid rounding chunk sizes down too aggressively.
            max_tokens = max(1, (chunk_size_chars + chars_per_token - 1) // chars_per_token)
            overlap_tokens = max(0, chunk_overlap_chars // chars_per_token)
            min_tokens_raw = config.get("min_chunk_size")

        # Get strategy from domain layer
        strategy_impl = get_strategy(factory_name)
        if not strategy_impl:
            raise ChunkingStrategyError(strategy=factory_name, reason=f"Strategy '{factory_name}' not found")

        # Execute chunking
        # Create a ChunkConfig object for strategies that need it
        from shared.chunking.domain.value_objects import ChunkConfig

        # Derive a safe min_tokens default that is compatible with the overlap.
        # The domain ChunkConfig enforces overlap_tokens < min_tokens <= max_tokens.
        if min_tokens_raw is None:
            derived_min = max(max_tokens // 5, overlap_tokens * 2, 10)
            min_tokens = min(derived_min, max_tokens)
        else:
            min_tokens = int(min_tokens_raw)
            min_tokens = max(1, min(min_tokens, max_tokens))

        # Ensure overlap constraints for domain validation.
        if overlap_tokens >= min_tokens:
            overlap_tokens = max(0, min_tokens - 1)
        if overlap_tokens >= max_tokens:
            overlap_tokens = max(0, max_tokens - 1)

        chunk_config = ChunkConfig(
            strategy_name=factory_name,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            separator=config.get("separator", " "),
        )

        # All strategies should accept content and ChunkConfig
        chunks = strategy_impl.chunk(content, chunk_config)

        # Convert to standardized format
        return self._format_chunks(chunks, strategy)

    def _apply_simple_fallback(
        self,
        content: str,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Apply simple character-based chunking as fallback."""
        chunk_size = int(config.get("chunk_size", self.DEFAULT_CHUNK_SIZE))
        chunk_overlap = int(config.get("chunk_overlap", self.DEFAULT_CHUNK_OVERLAP))

        chunks = self._simple_chunk_text(content, chunk_size, chunk_overlap)
        return self._format_chunks(chunks, "fallback")

    def _simple_chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Simple text chunking implementation."""
        if not text or chunk_size <= 0:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk = text[start:end]

            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

            if end >= text_length:
                break

            # Move start position considering overlap
            start = end - chunk_overlap if chunk_overlap > 0 else end

        return chunks

    def _format_chunks(
        self,
        chunks: list[Any],
        strategy: str,
    ) -> list[dict[str, Any]]:
        """Format chunks into a standardized structure."""
        formatted: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            content: str | None = None
            metadata: dict[str, Any] = {}
            token_count: int | None = None
            quality_score: float = 0.8
            chunk_id: str | None = None
            chunk_index = i
            start_offset: int | None = None
            end_offset: int | None = None

            if isinstance(chunk, DomainChunk):
                content = chunk.content
                meta = chunk.metadata
                metadata = {
                    "chunk_id": meta.chunk_id,
                    "document_id": meta.document_id,
                    "chunk_index": meta.chunk_index,
                    "start_offset": meta.start_offset,
                    "end_offset": meta.end_offset,
                    "token_count": meta.token_count,
                    "strategy_name": meta.strategy_name,
                    "semantic_score": meta.semantic_score,
                    "semantic_density": meta.semantic_density,
                    "confidence_score": meta.confidence_score,
                    "overlap_percentage": meta.overlap_percentage,
                    "hierarchy_level": meta.hierarchy_level,
                    "section_title": meta.section_title,
                }
                if meta.custom_attributes:
                    metadata["custom_attributes"] = meta.custom_attributes
                chunk_id = meta.chunk_id
                chunk_index = meta.chunk_index
                token_count = meta.token_count
                quality_score = meta.confidence_score or quality_score
                start_offset = meta.start_offset
                end_offset = meta.end_offset
            elif isinstance(chunk, ChunkResult):
                content = chunk.text
                metadata = dict(chunk.metadata or {})
                chunk_id = chunk.chunk_id
                chunk_index = metadata.get("chunk_index", i)
                token_count = metadata.get("token_count")
                start_offset = chunk.start_offset
                end_offset = chunk.end_offset
            elif isinstance(chunk, dict):
                content = chunk.get("content") or chunk.get("text") or str(chunk)
                metadata = chunk.get("metadata", {}).copy()
                chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id")
                chunk_index = chunk.get("chunk_index", chunk.get("index", i)) or i
                token_candidate = chunk.get("token_count")
                token_count = token_candidate if token_candidate is not None else metadata.get("token_count")
                quality_score = chunk.get("quality_score", metadata.get("quality_score", quality_score))
                start_offset = chunk.get("start_offset")
                end_offset = chunk.get("end_offset")
            elif isinstance(chunk, str):
                content = chunk
                metadata = {}
                token_count = len(content) // 4
            else:
                # Fallback for unexpected types
                content = str(chunk)
                metadata = {}
                token_count = len(content) // 4

            if content is None:
                content = ""

            # Normalize metadata
            metadata = metadata or {}
            effective_chunk_size = (
                end_offset - start_offset
                if start_offset is not None and end_offset is not None and end_offset > start_offset
                else len(content)
            )
            metadata.setdefault("chunk_size", effective_chunk_size)

            # Promote common hierarchical fields from custom_attributes for compatibility
            custom_attrs = metadata.get("custom_attributes")
            if isinstance(custom_attrs, dict):
                for key in ("parent_chunk_id", "child_chunk_ids", "chunk_sizes", "node_id", "is_leaf"):
                    if key in custom_attrs and key not in metadata:
                        metadata[key] = custom_attrs[key]

            # Calculate token count if missing
            token_count = int(token_count) if token_count is not None else len(content) // 4

            try:
                normalized_chunk_index = int(chunk_index)
            except (TypeError, ValueError):
                normalized_chunk_index = i
            metadata.setdefault("position", normalized_chunk_index)

            formatted_chunk: dict[str, Any] = {
                "content": content,
                "text": content,
                "index": normalized_chunk_index,
                "chunk_index": normalized_chunk_index,
                "strategy": strategy,
                "metadata": metadata,
                "char_count": len(content),
                "token_count": token_count,
                "quality_score": quality_score,
            }

            if chunk_id:
                formatted_chunk["chunk_id"] = chunk_id
            if start_offset is not None:
                formatted_chunk["start_offset"] = start_offset
            if end_offset is not None:
                formatted_chunk["end_offset"] = end_offset

            formatted.append(formatted_chunk)

        return formatted

    def _map_strategy_name(self, strategy: str) -> str:
        """Map API strategy name to factory strategy name."""
        return self.strategy_mapping.get(strategy, strategy)

    def calculate_statistics(
        self,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate statistics for a set of chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_size": 0,
                "size_variance": 0.0,
                "quality_score": 0.0,
            }

        sizes = [chunk.get("char_count") or len(chunk.get("content", "")) for chunk in chunks]

        total_size = sum(sizes)
        avg_size = total_size / len(sizes) if sizes else 0
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes) if len(sizes) > 1 else 0.0
        quality_score = 1.0 - min(1.0, variance / (avg_size**2)) if avg_size > 0 else 0.0

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": avg_size,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "total_size": total_size,
            "size_variance": variance,
            "quality_score": quality_score,
        }
