"""
Core chunking processor service.

Handles the actual document chunking logic, strategy execution,
and fallback mechanisms.
"""

import logging
from typing import Any

from packages.shared.chunking.domain.services.chunking_strategies import get_strategy
from packages.shared.chunking.infrastructure.exceptions import ChunkingStrategyError, DocumentTooLargeError

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

        # Get and validate configuration
        chunk_size = int(config.get("chunk_size", self.DEFAULT_CHUNK_SIZE))
        chunk_overlap = int(config.get("chunk_overlap", self.DEFAULT_CHUNK_OVERLAP))

        # Get strategy from domain layer
        strategy_impl = get_strategy(factory_name)
        if not strategy_impl:
            raise ChunkingStrategyError(strategy=factory_name, reason=f"Strategy '{factory_name}' not found")

        # Execute chunking
        # Create a ChunkConfig object for strategies that need it
        from packages.shared.chunking.domain.value_objects import ChunkConfig

        chunk_config = ChunkConfig(
            strategy_name=factory_name,
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
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
        """Format chunks into standardized structure."""
        formatted = []

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                content = chunk
                metadata: dict[str, Any] = {"chunk_size": len(content), "position": i}
                token_count = len(content) // 4
                quality_score = 0.8
            elif isinstance(chunk, dict):
                content = chunk.get("content") or chunk.get("text") or str(chunk)
                metadata = chunk.get("metadata", {}).copy()
                metadata.setdefault("chunk_size", len(content))
                metadata["position"] = i
                token_count = chunk.get("token_count")
                if token_count is None:
                    token_count = len(content) // 4
                quality_score = chunk.get("quality_score", 0.8)
            else:
                content = str(chunk)
                metadata = {"position": i, "chunk_size": len(content)}
                token_count = len(content) // 4
                quality_score = 0.8

            formatted.append(
                {
                    "content": content,
                    "text": content,
                    "index": i,
                    "strategy": strategy,
                    "metadata": metadata,
                    "char_count": len(content),
                    "token_count": token_count,
                    "quality_score": quality_score,
                }
            )

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
