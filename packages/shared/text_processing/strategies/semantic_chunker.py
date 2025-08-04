"""Semantic chunking strategy using LlamaIndex SemanticSplitterNodeParser.

This module implements semantic-aware text chunking that uses embeddings to find
natural topic boundaries in text, resulting in more coherent chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult
from packages.shared.text_processing.chunking_metrics import performance_monitor

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding

logger = logging.getLogger(__name__)


class SemanticChunker(BaseChunker):
    """Semantic chunking using embeddings to find natural topic boundaries."""

    def __init__(
        self,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1,
        max_chunk_size: int = 1000,
        embed_model: BaseEmbedding | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SemanticChunker.

        Args:
            breakpoint_percentile_threshold: Percentile threshold for semantic breakpoints (0-100)
            buffer_size: Number of sentences to group before calculating embeddings
            max_chunk_size: Maximum size of chunks in tokens
            embed_model: Embedding model to use (if None, will use default from factory)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        # Validate parameters
        if not 0 <= breakpoint_percentile_threshold <= 100:
            raise ValueError(f"breakpoint_percentile_threshold must be between 0 and 100, got {breakpoint_percentile_threshold}")

        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")

        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.buffer_size = buffer_size
        self.max_chunk_size = max_chunk_size
        self.embed_model = embed_model

        # Initialize splitter - will be created on first use
        self._splitter: SemanticSplitterNodeParser | None = None
        self._initialization_lock = asyncio.Lock()
        self._sync_initialization_lock = asyncio.Lock()  # For sync context

        logger.info(
            f"Initialized SemanticChunker with threshold={breakpoint_percentile_threshold}, "
            f"buffer_size={buffer_size}, max_chunk_size={max_chunk_size}"
        )

    def _get_splitter(self) -> SemanticSplitterNodeParser:
        """Get or create the semantic splitter.

        Returns:
            Initialized semantic splitter

        Raises:
            RuntimeError: If embedding model is not available
        """
        if self._splitter is None:
            if self.embed_model is None:
                raise RuntimeError(
                    "Embedding model not provided. Please provide embed_model parameter "
                    "or ensure it's configured in the factory."
                )

            self._splitter = SemanticSplitterNodeParser(
                embed_model=self.embed_model,
                buffer_size=self.buffer_size,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
            )

        return self._splitter

    async def _get_splitter_async(self) -> SemanticSplitterNodeParser:
        """Get or create the semantic splitter asynchronously.

        Returns:
            Initialized semantic splitter
        """
        async with self._initialization_lock:
            return self._get_splitter()

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous semantic chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        with performance_monitor.measure_chunking(
            strategy="semantic",
            doc_id=doc_id,
            text_length=len(text),
            metadata={"threshold": self.breakpoint_percentile_threshold},
        ) as metrics:
            try:
                # Get splitter
                splitter = self._get_splitter()

                # Create document
                doc = Document(text=text, metadata=metadata or {})

                # Perform semantic chunking with retry logic
                max_retries = 3
                retry_delay = 1.0

                for attempt in range(max_retries):
                    try:
                        nodes = splitter.get_nodes_from_documents([doc])
                        break  # Success
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Semantic chunking attempt {attempt + 1} failed: {e}. "
                                f"Retrying in {retry_delay}s..."
                            )
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            logger.error(f"Semantic chunking failed after {max_retries} attempts: {e}")
                            raise RuntimeError(f"Semantic chunking failed: {e}") from e

                # Convert nodes to ChunkResult
                results = []
                cumulative_length = 0

                for idx, node in enumerate(nodes):
                    content = node.get_content()
                    start_offset = cumulative_length
                    end_offset = start_offset + len(content)

                    # Add semantic chunking metadata
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata["semantic_boundary"] = True
                    chunk_metadata["breakpoint_threshold"] = self.breakpoint_percentile_threshold

                    result = self._create_chunk_result(
                        doc_id=doc_id,
                        chunk_index=idx,
                        text=content,
                        start_offset=start_offset,
                        end_offset=end_offset,
                        metadata=chunk_metadata,
                    )
                    results.append(result)

                    cumulative_length = end_offset

                # Update metrics
                metrics.output_chunks = len(results)

                logger.debug(f"Created {len(results)} semantic chunks from {len(text)} characters")
                return results

            except Exception as e:
                logger.error(f"Error in semantic chunking: {e}")
                # Fallback to character-based chunking on embedding failures
                if "embed" in str(e).lower() or "model" in str(e).lower():
                    logger.warning("Falling back to character-based chunking due to embedding error")
                    from .character_chunker import CharacterChunker

                    fallback_chunker = CharacterChunker(
                        chunk_size=self.max_chunk_size,
                        chunk_overlap=100,
                    )
                    return fallback_chunker.chunk_text(text, doc_id, metadata)
                raise

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous semantic chunking.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        try:
            # Get splitter asynchronously
            splitter = await self._get_splitter_async()

            # Run chunking in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._chunk_with_splitter,
                splitter,
                text,
                doc_id,
                metadata,
            )

        except Exception as e:
            logger.error(f"Error in async semantic chunking: {e}")
            # Fallback to character-based chunking
            if "embed" in str(e).lower() or "model" in str(e).lower():
                logger.warning("Falling back to character-based chunking due to embedding error")
                from .character_chunker import CharacterChunker

                fallback_chunker = CharacterChunker(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=100,
                )
                return await fallback_chunker.chunk_text_async(text, doc_id, metadata)
            raise

    def _chunk_with_splitter(
        self,
        splitter: SemanticSplitterNodeParser,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None,
    ) -> list[ChunkResult]:
        """Helper method to perform chunking with a given splitter."""
        doc = Document(text=text, metadata=metadata or {})

        # Perform chunking with retry
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                nodes = splitter.get_nodes_from_documents([doc])
                elapsed_time = time.time() - start_time

                chunks_per_sec = len(nodes) / elapsed_time if elapsed_time > 0 else 0
                logger.debug(
                    f"Async semantic chunking: {len(nodes)} chunks in {elapsed_time:.2f}s "
                    f"({chunks_per_sec:.1f} chunks/sec)"
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Chunking attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        # Convert to ChunkResult
        results = []
        cumulative_length = 0

        for idx, node in enumerate(nodes):
            content = node.get_content()
            start_offset = cumulative_length
            end_offset = start_offset + len(content)

            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["semantic_boundary"] = True
            chunk_metadata["breakpoint_threshold"] = self.breakpoint_percentile_threshold

            result = self._create_chunk_result(
                doc_id=doc_id,
                chunk_index=idx,
                text=content,
                start_offset=start_offset,
                end_offset=end_offset,
                metadata=chunk_metadata,
            )
            results.append(result)
            cumulative_length = end_offset

        return results

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate semantic chunker configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            threshold = config.get("breakpoint_percentile_threshold", self.breakpoint_percentile_threshold)
            buffer_size = config.get("buffer_size", self.buffer_size)
            max_chunk_size = config.get("max_chunk_size", self.max_chunk_size)

            # Validate threshold
            if not isinstance(threshold, int | float) or not 0 <= threshold <= 100:
                logger.error(f"Invalid breakpoint_percentile_threshold: {threshold}")
                return False

            # Validate buffer size
            if not isinstance(buffer_size, int) or buffer_size <= 0:
                logger.error(f"Invalid buffer_size: {buffer_size}")
                return False

            # Validate max chunk size
            if not isinstance(max_chunk_size, int) or max_chunk_size <= 0:
                logger.error(f"Invalid max_chunk_size: {max_chunk_size}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating config: {e}")
            return False

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning.

        Args:
            text_length: Length of text in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """
        max_chunk_size = config.get("max_chunk_size", self.max_chunk_size)

        # Semantic chunking typically creates fewer, more coherent chunks
        # Estimate ~500-800 chars per semantic chunk on average
        avg_chars_per_chunk = 650

        # Rough estimate based on typical semantic boundaries
        estimated_chunks = max(1, text_length // avg_chars_per_chunk)

        # Cap by max chunk size constraint
        min_chunks_by_size = max(1, text_length // (max_chunk_size * 4))  # ~4 chars/token

        return max(min_chunks_by_size, estimated_chunks)
