#!/usr/bin/env python3
"""
Unified recursive text chunking strategy.

This module merges the domain-based and LlamaIndex-based recursive chunking
implementations into a single unified strategy.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.unified.base import UnifiedChunkingStrategy

logger = logging.getLogger(__name__)


class RecursiveChunkingStrategy(UnifiedChunkingStrategy):
    """
    Unified recursive text chunking strategy.

    This strategy recursively splits text using a hierarchy of separators,
    preferring natural document boundaries like paragraphs, sentences, and words.
    Can optionally use LlamaIndex for enhanced text splitting.
    """

    def __init__(self, use_llama_index: bool = False) -> None:
        """
        Initialize the recursive chunking strategy.

        Args:
            use_llama_index: Whether to use LlamaIndex implementation
        """
        super().__init__("recursive")
        self._use_llama_index = use_llama_index
        self._llama_splitter = None

        # Define separator hierarchy (from most to least preferred)
        self.separators = [
            "\n\n\n",  # Multiple blank lines (section breaks)
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentences
            "! ",  # Exclamation sentences
            "? ",  # Question sentences
            "; ",  # Semicolon clauses
            ", ",  # Comma clauses
            " ",  # Words
            "",  # Characters (last resort)
        ]

        if use_llama_index:
            try:
                # Check if LlamaIndex is available
                import importlib.util

                spec = importlib.util.find_spec("llama_index.core.node_parser")
                self._llama_available = spec is not None
            except ImportError:
                logger.warning("LlamaIndex not available, falling back to domain implementation")
                self._llama_available = False
                self._use_llama_index = False
        else:
            self._llama_available = False

    def _init_llama_splitter(self, config: ChunkConfig) -> Any:
        """Initialize LlamaIndex splitter if needed."""
        if not self._use_llama_index or not self._llama_available:
            return None

        try:
            from llama_index.core.node_parser import SentenceSplitter

            # Use overlap_tokens directly from config
            overlap_tokens = config.overlap_tokens

            return SentenceSplitter(
                chunk_size=config.max_tokens,
                chunk_overlap=overlap_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex splitter: {e}")
            return None

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Recursively split content into chunks.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Try LlamaIndex implementation if enabled
        if self._use_llama_index and self._llama_available:
            chunks = self._chunk_with_llama_index(content, config, progress_callback)
            if chunks is not None:
                return chunks

        # Fall back to domain implementation
        return self._chunk_with_domain(content, config, progress_callback)

    async def chunk_async(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Asynchronous chunking.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Run synchronous method in executor to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.chunk,
            content,
            config,
            progress_callback,
        )

    def _chunk_with_llama_index(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk] | None:
        """
        Chunk using LlamaIndex SentenceSplitter.

        Returns None if LlamaIndex is not available or fails.
        """
        try:
            from llama_index.core import Document

            # Initialize splitter
            splitter = self._init_llama_splitter(config)
            if not splitter:
                return None

            # Create a temporary document
            doc = Document(text=content)

            # Get nodes using sentence splitter
            nodes = splitter.get_nodes_from_documents([doc])

            if not nodes:
                return []

            chunks: list[Chunk] = []
            total_chars = len(content)

            # Convert LlamaIndex nodes to domain chunks
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()

                # Calculate offsets
                if idx == 0:
                    start_offset = 0
                else:
                    # Find the chunk text in the original content
                    prev_end = chunks[-1].metadata.end_offset
                    start_offset = content.find(chunk_text, prev_end - 100)  # Look near previous end
                    if start_offset == -1:
                        start_offset = prev_end

                end_offset = min(start_offset + len(chunk_text), total_chars)

                # Create chunk metadata
                token_count = self.count_tokens(chunk_text)

                metadata = ChunkMetadata(
                    chunk_id=f"{config.strategy_name}_{idx:04d}",
                    document_id="doc",
                    chunk_index=idx,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=token_count,
                    strategy_name=self.name,
                    semantic_density=0.6,  # Higher than character-based
                    confidence_score=0.95,  # Higher confidence with LlamaIndex
                    created_at=datetime.now(tz=UTC),
                )

                # Create chunk entity
                effective_min_tokens = max(1, min(config.min_tokens, token_count))

                chunk = Chunk(
                    content=chunk_text,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=config.max_tokens,
                )

                chunks.append(chunk)

                # Report progress
                if progress_callback:
                    progress = ((idx + 1) / len(nodes)) * 100
                    progress_callback(min(progress, 100.0))

            return chunks

        except Exception as e:
            logger.warning(f"LlamaIndex chunking failed, falling back to domain: {e}")
            return None

    def _chunk_with_domain(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk using domain implementation (recursive splitting).
        """
        # Convert token limits to character estimates
        chars_per_token = 4
        max_chars = config.max_tokens * chars_per_token
        min_chars = config.min_tokens * chars_per_token
        # Ensure small documents aren't discarded entirely when the min size exceeds
        # their length. Clamping keeps recursive splitting from returning [].
        if content:
            min_chars = max(1, min(min_chars, len(content)))
        overlap_chars = config.overlap_tokens * chars_per_token

        # Start recursive splitting.
        #
        # Splits are later combined with overlap. Ensure the split size budget
        # leaves room for the overlap so the final chunk doesn't violate the
        # max_tokens constraint enforced by the domain Chunk entity.
        split_budget = max_chars
        if overlap_chars > 0:
            split_budget = max(1, max_chars - overlap_chars)
            min_chars = min(min_chars, split_budget)

        splits = self._recursive_split(content, self.separators, split_budget, min_chars)

        if not splits:
            splits = [content]

        # Convert splits to chunks with overlap
        chunks: list[Chunk] = []
        for idx, split in enumerate(splits):
            # Calculate start offset
            if idx == 0:
                start_offset = 0
            else:
                # Find this split in the original content
                prev_end = chunks[-1].metadata.end_offset
                # Apply overlap by going back
                start_offset = max(0, prev_end - overlap_chars)
                # Find word boundary for clean overlap
                start_offset = self.find_word_boundary(content, start_offset, prefer_before=False)

            # Calculate end offset
            end_offset = content.find(split, start_offset) + len(split)
            if end_offset < start_offset:
                # If we can't find the split, estimate the position
                end_offset = min(start_offset + len(split), len(content))

            # Ensure end_offset is always greater than start_offset
            if end_offset <= start_offset:
                # Make sure we have at least 1 character
                end_offset = min(start_offset + max(1, len(split) if split else 1), len(content))
                if end_offset <= start_offset and start_offset < len(content):
                    end_offset = start_offset + 1

            # Add overlap content from previous chunk if needed
            if idx > 0 and overlap_chars > 0 and start_offset < chunks[-1].metadata.end_offset:
                overlap_text = content[start_offset : chunks[-1].metadata.end_offset]
                chunk_text = overlap_text + split
            else:
                chunk_text = split

            chunk_text = self.clean_chunk_text(chunk_text)
            if not chunk_text:
                continue

            # Final validation: skip if offsets are invalid
            if end_offset <= start_offset:
                continue

            # Create chunk metadata
            token_count = self.count_tokens(chunk_text)

            metadata = ChunkMetadata(
                chunk_id=f"{config.strategy_name}_{idx:04d}",
                document_id="doc",
                chunk_index=idx,
                start_offset=start_offset,
                end_offset=end_offset,
                token_count=token_count,
                strategy_name=self.name,
                semantic_density=0.6,  # Higher for recursive strategy
                confidence_score=0.9,
                created_at=datetime.now(tz=UTC),
            )

            # Create chunk entity
            effective_min_tokens = max(1, min(config.min_tokens, token_count))

            chunk = Chunk(
                content=chunk_text,
                metadata=metadata,
                min_tokens=effective_min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)

            # Report progress
            if progress_callback:
                progress = ((idx + 1) / len(splits)) * 100
                progress_callback(min(progress, 100.0))

        return chunks

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        max_size: int,
        min_size: int,
    ) -> list[str]:
        """
        Recursively split text using a hierarchy of separators.

        Args:
            text: Text to split
            separators: List of separators in priority order
            max_size: Maximum chunk size in characters
            min_size: Minimum chunk size in characters

        Returns:
            List of text chunks
        """
        # Base case: text is within size limits
        if len(text) <= max_size:
            return [text] if len(text) >= min_size else []

        # Try each separator in order
        for separator in separators:
            if separator and separator in text:
                splits = text.split(separator)

                # Recombine splits to respect size limits
                result = []
                current_chunk = ""

                for i, split in enumerate(splits):
                    # Add separator back (except for last split)
                    if i > 0:
                        split = separator + split

                    # Check if adding this split would exceed max size
                    if current_chunk and len(current_chunk) + len(split) > max_size:
                        # Save current chunk and start new one
                        if len(current_chunk) >= min_size:
                            result.append(current_chunk)
                        current_chunk = split
                    else:
                        # Add to current chunk
                        current_chunk += split

                # Add final chunk
                if current_chunk and len(current_chunk) >= min_size:
                    result.append(current_chunk)

                # Recursively process chunks that are still too large
                final_result = []
                for chunk in result:
                    if len(chunk) > max_size:
                        # Recursively split with remaining separators
                        sub_splits = self._recursive_split(
                            chunk,
                            separators[separators.index(separator) + 1 :],
                            max_size,
                            min_size,
                        )
                        final_result.extend(sub_splits)
                    else:
                        final_result.append(chunk)

                return final_result

        # Last resort: split by max_size
        result = []
        for i in range(0, len(text), max_size):
            chunk = text[i : i + max_size]
            if len(chunk) >= min_size:
                result.append(chunk)

        return result

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for recursive chunking.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content:
            return False, "Content cannot be empty"

        if len(content) > 50_000_000:  # 50MB limit
            return False, f"Content too large: {len(content)} characters"

        return True, None

    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """
        Estimate the number of chunks.

        Args:
            content_length: Length of content in characters
            config: Chunking configuration

        Returns:
            Estimated chunk count
        """
        if content_length == 0:
            return 0

        # Convert character length to estimated tokens
        estimated_tokens = content_length // 4

        # Recursive chunking tends to create slightly fewer chunks than character-based
        base_estimate = config.estimate_chunks(estimated_tokens)
        return max(1, int(base_estimate * 0.9))
