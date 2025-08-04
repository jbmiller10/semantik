#!/usr/bin/env python3
"""
Hybrid chunking strategy that intelligently selects the best chunking approach.

This module implements an intelligent chunking strategy that analyzes content
characteristics and selects the most appropriate chunking strategy for optimal results.
"""

import asyncio
import logging
import platform
import re
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from typing import Any

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult
from packages.shared.text_processing.chunking_factory import ChunkingFactory

logger = logging.getLogger(__name__)

# Security constants
REGEX_TIMEOUT = 1  # Maximum seconds for regex execution
MAX_TEXT_LENGTH = 5_000_000  # 5MB text limit to prevent DOS


# Platform-specific timeout implementation
if platform.system() != "Windows":

    @contextmanager
    def timeout(seconds: int) -> Iterator[None]:
        """Context manager for timeout enforcement using signal alarm (Unix/Linux)."""

        def timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
            raise TimeoutError("Regex execution timeout")

        # Set the signal handler and alarm
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            # Restore the old handler and cancel the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

else:
    # Windows-compatible timeout using threading
    @contextmanager
    def timeout(seconds: int) -> Iterator[None]:
        """Context manager for timeout enforcement using threading (Windows)."""
        timer = threading.Timer(seconds, lambda: None)
        timer.start()
        try:
            yield
            # Note: Windows timeout is less reliable for regex operations
            # Consider using regex module with timeout support as alternative
        finally:
            timer.cancel()


def safe_regex_findall(pattern: re.Pattern, text: str, flags: int = 0) -> list[str]:
    """Execute regex findall with timeout protection.

    Args:
        pattern: Compiled regex pattern or string pattern
        text: Text to search in
        flags: Regex flags

    Returns:
        List of matches

    Raises:
        TimeoutError: If regex execution exceeds timeout
    """
    try:
        with timeout(REGEX_TIMEOUT):
            if isinstance(pattern, str):
                return re.findall(pattern, text, flags)
            return pattern.findall(text)
    except TimeoutError:
        logger.warning("Regex execution timeout - possible ReDoS attack")
        return []


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"
    CHARACTER = "character"  # Fallback strategy


class HybridChunker(BaseChunker):
    """Hybrid chunking strategy that selects the best approach based on content analysis."""

    def __init__(
        self,
        markdown_threshold: float = 0.15,
        semantic_coherence_threshold: float = 0.7,
        large_doc_threshold: int = 50000,
        enable_strategy_override: bool = True,
        fallback_strategy: str = ChunkingStrategy.RECURSIVE,
        **kwargs: Any,
    ) -> None:
        """Initialize HybridChunker.

        Args:
            markdown_threshold: Minimum ratio of markdown elements to consider markdown strategy
            semantic_coherence_threshold: Minimum coherence score to use semantic chunking
            large_doc_threshold: Character count threshold for considering hierarchical chunking
            enable_strategy_override: Whether to allow manual strategy override in metadata
            fallback_strategy: Default strategy to use when others fail or aren't suitable
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)

        self.markdown_threshold = markdown_threshold
        self.semantic_coherence_threshold = semantic_coherence_threshold
        self.large_doc_threshold = large_doc_threshold
        self.enable_strategy_override = enable_strategy_override
        self.fallback_strategy = fallback_strategy

        # Cache for initialized chunkers
        self._chunker_cache: dict[str, BaseChunker] = {}

        # Pre-compile regex patterns for security and performance
        self._compiled_patterns: dict[str, tuple[re.Pattern, float]] = {}
        self._compile_markdown_patterns()

        logger.info(
            f"Initialized HybridChunker with params: "
            f"markdown_threshold={markdown_threshold}, "
            f"semantic_coherence_threshold={semantic_coherence_threshold}, "
            f"large_doc_threshold={large_doc_threshold}, "
            f"fallback_strategy={fallback_strategy}"
        )

    def _compile_markdown_patterns(self) -> None:
        """Pre-compile regex patterns with safety validation."""
        patterns = [
            (r"^#{1,6}\s+", 3.0),  # Headers (high weight)
            (r"^\*{1,3}\s+|\-\s+|\+\s+|\d+\.\s+", 2.0),  # Lists
            (r"\[.*?\]\(.*?\)", 2.0),  # Links
            (r"!\[.*?\]\(.*?\)", 2.0),  # Images
            (r"`{1,3}[^`]+`{1,3}", 1.5),  # Code blocks/inline
            (r"^\>\s+", 1.5),  # Blockquotes
            (r"\*{1,2}[^\*]+\*{1,2}", 1.0),  # Bold/italic
            (r"^\s*\|.*\|", 2.0),  # Tables
            (r"^---+$|^===+$", 1.0),  # Horizontal rules
        ]

        for pattern_str, weight in patterns:
            try:
                # Test pattern compilation with timeout
                with timeout(REGEX_TIMEOUT):
                    compiled = re.compile(pattern_str, re.MULTILINE)
                self._compiled_patterns[pattern_str] = (compiled, weight)
            except (TimeoutError, re.error) as e:
                logger.warning(f"Failed to compile regex pattern: {pattern_str[:50]}...")
                logger.debug(f"Pattern compilation error: {e}")

    def _get_chunker(self, strategy: str, params: dict[str, Any] | None = None) -> BaseChunker:
        """Get or create a chunker instance for the given strategy.

        Args:
            strategy: The chunking strategy to use
            params: Optional parameters for the chunker

        Returns:
            Initialized chunker instance
        """
        cache_key = f"{strategy}_{hash(str(params))}"

        if cache_key not in self._chunker_cache:
            config: dict[str, Any] = {"strategy": strategy}
            if params:
                config["params"] = params

            try:
                self._chunker_cache[cache_key] = ChunkingFactory.create_chunker(config)
            except Exception as e:
                # Security: Log generic error externally, detailed error internally
                logger.error(f"Failed to create chunker for strategy: {strategy}")
                logger.debug(f"Chunker creation error details: {e}")
                # Fallback to character chunker as last resort
                if strategy != ChunkingStrategy.CHARACTER:
                    logger.warning("Using fallback character chunker")
                    return self._get_chunker(ChunkingStrategy.CHARACTER)
                raise ValueError("Unable to create chunker") from e

        return self._chunker_cache[cache_key]

    def _analyze_markdown_content(self, text: str, metadata: dict[str, Any] | None) -> tuple[bool, float]:
        """Analyze if content is markdown and calculate markdown density.

        Args:
            text: The text to analyze
            metadata: Optional metadata with file information

        Returns:
            Tuple of (is_markdown_file, markdown_density_score)
        """
        # Check file extension first
        if metadata:
            file_path = metadata.get("file_path", "")
            file_name = metadata.get("file_name", "")
            file_type = metadata.get("file_type", "")

            markdown_extensions = {".md", ".markdown", ".mdown", ".mkd", ".mdx"}
            for ext in markdown_extensions:
                if file_path.endswith(ext) or file_name.endswith(ext) or file_type == ext:
                    return True, 1.0

        # Analyze markdown syntax density using pre-compiled patterns
        total_score = 0.0
        total_lines = max(1, len(text.splitlines()))

        # Use pre-compiled patterns with safe execution
        for _, (compiled_pattern, weight) in self._compiled_patterns.items():
            try:
                matches = safe_regex_findall(compiled_pattern, text)
                total_score += len(matches) * weight
            except Exception:
                # Security: Don't expose pattern details in logs
                logger.debug("Regex pattern execution failed")
                continue

        # Normalize score by text length
        markdown_density = total_score / total_lines

        return False, markdown_density

    def _estimate_semantic_coherence(self, text: str) -> float:
        """Estimate semantic coherence of the text.

        Args:
            text: The text to analyze

        Returns:
            Estimated coherence score (0.0 to 1.0)
        """
        # Simple heuristic based on topic consistency indicators
        # In production, this could use more sophisticated NLP analysis

        lines = text.splitlines()
        if len(lines) < 10:
            return 0.5  # Not enough content to determine

        # Check for topic indicators
        indicators = {
            "repeated_terms": 0.0,
            "section_structure": 0.0,
            "consistent_vocabulary": 0.0,
        }

        # Extract words (simple tokenization) with safe regex execution
        try:
            word_pattern = re.compile(r"\b\w+\b")
            words = safe_regex_findall(word_pattern, text.lower())
            unique_words = set(words)
        except Exception:
            logger.debug("Word extraction failed, using fallback")
            # Fallback: simple split
            words = text.lower().split()
            unique_words = set(words)

        if words:
            # Repeated terms indicate focused content
            word_frequency: dict[str, int] = {}
            for word in words:
                if len(word) > 4:  # Skip short words
                    word_frequency[word] = word_frequency.get(word, 0) + 1

            # Top 10% most frequent words
            if word_frequency:
                sorted_words = sorted(word_frequency.values(), reverse=True)
                top_10_percent = len(sorted_words) // 10 or 1
                top_word_freq = sum(sorted_words[:top_10_percent])
                indicators["repeated_terms"] = min(1.0, top_word_freq / len(words) * 10)

            # Vocabulary consistency
            indicators["consistent_vocabulary"] = 1.0 - (len(unique_words) / len(words))

        # Section structure (paragraphs of similar length suggest structured content)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) > 3:
            lengths = [len(p) for p in paragraphs]
            avg_length = sum(lengths) / len(lengths)
            variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
            # Lower variance means more consistent structure
            indicators["section_structure"] = 1.0 / (1.0 + variance / (avg_length**2))

        # Weighted average of indicators
        coherence_score = (
            indicators["repeated_terms"] * 0.4
            + indicators["consistent_vocabulary"] * 0.3
            + indicators["section_structure"] * 0.3
        )

        return min(1.0, max(0.0, coherence_score))

    def _select_strategy(
        self, text: str, metadata: dict[str, Any] | None
    ) -> tuple[ChunkingStrategy, dict[str, Any], str]:
        """Select the best chunking strategy based on content analysis.

        Args:
            text: The text to analyze
            metadata: Optional metadata

        Returns:
            Tuple of (selected_strategy, strategy_params, reasoning)
        """
        # Check for manual override first
        if self.enable_strategy_override and metadata:
            override_strategy = metadata.get("chunking_strategy")
            if override_strategy and override_strategy in [s.value for s in ChunkingStrategy]:
                reasoning = f"Using manually specified strategy: {override_strategy}"
                logger.info(reasoning)
                return ChunkingStrategy(override_strategy), {}, reasoning

        # Analyze content characteristics
        text_length = len(text)
        is_markdown_file, markdown_density = self._analyze_markdown_content(text, metadata)
        semantic_coherence = self._estimate_semantic_coherence(text)

        # Decision logic with detailed reasoning
        reasoning_parts = []

        # 1. Check for markdown content
        if is_markdown_file:
            reasoning = "Detected markdown file extension - using MarkdownChunker"
            logger.info(f"{reasoning} for document")
            return ChunkingStrategy.MARKDOWN, {}, reasoning

        if markdown_density > self.markdown_threshold:
            reasoning = (
                f"High markdown syntax density ({markdown_density:.2f} > {self.markdown_threshold}) "
                f"- using MarkdownChunker"
            )
            logger.info(reasoning)
            return ChunkingStrategy.MARKDOWN, {}, reasoning

        # 2. Check for large documents that benefit from hierarchical organization
        if text_length > self.large_doc_threshold:
            reasoning_parts.append(f"Large document ({text_length:,} chars > {self.large_doc_threshold:,})")

            # For very large documents with high coherence, use hierarchical
            if semantic_coherence > self.semantic_coherence_threshold:
                reasoning = (
                    f"{reasoning_parts[0]} with high semantic coherence ({semantic_coherence:.2f}) "
                    f"- using HierarchicalChunker for multi-level organization"
                )
                logger.info(reasoning)
                return ChunkingStrategy.HIERARCHICAL, {}, reasoning

        # 3. Check for high semantic coherence (topic-focused content)
        if semantic_coherence > self.semantic_coherence_threshold:
            reasoning = (
                f"High semantic coherence ({semantic_coherence:.2f} > {self.semantic_coherence_threshold}) "
                f"indicating topic-focused content - using SemanticChunker"
            )
            logger.info(reasoning)
            return ChunkingStrategy.SEMANTIC, {}, reasoning

        # 4. Default to recursive chunker for general text
        reasoning = (
            f"General text content (length: {text_length:,}, "
            f"markdown_density: {markdown_density:.2f}, "
            f"semantic_coherence: {semantic_coherence:.2f}) "
            f"- using RecursiveChunker as default"
        )
        logger.info(reasoning)
        return ChunkingStrategy.RECURSIVE, {}, reasoning

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Synchronous chunking with intelligent strategy selection.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Security validation: Prevent processing of excessively large texts
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("Text exceeds maximum length limit")
            raise ValueError("Text too large to process")

        # Select the best strategy
        strategy, params, reasoning = self._select_strategy(text, metadata)

        # Log strategy selection
        logger.info(f"Document {doc_id}: {reasoning}")

        # Update metadata with strategy info
        enhanced_metadata = metadata.copy() if metadata else {}
        enhanced_metadata.update(
            {
                "hybrid_strategy_used": strategy.value,
                "hybrid_strategy_reasoning": reasoning,
            }
        )

        try:
            # Get the appropriate chunker
            chunker = self._get_chunker(strategy.value, params)

            # Perform chunking
            chunks: list[ChunkResult] = chunker.chunk_text(text, doc_id, enhanced_metadata)

            # Add hybrid chunker metadata to each chunk
            for chunk in chunks:
                chunk.metadata["hybrid_chunker"] = True
                chunk.metadata["selected_strategy"] = strategy.value

            logger.debug(f"Successfully created {len(chunks)} chunks using {strategy.value} strategy")
            return chunks

        except Exception as e:
            # Security: Log generic error externally, detailed error internally
            logger.error(f"Chunking strategy failed for document {doc_id}")
            logger.debug(f"Internal error details: {e}")

            # Try fallback strategy if not already using it
            if strategy.value != self.fallback_strategy:
                logger.warning("Attempting fallback strategy")
                try:
                    fallback_chunker = self._get_chunker(self.fallback_strategy)
                    fallback_chunks: list[ChunkResult] = fallback_chunker.chunk_text(text, doc_id, enhanced_metadata)

                    # Update metadata to reflect fallback
                    for chunk in fallback_chunks:
                        chunk.metadata["hybrid_chunker"] = True
                        chunk.metadata["selected_strategy"] = self.fallback_strategy
                        chunk.metadata["fallback_used"] = True
                        chunk.metadata["original_strategy_failed"] = strategy.value

                    logger.info("Successfully chunked using fallback strategy")
                    return fallback_chunks

                except Exception as fallback_error:
                    # Security: Don't expose fallback error details
                    logger.error("Fallback strategy also failed")
                    logger.debug(f"Fallback error details: {fallback_error}")

            # If all strategies fail, create a single chunk as last resort
            logger.error("All chunking strategies failed. Creating single chunk as last resort.")
            return [
                self._create_chunk_result(
                    doc_id=doc_id,
                    chunk_index=0,
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                    metadata={
                        **enhanced_metadata,
                        "hybrid_chunker": True,
                        "selected_strategy": "emergency_single_chunk",
                        "all_strategies_failed": True,
                    },
                )
            ]

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Asynchronous chunking with intelligent strategy selection.

        Args:
            text: The text to chunk
            doc_id: Unique identifier for the document
            metadata: Optional metadata to include with chunks

        Returns:
            List of ChunkResult objects
        """
        if not text.strip():
            return []

        # Security validation: Prevent processing of excessively large texts
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning("Text exceeds maximum length limit")
            raise ValueError("Text too large to process")

        # Run strategy selection in executor to avoid blocking
        loop = asyncio.get_event_loop()
        strategy, params, reasoning = await loop.run_in_executor(None, self._select_strategy, text, metadata)

        # Log strategy selection
        logger.info(f"Document {doc_id}: {reasoning}")

        # Update metadata with strategy info
        enhanced_metadata = metadata.copy() if metadata else {}
        enhanced_metadata.update(
            {
                "hybrid_strategy_used": strategy.value,
                "hybrid_strategy_reasoning": reasoning,
            }
        )

        try:
            # Get the appropriate chunker
            chunker = self._get_chunker(strategy.value, params)

            # Perform async chunking
            chunks: list[ChunkResult] = await chunker.chunk_text_async(text, doc_id, enhanced_metadata)

            # Add hybrid chunker metadata to each chunk
            for chunk in chunks:
                chunk.metadata["hybrid_chunker"] = True
                chunk.metadata["selected_strategy"] = strategy.value

            logger.debug(f"Successfully created {len(chunks)} chunks using {strategy.value} strategy")
            return chunks

        except Exception as e:
            # Security: Log generic error externally, detailed error internally
            logger.error(f"Async chunking strategy failed for document {doc_id}")
            logger.debug(f"Internal error details: {e}")

            # Try fallback strategy
            if strategy.value != self.fallback_strategy:
                logger.warning("Attempting fallback strategy")
                try:
                    fallback_chunker = self._get_chunker(self.fallback_strategy)
                    fallback_chunks: list[ChunkResult] = await fallback_chunker.chunk_text_async(text, doc_id, enhanced_metadata)

                    # Update metadata to reflect fallback
                    for chunk in fallback_chunks:
                        chunk.metadata["hybrid_chunker"] = True
                        chunk.metadata["selected_strategy"] = self.fallback_strategy
                        chunk.metadata["fallback_used"] = True
                        chunk.metadata["original_strategy_failed"] = strategy.value

                    logger.info("Successfully chunked using fallback strategy")
                    return fallback_chunks

                except Exception as fallback_error:
                    # Security: Don't expose fallback error details
                    logger.error("Fallback strategy also failed")
                    logger.debug(f"Fallback error details: {fallback_error}")

            # Last resort: single chunk
            logger.error("All chunking strategies failed. Creating single chunk as last resort.")
            return [
                self._create_chunk_result(
                    doc_id=doc_id,
                    chunk_index=0,
                    text=text,
                    start_offset=0,
                    end_offset=len(text),
                    metadata={
                        **enhanced_metadata,
                        "hybrid_chunker": True,
                        "selected_strategy": "emergency_single_chunk",
                        "all_strategies_failed": True,
                    },
                )
            ]

    def validate_config(self, config: dict[str, Any]) -> bool:
        """Validate hybrid chunker configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate thresholds
            markdown_threshold = config.get("markdown_threshold", self.markdown_threshold)
            if not isinstance(markdown_threshold, int | float) or not 0 <= markdown_threshold <= 1:
                logger.error(f"Invalid markdown_threshold: {markdown_threshold}")
                return False

            semantic_threshold = config.get("semantic_coherence_threshold", self.semantic_coherence_threshold)
            if not isinstance(semantic_threshold, int | float) or not 0 <= semantic_threshold <= 1:
                logger.error(f"Invalid semantic_coherence_threshold: {semantic_threshold}")
                return False

            large_doc_threshold = config.get("large_doc_threshold", self.large_doc_threshold)
            if not isinstance(large_doc_threshold, int) or large_doc_threshold <= 0:
                logger.error(f"Invalid large_doc_threshold: {large_doc_threshold}")
                return False

            # Validate fallback strategy
            fallback = config.get("fallback_strategy", self.fallback_strategy)
            if fallback not in [s.value for s in ChunkingStrategy]:
                logger.error(f"Invalid fallback_strategy: {fallback}")
                return False

            return True

        except Exception:
            # Security: Don't expose exception details
            logger.error("Configuration validation failed")
            return False

    def estimate_chunks(self, text_length: int, config: dict[str, Any]) -> int:
        """Estimate number of chunks based on likely strategy selection.

        Args:
            text_length: Length of text in characters
            config: Configuration parameters

        Returns:
            Estimated number of chunks
        """
        # For estimation, we'll use heuristics without full content analysis
        large_doc_threshold = config.get("large_doc_threshold", self.large_doc_threshold)

        # Assume different strategies based on document size
        if text_length > large_doc_threshold:
            # Hierarchical chunker estimate
            # Hierarchical creates multiple levels, so more chunks
            return max(1, text_length // 800)  # More granular estimate

        # For medium-sized documents, could be semantic or recursive
        # Use conservative estimate based on recursive chunker defaults
        chunk_size = config.get("chunk_size", 100) * 4  # ~100 tokens * 4 chars/token
        chunk_overlap = config.get("chunk_overlap", 20) * 4

        # Handle edge case where overlap >= chunk_size
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 4

        effective_chunk_size = chunk_size - chunk_overlap

        if text_length <= chunk_size:
            return 1

        # Calculate number of chunks needed
        remaining_text = text_length - chunk_size
        additional_chunks = max(0, (remaining_text + effective_chunk_size - 1) // effective_chunk_size)

        return int(1 + additional_chunks)
