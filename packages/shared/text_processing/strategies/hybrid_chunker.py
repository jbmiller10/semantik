#!/usr/bin/env python3
"""
Hybrid chunking strategy that intelligently selects the best chunking approach.

This module implements an intelligent chunking strategy that analyzes content
characteristics and selects the most appropriate chunking strategy for optimal results.
"""

import asyncio
import logging
import re
import signal
from contextlib import contextmanager
from enum import Enum
from typing import Any

from packages.shared.chunking.utils.input_validator import ChunkingInputValidator
from packages.shared.chunking.utils.regex_monitor import RegexPerformanceMonitor
from packages.shared.chunking.utils.safe_regex import RegexTimeout, SafeRegex
from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult
from packages.shared.text_processing.chunking_factory import ChunkingFactory

logger = logging.getLogger(__name__)

# Security constants
MAX_TEXT_LENGTH = 5_000_000  # 5MB text limit to prevent DOS
REGEX_TIMEOUT = 1  # Default timeout for regex operations


def safe_regex_findall(pattern, text, flags=None):
    """Helper function for executing regex with timeout protection.

    Args:
        pattern: Regex pattern (string or compiled)
        text: Text to search
        flags: Optional regex flags

    Returns:
        List of matches or empty list on timeout
    """
    safe_regex = SafeRegex(timeout=REGEX_TIMEOUT)
    if isinstance(pattern, str):
        pattern = re.compile(pattern, flags) if flags else safe_regex.compile_safe(pattern)
    try:
        return safe_regex.findall_safe(pattern.pattern if hasattr(pattern, "pattern") else str(pattern), text)
    except RegexTimeout:
        logger.warning(f"Regex timeout for pattern: {pattern}")
        return []
    except Exception as e:
        logger.warning(f"Regex error: {e}")
        return []


@contextmanager
def timeout(seconds):
    """Context manager for timeout operations.

    Args:
        seconds: Timeout duration in seconds

    Yields:
        None

    Raises:
        TimeoutError: If operation exceeds timeout
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Restore the original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


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

        # Initialize SafeRegex and monitoring
        self.safe_regex = SafeRegex(timeout=1.0)
        self.regex_monitor = RegexPerformanceMonitor()
        self.input_validator = ChunkingInputValidator()

        # Pre-compile safe regex patterns for security and performance
        self._compiled_patterns: dict[str, tuple[Any, float]] = {}
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
        # Use safer, bounded patterns
        patterns = [
            (r"^#{1,6}\s+\S.*$", 3.0),  # Headers (bounded)
            (r"^[\*\-\+]\s+\S.*$", 2.0),  # Lists (simplified)
            (r"^\d+\.\s+\S.*$", 2.0),  # Numbered lists
            (r"\[([^\]]+)\]\(([^)]+)\)", 2.0),  # Links (bounded)
            (r"!\[([^\]]*)\]\(([^)]+)\)", 2.0),  # Images (bounded)
            (r"`([^`]+)`", 1.5),  # Inline code (bounded)
            (r"^>\s*\S.*$", 1.5),  # Blockquotes (bounded)
            (r"\*\*([^*]+)\*\*", 1.0),  # Bold (bounded)
            (r"\*([^*]+)\*", 1.0),  # Italic (bounded)
            (r"^\s*\|[^|]+\|", 2.0),  # Tables (simplified)
            (r"^(?:---|\\*\\*\\*|___)$", 1.0),  # Horizontal rules (fixed)
        ]

        for pattern_str, weight in patterns:
            try:
                # Compile with SafeRegex for ReDoS protection, using MULTILINE for line anchors
                compiled = self.safe_regex.compile_safe(pattern_str, use_re2=True, flags=re.MULTILINE)
                self._compiled_patterns[pattern_str] = (compiled, weight)
            except (ValueError, Exception) as e:
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

        # Analyze markdown density as the fraction of lines that look like markdown
        lines = text.splitlines()
        total_lines = max(1, len(lines))
        matched_lines = 0

        for line in lines:
            l = line.strip()
            if not l:
                continue
            # Quick heuristics for common markdown constructs
            if (
                l.startswith(("#", ">", "* ", "- ", "+ "))
                or re.match(r"^\d+\.\s+", l) is not None
                or ("|" in l and l.count("|") >= 2)
                or ("[" in l and "]" in l and "(" in l and ")" in l)
                or ("`" in l)
            ):
                matched_lines += 1

        markdown_density = matched_lines / total_lines
        return False, float(markdown_density)

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
            import time

            start_time = time.time()
            # Use a simpler, bounded pattern for word extraction
            words = self.safe_regex.findall_safe(r"\w+", text.lower(), max_matches=10000)
            execution_time = time.time() - start_time

            self.regex_monitor.record_execution(
                pattern=r"\w+", execution_time=execution_time, input_size=len(text), matched=len(words) > 0
            )

            unique_words = set(words)
        except (RegexTimeout, Exception) as e:
            logger.debug(f"Word extraction failed: {e}, using fallback")
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

        # 1. Check for markdown content by explicit file indication
        if is_markdown_file:
            reasoning = "Detected markdown file extension - using MarkdownChunker"
            logger.info(f"{reasoning} for document")
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

        # 3. Prefer semantic for high coherence before markdown density
        if semantic_coherence > self.semantic_coherence_threshold:
            reasoning = (
                f"High semantic coherence ({semantic_coherence:.2f} > {self.semantic_coherence_threshold}) "
                f"indicating topic-focused content - using SemanticChunker"
            )
            logger.info(reasoning)
            return ChunkingStrategy.SEMANTIC, {}, reasoning

        # 4. Consider markdown density if not a markdown file
        if markdown_density > self.markdown_threshold:
            reasoning = (
                f"High markdown syntax density ({markdown_density:.2f} > {self.markdown_threshold}) "
                f"- using MarkdownChunker"
            )
            logger.info(reasoning)
            return ChunkingStrategy.MARKDOWN, {}, reasoning

        # 5. Default to recursive chunker for general text
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
        try:
            self.input_validator.validate_document(text)
        except ValueError as e:
            logger.warning(f"Input validation failed: {e}")
            # For very large documents, try to process with character chunker as fallback
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError("Text too large to process") from e

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
        try:
            self.input_validator.validate_document(text)
        except ValueError as e:
            logger.warning(f"Input validation failed: {e}")
            # For very large documents, try to process with character chunker as fallback
            if len(text) > MAX_TEXT_LENGTH:
                raise ValueError("Text too large to process") from e

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
                    fallback_chunks: list[ChunkResult] = await fallback_chunker.chunk_text_async(
                        text, doc_id, enhanced_metadata
                    )

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
