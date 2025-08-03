#!/usr/bin/env python3
"""
Hybrid chunking strategy with intelligent strategy selection.

This module implements intelligent strategy selection based on content analysis,
choosing the optimal chunking approach for different document types and characteristics.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from packages.shared.text_processing.base_chunker import BaseChunker, ChunkResult

logger = logging.getLogger(__name__)


class HybridChunker(BaseChunker):
    """Intelligently selects optimal chunking strategy based on content analysis."""

    # Pre-compiled safe patterns to prevent ReDoS
    SAFE_MARKDOWN_PATTERNS = {
        'headers': re.compile(r"^#{1,6}\s", re.MULTILINE),
        'code_blocks': re.compile(r"```"),
        'lists': re.compile(r"^\s*[-*+]\s", re.MULTILINE),
        'numbered_lists': re.compile(r"^\s*\d+\.\s", re.MULTILINE),
        'links': re.compile(r"\[.{1,100}\]\(.{1,200}\)"),  # Bounded quantifiers
        'images': re.compile(r"!\[.{0,100}\]\(.{1,200}\)"),  # Bounded quantifiers
        'bold': re.compile(r"\*\*.{1,100}\*\*"),  # Bounded
        'bold_underscore': re.compile(r"__.{1,100}__"),  # Bounded
        'italic': re.compile(r"(?<!\*)\*.{1,100}\*(?!\*)"),  # Bounded with negative lookaround
        'italic_underscore': re.compile(r"(?<!_)_.{1,100}_(?!_)"),  # Bounded with negative lookaround
        'tables': re.compile(r"^\|.*\|$", re.MULTILINE),
        'blockquotes': re.compile(r"^>\s", re.MULTILINE),
    }

    def __init__(
        self,
        markdown_density_threshold: float = 0.1,
        topic_diversity_threshold: float = 0.7,
        semantic_min_length: int = 1000,
        enable_analytics: bool = True,
    ):
        """Initialize hybrid chunker with strategy selection parameters.

        Args:
            markdown_density_threshold: Minimum markdown element density to use markdown strategy
            topic_diversity_threshold: Minimum topic diversity to use semantic strategy
            semantic_min_length: Minimum text length to consider semantic chunking
            enable_analytics: Whether to log strategy selection analytics
        """
        self.markdown_density_threshold = markdown_density_threshold
        self.topic_diversity_threshold = topic_diversity_threshold
        self.semantic_min_length = semantic_min_length
        self.enable_analytics = enable_analytics

        # Initialize strategy instances (lazy loaded)
        self._strategies: Dict[str, BaseChunker] = {}
        self.strategy_name = "hybrid"

        # Analytics tracking
        self._selection_stats = {
            "markdown": 0,
            "semantic": 0,
            "recursive": 0,
            "fallback": 0,
        }

        logger.info(
            f"Initialized HybridChunker with markdown_threshold={markdown_density_threshold}, "
            f"topic_threshold={topic_diversity_threshold}, semantic_min_length={semantic_min_length}"
        )

    def _get_strategy(self, strategy_name: str) -> BaseChunker:
        """Get or create strategy instance.

        Args:
            strategy_name: Name of the strategy to get

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy is not supported
        """
        if strategy_name not in self._strategies:
            if strategy_name == "markdown":
                from packages.shared.text_processing.strategies.markdown_chunker import MarkdownChunker

                self._strategies["markdown"] = MarkdownChunker()

            elif strategy_name == "semantic":
                from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker

                # Use moderate settings for hybrid mode
                self._strategies["semantic"] = SemanticChunker(
                    breakpoint_percentile_threshold=90,
                    max_chunk_size=2000,
                )

            elif strategy_name == "recursive":
                from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker

                self._strategies["recursive"] = RecursiveChunker(
                    chunk_size=600,
                    chunk_overlap=100,
                )

            else:
                raise ValueError(f"Unsupported strategy: {strategy_name}")

        return self._strategies[strategy_name]

    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """Asynchronously chunk text using intelligently selected strategy.

        Args:
            text: Text to chunk
            doc_id: Document identifier for chunk IDs
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects from selected strategy
        """
        # Validate inputs
        self._validate_input(text, doc_id, metadata)
        
        if not text.strip():
            return []

        try:
            # Analyze content to select optimal strategy
            analysis_result = await self._analyze_content_async(text, metadata)
            selected_strategy = analysis_result["selected_strategy"]
            selection_reason = analysis_result["selection_reason"]
            content_characteristics = analysis_result["characteristics"]

            logger.info(
                f"Hybrid chunker selected '{selected_strategy}' for document {doc_id}: {selection_reason}"
            )

            # Update analytics
            if self.enable_analytics:
                self._selection_stats[selected_strategy] = self._selection_stats.get(selected_strategy, 0) + 1

            # Get selected strategy and execute chunking
            chunker = self._get_strategy(selected_strategy)
            chunks = await chunker.chunk_text_async(text, doc_id, metadata)

            # Enhance metadata with hybrid information
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "strategy": self.strategy_name,
                        "sub_strategy": selected_strategy,
                        "selection_reason": selection_reason,
                        "content_characteristics": content_characteristics,
                        "hybrid_analytics": self._get_analytics_summary() if self.enable_analytics else None,
                    }
                )

            logger.info(
                f"Hybrid chunking ({selected_strategy}) created {len(chunks)} chunks "
                f"for document {doc_id} ({len(text)} characters)"
            )

            return chunks

        except Exception as e:
            logger.error(f"Hybrid chunking failed for document {doc_id}: {e}")
            # Emergency fallback to recursive
            return await self._emergency_fallback_async(text, doc_id, metadata, str(e))

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkResult]:
        """Synchronously chunk text using intelligently selected strategy.

        Args:
            text: Text to chunk
            doc_id: Document identifier for chunk IDs
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects from selected strategy
        """
        # Validate inputs
        self._validate_input(text, doc_id, metadata)
        
        if not text.strip():
            return []

        try:
            # Analyze content to select optimal strategy
            analysis_result = self._analyze_content_sync(text, metadata)
            selected_strategy = analysis_result["selected_strategy"]
            selection_reason = analysis_result["selection_reason"]
            content_characteristics = analysis_result["characteristics"]

            logger.info(
                f"Hybrid chunker selected '{selected_strategy}' for document {doc_id}: {selection_reason}"
            )

            # Update analytics
            if self.enable_analytics:
                self._selection_stats[selected_strategy] = self._selection_stats.get(selected_strategy, 0) + 1

            # Get selected strategy and execute chunking
            chunker = self._get_strategy(selected_strategy)
            chunks = chunker.chunk_text(text, doc_id, metadata)

            # Enhance metadata with hybrid information
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "strategy": self.strategy_name,
                        "sub_strategy": selected_strategy,
                        "selection_reason": selection_reason,
                        "content_characteristics": content_characteristics,
                        "hybrid_analytics": self._get_analytics_summary() if self.enable_analytics else None,
                    }
                )

            logger.info(
                f"Hybrid chunking ({selected_strategy}) created {len(chunks)} chunks "
                f"for document {doc_id} ({len(text)} characters)"
            )

            return chunks

        except Exception as e:
            logger.error(f"Hybrid chunking failed for document {doc_id}: {e}")
            # Emergency fallback to recursive
            return self._emergency_fallback_sync(text, doc_id, metadata, str(e))

    async def _analyze_content_async(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Asynchronously analyze content to select optimal strategy.

        Args:
            text: Text to analyze
            metadata: Optional metadata with file type hints

        Returns:
            Dictionary with selected strategy and analysis details
        """
        # Run analysis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_content_sync, text, metadata)

    def _analyze_content_sync(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Synchronously analyze content to select optimal strategy.

        Args:
            text: Text to analyze
            metadata: Optional metadata with file type hints

        Returns:
            Dictionary with selected strategy and analysis details
        """
        characteristics = self._compute_content_characteristics(text, metadata)

        # Strategy selection logic
        selected_strategy = "recursive"  # Default
        selection_reason = "default choice for general text"

        # Rule 1: Check file type from metadata
        file_type = metadata.get("file_type", "") if metadata else ""
        file_name = metadata.get("file_name", "") if metadata else ""

        if file_type in [".md", ".markdown", ".mdown", ".mkd", ".mdx"] or file_name.endswith(
            (".md", ".markdown", ".mdown", ".mkd", ".mdx")
        ):
            selected_strategy = "markdown"
            selection_reason = f"file type indicates markdown ({file_type or 'inferred from filename'})"

        # Rule 2: Check markdown structure density
        elif characteristics["markdown_density"] > self.markdown_density_threshold:
            selected_strategy = "markdown"
            selection_reason = f"high markdown density ({characteristics['markdown_density']:.2f} > {self.markdown_density_threshold})"

        # Rule 3: Check for topic diversity (semantic chunking)
        elif (
            characteristics["topic_diversity"] > self.topic_diversity_threshold
            and len(text) >= self.semantic_min_length
        ):
            selected_strategy = "semantic"
            selection_reason = (
                f"high topic diversity ({characteristics['topic_diversity']:.2f} > {self.topic_diversity_threshold}) "
                f"and sufficient length ({len(text)} >= {self.semantic_min_length})"
            )

        # Rule 4: Code files get recursive (already default)
        elif characteristics.get("is_code_like", False):
            selected_strategy = "recursive"
            selection_reason = "code-like content detected, using recursive for better structure handling"

        return {
            "selected_strategy": selected_strategy,
            "selection_reason": selection_reason,
            "characteristics": characteristics,
        }

    def _compute_content_characteristics(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute various characteristics of the text content.

        Args:
            text: Text to analyze
            metadata: Optional metadata

        Returns:
            Dictionary with content characteristics
        """
        characteristics = {}

        # Basic text statistics
        lines = text.split("\n")
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        characteristics.update(
            {
                "text_length": len(text),
                "line_count": len(lines),
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_line_length": sum(len(line) for line in lines) / max(len(lines), 1),
                "avg_sentence_length": len(words) / max(len([s for s in sentences if s.strip()]), 1),
            }
        )

        # Markdown structure detection
        characteristics["markdown_density"] = self._calculate_markdown_density(text)

        # Topic diversity estimation
        characteristics["topic_diversity"] = self._estimate_topic_diversity(text, words)

        # Code-like content detection
        characteristics["is_code_like"] = self._detect_code_like_content(text, metadata)

        # Language/structure patterns
        characteristics.update(self._analyze_language_patterns(text))

        return characteristics

    def _calculate_markdown_density(self, text: str) -> float:
        """Calculate the density of markdown elements in text.

        Args:
            text: Text to analyze

        Returns:
            Markdown element density (0.0 to 1.0)
        """
        lines = text.split("\n")
        total_lines = len(lines)

        # Limit analysis to first 10KB to prevent performance issues
        text_sample = text[:10000]
        
        # Count various markdown indicators using pre-compiled safe patterns
        markdown_elements = 0
        
        # Use pre-compiled patterns with bounded matching
        for pattern_name, pattern in self.SAFE_MARKDOWN_PATTERNS.items():
            try:
                matches = pattern.findall(text_sample)
                markdown_elements += len(matches)
                
                # Additional safety: limit number of matches processed
                if len(matches) > 1000:
                    logger.warning(f"Excessive {pattern_name} matches found, capping at 1000")
                    markdown_elements = markdown_elements - len(matches) + 1000
                    
            except Exception as e:
                logger.warning(f"Error in markdown pattern {pattern_name}: {e}")
                continue

        # Calculate density
        density = markdown_elements / max(total_lines, 1)
        return min(density, 1.0)  # Cap at 1.0

    def _estimate_topic_diversity(self, text: str, words: List[str]) -> float:
        """Estimate topic diversity in the text.

        Args:
            text: Full text
            words: List of words

        Returns:
            Topic diversity estimate (0.0 to 1.0)
        """
        if not words:
            return 0.0

        # Simple lexical diversity measure
        unique_words = set(word.lower().strip(".,!?;:()[]{}\"'") for word in words if len(word) > 3)
        lexical_diversity = len(unique_words) / max(len(words), 1)

        # Paragraph transition analysis (simplified)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) < 2:
            paragraph_diversity = 0.0
        else:
            # Count topic transition indicators
            transitions = 0
            transition_words = [
                "however",
                "furthermore",
                "moreover",
                "additionally",
                "meanwhile",
                "in contrast",
                "on the other hand",
                "subsequently",
                "consequently",
                "therefore",
                "thus",
                "finally",
                "in conclusion",
            ]

            for word in transition_words:
                transitions += text.lower().count(word)

            paragraph_diversity = min(transitions / max(len(paragraphs), 1), 1.0)

        # Sentence length variety (indicates complexity/topic changes)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) < 2:
            length_variety = 0.0
        else:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
            length_variety = min(variance / (avg_length + 1), 1.0)

        # Combine metrics
        topic_diversity = (lexical_diversity * 0.5) + (paragraph_diversity * 0.3) + (length_variety * 0.2)

        return min(topic_diversity, 1.0)

    def _detect_code_like_content(self, text: str, metadata: Optional[Dict[str, Any]]) -> bool:
        """Detect if content appears to be code or technical documentation.

        Args:
            text: Text to analyze
            metadata: Optional metadata with file type

        Returns:
            True if content appears code-like
        """
        # Check file type
        file_type = metadata.get("file_type", "") if metadata else ""
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".cs",
            ".rb",
            ".go",
            ".rs",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".m",
            ".lua",
            ".dart",
            ".jsx",
            ".tsx",
            ".vue",
            ".sql",
            ".sh",
            ".bash",
            ".ps1",
            ".yaml",
            ".yml",
            ".json",
            ".xml",
            ".html",
            ".css",
        }

        if file_type in code_extensions:
            return True

        # Analyze content patterns
        code_indicators = 0

        # Function/method definitions
        code_indicators += len(re.findall(r"def\s+\w+\s*\(", text))
        code_indicators += len(re.findall(r"function\s+\w+\s*\(", text))
        code_indicators += len(re.findall(r"class\s+\w+", text))

        # Common programming constructs
        code_indicators += len(re.findall(r"\bif\s*\(.*\)\s*\{", text))
        code_indicators += len(re.findall(r"\bfor\s*\(.*\)\s*\{", text))
        code_indicators += len(re.findall(r"\bwhile\s*\(.*\)\s*\{", text))

        # Code-like syntax
        code_indicators += len(re.findall(r"[;}]\s*$", text, re.MULTILINE))
        code_indicators += len(re.findall(r"^\s*[#//]", text, re.MULTILINE))  # Comments

        # Calculate ratio
        lines = text.split("\n")
        code_density = code_indicators / max(len(lines), 1)

        return code_density > 0.1  # Threshold for code-like content

    def _analyze_language_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze language and structural patterns.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with language pattern analysis
        """
        patterns = {}

        # Question density (indicates FAQ or educational content)
        questions = len(re.findall(r"\?", text))
        patterns["question_density"] = questions / max(len(text), 1)

        # URL/link density
        links = len(re.findall(r"https?://\S+", text))
        patterns["link_density"] = links / max(len(text.split()), 1)

        # Number/data density (indicates technical/scientific content)
        numbers = len(re.findall(r"\b\d+\.?\d*\b", text))
        patterns["number_density"] = numbers / max(len(text.split()), 1)

        # Capitalization patterns (indicates proper nouns, technical terms)
        caps = len(re.findall(r"\b[A-Z][a-z]+\b", text))
        patterns["capitalization_density"] = caps / max(len(text.split()), 1)

        return patterns

    async def _emergency_fallback_async(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
        error_msg: str,
    ) -> List[ChunkResult]:
        """Emergency fallback when all strategy selection fails.

        Args:
            text: Text to chunk
            doc_id: Document identifier
            metadata: Metadata to preserve
            error_msg: Error message from failed attempt

        Returns:
            List of ChunkResult from emergency fallback
        """
        logger.warning(f"Using emergency fallback for document {doc_id}")

        try:
            # Always fallback to recursive as it's most reliable
            chunker = self._get_strategy("recursive")
            chunks = await chunker.chunk_text_async(text, doc_id, metadata)

            # Update analytics
            if self.enable_analytics:
                self._selection_stats["fallback"] = self._selection_stats.get("fallback", 0) + 1

            # Update metadata
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "strategy": self.strategy_name,
                        "sub_strategy": "recursive",
                        "selection_reason": "emergency_fallback",
                        "fallback_error": error_msg,
                        "hybrid_analytics": self._get_analytics_summary() if self.enable_analytics else None,
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Emergency fallback also failed for document {doc_id}: {e}")
            # Absolute last resort
            return [
                ChunkResult(
                    chunk_id=f"{doc_id}_emergency",
                    text=text[:5000],  # Limit size
                    start_offset=0,
                    end_offset=min(len(text), 5000),
                    metadata={
                        **(metadata or {}),
                        "strategy": "absolute_emergency",
                        "error": f"All chunking failed: {e}",
                    },
                )
            ]

    def _emergency_fallback_sync(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]],
        error_msg: str,
    ) -> List[ChunkResult]:
        """Synchronous emergency fallback."""
        logger.warning(f"Using emergency fallback for document {doc_id}")

        try:
            chunker = self._get_strategy("recursive")
            chunks = chunker.chunk_text(text, doc_id, metadata)

            if self.enable_analytics:
                self._selection_stats["fallback"] = self._selection_stats.get("fallback", 0) + 1

            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "strategy": self.strategy_name,
                        "sub_strategy": "recursive",
                        "selection_reason": "emergency_fallback",
                        "fallback_error": error_msg,
                        "hybrid_analytics": self._get_analytics_summary() if self.enable_analytics else None,
                    }
                )

            return chunks

        except Exception as e:
            logger.error(f"Emergency fallback also failed for document {doc_id}: {e}")
            return [
                ChunkResult(
                    chunk_id=f"{doc_id}_emergency",
                    text=text[:5000],
                    start_offset=0,
                    end_offset=min(len(text), 5000),
                    metadata={
                        **(metadata or {}),
                        "strategy": "absolute_emergency",
                        "error": f"All chunking failed: {e}",
                    },
                )
            ]

    def _get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of strategy selection analytics.

        Returns:
            Dictionary with selection statistics
        """
        total_selections = sum(self._selection_stats.values())
        if total_selections == 0:
            return {"total_selections": 0}

        return {
            "total_selections": total_selections,
            "strategy_percentages": {
                strategy: (count / total_selections) * 100 for strategy, count in self._selection_stats.items()
            },
            "most_used_strategy": max(self._selection_stats.items(), key=lambda x: x[1])[0],
        }

    def validate_config(self, params: Dict[str, Any]) -> bool:
        """Validate hybrid chunking configuration parameters.

        Args:
            params: Configuration parameters to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate thresholds
            markdown_threshold = params.get("markdown_density_threshold", self.markdown_density_threshold)
            if not isinstance(markdown_threshold, (int, float)) or not (0.0 <= markdown_threshold <= 1.0):
                logger.error(f"Invalid markdown_density_threshold: {markdown_threshold}")
                return False

            topic_threshold = params.get("topic_diversity_threshold", self.topic_diversity_threshold)
            if not isinstance(topic_threshold, (int, float)) or not (0.0 <= topic_threshold <= 1.0):
                logger.error(f"Invalid topic_diversity_threshold: {topic_threshold}")
                return False

            # Validate semantic minimum length
            semantic_min_length = params.get("semantic_min_length", self.semantic_min_length)
            if not isinstance(semantic_min_length, int) or semantic_min_length < 0:
                logger.error(f"Invalid semantic_min_length: {semantic_min_length}")
                return False

            return True

        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    def estimate_chunks(self, text_length: int, params: Dict[str, Any]) -> int:
        """Estimate number of chunks based on likely strategy selection.

        Args:
            text_length: Length of text in characters
            params: Chunking parameters

        Returns:
            Estimated number of chunks
        """
        # This is a rough estimate since actual strategy depends on content analysis
        # Assume most content will use recursive chunking (conservative estimate)
        avg_chunk_size = 600 * 4  # 600 tokens * ~4 chars/token
        base_estimate = max(1, text_length // avg_chunk_size)

        # Add some variance for different strategies
        # Semantic might create fewer, larger chunks
        # Markdown might create more, structured chunks
        estimated = max(1, int(base_estimate * 1.1))  # 10% buffer

        logger.debug(f"Estimated {estimated} chunks for {text_length} characters (hybrid strategy)")

        return estimated

    def get_selection_analytics(self) -> Dict[str, Any]:
        """Get detailed analytics about strategy selections.

        Returns:
            Dictionary with detailed analytics
        """
        return {
            "selection_stats": self._selection_stats.copy(),
            "analytics_summary": self._get_analytics_summary(),
            "configuration": {
                "markdown_density_threshold": self.markdown_density_threshold,
                "topic_diversity_threshold": self.topic_diversity_threshold,
                "semantic_min_length": self.semantic_min_length,
                "enable_analytics": self.enable_analytics,
            },
        }