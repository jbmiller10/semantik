#!/usr/bin/env python3
"""
Hybrid chunking strategy combining multiple approaches.

This strategy intelligently combines different chunking methods based on
content characteristics to achieve optimal results.
"""

from collections.abc import Callable
from datetime import datetime

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.character import (
    CharacterChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.markdown import (
    MarkdownChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.recursive import (
    RecursiveChunkingStrategy,
)
from packages.shared.chunking.domain.services.chunking_strategies.semantic import (
    SemanticChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class HybridChunkingStrategy(ChunkingStrategy):
    """
    Hybrid chunking strategy.

    Combines multiple chunking strategies based on content analysis,
    choosing the best approach for different sections of the document.
    """

    def __init__(self) -> None:
        """Initialize the hybrid chunking strategy."""
        super().__init__("hybrid")

        # Initialize component strategies
        self._character_strategy = CharacterChunkingStrategy()
        self._recursive_strategy = RecursiveChunkingStrategy()
        self._semantic_strategy = SemanticChunkingStrategy()
        self._markdown_strategy = MarkdownChunkingStrategy()

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Apply hybrid chunking using the best strategy for each section.

        Args:
            content: The text content to chunk
            config: Configuration parameters
            progress_callback: Optional progress callback

        Returns:
            List of chunks
        """
        if not content:
            return []

        # Analyze content characteristics
        content_analysis = self._analyze_content(content)

        # Determine primary strategy based on analysis
        primary_strategy = self._select_primary_strategy(content_analysis)

        # Apply chunking based on content type
        if content_analysis["is_mixed"]:
            # Mixed content: segment and apply different strategies
            chunks = self._chunk_mixed_content(
                content,
                config,
                content_analysis,
                progress_callback,
            )
        elif config.strategies and len(config.strategies) > 1:
            # Multiple strategies specified: use consensus building
            strategy_results = {}
            for strategy_name in config.strategies:
                try:
                    if strategy_name == "character":
                        strategy_chunks = self._character_strategy.chunk(content, config, None)
                    elif strategy_name == "semantic":
                        strategy_chunks = self._semantic_strategy.chunk(content, config, None)
                    elif strategy_name == "markdown":
                        strategy_chunks = self._markdown_strategy.chunk(content, config, None)
                    elif strategy_name == "recursive":
                        strategy_chunks = self._recursive_strategy.chunk(content, config, None)
                    else:
                        continue
                    strategy_results[strategy_name] = strategy_chunks
                except Exception as e:
                    print(f"Strategy {strategy_name} failed: {e}")

            if strategy_results:
                chunks = self._build_consensus(strategy_results)
            else:
                # All strategies failed, fall back to character
                chunks = self._character_strategy.chunk(content, config, progress_callback)
        else:
            # Uniform content: use primary strategy with enhancements
            chunks = self._chunk_uniform_content(
                content,
                config,
                primary_strategy,
                progress_callback,
            )

        # If no chunks were created, fall back to character strategy
        if not chunks:
            chunks = self._character_strategy.chunk(content, config, progress_callback)
            # Update metadata to reflect hybrid processing
            for i, chunk in enumerate(chunks):
                custom_attrs = {
                    "strategies_used": ["character"],
                    "fallback": True,
                }

                hybrid_metadata = ChunkMetadata(
                    chunk_id=f"hybrid_{i:04d}",
                    document_id=chunk.metadata.document_id,
                    chunk_index=i,
                    start_offset=chunk.metadata.start_offset,
                    end_offset=chunk.metadata.end_offset,
                    token_count=chunk.metadata.token_count,
                    strategy_name="hybrid",
                    custom_attributes=custom_attrs,
                    semantic_density=0.6,
                    confidence_score=0.8,
                    created_at=datetime.utcnow(),
                )

                chunks[i] = Chunk(
                    content=chunk.content,
                    metadata=hybrid_metadata,
                    min_tokens=config.min_tokens,
                    max_tokens=config.max_tokens,
                )

        # Post-process chunks for consistency
        chunks = self._post_process_chunks(chunks, config)

        return chunks

    def _analyze_content(self, content: str) -> dict:
        """
        Analyze content characteristics.

        Args:
            content: Content to analyze

        Returns:
            Dictionary with content analysis results
        """
        analysis = {
            "total_chars": len(content),
            "total_lines": content.count("\n") + 1,
            "has_markdown": False,
            "has_code": False,
            "has_structure": False,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "is_mixed": False,
            "sections": [],
        }

        # Check for markdown elements
        markdown_indicators = ["#", "```", "|", ">", "- ", "* ", "1. "]
        for indicator in markdown_indicators:
            if indicator in content:
                analysis["has_markdown"] = True
                break

        # Check for code blocks
        if "```" in content or "def " in content or "function " in content:
            analysis["has_code"] = True

        # Count sentences
        sentences = content.count(".") + content.count("!") + content.count("?")
        analysis["sentence_count"] = sentences

        if sentences > 0:
            analysis["avg_sentence_length"] = len(content) / sentences

        # Check for structural elements
        if "\n\n" in content or analysis["has_markdown"]:
            analysis["has_structure"] = True

        # Determine if content is mixed
        if analysis["has_code"] and analysis["sentence_count"] > 10:
            analysis["is_mixed"] = True
        elif analysis["has_markdown"] and not analysis["has_code"] or analysis["sentence_count"] < 5:
            analysis["is_mixed"] = False
        else:
            # Analyze section variability
            sections = self._identify_sections(content)
            if len(sections) > 1:
                # Check if sections have different characteristics
                section_types = set(s["type"] for s in sections)
                if len(section_types) > 2:
                    analysis["is_mixed"] = True
            analysis["sections"] = sections

        return analysis

    def _identify_sections(self, content: str) -> list[dict]:
        """
        Identify distinct sections in content.

        Args:
            content: Content to analyze

        Returns:
            List of section dictionaries
        """
        sections = []
        current_pos = 0

        # Split by major breaks
        parts = content.split("\n\n")

        for part in parts:
            if not part.strip():
                current_pos += len(part) + 2
                continue

            section_type = self._classify_section(part)

            sections.append(
                {
                    "type": section_type,
                    "start": current_pos,
                    "end": current_pos + len(part),
                    "content": part,
                }
            )

            current_pos += len(part) + 2  # +2 for \n\n

        return sections

    def _classify_section(self, text: str) -> str:
        """
        Classify a section of text.

        Args:
            text: Text section to classify

        Returns:
            Section type
        """
        stripped = text.strip()

        # Code detection
        if stripped.startswith("```") or "def " in stripped or "function " in stripped:
            return "code"

        # Markdown header
        if stripped.startswith("#"):
            return "header"

        # List
        if stripped.startswith(("- ", "* ", "1. ")):
            return "list"

        # Table (simple detection)
        if "|" in stripped and stripped.count("|") > 2:
            return "table"

        # Quote
        if stripped.startswith(">"):
            return "quote"

        # Default to prose
        return "prose"

    def _select_primary_strategy(self, analysis: dict) -> str:
        """
        Select the primary chunking strategy based on content analysis.

        Args:
            analysis: Content analysis results

        Returns:
            Name of the primary strategy
        """
        # For structured markdown documents
        if analysis["has_markdown"] and analysis["has_structure"]:
            return "markdown"

        # For code-heavy content
        if analysis["has_code"] and not analysis["is_mixed"]:
            return "character"  # Fixed-size for code

        # For prose with good sentence structure
        if analysis["sentence_count"] > 20 and analysis["avg_sentence_length"] < 200:
            return "semantic"

        # Default to recursive for general content
        return "recursive"

    def _chunk_mixed_content(
        self,
        content: str,
        config: ChunkConfig,
        analysis: dict,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk mixed content using appropriate strategies for each section.

        Args:
            content: Content to chunk
            config: Configuration
            analysis: Content analysis
            progress_callback: Progress callback

        Returns:
            List of chunks
        """
        all_chunks = []
        sections = analysis.get("sections", [])

        if not sections:
            # Fall back to recursive if no sections identified
            return self._recursive_strategy.chunk(content, config, progress_callback)

        total_sections = len(sections)
        chunk_index = 0

        for i, section in enumerate(sections):
            section_content = section["content"]
            section_type = section["type"]

            # Select strategy for this section
            if section_type == "code":
                strategy = self._character_strategy
            elif section_type in ["header", "list", "table"]:
                strategy = self._markdown_strategy
            elif section_type == "prose":
                strategy = self._semantic_strategy
            else:
                strategy = self._recursive_strategy

            # Chunk the section with error handling
            try:
                section_chunks = strategy.chunk(section_content, config, None)
            except Exception as e:
                # Fall back to character strategy if the selected strategy fails
                print(f"Strategy {section_type} failed: {e}. Falling back to character strategy.")
                section_chunks = self._character_strategy.chunk(section_content, config, None)

            # Adjust chunk metadata for correct offsets
            for chunk in section_chunks:
                # Create new metadata with adjusted offsets
                custom_attrs = {
                    "strategies_used": [section_type],
                    "section_type": section_type,
                }

                adjusted_metadata = ChunkMetadata(
                    chunk_id=f"hybrid_{chunk_index:04d}",
                    document_id=chunk.metadata.document_id,
                    chunk_index=chunk_index,
                    start_offset=section["start"] + chunk.metadata.start_offset,
                    end_offset=section["start"] + chunk.metadata.end_offset,
                    token_count=chunk.metadata.token_count,
                    strategy_name="hybrid",
                    section_title=f"{section_type}_section",
                    custom_attributes=custom_attrs,
                    semantic_density=0.6,
                    confidence_score=0.8,
                    created_at=datetime.utcnow(),
                )

                # Create new chunk with adjusted metadata
                adjusted_chunk = Chunk(
                    content=chunk.content,
                    metadata=adjusted_metadata,
                    min_tokens=config.min_tokens,
                    max_tokens=config.max_tokens,
                )

                all_chunks.append(adjusted_chunk)
                chunk_index += 1

            # Report progress
            if progress_callback:
                progress = ((i + 1) / total_sections) * 100
                progress_callback(min(progress, 100.0))

        return all_chunks

    def _chunk_uniform_content(
        self,
        content: str,
        config: ChunkConfig,
        primary_strategy: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk uniform content using the primary strategy.

        Args:
            content: Content to chunk
            config: Configuration
            primary_strategy: Name of primary strategy
            progress_callback: Progress callback

        Returns:
            List of chunks
        """
        # Select and apply primary strategy with error handling
        chunks = []
        try:
            if primary_strategy == "markdown":
                chunks = self._markdown_strategy.chunk(content, config, progress_callback)
            elif primary_strategy == "semantic":
                chunks = self._semantic_strategy.chunk(content, config, progress_callback)
            elif primary_strategy == "character":
                chunks = self._character_strategy.chunk(content, config, progress_callback)
            else:
                chunks = self._recursive_strategy.chunk(content, config, progress_callback)
        except Exception as e:
            # Fall back to character strategy if primary fails
            print(f"Primary strategy {primary_strategy} failed: {e}. Falling back to character strategy.")
            chunks = self._character_strategy.chunk(content, config, progress_callback)

        # Update strategy name in metadata
        updated_chunks = []
        for i, chunk in enumerate(chunks):
            # Create new metadata with hybrid strategy name
            custom_attrs = {
                "strategies_used": [primary_strategy],
                "primary_strategy": primary_strategy,
            }

            hybrid_metadata = ChunkMetadata(
                chunk_id=f"hybrid_{i:04d}",
                document_id=chunk.metadata.document_id,
                chunk_index=i,
                start_offset=chunk.metadata.start_offset,
                end_offset=chunk.metadata.end_offset,
                token_count=chunk.metadata.token_count,
                strategy_name="hybrid",
                semantic_score=chunk.metadata.semantic_score,
                hierarchy_level=chunk.metadata.hierarchy_level,
                section_title=chunk.metadata.section_title,
                custom_attributes=custom_attrs,
                semantic_density=0.6,
                confidence_score=0.8,
                created_at=datetime.utcnow(),
            )

            # Create new chunk with updated metadata
            hybrid_chunk = Chunk(
                content=chunk.content,
                metadata=hybrid_metadata,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
            )

            updated_chunks.append(hybrid_chunk)

        return updated_chunks

    def _post_process_chunks(self, chunks: list[Chunk], config: ChunkConfig) -> list[Chunk]:
        """
        Post-process chunks for consistency.

        Args:
            chunks: Chunks to post-process
            config: Configuration

        Returns:
            Post-processed chunks
        """
        if not chunks:
            return chunks

        processed = []

        for chunk in chunks:
            # Skip chunks that are too small (unless they're the only chunk)
            if chunk.metadata.token_count < config.min_tokens and len(chunks) > 1:
                # Try to merge with adjacent chunk
                if processed and processed[-1].metadata.token_count + chunk.metadata.token_count <= config.max_tokens:
                    # Merge with previous chunk
                    prev_chunk = processed[-1]
                    merged_content = prev_chunk.content + "\n" + chunk.content
                    merged_tokens = prev_chunk.metadata.token_count + chunk.metadata.token_count

                    # Create merged metadata
                    merged_metadata = ChunkMetadata(
                        chunk_id=prev_chunk.metadata.chunk_id,
                        document_id=prev_chunk.metadata.document_id,
                        chunk_index=prev_chunk.metadata.chunk_index,
                        start_offset=prev_chunk.metadata.start_offset,
                        end_offset=chunk.metadata.end_offset,
                        token_count=merged_tokens,
                        strategy_name="hybrid",
                        custom_attributes=prev_chunk.metadata.custom_attributes,
                        semantic_density=0.6,
                        confidence_score=0.8,
                        created_at=datetime.utcnow(),
                    )

                    # Replace previous chunk with merged version
                    processed[-1] = Chunk(
                        content=merged_content,
                        metadata=merged_metadata,
                        min_tokens=config.min_tokens,
                        max_tokens=config.max_tokens,
                    )
                    continue

            processed.append(chunk)

        # Re-index chunks
        final_chunks = []
        for i, chunk in enumerate(processed):
            # Create new metadata with correct index
            reindexed_metadata = ChunkMetadata(
                chunk_id=f"hybrid_{i:04d}",
                document_id=chunk.metadata.document_id,
                chunk_index=i,
                start_offset=chunk.metadata.start_offset,
                end_offset=chunk.metadata.end_offset,
                token_count=chunk.metadata.token_count,
                strategy_name="hybrid",
                semantic_score=chunk.metadata.semantic_score,
                hierarchy_level=chunk.metadata.hierarchy_level,
                section_title=chunk.metadata.section_title,
                custom_attributes=chunk.metadata.custom_attributes,
                semantic_density=chunk.metadata.semantic_density,
                confidence_score=chunk.metadata.confidence_score,
                created_at=datetime.utcnow(),
            )

            final_chunk = Chunk(
                content=chunk.content,
                metadata=reindexed_metadata,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
            )

            final_chunks.append(final_chunk)

        return final_chunks

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for hybrid chunking.

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

        # Hybrid typically produces similar to recursive
        estimated_tokens = content_length // 4
        return config.estimate_chunks(estimated_tokens)

    def _build_consensus(self, strategy_results: dict[str, list[Chunk]]) -> list[Chunk]:
        """
        Build consensus from multiple strategy results.

        This method combines chunks from different strategies to create
        an optimal chunking solution.

        Args:
            strategy_results: Dictionary mapping strategy names to their chunks

        Returns:
            Consensus list of chunks
        """
        if not strategy_results:
            return []

        # If only one strategy, return its results
        if len(strategy_results) == 1:
            return list(strategy_results.values())[0]

        # Simple consensus: use the strategy that produced the most reasonable chunks
        # (not too many, not too few)
        best_chunks = None
        best_score = float("inf")

        for strategy_name, chunks in strategy_results.items():
            if not chunks:
                continue

            # Score based on chunk count (prefer moderate numbers)
            chunk_count = len(chunks)
            if chunk_count == 0:
                score = float("inf")
            elif chunk_count < 3:
                score = 10 - chunk_count  # Penalize too few chunks
            elif chunk_count > 50:
                score = chunk_count  # Penalize too many chunks
            else:
                score = 0  # Ideal range

            if score < best_score:
                best_score = score
                best_chunks = chunks

        return best_chunks if best_chunks else []
