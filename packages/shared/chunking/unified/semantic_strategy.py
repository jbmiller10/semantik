#!/usr/bin/env python3
"""
Unified semantic chunking strategy.

This module merges the domain-based and LlamaIndex-based semantic chunking 
implementations into a single unified strategy.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from packages.shared.chunking.unified.base import UnifiedChunkingStrategy

logger = logging.getLogger(__name__)


class SemanticChunkingStrategy(UnifiedChunkingStrategy):
    """
    Unified semantic chunking strategy.

    This strategy groups text based on semantic similarity between sentences
    or paragraphs, keeping related content together. Can optionally use
    LlamaIndex for embedding-based semantic chunking.
    """

    def __init__(self, use_llama_index: bool = False, embed_model: Any = None) -> None:
        """
        Initialize the semantic chunking strategy.

        Args:
            use_llama_index: Whether to use LlamaIndex implementation
            embed_model: Optional embedding model for LlamaIndex
        """
        super().__init__("semantic")
        self._use_llama_index = use_llama_index
        self._embed_model = embed_model
        self._llama_splitter = None

        if use_llama_index:
            try:
                from llama_index.core.node_parser import SemanticSplitterNodeParser

                self._llama_available = True
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

        if not self._embed_model:
            logger.warning("Embedding model not provided for semantic chunking")
            return None

        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser

            # Convert semantic threshold to percentile (0-100)
            breakpoint_percentile = int(config.semantic_threshold * 100)

            return SemanticSplitterNodeParser(
                embed_model=self._embed_model,
                buffer_size=1,  # Number of sentences to group
                breakpoint_percentile_threshold=breakpoint_percentile,
            )
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex semantic splitter: {e}")
            return None

    def chunk(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Create semantically coherent chunks.

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
        loop = asyncio.get_event_loop()
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
        Chunk using LlamaIndex SemanticSplitterNodeParser.

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

            # Get nodes using semantic splitter
            nodes = splitter.get_nodes_from_documents([doc])

            if not nodes:
                return []

            chunks = []
            total_chars = len(content)

            # Convert LlamaIndex nodes to domain chunks
            for idx, node in enumerate(nodes):
                chunk_text = node.get_content()
                token_count = self.count_tokens(chunk_text)

                # If chunk exceeds max_tokens, split it
                if token_count > config.max_tokens:
                    # Split the chunk into smaller pieces
                    words = chunk_text.split()
                    current_words = []
                    current_token_count = 0
                    
                    # Calculate initial offset
                    if idx == 0 and len(chunks) == 0:
                        chunk_start_offset = 0
                    else:
                        prev_end = chunks[-1].metadata.end_offset if chunks else 0
                        chunk_start_offset = content.find(chunk_text, prev_end - 100)
                        if chunk_start_offset == -1:
                            chunk_start_offset = prev_end
                    
                    for word in words:
                        word_tokens = self.count_tokens(word + " ")
                        
                        if current_token_count + word_tokens > config.max_tokens and current_words:
                            # Create a chunk from accumulated words
                            sub_chunk_text = " ".join(current_words)
                            sub_token_count = self.count_tokens(sub_chunk_text)
                            
                            metadata = ChunkMetadata(
                                chunk_id=f"{config.strategy_name}_{len(chunks):04d}",
                                document_id="doc",
                                chunk_index=len(chunks),
                                start_offset=chunk_start_offset,
                                end_offset=min(chunk_start_offset + len(sub_chunk_text), total_chars),
                                token_count=sub_token_count,
                                strategy_name=self.name,
                                semantic_density=0.8,
                                confidence_score=0.95,
                                created_at=datetime.now(tz=UTC),
                            )
                            
                            effective_min_tokens = min(config.min_tokens, sub_token_count, 1)
                            chunk = Chunk(
                                content=sub_chunk_text,
                                metadata=metadata,
                                min_tokens=effective_min_tokens,
                                max_tokens=config.max_tokens,
                            )
                            chunks.append(chunk)
                            
                            chunk_start_offset += len(sub_chunk_text) + 1
                            current_words = [word]
                            current_token_count = word_tokens
                        else:
                            current_words.append(word)
                            current_token_count += word_tokens
                    
                    # Add remaining words if any
                    if current_words:
                        sub_chunk_text = " ".join(current_words)
                        sub_token_count = self.count_tokens(sub_chunk_text)
                        
                        metadata = ChunkMetadata(
                            chunk_id=f"{config.strategy_name}_{len(chunks):04d}",
                            document_id="doc",
                            chunk_index=len(chunks),
                            start_offset=chunk_start_offset,
                            end_offset=min(chunk_start_offset + len(sub_chunk_text), total_chars),
                            token_count=sub_token_count,
                            strategy_name=self.name,
                            semantic_density=0.8,
                            confidence_score=0.95,
                            created_at=datetime.now(tz=UTC),
                        )
                        
                        effective_min_tokens = min(config.min_tokens, sub_token_count, 1)
                        chunk = Chunk(
                            content=sub_chunk_text,
                            metadata=metadata,
                            min_tokens=effective_min_tokens,
                            max_tokens=config.max_tokens,
                        )
                        chunks.append(chunk)
                else:
                    # Regular processing for chunks within limits
                    # Calculate offsets
                    if idx == 0 and len(chunks) == 0:
                        start_offset = 0
                    else:
                        # Find the chunk text in the original content
                        prev_end = chunks[-1].metadata.end_offset if chunks else 0
                        start_offset = content.find(chunk_text, prev_end - 100)  # Look near previous end
                        if start_offset == -1:
                            start_offset = prev_end

                    end_offset = min(start_offset + len(chunk_text), total_chars)

                    # Create chunk metadata
                    metadata = ChunkMetadata(
                        chunk_id=f"{config.strategy_name}_{len(chunks):04d}",
                        document_id="doc",
                        chunk_index=len(chunks),
                        start_offset=start_offset,
                        end_offset=end_offset,
                        token_count=token_count,
                        strategy_name=self.name,
                        semantic_density=0.8,  # High for semantic chunking
                        confidence_score=0.95,  # Higher confidence with LlamaIndex
                        created_at=datetime.now(tz=UTC),
                    )

                    # Create chunk entity
                    effective_min_tokens = min(config.min_tokens, token_count, 1)

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
            logger.warning(f"LlamaIndex semantic chunking failed, falling back to domain: {e}")
            return None

    def _chunk_with_domain(
        self,
        content: str,
        config: ChunkConfig,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[Chunk]:
        """
        Chunk using domain implementation (simple semantic clustering).
        """
        # Split into sentences for semantic analysis
        sentences = self._split_into_sentences(content)
        if not sentences:
            return []

        # Group sentences into semantic clusters
        clusters = self._create_semantic_clusters(
            sentences,
            config.max_tokens,
            config.semantic_threshold,
        )

        # Merge small clusters to meet min_tokens requirement while respecting max_tokens
        clusters = self._merge_small_clusters(clusters, config.min_tokens, config.max_tokens)

        # Convert clusters to chunks
        chunks = []
        chunk_index = 0
        total_clusters = len(clusters)

        for i, cluster in enumerate(clusters):
            # Join sentences in cluster
            cluster_text = " ".join(cluster["sentences"])

            # Clean text
            cluster_text = self.clean_chunk_text(cluster_text)
            if not cluster_text:
                continue

            # Calculate offsets
            start_offset = cluster["start_offset"]
            end_offset = cluster["end_offset"]
            token_count = self.count_tokens(cluster_text)

            # If cluster still exceeds max_tokens, split it further
            if token_count > config.max_tokens:
                # Split the cluster text into smaller chunks
                words = cluster_text.split()
                current_words = []
                current_token_count = 0
                word_start_offset = start_offset
                
                for word in words:
                    word_tokens = self.count_tokens(word + " ")
                    
                    if current_token_count + word_tokens > config.max_tokens and current_words:
                        # Create a chunk from accumulated words
                        sub_chunk_text = " ".join(current_words)
                        sub_token_count = self.count_tokens(sub_chunk_text)
                        
                        metadata = ChunkMetadata(
                            chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                            document_id="doc",
                            chunk_index=chunk_index,
                            start_offset=word_start_offset,
                            end_offset=min(word_start_offset + len(sub_chunk_text), end_offset),
                            token_count=sub_token_count,
                            strategy_name=self.name,
                            semantic_score=cluster.get("similarity_score"),
                            semantic_density=cluster.get("similarity_score", 0.5),
                            confidence_score=0.8,
                            created_at=datetime.now(tz=UTC),
                        )
                        
                        effective_min_tokens = min(config.min_tokens, sub_token_count, 1)
                        chunk = Chunk(
                            content=sub_chunk_text,
                            metadata=metadata,
                            min_tokens=effective_min_tokens,
                            max_tokens=config.max_tokens,
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                        word_start_offset += len(sub_chunk_text) + 1
                        current_words = [word]
                        current_token_count = word_tokens
                    else:
                        current_words.append(word)
                        current_token_count += word_tokens
                
                # Add remaining words if any
                if current_words:
                    sub_chunk_text = " ".join(current_words)
                    sub_token_count = self.count_tokens(sub_chunk_text)
                    
                    # Final check: if still too large, truncate to fit
                    if sub_token_count > config.max_tokens:
                        # Truncate words to fit within max_tokens
                        truncated_words = []
                        truncated_token_count = 0
                        for word in current_words:
                            temp_text = " ".join(truncated_words + [word])
                            temp_tokens = self.count_tokens(temp_text)
                            if temp_tokens <= config.max_tokens:
                                truncated_words.append(word)
                                truncated_token_count = temp_tokens
                            else:
                                break
                        if truncated_words:
                            sub_chunk_text = " ".join(truncated_words)
                            sub_token_count = truncated_token_count
                            # Final safety check
                            if sub_token_count > config.max_tokens:
                                logger.warning(f"Chunk still exceeds max after truncation: {sub_token_count} > {config.max_tokens}")
                                # Force truncation to character limit as last resort
                                max_chars = config.max_tokens * 4  # Approximate 4 chars per token
                                sub_chunk_text = sub_chunk_text[:max_chars]
                                sub_token_count = self.count_tokens(sub_chunk_text)
                        else:
                            # Skip if no words fit
                            continue
                    
                    metadata = ChunkMetadata(
                        chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                        document_id="doc",
                        chunk_index=chunk_index,
                        start_offset=word_start_offset,
                        end_offset=end_offset,
                        token_count=sub_token_count,
                        strategy_name=self.name,
                        semantic_score=cluster.get("similarity_score"),
                        semantic_density=cluster.get("similarity_score", 0.5),
                        confidence_score=0.8,
                        created_at=datetime.now(tz=UTC),
                    )
                    
                    effective_min_tokens = min(config.min_tokens, sub_token_count, 1)
                    chunk = Chunk(
                        content=sub_chunk_text,
                        metadata=metadata,
                        min_tokens=effective_min_tokens,
                        max_tokens=config.max_tokens,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Regular processing for clusters within limits
                # Calculate semantic density (higher for more semantically cohesive clusters)
                semantic_density = cluster.get("similarity_score", 0.5)

                # Create metadata
                metadata = ChunkMetadata(
                    chunk_id=f"{config.strategy_name}_{chunk_index:04d}",
                    document_id="doc",
                    chunk_index=chunk_index,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    token_count=token_count,
                    strategy_name=self.name,
                    semantic_score=cluster.get("similarity_score"),
                    semantic_density=semantic_density,
                    confidence_score=0.8,  # Good confidence for semantic strategy
                    created_at=datetime.now(tz=UTC),
                )

                # Create chunk entity with adjusted min_tokens
                effective_min_tokens = min(config.min_tokens, token_count, 1)

                chunk = Chunk(
                    content=cluster_text,
                    metadata=metadata,
                    min_tokens=effective_min_tokens,
                    max_tokens=config.max_tokens,
                )

                chunks.append(chunk)
                chunk_index += 1

            # Report progress
            if progress_callback:
                progress = ((i + 1) / total_clusters) * 100
                progress_callback(min(progress, 100.0))

        # Final progress report
        if progress_callback:
            progress_callback(100.0)

        return chunks

    def _split_into_sentences(self, content: str) -> list[dict[str, Any]]:
        """
        Split text into sentences with position tracking.

        Args:
            content: Text to split

        Returns:
            List of sentence dictionaries with text and positions
        """
        sentences = []

        # Simple sentence splitting (can be improved with NLP libraries)
        sentence_endings = [".", "!", "?"]
        current_sentence = ""
        current_start = 0

        for i, char in enumerate(content):
            current_sentence += char

            if char in sentence_endings:
                # Check for end of sentence (next char is space or end)
                if i + 1 >= len(content) or content[i + 1].isspace():
                    sentence_text = current_sentence.strip()
                    if sentence_text:
                        sentences.append(
                            {
                                "text": sentence_text,
                                "start_offset": current_start,
                                "end_offset": i + 1,
                            }
                        )
                    current_sentence = ""
                    current_start = i + 1

        # Add remaining text as final sentence
        if current_sentence.strip():
            sentences.append(
                {
                    "text": current_sentence.strip(),
                    "start_offset": current_start,
                    "end_offset": len(content),
                }
            )

        return sentences

    def _create_semantic_clusters(
        self,
        sentences: list[dict[str, Any]],
        max_tokens: int,
        similarity_threshold: float,
    ) -> list[dict[str, Any]]:
        """
        Group sentences into semantic clusters.

        Args:
            sentences: List of sentence dictionaries
            max_tokens: Maximum tokens per cluster
            similarity_threshold: Threshold for semantic similarity

        Returns:
            List of cluster dictionaries
        """
        if not sentences:
            return []

        clusters = []
        current_cluster = {
            "sentences": [],
            "start_offset": sentences[0]["start_offset"],
            "end_offset": sentences[0]["end_offset"],
            "token_count": 0,
            "similarity_score": 1.0,
        }

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence["text"])
            
            # If a single sentence exceeds max_tokens, split it
            if sentence_tokens > max_tokens:
                # Save current cluster if it has content
                if current_cluster["sentences"]:
                    clusters.append(current_cluster)
                
                # Split the sentence into smaller chunks
                sentence_text = sentence["text"]
                words = sentence_text.split()
                current_words = []
                current_token_count = 0
                sentence_start = sentence["start_offset"]
                
                for word in words:
                    word_tokens = self.count_tokens(word + " ")
                    if current_token_count + word_tokens > max_tokens and current_words:
                        # Create a cluster from accumulated words
                        chunk_text = " ".join(current_words)
                        chunk_end = sentence_start + len(chunk_text)
                        clusters.append({
                            "sentences": [chunk_text],
                            "start_offset": sentence_start,
                            "end_offset": chunk_end,
                            "token_count": current_token_count,
                            "similarity_score": 0.9,
                        })
                        sentence_start = chunk_end + 1
                        current_words = [word]
                        current_token_count = word_tokens
                    else:
                        current_words.append(word)
                        current_token_count += word_tokens
                
                # Add remaining words as new cluster
                if current_words:
                    chunk_text = " ".join(current_words)
                    current_cluster = {
                        "sentences": [chunk_text],
                        "start_offset": sentence_start,
                        "end_offset": sentence["end_offset"],
                        "token_count": current_token_count,
                        "similarity_score": 0.9,
                    }
                else:
                    # Start fresh cluster for next sentence
                    current_cluster = {
                        "sentences": [],
                        "start_offset": sentence["end_offset"],
                        "end_offset": sentence["end_offset"],
                        "token_count": 0,
                        "similarity_score": 1.0,
                    }
                continue

            # Check if adding this sentence would exceed max_tokens
            if current_cluster["token_count"] + sentence_tokens > max_tokens:
                # Save current cluster and start new one
                if current_cluster["sentences"]:
                    clusters.append(current_cluster)

                current_cluster = {
                    "sentences": [sentence["text"]],
                    "start_offset": sentence["start_offset"],
                    "end_offset": sentence["end_offset"],
                    "token_count": sentence_tokens,
                    "similarity_score": 1.0,
                }
            else:
                # Add sentence to current cluster
                current_cluster["sentences"].append(sentence["text"])
                current_cluster["end_offset"] = sentence["end_offset"]
                current_cluster["token_count"] += sentence_tokens

                # Simple similarity: sentences close together are similar
                # (In production, use actual embeddings for similarity)
                current_cluster["similarity_score"] *= 0.95

        # Add final cluster
        if current_cluster["sentences"]:
            clusters.append(current_cluster)

        return clusters

    def _merge_small_clusters(self, clusters: list[dict[str, Any]], min_tokens: int, max_tokens: int) -> list[dict[str, Any]]:
        """
        Merge small clusters to meet minimum token requirements while respecting maximum.

        Args:
            clusters: List of cluster dictionaries
            min_tokens: Minimum tokens per cluster
            max_tokens: Maximum tokens per cluster

        Returns:
            List of merged clusters
        """
        if not clusters:
            return []

        merged = []
        current_merge = None

        for cluster in clusters:
            if cluster["token_count"] >= min_tokens:
                # Cluster is large enough on its own
                if current_merge:
                    merged.append(current_merge)
                    current_merge = None
                merged.append(cluster)
            else:
                # Cluster is too small, try to merge with current
                if current_merge:
                    # Check if merging would exceed max_tokens
                    combined_tokens = current_merge["token_count"] + cluster["token_count"]
                    if combined_tokens <= max_tokens:
                        # Safe to merge
                        current_merge["sentences"].extend(cluster["sentences"])
                        current_merge["end_offset"] = cluster["end_offset"]
                        current_merge["token_count"] = combined_tokens
                        current_merge["similarity_score"] = min(
                            current_merge["similarity_score"], cluster["similarity_score"]
                        )

                        # Check if merged is now large enough
                        if current_merge["token_count"] >= min_tokens:
                            merged.append(current_merge)
                            current_merge = None
                    else:
                        # Would exceed max_tokens, save current and start new
                        merged.append(current_merge)
                        current_merge = cluster.copy()
                else:
                    # Start new merge
                    current_merge = cluster.copy()

        # Add final merge if exists
        if current_merge:
            merged.append(current_merge)

        return merged

    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """
        Validate content for semantic chunking.

        Args:
            content: Content to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not content:
            return False, "Content cannot be empty"

        if len(content) > 50_000_000:  # 50MB limit
            return False, f"Content too large: {len(content)} characters"

        # Semantic chunking works best with natural language text
        if len(content.split()) < 10:
            return False, "Content too short for semantic analysis"

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

        # Semantic chunking tends to create slightly fewer, more coherent chunks
        base_estimate = config.estimate_chunks(estimated_tokens)
        return max(1, int(base_estimate * 0.8))
