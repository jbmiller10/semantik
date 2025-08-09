#!/usr/bin/env python3
"""
Semantic chunking strategy based on meaning and context.

This strategy groups text into chunks based on semantic similarity,
ensuring that related content stays together.
"""

from collections.abc import Callable
from datetime import datetime

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import (
    ChunkingStrategy,
)
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Semantic chunking strategy.

    This strategy groups text based on semantic similarity between sentences
    or paragraphs, keeping related content together.
    """

    def __init__(self) -> None:
        """Initialize the semantic chunking strategy."""
        super().__init__("semantic")

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

        # Merge small clusters to meet min_tokens requirement
        clusters = self._merge_small_clusters(clusters, config.min_tokens)

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
                created_at=datetime.utcnow(),
            )

            # For semantic chunking, use a lower min_tokens to allow for natural boundaries
            # but ensure chunks aren't too small
            effective_min_tokens = min(config.min_tokens, token_count, 1)
            
            # Create chunk with adjusted min_tokens for semantic boundaries
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

        return chunks

    def _split_into_sentences(self, text: str) -> list[dict]:
        """
        Split text into sentences with position tracking.

        Args:
            text: Text to split

        Returns:
            List of sentence dictionaries with text and position
        """
        sentences = []
        current_pos = 0
        current_sentence = []

        for i, char in enumerate(text):
            current_sentence.append(char)

            # Check for sentence ending
            if char in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                sentence_text = ''.join(current_sentence).strip()
                if sentence_text:
                    sentences.append({
                        "text": sentence_text,
                        "start": current_pos,
                        "end": i + 1,
                    })
                current_pos = i + 1
                current_sentence = []

        # Add remaining text as final sentence
        if current_sentence:
            sentence_text = ''.join(current_sentence).strip()
            if sentence_text:
                sentences.append({
                    "text": sentence_text,
                    "start": current_pos,
                    "end": len(text),
                })

        return sentences

    def _create_semantic_clusters(
        self,
        sentences: list[dict],
        max_tokens: int,
        threshold: float,
    ) -> list[dict]:
        """
        Group sentences into semantic clusters.

        Args:
            sentences: List of sentence dictionaries
            max_tokens: Maximum tokens per cluster
            threshold: Similarity threshold for grouping

        Returns:
            List of cluster dictionaries
        """
        if not sentences:
            return []

        clusters = []
        current_cluster = {
            "sentences": [],
            "tokens": 0,
            "start_offset": sentences[0]["start"],
            "end_offset": sentences[0]["end"],
            "similarity_score": 1.0,
        }

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence["text"])

            # Check if adding sentence would exceed token limit
            if current_cluster["tokens"] + sentence_tokens > max_tokens:
                # Save current cluster if it has content
                if current_cluster["sentences"]:
                    clusters.append(current_cluster)

                # Start new cluster
                current_cluster = {
                    "sentences": [sentence["text"]],
                    "tokens": sentence_tokens,
                    "start_offset": sentence["start"],
                    "end_offset": sentence["end"],
                    "similarity_score": 1.0,
                }
            else:
                # Calculate semantic similarity (simplified without embeddings)
                if current_cluster["sentences"]:
                    similarity = self._calculate_similarity(
                        current_cluster["sentences"][-1],
                        sentence["text"],
                    )

                    # If similarity is below threshold, start new cluster
                    if similarity < threshold:
                        clusters.append(current_cluster)
                        current_cluster = {
                            "sentences": [sentence["text"]],
                            "tokens": sentence_tokens,
                            "start_offset": sentence["start"],
                            "end_offset": sentence["end"],
                            "similarity_score": 1.0,
                        }
                    else:
                        # Add to current cluster
                        current_cluster["sentences"].append(sentence["text"])
                        current_cluster["tokens"] += sentence_tokens
                        current_cluster["end_offset"] = sentence["end"]
                        current_cluster["similarity_score"] = min(
                            current_cluster["similarity_score"],
                            similarity,
                        )
                else:
                    # First sentence in cluster
                    current_cluster["sentences"].append(sentence["text"])
                    current_cluster["tokens"] = sentence_tokens
                    current_cluster["end_offset"] = sentence["end"]

        # Add final cluster
        if current_cluster["sentences"]:
            clusters.append(current_cluster)

        return clusters

    def _merge_small_clusters(
        self,
        clusters: list[dict],
        min_tokens: int,
    ) -> list[dict]:
        """
        Merge clusters that are too small to meet min_tokens requirement.

        Args:
            clusters: List of cluster dictionaries
            min_tokens: Minimum tokens per cluster

        Returns:
            List of merged clusters
        """
        if not clusters:
            return []

        merged = []
        current = None

        for cluster in clusters:
            if current is None:
                current = cluster.copy()
            elif current["tokens"] < min_tokens:
                # Merge with next cluster if current is too small
                current["sentences"].extend(cluster["sentences"])
                current["tokens"] += cluster["tokens"]
                current["end_offset"] = cluster["end_offset"]
                # Update similarity score to be the minimum
                current["similarity_score"] = min(
                    current.get("similarity_score", 1.0),
                    cluster.get("similarity_score", 1.0),
                )
            else:
                # Current cluster is big enough, save it
                merged.append(current)
                current = cluster.copy()

        # Add the last cluster
        if current:
            # If the last cluster is too small, merge it with the previous one
            if current["tokens"] < min_tokens and merged:
                last = merged[-1]
                last["sentences"].extend(current["sentences"])
                last["tokens"] += current["tokens"]
                last["end_offset"] = current["end_offset"]
                last["similarity_score"] = min(
                    last.get("similarity_score", 1.0),
                    current.get("similarity_score", 1.0),
                )
            else:
                merged.append(current)

        return merged

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.

        This is a simplified version that uses word overlap as a proxy
        for semantic similarity, since we can't use external embedding models
        in the pure domain layer.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'as',
        }

        words1 = words1 - stop_words
        words2 = words2 - stop_words

        # If either set is empty, return low similarity
        if not words1 or not words2:
            return 0.3

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Also consider shared n-grams for better similarity
        ngrams1 = self._get_ngrams(text1.lower(), 3)
        ngrams2 = self._get_ngrams(text2.lower(), 3)

        if ngrams1 and ngrams2:
            ngram_intersection = len(ngrams1 & ngrams2)
            ngram_union = len(ngrams1 | ngrams2)
            ngram_similarity = ngram_intersection / ngram_union if ngram_union > 0 else 0

            # Weighted average of word and n-gram similarity
            similarity = 0.6 * jaccard + 0.4 * ngram_similarity
        else:
            similarity = jaccard

        # Apply smoothing to avoid harsh boundaries
        return min(1.0, max(0.0, similarity * 1.2))

    def _get_ngrams(self, text: str, n: int) -> set[str]:
        """
        Extract n-grams from text.

        Args:
            text: Text to extract n-grams from
            n: Size of n-grams

        Returns:
            Set of n-grams
        """
        if len(text) < n:
            return set()

        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i + n])

        return ngrams

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

        if len(content) > 10_000_000:  # 10MB limit for semantic analysis
            return False, f"Content too large for semantic analysis: {len(content)} characters"

        # Check if content has enough sentences for semantic chunking
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        if sentence_count < 2:
            return False, "Content must have at least 2 sentences for semantic chunking"

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

        # Semantic chunking typically produces fewer, more coherent chunks
        estimated_tokens = content_length // 4
        base_estimate = config.estimate_chunks(estimated_tokens)

        # Reduce by 20-30% due to semantic grouping
        return max(1, int(base_estimate * 0.75))
