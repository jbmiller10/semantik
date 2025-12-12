#!/usr/bin/env python3
"""
Streaming semantic chunking strategy.

This strategy groups semantically related sentences together while maintaining
bounded memory usage by buffering up to 10 sentences (max 50KB).
"""

import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata
from shared.chunking.infrastructure.streaming.window import StreamingWindow

from .base import StreamingChunkingStrategy


class StreamingSemanticStrategy(StreamingChunkingStrategy):
    """
    Streaming semantic chunking strategy.

    Groups semantically similar sentences together by analyzing sentence
    similarity and topic coherence, maintaining a buffer of up to 10 sentences
    (max 50KB) for context.
    """

    MAX_BUFFER_SIZE = 50 * 1024  # 50KB max buffer
    MAX_SENTENCES = 10  # Maximum sentences in buffer

    def __init__(self) -> None:
        """Initialize the streaming semantic strategy."""
        super().__init__("semantic")
        self._sentence_buffer: list[tuple[str, int, dict]] = []  # Buffer of (sentence, tokens, features)
        self._incomplete_sentence = ""  # Incomplete sentence from previous window
        self._chunk_index = 0
        self._char_offset = 0

    async def process_window(self, window: StreamingWindow, config: ChunkConfig, is_final: bool = False) -> list[Chunk]:
        """
        Process a window using semantic grouping.

        Args:
            window: StreamingWindow containing the data
            config: Chunk configuration parameters
            is_final: Whether this is the final window

        Returns:
            List of chunks produced from this window
        """
        chunks: list[Chunk] = []

        # Get text from window
        text = window.decode_safe()
        if not text and not is_final:
            return chunks

        # Combine with incomplete sentence from previous window
        if self._incomplete_sentence:
            text = self._incomplete_sentence + text
            self._incomplete_sentence = ""

        # Extract sentences
        sentences = self._extract_sentences(text, is_final)

        # Add sentences to buffer with features
        for sentence in sentences:
            if not sentence.strip():
                continue

            # Calculate features for semantic similarity
            features = self._extract_features(sentence)
            tokens = self.count_tokens(sentence)

            self._sentence_buffer.append((sentence, tokens, features))

            # Check buffer constraints
            if len(self._sentence_buffer) >= self.MAX_SENTENCES:
                # Process buffered sentences
                chunk = await self._process_sentence_buffer(config, force_emit=True)
                if chunk:
                    chunks.append(chunk)

        # Check if we should emit based on semantic boundaries
        if self._should_emit_chunk():
            chunk = await self._process_sentence_buffer(config, force_emit=False)
            if chunk:
                chunks.append(chunk)

        # If final, process remaining buffer
        if is_final and self._sentence_buffer:
            chunk = await self._process_sentence_buffer(config, force_emit=True)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _extract_sentences(self, text: str, is_final: bool) -> list[str]:
        """
        Extract complete sentences from text.

        Args:
            text: Text to extract sentences from
            is_final: Whether this is the final window

        Returns:
            List of complete sentences
        """
        sentences = []

        # Simple sentence splitting pattern
        sentence_pattern = r"[.!?]+[\s]+"

        # Split by sentence endings
        parts = re.split(sentence_pattern, text)

        # Find the actual delimiters
        delimiters = re.findall(sentence_pattern, text)

        # Reconstruct sentences with their delimiters
        for i, part in enumerate(parts[:-1]):
            if part.strip():
                # Add back the delimiter
                sentence = part + (delimiters[i].strip() if i < len(delimiters) else ".")
                sentences.append(sentence.strip())

        # Handle last part
        if parts and parts[-1].strip():
            last_part = parts[-1].strip()

            # Check if it's a complete sentence
            if is_final or last_part[-1] in ".!?":
                sentences.append(last_part)
            else:
                # Save incomplete sentence for next window
                self._incomplete_sentence = last_part

        return sentences

    def _extract_features(self, sentence: str) -> dict:
        """
        Extract semantic features from a sentence.

        Args:
            sentence: Sentence to analyze

        Returns:
            Dictionary of features for similarity comparison
        """
        features: dict[str, Any] = {
            "length": len(sentence.split()),
            "has_number": bool(re.search(r"\d", sentence)),
            "has_quote": '"' in sentence or "'" in sentence,
            "is_question": sentence.strip().endswith("?"),
            "is_exclamation": sentence.strip().endswith("!"),
            "starts_with_capital": sentence and sentence[0].isupper(),
            "keywords": set(),
        }

        # Extract key terms (simple approach - nouns and important words)
        words = sentence.lower().split()

        # Common stop words to exclude
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
        }

        # Extract keywords (words > 3 chars, not stop words)
        keywords = {word for word in words if len(word) > 3 and word not in stop_words}
        features["keywords"] = keywords

        return features

    def _calculate_similarity(self, features1: dict, features2: dict) -> float:
        """
        Calculate semantic similarity between two feature sets.

        Args:
            features1: First feature set
            features2: Second feature set

        Returns:
            Similarity score (0-1)
        """
        score = 0.0

        # Length similarity
        len_diff = abs(features1["length"] - features2["length"])
        len_similarity = 1.0 / (1.0 + len_diff / 10)
        score += len_similarity * 0.2

        # Type similarity (question, exclamation, etc.)
        if features1["is_question"] == features2["is_question"]:
            score += 0.1
        if features1["is_exclamation"] == features2["is_exclamation"]:
            score += 0.1

        # Feature similarity
        if features1["has_number"] == features2["has_number"]:
            score += 0.1
        if features1["has_quote"] == features2["has_quote"]:
            score += 0.1

        # Keyword overlap (most important)
        keywords1 = features1["keywords"]
        keywords2 = features2["keywords"]

        if keywords1 or keywords2:
            intersection = len(keywords1 & keywords2)
            union = len(keywords1 | keywords2)
            if union > 0:
                jaccard = intersection / union
                score += jaccard * 0.4

        return float(min(1.0, score))

    def _should_emit_chunk(self) -> bool:
        """
        Determine if we should emit a chunk based on semantic boundaries.

        Returns:
            True if a semantic boundary is detected
        """
        if len(self._sentence_buffer) < 3:
            return False

        # Check for semantic shift in recent sentences
        if len(self._sentence_buffer) >= 5:
            # Compare last 2 sentences with previous 3
            recent_features = [s[2] for s in self._sentence_buffer[-2:]]
            previous_features = [s[2] for s in self._sentence_buffer[-5:-2]]

            # Calculate average similarity
            total_similarity = 0.0
            comparisons = 0

            for recent in recent_features:
                for previous in previous_features:
                    total_similarity += self._calculate_similarity(recent, previous)
                    comparisons += 1

            if comparisons > 0:
                avg_similarity = total_similarity / comparisons

                # Emit if similarity is low (topic shift)
                if avg_similarity < 0.3:
                    return True

        # Check total token count
        total_tokens = sum(s[1] for s in self._sentence_buffer)
        if total_tokens >= self._state.get("target_tokens", 1000) * 0.9:
            return True

        return False

    async def _process_sentence_buffer(self, config: ChunkConfig, force_emit: bool = False) -> Chunk | None:
        """
        Process buffered sentences into a chunk.

        Args:
            config: Chunk configuration
            force_emit: Whether to force emission regardless of size

        Returns:
            Chunk if created, None otherwise
        """
        if not self._sentence_buffer:
            return None

        # Determine how many sentences to include
        if force_emit:
            # Take all sentences up to max tokens
            sentences_to_include = []
            total_tokens = 0

            for sentence, tokens, features in self._sentence_buffer:
                if total_tokens + tokens <= config.max_tokens:
                    sentences_to_include.append((sentence, tokens, features))
                    total_tokens += tokens
                else:
                    break
        else:
            # Find semantic boundary
            boundary_index = self._find_semantic_boundary(config)
            sentences_to_include = self._sentence_buffer[:boundary_index]

        if not sentences_to_include:
            return None

        # Create chunk from sentences
        chunk_text = " ".join(s[0] for s in sentences_to_include)
        chunk_text = self.clean_chunk_text(chunk_text)

        if not chunk_text:
            return None

        # Calculate semantic density based on feature coherence
        density = self._calculate_semantic_density(sentences_to_include)

        # Create metadata
        token_count = sum(s[1] for s in sentences_to_include)
        effective_min_tokens = min(config.min_tokens, token_count, 1)

        metadata = ChunkMetadata(
            chunk_id=str(uuid4()),
            document_id="doc",
            chunk_index=self._chunk_index,
            start_offset=self._char_offset,
            end_offset=self._char_offset + len(chunk_text),
            token_count=token_count,
            strategy_name=self.name,
            semantic_density=density,
            confidence_score=0.8,  # Good confidence for semantic strategy
            created_at=datetime.now(tz=UTC),
        )

        # Create chunk
        chunk = Chunk(
            content=chunk_text,
            metadata=metadata,
            min_tokens=effective_min_tokens,
            max_tokens=config.max_tokens,
        )

        # Update state
        self._chunk_index += 1
        self._char_offset += len(chunk_text)

        # Remove processed sentences and apply overlap
        self._sentence_buffer = self._sentence_buffer[len(sentences_to_include) :]

        # Keep last 2 sentences for context if we have overlap
        if config.overlap_tokens > 0 and len(sentences_to_include) > 2:
            overlap_sentences = sentences_to_include[-2:]
            self._sentence_buffer = overlap_sentences + self._sentence_buffer

        return chunk

    def _find_semantic_boundary(self, config: ChunkConfig) -> int:
        """
        Find the best semantic boundary in the buffer.

        Args:
            config: Chunk configuration

        Returns:
            Index of boundary
        """
        if len(self._sentence_buffer) <= 2:
            return len(self._sentence_buffer)

        best_boundary = len(self._sentence_buffer)
        best_score = 0.0
        total_tokens = 0

        for i in range(2, len(self._sentence_buffer)):
            total_tokens += self._sentence_buffer[i - 1][1]

            # Don't exceed max tokens
            if total_tokens > config.max_tokens:
                break

            # Calculate coherence before and after boundary
            before_coherence = self._calculate_group_coherence(self._sentence_buffer[:i])

            # Prefer boundaries near target size
            size_score = 1.0 - abs(total_tokens - config.max_tokens * 0.8) / config.max_tokens

            # Combined score
            score = before_coherence * 0.7 + size_score * 0.3

            if score > best_score:
                best_score = score
                best_boundary = i

        return best_boundary

    def _calculate_group_coherence(self, sentences: list[tuple[str, int, dict]]) -> float:
        """
        Calculate coherence of a group of sentences.

        Args:
            sentences: List of (sentence, tokens, features) tuples

        Returns:
            Coherence score (0-1)
        """
        if len(sentences) <= 1:
            return 1.0

        total_similarity = 0.0
        comparisons = 0

        for i in range(len(sentences) - 1):
            for j in range(i + 1, len(sentences)):
                similarity = self._calculate_similarity(sentences[i][2], sentences[j][2])
                total_similarity += similarity
                comparisons += 1

        return total_similarity / comparisons if comparisons > 0 else 0.5

    def _calculate_semantic_density(self, sentences: list[tuple[str, int, dict]]) -> float:
        """
        Calculate semantic density of a chunk.

        Args:
            sentences: Sentences in the chunk

        Returns:
            Density score (0-1)
        """
        if not sentences:
            return 0.5

        # Collect all keywords
        all_keywords = set()
        for _, _, features in sentences:
            all_keywords.update(features["keywords"])

        # Calculate density based on keyword diversity and coherence
        coherence = self._calculate_group_coherence(sentences)
        diversity = len(all_keywords) / max(1, sum(s[1] for s in sentences) / 4)

        # Balance coherence and diversity
        return coherence * 0.6 + min(1.0, diversity) * 0.4

    async def finalize(self, config: ChunkConfig) -> list[Chunk]:
        """
        Process any remaining buffered sentences.

        Args:
            config: Chunk configuration parameters

        Returns:
            List of final chunks
        """
        chunks: list[Chunk] = []

        # Process incomplete sentence if any
        if self._incomplete_sentence:
            sentences = self._extract_sentences(self._incomplete_sentence, is_final=True)
            for sentence in sentences:
                if sentence.strip():
                    features = self._extract_features(sentence)
                    tokens = self.count_tokens(sentence)
                    self._sentence_buffer.append((sentence, tokens, features))
            self._incomplete_sentence = ""

        # Process remaining buffer
        while self._sentence_buffer:
            chunk = await self._process_sentence_buffer(config, force_emit=True)
            if chunk:
                chunks.append(chunk)
            else:
                break

        self._is_finalized = True
        return chunks

    def get_buffer_size(self) -> int:
        """
        Return the current buffer size in bytes.

        Returns:
            Size of sentence buffer in bytes
        """
        size = 0

        # Sentence buffer
        for sentence, _, _ in self._sentence_buffer:
            size += len(sentence.encode("utf-8"))

        # Incomplete sentence
        if self._incomplete_sentence:
            size += len(self._incomplete_sentence.encode("utf-8"))

        return size

    def get_max_buffer_size(self) -> int:
        """
        Return the maximum allowed buffer size.

        Returns:
            50KB maximum buffer size
        """
        return self.MAX_BUFFER_SIZE

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self._sentence_buffer = []
        self._incomplete_sentence = ""
        self._chunk_index = 0
        self._char_offset = 0
        self._state = {"target_tokens": 1000}
