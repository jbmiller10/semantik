"""Keyword Extractor Plugin - lightweight keyword extraction using RAKE algorithm."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, ClassVar

from shared.plugins.manifest import AgentHints, PluginManifest
from shared.plugins.types.extractor import ExtractionResult, ExtractionType, ExtractorPlugin

logger = logging.getLogger(__name__)

# Common English stop words for RAKE algorithm
STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting on common delimiters
    return re.split(r"[.!?\n]+", text)


def _generate_candidate_keywords(sentences: list[str], stop_words: frozenset[str]) -> list[list[str]]:
    """Generate candidate keywords from sentences by splitting on stop words."""
    phrase_list = []

    for sentence in sentences:
        # Normalize and tokenize
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*(?:'[a-zA-Z]+)?\b", sentence.lower())

        # Build phrases by splitting on stop words
        current_phrase: list[str] = []
        for word in words:
            if word in stop_words:
                if current_phrase:
                    phrase_list.append(current_phrase)
                    current_phrase = []
            else:
                current_phrase.append(word)

        if current_phrase:
            phrase_list.append(current_phrase)

    return phrase_list


def _calculate_word_scores(phrase_list: list[list[str]]) -> dict[str, float]:
    """Calculate RAKE word scores based on degree and frequency."""
    word_frequency: Counter[str] = Counter()
    word_degree: Counter[str] = Counter()

    for phrase in phrase_list:
        degree = len(phrase) - 1  # Degree is phrase length minus 1
        for word in phrase:
            word_frequency[word] += 1
            word_degree[word] += degree

    # Score = degree / frequency
    word_scores: dict[str, float] = {}
    for word in word_frequency:
        word_scores[word] = (word_degree[word] + word_frequency[word]) / word_frequency[word]

    return word_scores


def _calculate_phrase_scores(
    phrase_list: list[list[str]],
    word_scores: dict[str, float],
) -> dict[str, float]:
    """Calculate scores for each phrase by summing word scores."""
    phrase_scores: dict[str, float] = {}

    for phrase in phrase_list:
        phrase_str = " ".join(phrase)
        if phrase_str not in phrase_scores:
            phrase_scores[phrase_str] = sum(word_scores.get(word, 0) for word in phrase)

    return phrase_scores


def extract_keywords_rake(
    text: str,
    top_k: int = 10,
    min_chars: int = 3,
    max_words: int = 4,
    stop_words: frozenset[str] | None = None,
) -> list[str]:
    """Extract keywords using RAKE (Rapid Automatic Keyword Extraction).

    Args:
        text: Input text to extract keywords from.
        top_k: Maximum number of keywords to return.
        min_chars: Minimum characters for a keyword.
        max_words: Maximum words in a keyphrase.
        stop_words: Custom stop words (uses default if None).

    Returns:
        List of keywords/keyphrases ordered by relevance.
    """
    if not text or not text.strip():
        return []

    if stop_words is None:
        stop_words = STOP_WORDS

    # Split into sentences
    sentences = _split_sentences(text)

    # Generate candidate keywords
    phrase_list = _generate_candidate_keywords(sentences, stop_words)

    # Filter by max words and min chars
    phrase_list = [phrase for phrase in phrase_list if len(phrase) <= max_words and len(" ".join(phrase)) >= min_chars]

    if not phrase_list:
        return []

    # Calculate word scores
    word_scores = _calculate_word_scores(phrase_list)

    # Calculate phrase scores
    phrase_scores = _calculate_phrase_scores(phrase_list, word_scores)

    # Sort by score and return top_k
    sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)

    return [phrase for phrase, _score in sorted_phrases[:top_k]]


class KeywordExtractorPlugin(ExtractorPlugin):
    """Keyword extraction plugin using RAKE algorithm.

    RAKE (Rapid Automatic Keyword Extraction) is a domain-independent
    keyword extraction algorithm that identifies key phrases in text
    based on word co-occurrence patterns.
    """

    PLUGIN_TYPE: ClassVar[str] = "extractor"
    PLUGIN_ID: ClassVar[str] = "keyword-extractor"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"

    AGENT_HINTS: ClassVar[AgentHints] = AgentHints(
        purpose="Extracts keywords and keyphrases from text for metadata enrichment. "
        "Enables keyword-based filtering and faceted search.",
        best_for=[
            "improving keyword search alongside semantic",
            "faceted filtering by topic",
            "document categorization",
            "extracting key concepts for metadata",
        ],
        not_recommended_for=[
            "when only semantic search is needed",
            "very short text snippets",
            "highly technical jargon (may miss domain terms)",
        ],
        input_types=["text/plain"],
        output_type="keywords",
        tradeoffs="Adds metadata overhead. Most useful when users will filter by keywords. "
        "RAKE algorithm is fast but may miss domain-specific terms.",
    )

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Keyword Extractor",
        "description": "Extract keywords and keyphrases using RAKE algorithm",
        "author": "Semantik",
        "license": "Apache-2.0",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize keyword extractor.

        Args:
            config: Plugin configuration with optional keys:
                - top_k: Maximum keywords to extract (default: 10)
                - min_chars: Minimum keyword length (default: 3)
                - max_words: Maximum words per keyphrase (default: 4)
        """
        super().__init__(config)
        self._top_k = self._config.get("top_k", 10)
        self._min_chars = self._config.get("min_chars", 3)
        self._max_words = self._config.get("max_words", 4)

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest."""
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            author=cls.METADATA.get("author"),
            license=cls.METADATA.get("license"),
            capabilities={
                "supported_extractions": ["keywords"],
            },
        )

    @classmethod
    def supported_extractions(cls) -> list[ExtractionType]:
        """Return supported extraction types."""
        return [ExtractionType.KEYWORDS]

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for plugin configuration."""
        return {
            "type": "object",
            "properties": {
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of keywords to extract",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                },
                "min_chars": {
                    "type": "integer",
                    "description": "Minimum characters for a keyword",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 3,
                },
                "max_words": {
                    "type": "integer",
                    "description": "Maximum words in a keyphrase",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 4,
                },
            },
        }

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        """Check if extractor is healthy (always true for this lightweight plugin)."""
        return True

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the extractor."""
        await super().initialize(config)

        # Update config from initialize call
        if config:
            self._top_k = config.get("top_k", self._top_k)
            self._min_chars = config.get("min_chars", self._min_chars)
            self._max_words = config.get("max_words", self._max_words)

        logger.info(
            "Keyword extractor initialized (top_k=%d, min_chars=%d, max_words=%d)",
            self._top_k,
            self._min_chars,
            self._max_words,
        )

    async def extract(
        self,
        text: str,
        extraction_types: list[ExtractionType] | None = None,
        options: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Extract keywords from text.

        Args:
            text: Text to extract from.
            extraction_types: Which extractions to perform (only KEYWORDS supported).
            options: Override options (top_k, min_chars, max_words).

        Returns:
            ExtractionResult with extracted keywords.
        """
        # Check if keywords extraction is requested
        if extraction_types is not None and ExtractionType.KEYWORDS not in extraction_types:
            return ExtractionResult()

        # Get options with fallback to config
        opts = options or {}
        top_k = opts.get("top_k", self._top_k)
        min_chars = opts.get("min_chars", self._min_chars)
        max_words = opts.get("max_words", self._max_words)

        # Extract keywords
        keywords = extract_keywords_rake(
            text=text,
            top_k=top_k,
            min_chars=min_chars,
            max_words=max_words,
        )

        return ExtractionResult(keywords=keywords)

    async def extract_batch(
        self,
        texts: list[str],
        extraction_types: list[ExtractionType] | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[ExtractionResult]:
        """Batch extract keywords from multiple texts."""
        return [await self.extract(text, extraction_types, options) for text in texts]
