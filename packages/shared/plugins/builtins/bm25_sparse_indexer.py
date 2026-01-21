"""BM25 Sparse Indexer Plugin - Classic term-frequency based sparse retrieval.

This plugin implements the BM25 (Best Matching 25) algorithm for sparse vector
generation. BM25 is a probabilistic retrieval model that scores documents based on
term frequency and inverse document frequency (TF-IDF variant).

The plugin generates sparse vectors where:
- Indices are term IDs from a vocabulary mapping
- Values are BM25 scores (TF-IDF with saturation)

BM25 Formula:
    score(D, Q) = sum( IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D|/avgdl)) )

Where:
- f(qi, D) = term frequency of qi in document D
- |D| = document length
- avgdl = average document length in collection
- k1 = term saturation parameter (default: 1.5)
- b = document length normalization (default: 0.75)
- IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
- N = total documents
- n(qi) = documents containing term qi
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import math
import re
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from shared.plugins.manifest import PluginManifest
from shared.plugins.types.sparse_indexer import (
    SparseIndexerCapabilities,
    SparseIndexerPlugin,
    SparseQueryVector,
    SparseVector,
)

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# Default BM25 parameters
DEFAULT_K1 = 1.5  # Term saturation - higher = more weight to term frequency
DEFAULT_B = 0.75  # Length normalization - 0 = no normalization, 1 = full normalization

# Default stopwords (English)
# We use a minimal set to avoid dependencies on NLTK
ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "can",
        "should",
        "now",
        "or",
        "if",
        "then",
        "also",
        "been",
        "being",
        "would",
        "could",
        "does",
        "did",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "because",
        "while",
    }
)


class BM25SparseIndexerPlugin(SparseIndexerPlugin):
    """BM25 sparse indexer plugin.

    Generates sparse vectors using the BM25 algorithm. BM25 is a bag-of-words
    model that scores terms based on their frequency in a document, normalized
    by document length and inverse document frequency across the corpus.

    Configuration options:
        k1: Term saturation parameter (default: 1.5)
        b: Document length normalization (default: 0.75)
        lowercase: Convert text to lowercase (default: True)
        remove_stopwords: Remove English stopwords (default: True)
        min_token_length: Minimum token length to include (default: 2)
        idf_path: Path to store IDF statistics (optional, auto-generated)
        collection_name: Collection name for IDF file path (optional)

    Usage:
        plugin = BM25SparseIndexerPlugin()
        await plugin.initialize({"k1": 1.5, "b": 0.75})

        vectors = await plugin.encode_documents([
            {"content": "hello world", "chunk_id": "chunk-1"},
        ])

        query_vector = await plugin.encode_query("hello")
    """

    PLUGIN_TYPE: ClassVar[str] = "sparse_indexer"
    PLUGIN_ID: ClassVar[str] = "bm25-local"
    PLUGIN_VERSION: ClassVar[str] = "1.0.0"
    SPARSE_TYPE: ClassVar[str] = "bm25"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "BM25 Sparse Indexer",
        "description": "Classic BM25 term-frequency based sparse retrieval with configurable parameters",
        "author": "Semantik",
        "license": "Apache-2.0",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the BM25 plugin.

        Args:
            config: Plugin configuration. See class docstring for options.
        """
        super().__init__(config)

        # BM25 parameters
        self._k1 = self._config.get("k1", DEFAULT_K1)
        self._b = self._config.get("b", DEFAULT_B)

        # Tokenization settings
        self._lowercase = self._config.get("lowercase", True)
        self._remove_stopwords = self._config.get("remove_stopwords", True)
        self._min_token_length = self._config.get("min_token_length", 2)

        # IDF state
        self._term_to_id: dict[str, int] = {}
        self._term_doc_freqs: dict[str, int] = {}
        self._document_count = 0
        self._avg_doc_length = 0.0
        self._total_doc_length = 0  # For computing average

        # Chunk tracking for remove_documents
        self._chunk_terms: dict[str, set[str]] = {}  # chunk_id -> set of terms
        self._chunk_lengths: dict[str, int] = {}  # chunk_id -> token count

        # Thread safety for IDF updates
        self._idf_lock = threading.Lock()

        # IDF persistence
        self._idf_path: Path | None = None
        self._idf_version = 0
        self._dirty = False  # Track if IDF needs saving

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest for discovery and UI."""
        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=cls.METADATA["display_name"],
            description=cls.METADATA["description"],
            author=cls.METADATA.get("author"),
            license=cls.METADATA.get("license"),
            capabilities={
                "sparse_type": cls.SPARSE_TYPE,
                "max_tokens": 8192,
                "vocabulary_handling": "direct",
                "supports_batching": True,
                "max_batch_size": 100,
                "requires_corpus_stats": True,
                "idf_storage": "file",
            },
        )

    @classmethod
    def get_capabilities(cls) -> SparseIndexerCapabilities:
        """Return sparse indexer capabilities."""
        return SparseIndexerCapabilities(
            sparse_type=cls.SPARSE_TYPE,
            max_tokens=8192,
            vocabulary_handling="direct",
            supports_batching=True,
            max_batch_size=100,
            requires_corpus_stats=True,
            idf_storage="file",
        )

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Return JSON Schema for plugin configuration."""
        return {
            "type": "object",
            "properties": {
                "k1": {
                    "type": "number",
                    "description": "BM25 term saturation parameter. Higher values give more weight to term frequency.",
                    "minimum": 0.0,
                    "maximum": 10.0,
                    "default": DEFAULT_K1,
                },
                "b": {
                    "type": "number",
                    "description": "BM25 document length normalization. 0 = no normalization, 1 = full normalization.",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": DEFAULT_B,
                },
                "lowercase": {
                    "type": "boolean",
                    "description": "Convert text to lowercase before tokenization.",
                    "default": True,
                },
                "remove_stopwords": {
                    "type": "boolean",
                    "description": "Remove common English stopwords.",
                    "default": True,
                },
                "min_token_length": {
                    "type": "integer",
                    "description": "Minimum token length to include.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 2,
                },
                "collection_name": {
                    "type": "string",
                    "description": "Collection name for IDF file path. If not provided, IDF stats are not persisted.",
                },
            },
        }

    @classmethod
    async def health_check(cls, config: dict[str, Any] | None = None) -> bool:  # noqa: ARG003
        """Check if the BM25 plugin is healthy.

        BM25 has no external dependencies, so it's always healthy.
        """
        return True

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the plugin with configuration.

        Loads IDF statistics from file if available.

        Args:
            config: Plugin configuration.
        """
        await super().initialize(config)

        # Update config-dependent settings
        if config:
            self._k1 = config.get("k1", self._k1)
            self._b = config.get("b", self._b)
            self._lowercase = config.get("lowercase", self._lowercase)
            self._remove_stopwords = config.get("remove_stopwords", self._remove_stopwords)
            self._min_token_length = config.get("min_token_length", self._min_token_length)

            # Set up IDF path based on collection name
            collection_name = config.get("collection_name")
            if collection_name:
                self._idf_path = self._get_idf_path(collection_name)
                await self._load_idf_stats()

        logger.info(
            "BM25 plugin initialized (k1=%.2f, b=%.2f, lowercase=%s, stopwords=%s)",
            self._k1,
            self._b,
            self._lowercase,
            self._remove_stopwords,
        )

    async def cleanup(self) -> None:
        """Clean up plugin resources.

        Persists IDF statistics if modified.
        """
        if self._dirty and self._idf_path:
            await self._save_idf_stats()

        await super().cleanup()

    async def encode_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[SparseVector]:
        """Generate sparse vectors for documents.

        Updates IDF statistics incrementally as documents are encoded.

        Args:
            documents: List of documents with 'content', 'chunk_id', and optional 'metadata'.

        Returns:
            List of SparseVector instances, one per document.
        """
        if not documents:
            return []

        results = []

        with self._idf_lock:
            for doc in documents:
                content = doc.get("content", "")
                chunk_id = doc.get("chunk_id", "")
                metadata = doc.get("metadata", {})

                # Tokenize
                tokens = self._tokenize(content)
                doc_length = len(tokens)

                if doc_length == 0:
                    # Empty document - return empty sparse vector
                    results.append(
                        SparseVector(
                            indices=(),
                            values=(),
                            chunk_id=chunk_id,
                            metadata=metadata,
                        )
                    )
                    continue

                # Update IDF statistics
                self._update_idf_for_document(tokens, chunk_id, doc_length)

                # Compute BM25 sparse vector
                indices, values = self._compute_bm25_vector(tokens, doc_length)

                results.append(
                    SparseVector(
                        indices=tuple(indices),
                        values=tuple(values),
                        chunk_id=chunk_id,
                        metadata=metadata,
                    )
                )

            self._dirty = True

        return results

    async def encode_query(self, query: str) -> SparseQueryVector:
        """Generate sparse vector for a search query.

        Query encoding uses the same tokenization but different scoring.
        For queries, we use simple TF-IDF without length normalization.

        Args:
            query: Search query text.

        Returns:
            SparseQueryVector with indices and values.
        """
        tokens = self._tokenize(query)

        if not tokens:
            return SparseQueryVector(indices=(), values=())

        # Count term frequencies in query
        term_freqs: dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        indices = []
        values = []

        with self._idf_lock:
            for term, tf in term_freqs.items():
                if term not in self._term_to_id:
                    # Unknown term - skip (not in vocabulary)
                    continue

                term_id = self._term_to_id[term]
                idf = self._compute_idf(term)

                # Query weight: TF * IDF (no length normalization)
                weight = tf * idf

                if weight > 0:
                    indices.append(term_id)
                    values.append(weight)

        # Sort by indices for consistent ordering
        if indices:
            sorted_pairs = sorted(zip(indices, values, strict=True))
            sorted_indices, sorted_values = zip(*sorted_pairs, strict=True)
            return SparseQueryVector(
                indices=tuple(sorted_indices),
                values=tuple(sorted_values),
            )

        return SparseQueryVector(indices=(), values=())

    async def remove_documents(self, chunk_ids: list[str]) -> None:
        """Update IDF statistics when chunks are removed.

        Decrements term document frequencies and document count for
        the terms that appeared in the removed chunks.

        Args:
            chunk_ids: List of chunk IDs being removed.
        """
        if not chunk_ids:
            return

        with self._idf_lock:
            for chunk_id in chunk_ids:
                if chunk_id not in self._chunk_terms:
                    # Chunk not in our tracking - skip
                    continue

                terms = self._chunk_terms[chunk_id]
                doc_length = self._chunk_lengths.get(chunk_id, 0)

                # Decrement document frequencies
                for term in terms:
                    if term in self._term_doc_freqs:
                        self._term_doc_freqs[term] -= 1
                        if self._term_doc_freqs[term] <= 0:
                            del self._term_doc_freqs[term]

                # Update document count and average length
                if self._document_count > 0:
                    self._total_doc_length -= doc_length
                    self._document_count -= 1
                    if self._document_count > 0:
                        self._avg_doc_length = self._total_doc_length / self._document_count
                    else:
                        self._avg_doc_length = 0.0

                # Clean up tracking
                del self._chunk_terms[chunk_id]
                if chunk_id in self._chunk_lengths:
                    del self._chunk_lengths[chunk_id]

            self._dirty = True

        logger.debug("Removed %d chunks from BM25 IDF stats", len(chunk_ids))

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        if not text:
            return []

        # Lowercase if configured
        if self._lowercase:
            text = text.lower()

        # Simple word tokenization using regex
        # Matches word characters, allows hyphens within words
        tokens = re.findall(r"\b[\w]+(?:-[\w]+)*\b", text)

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self._min_token_length]

        # Remove stopwords if configured
        if self._remove_stopwords:
            tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]

        return tokens

    def _update_idf_for_document(
        self,
        tokens: list[str],
        chunk_id: str,
        doc_length: int,
    ) -> None:
        """Update IDF statistics for a new document.

        Must be called within _idf_lock.

        Args:
            tokens: Document tokens.
            chunk_id: Chunk identifier.
            doc_length: Number of tokens in document.
        """
        # Get unique terms in this document
        unique_terms = set(tokens)

        # Check if this is a re-indexing of an existing chunk
        if chunk_id in self._chunk_terms:
            # Remove old stats first
            old_terms = self._chunk_terms[chunk_id]
            old_length = self._chunk_lengths.get(chunk_id, 0)

            for term in old_terms:
                if term in self._term_doc_freqs:
                    self._term_doc_freqs[term] -= 1
                    if self._term_doc_freqs[term] <= 0:
                        del self._term_doc_freqs[term]

            self._total_doc_length -= old_length
            self._document_count -= 1

        # Update term document frequencies
        for term in unique_terms:
            if term not in self._term_to_id:
                self._term_to_id[term] = len(self._term_to_id)
            self._term_doc_freqs[term] = self._term_doc_freqs.get(term, 0) + 1

        # Update document count and average length
        self._document_count += 1
        self._total_doc_length += doc_length
        self._avg_doc_length = self._total_doc_length / self._document_count

        # Track chunk terms for removal
        self._chunk_terms[chunk_id] = unique_terms
        self._chunk_lengths[chunk_id] = doc_length

    def _compute_idf(self, term: str) -> float:
        """Compute IDF for a term.

        Uses BM25's IDF formula:
            IDF(q) = log((N - n(q) + 0.5) / (n(q) + 0.5) + 1)

        Must be called within _idf_lock.

        Args:
            term: Term to compute IDF for.

        Returns:
            IDF value.
        """
        if self._document_count == 0:
            return 0.0

        doc_freq = self._term_doc_freqs.get(term, 0)
        if doc_freq == 0:
            return 0.0

        # BM25 IDF formula
        n = self._document_count
        idf = math.log((n - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        return max(0.0, idf)  # Ensure non-negative

    def _compute_bm25_vector(
        self,
        tokens: list[str],
        doc_length: int,
    ) -> tuple[list[int], list[float]]:
        """Compute BM25 sparse vector for a document.

        Must be called within _idf_lock.

        Args:
            tokens: Document tokens.
            doc_length: Number of tokens.

        Returns:
            Tuple of (indices, values) for sparse vector.
        """
        # Count term frequencies
        term_freqs: dict[str, int] = {}
        for token in tokens:
            term_freqs[token] = term_freqs.get(token, 0) + 1

        indices = []
        values = []

        # Length normalization factor
        length_norm = 1 - self._b + self._b * (doc_length / self._avg_doc_length) if self._avg_doc_length > 0 else 1.0

        for term, tf in term_freqs.items():
            term_id = self._term_to_id.get(term)
            if term_id is None:
                continue

            idf = self._compute_idf(term)
            if idf <= 0:
                continue

            # BM25 term weight
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * length_norm
            weight = idf * (numerator / denominator)

            if weight > 0:
                indices.append(term_id)
                values.append(weight)

        # Sort by indices for consistent ordering
        if indices:
            sorted_pairs = sorted(zip(indices, values, strict=True))
            indices_tuple, values_tuple = zip(*sorted_pairs, strict=True)
            return list(indices_tuple), list(values_tuple)

        return [], []

    def _get_idf_path(self, collection_name: str) -> Path:
        """Get the path for IDF statistics file.

        Args:
            collection_name: Collection name.

        Returns:
            Path to IDF stats file.
        """
        # Use data directory from environment or default
        import os

        data_dir = os.environ.get("SEMANTIK_DATA_DIR", "data")
        return Path(data_dir) / "sparse_indexes" / collection_name / "idf_stats.json"

    @contextmanager
    def _idf_file_lock(self, timeout: float = 30.0) -> Generator[None, None, None]:
        """Acquire exclusive file lock for IDF stats access.

        Uses fcntl file locking to prevent concurrent access from multiple
        Celery workers. The lock is automatically released when the context
        manager exits or if the process dies.

        Args:
            timeout: Maximum seconds to wait for lock acquisition.

        Yields:
            None when lock is acquired.

        Raises:
            TimeoutError: If lock cannot be acquired within timeout.
        """
        if self._idf_path is None:
            yield
            return

        lock_path = self._idf_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        lock_file = lock_path.open("w")  # noqa: SIM115 - need file handle for fcntl
        start = time.time()
        try:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.debug("Acquired IDF file lock for %s", self._idf_path)
                    break
                except OSError:
                    if time.time() - start > timeout:
                        raise TimeoutError(
                            f"Could not acquire IDF lock for {self._idf_path}. "
                            "Another process may be updating IDF stats."
                        ) from None
                    time.sleep(0.1)
            yield
        finally:
            with contextlib.suppress(Exception):
                # Best-effort cleanup: if the fd is already closed or unlock fails, we still
                # want to close the handle without masking the original exception.
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                lock_file.close()
            logger.debug("Released IDF file lock for %s", self._idf_path)

    async def _load_idf_stats(self) -> None:
        """Load IDF statistics from file.

        Uses file locking to prevent race conditions with concurrent workers.
        """
        if self._idf_path is None or not self._idf_path.exists():
            logger.debug("No IDF stats file found, starting fresh")
            return

        try:
            with self._idf_file_lock():
                # Re-check existence after acquiring lock (file may have been deleted)
                if not self._idf_path.exists():
                    logger.debug("IDF stats file disappeared after acquiring lock")
                    return

                data = json.loads(self._idf_path.read_text())

                with self._idf_lock:
                    self._term_to_id = data.get("term_to_id", {})
                    self._term_doc_freqs = data.get("term_doc_frequencies", {})
                    self._document_count = data.get("document_count", 0)
                    self._avg_doc_length = data.get("avg_doc_length", 0.0)
                    self._total_doc_length = int(self._avg_doc_length * self._document_count)
                    self._chunk_terms = {k: set(v) for k, v in data.get("chunk_terms", {}).items()}
                    self._chunk_lengths = data.get("chunk_lengths", {})
                    self._idf_version = data.get("version", 0)

                logger.info(
                    "Loaded IDF stats: %d documents, %d terms",
                    self._document_count,
                    len(self._term_to_id),
                )
        except TimeoutError:
            logger.warning("Timeout acquiring IDF file lock for load, starting fresh")
        except Exception as e:
            logger.warning("Failed to load IDF stats: %s", e)

    async def _save_idf_stats(self) -> None:
        """Save IDF statistics to file.

        Uses file locking to prevent race conditions with concurrent workers,
        and atomic write (temp file + rename) for crash safety.
        """
        if self._idf_path is None:
            return

        try:
            with self._idf_file_lock():
                with self._idf_lock:
                    data = {
                        "plugin_id": self.PLUGIN_ID,
                        "document_count": self._document_count,
                        "avg_doc_length": self._avg_doc_length,
                        "term_doc_frequencies": self._term_doc_freqs,
                        "term_to_id": self._term_to_id,
                        "chunk_terms": {k: list(v) for k, v in self._chunk_terms.items()},
                        "chunk_lengths": self._chunk_lengths,
                        "version": self._idf_version + 1,
                        "last_updated_at": datetime.now(UTC).isoformat(),
                    }
                    self._idf_version += 1

                # Ensure directory exists
                self._idf_path.parent.mkdir(parents=True, exist_ok=True)

                # Atomic write: write to temp file, then rename
                tmp_path = self._idf_path.with_suffix(".tmp")
                tmp_path.write_text(json.dumps(data, indent=2))
                tmp_path.rename(self._idf_path)

                self._dirty = False
                logger.debug(
                    "Saved IDF stats: %d documents, %d terms (version %d)",
                    self._document_count,
                    len(self._term_to_id),
                    self._idf_version,
                )
        except TimeoutError:
            logger.error("Timeout acquiring IDF file lock for save, stats not persisted")
        except Exception as e:
            logger.exception("Failed to save IDF stats: %s", e)
