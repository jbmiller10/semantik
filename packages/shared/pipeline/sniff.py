"""Pre-routing content sniffing for pipeline routing decisions.

This module provides content detection capabilities to enrich FileReference
metadata with detected characteristics (is_scanned_pdf, is_code, is_structured_data)
before routing decisions are made.

The sniff step runs after content loading but before routing, enabling
predicates to route based on detected file characteristics rather than
just surface-level metadata.

Example:
    >>> sniffer = ContentSniffer(SniffConfig())
    >>> result = await sniffer.sniff(content, file_ref)
    >>> sniffer.enrich_file_ref(file_ref, result)
    >>> # Now file_ref.metadata["detected"] contains sniff results
    >>> matches_predicate(file_ref, {"metadata.detected.is_scanned_pdf": True})

Caching:
    The sniffer supports optional result caching based on content_hash.
    This can significantly speed up reindexing operations where the same
    content is processed multiple times.

    >>> cache = SniffCache(maxsize=10000, ttl=3600)
    >>> sniffer = ContentSniffer(config, cache=cache)
    >>> result = await sniffer.sniff(content, file_ref, content_hash="abc123...")
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shared.pipeline.types import FileReference

logger = logging.getLogger(__name__)


class SniffCache:
    """Thread-safe LRU cache with TTL for sniff results.

    Caches SniffResult objects keyed by content_hash. Items are evicted
    when the cache reaches maxsize (LRU eviction) or when items exceed
    their TTL.

    Attributes:
        maxsize: Maximum number of items to store (default: 10000)
        ttl: Time-to-live in seconds for cached items (default: 3600)

    Example:
        >>> cache = SniffCache(maxsize=10000, ttl=3600)
        >>> cache.set("content_hash_123", sniff_result)
        >>> cached = cache.get("content_hash_123")
    """

    def __init__(self, maxsize: int = 10000, ttl: int = 3600) -> None:
        """Initialize the cache.

        Args:
            maxsize: Maximum number of items to store
            ttl: Time-to-live in seconds for cached items

        Raises:
            ValueError: If maxsize < 1 or ttl < 0
        """
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        if ttl < 0:
            raise ValueError(f"ttl cannot be negative, got {ttl}")
        self._maxsize = maxsize
        self._ttl = ttl
        self._cache: OrderedDict[str, tuple[SniffResult, float]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, content_hash: str) -> SniffResult | None:
        """Retrieve a cached sniff result.

        Args:
            content_hash: SHA-256 hash of the content

        Returns:
            Cached SniffResult if found and not expired, None otherwise
        """
        with self._lock:
            if content_hash not in self._cache:
                self._misses += 1
                return None

            result, timestamp = self._cache[content_hash]

            # Check TTL
            if time.time() - timestamp > self._ttl:
                del self._cache[content_hash]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(content_hash)
            self._hits += 1
            return result

    def set(self, content_hash: str, result: SniffResult) -> None:
        """Store a sniff result in the cache.

        Args:
            content_hash: SHA-256 hash of the content
            result: SniffResult to cache
        """
        with self._lock:
            # Remove oldest items if at capacity
            while len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)

            # Store with current timestamp
            self._cache[content_hash] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, and size
        """
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
            }


@dataclass
class SniffConfig:
    """Configuration for content sniffing.

    Attributes:
        timeout_seconds: Maximum time for all sniffing operations (default: 5.0)
        pdf_sample_pages: Number of PDF pages to sample for text detection (default: 3)
        structured_sample_bytes: Bytes to sample for structured data detection (default: 4096)
        enabled: Whether sniffing is enabled (default: True)
    """

    timeout_seconds: float = 5.0
    pdf_sample_pages: int = 3
    structured_sample_bytes: int = 4096
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        if self.pdf_sample_pages < 1:
            raise ValueError(f"pdf_sample_pages must be at least 1, got {self.pdf_sample_pages}")
        if self.structured_sample_bytes < 1:
            raise ValueError(f"structured_sample_bytes must be at least 1, got {self.structured_sample_bytes}")


@dataclass
class SniffResult:
    """Result of content sniffing.

    All detection fields return None if the detection doesn't apply
    (e.g., is_scanned_pdf is None for non-PDF files).

    Attributes:
        is_scanned_pdf: True if PDF with no/minimal text layer, None if not a PDF
        is_code: True if detected as source code
        is_structured_data: True if detected as structured data (JSON, CSV, etc.)
        structured_format: Format name if structured data ("json", "csv", "xml", "yaml")
        sniff_duration_ms: Time taken for sniffing in milliseconds
        errors: List of non-fatal errors encountered during sniffing
    """

    is_scanned_pdf: bool | None = None
    is_code: bool = False
    is_structured_data: bool = False
    structured_format: str | None = None
    sniff_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to a dictionary suitable for metadata.detected namespace.

        Only includes fields that have meaningful values (not None and not
        default False values unless explicitly detected).

        Returns:
            Dictionary with detected metadata fields
        """
        result: dict[str, Any] = {}

        # is_scanned_pdf is only set for PDFs
        if self.is_scanned_pdf is not None:
            result["is_scanned_pdf"] = self.is_scanned_pdf

        # Only include is_code if True (avoid cluttering metadata)
        if self.is_code:
            result["is_code"] = True

        # Only include structured data info if detected
        if self.is_structured_data:
            result["is_structured_data"] = True
            if self.structured_format:
                result["structured_format"] = self.structured_format

        return result


class ContentSniffer:
    """Content sniffer for pre-routing file characteristic detection.

    The sniffer analyzes file content to detect characteristics that can
    inform routing decisions, such as whether a PDF is scanned (needs OCR)
    or whether a file contains structured data.

    Supports optional caching of results by content_hash to speed up
    reindexing operations.

    Example:
        >>> config = SniffConfig(timeout_seconds=3.0)
        >>> cache = SniffCache(maxsize=10000, ttl=3600)
        >>> sniffer = ContentSniffer(config, cache=cache)
        >>> result = await sniffer.sniff(pdf_bytes, file_ref, content_hash="abc...")
        >>> if result.is_scanned_pdf:
        ...     print("PDF needs OCR processing")
    """

    # Code file extensions (reused from FileTypeDetector)
    CODE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
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
        ".mm",
        ".lua",
        ".dart",
        ".jsx",
        ".tsx",
        ".vue",
        ".sql",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
    }

    # Shebang patterns for code detection
    SHEBANG_PATTERN = re.compile(rb"^#!\s*/(?:usr/)?(?:bin|local/bin)/(?:env\s+)?(\w+)")

    # Syntax patterns for code detection (conservative patterns)
    CODE_SYNTAX_PATTERNS = [
        re.compile(rb"^(?:import|from)\s+\w+", re.MULTILINE),  # Python imports
        re.compile(rb"^(?:const|let|var)\s+\w+\s*=", re.MULTILINE),  # JS/TS vars
        re.compile(rb"^(?:func|fn|def|function)\s+\w+\s*\(", re.MULTILINE),  # Function defs
        re.compile(rb"^(?:class|struct|interface)\s+\w+", re.MULTILINE),  # Class/struct defs
        re.compile(rb"^package\s+\w+", re.MULTILINE),  # Go/Java package
        re.compile(rb"^use\s+\w+(?:::\w+)*;", re.MULTILINE),  # Rust use
        re.compile(rb"^#include\s*[<\"]", re.MULTILINE),  # C/C++ include
    ]

    # Minimum chars per page threshold for native PDF detection.
    # Rationale: 50 chars/page distinguishes native PDFs (with extractable text)
    # from scanned PDFs (image-only, no text layer). Native PDFs typically have
    # hundreds to thousands of chars per page. Scanned PDFs have 0-10 chars from
    # OCR artifacts or watermarks. The 50 threshold provides a comfortable margin
    # to catch PDFs with sparse text layers (e.g., forms with few fields filled).
    # If false positives occur (native PDFs misclassified as scanned), increase
    # this value. If false negatives occur (scanned PDFs not detected), decrease it.
    PDF_MIN_CHARS_PER_PAGE = 50

    def __init__(
        self,
        config: SniffConfig | None = None,
        cache: SniffCache | None = None,
    ) -> None:
        """Initialize the content sniffer.

        Args:
            config: Sniff configuration (uses defaults if not provided)
            cache: Optional cache for sniff results
        """
        self.config = config or SniffConfig()
        self._cache = cache

    async def sniff(
        self,
        content: bytes,
        file_ref: FileReference,
        content_hash: str | None = None,
    ) -> SniffResult:
        """Sniff content to detect file characteristics.

        Runs content detection algorithms to identify characteristics like
        scanned PDFs, source code, or structured data (JSON, CSV, etc.).
        Results are used to enrich FileReference metadata for routing decisions.

        Args:
            content: Raw file content bytes
            file_ref: File reference with metadata
            content_hash: Optional SHA-256 hash for cache lookup

        Returns:
            SniffResult with detected characteristics

        Error Handling:
            - **Timeout**: If sniffing exceeds ``config.timeout_seconds``, returns
              a partial SniffResult with the timeout recorded in ``errors``.
            - **Individual detector failures**: If a specific detector (PDF, code,
              structured data) fails, that detection is skipped and the error is
              recorded in ``errors``. Other detectors continue normally.
            - Sniff failures are non-fatal to the pipeline; routing proceeds with
              whatever metadata was successfully detected.

        Caching:
            Results are cached by ``content_hash`` when provided. Cache hits
            return immediately without re-running detection. Results with errors
            are NOT cached to allow retry on transient failures. Use caching
            during reindexing operations to avoid redundant detection.

        Example:
            >>> sniffer = ContentSniffer(SniffConfig(timeout_seconds=3.0))
            >>> result = await sniffer.sniff(pdf_bytes, file_ref, content_hash="abc...")
            >>> if result.is_scanned_pdf:
            ...     print("PDF needs OCR processing")
            >>> if result.errors:
            ...     print(f"Detection warnings: {result.errors}")
        """
        if not self.config.enabled:
            return SniffResult()

        # Check cache first
        if content_hash and self._cache:
            cached = self._cache.get(content_hash)
            if cached is not None:
                logger.debug("Sniff cache hit for %s", file_ref.uri)
                return cached

        start_time = time.perf_counter()
        result = SniffResult()

        try:
            # Wrap in timeout
            result = await asyncio.wait_for(
                self._do_sniff(content, file_ref, result),
                timeout=self.config.timeout_seconds,
            )
        except TimeoutError:
            result.errors.append(f"Sniff timed out after {self.config.timeout_seconds}s")
            logger.warning("Sniff timed out for %s", file_ref.uri)
        except Exception as e:
            result.errors.append(f"Sniff failed: {e}")
            logger.warning("Sniff failed for %s: %s", file_ref.uri, e)

        result.sniff_duration_ms = (time.perf_counter() - start_time) * 1000

        # Store in cache (only cache successful results without errors)
        if content_hash and self._cache and not result.errors:
            self._cache.set(content_hash, result)

        return result

    async def _do_sniff(
        self,
        content: bytes,
        file_ref: FileReference,
        result: SniffResult,
    ) -> SniffResult:
        """Perform actual sniffing operations.

        Args:
            content: Raw file content
            file_ref: File reference
            result: Result object to populate

        Returns:
            Populated SniffResult
        """
        mime_type = file_ref.mime_type or ""
        extension = file_ref.extension or ""

        # 1. PDF detection (only for PDFs)
        if self._is_pdf(mime_type, extension):
            try:
                result.is_scanned_pdf = self._detect_scanned_pdf(content)
            except Exception as e:
                result.errors.append(f"PDF detection failed: {e}")
                logger.debug("PDF detection failed for %s: %s", file_ref.uri, e)

        # 2. Code detection
        try:
            result.is_code = self._detect_code(content, extension)
        except Exception as e:
            result.errors.append(f"Code detection failed: {e}")
            logger.debug("Code detection failed for %s: %s", file_ref.uri, e)

        # 3. Structured data detection (skip for PDFs and known code files)
        if not self._is_pdf(mime_type, extension) and not result.is_code:
            try:
                is_structured, format_name = self._detect_structured_data(content)
                result.is_structured_data = is_structured
                result.structured_format = format_name
            except Exception as e:
                result.errors.append(f"Structured data detection failed: {e}")
                logger.debug("Structured data detection failed for %s: %s", file_ref.uri, e)

        return result

    def enrich_file_ref(self, file_ref: FileReference, sniff_result: SniffResult) -> None:
        """Enrich a FileReference with sniff results.

        Populates file_ref.metadata["detected"] with sniff results and
        file_ref.metadata["errors"]["sniff"] with any errors.

        Args:
            file_ref: FileReference to enrich (modified in place)
            sniff_result: Sniff results to add
        """
        detected_metadata = sniff_result.to_metadata_dict()
        if detected_metadata:
            if "detected" not in file_ref.metadata:
                file_ref.metadata["detected"] = {}
            file_ref.metadata["detected"].update(detected_metadata)

        # Also record any errors for visibility
        if sniff_result.errors:
            if "errors" not in file_ref.metadata:
                file_ref.metadata["errors"] = {}
            file_ref.metadata["errors"]["sniff"] = sniff_result.errors

    def _is_pdf(self, mime_type: str, extension: str) -> bool:
        """Check if file is a PDF.

        Args:
            mime_type: MIME type string
            extension: File extension

        Returns:
            True if file appears to be a PDF
        """
        return mime_type == "application/pdf" or extension.lower() == ".pdf"

    def _detect_scanned_pdf(self, content: bytes) -> bool:
        """Detect if a PDF is scanned (lacks text layer).

        Uses pypdf to extract text from the first few pages and checks
        if the average characters per page is below threshold.

        Args:
            content: PDF file content

        Returns:
            True if PDF appears to be scanned, False if has text layer

        Raises:
            ValueError: If all sampled pages fail text extraction
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.debug("pypdf not available for scanned PDF detection")
            raise

        reader = PdfReader(io.BytesIO(content))
        total_pages = len(reader.pages)

        if total_pages == 0:
            # Empty PDF - treat as scanned
            return True

        # Sample first N pages
        pages_to_check = min(self.config.pdf_sample_pages, total_pages)
        total_chars = 0
        successful_pages = 0
        failed_pages: list[tuple[int, str]] = []

        for i in range(pages_to_check):
            try:
                page = reader.pages[i]
                text = page.extract_text() or ""
                total_chars += len(text.strip())
                successful_pages += 1
            except Exception as e:
                failed_pages.append((i, str(e)))

        # If all pages failed, raise an error
        if successful_pages == 0:
            error_details = "; ".join(f"page {p}: {e}" for p, e in failed_pages)
            raise ValueError(f"All {pages_to_check} pages failed extraction: {error_details}")

        # Calculate average only over successfully extracted pages
        avg_chars_per_page = total_chars / successful_pages
        return avg_chars_per_page < self.PDF_MIN_CHARS_PER_PAGE

    def _detect_code(self, content: bytes, extension: str) -> bool:
        """Detect if content is source code.

        Uses extension, shebang, and syntax pattern detection.

        Args:
            content: File content
            extension: File extension

        Returns:
            True if detected as code
        """
        # 1. Check extension
        if extension.lower() in self.CODE_EXTENSIONS:
            return True

        # 2. Check for shebang (first line starting with #!)
        # Only check first 256 bytes for shebang
        first_bytes = content[:256]
        if self.SHEBANG_PATTERN.match(first_bytes):
            return True

        # 3. Check for code syntax patterns
        # Sample first 4KB for pattern matching
        sample = content[: self.config.structured_sample_bytes]
        return any(pattern.search(sample) for pattern in self.CODE_SYNTAX_PATTERNS)

    def _detect_structured_data(self, content: bytes) -> tuple[bool, str | None]:
        """Detect if content is structured data.

        Tries JSON, XML, YAML, and CSV detection in order.

        Args:
            content: File content

        Returns:
            Tuple of (is_structured, format_name)
        """
        # Sample content for detection
        sample = content[: self.config.structured_sample_bytes]

        # Skip binary content (check for null bytes in first chunk)
        if b"\x00" in sample[:512]:
            return False, None

        # Try decoding as text
        try:
            text_sample = sample.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            try:
                text_sample = sample.decode("latin-1")
            except Exception:
                return False, None

        text_sample = text_sample.strip()
        if not text_sample:
            return False, None

        # 1. JSON detection (most common)
        if self._is_json(text_sample, content):
            return True, "json"

        # 2. XML detection
        if self._is_xml(text_sample):
            return True, "xml"

        # 3. YAML detection
        if self._is_yaml(text_sample):
            return True, "yaml"

        # 4. CSV detection
        if self._is_csv(text_sample):
            return True, "csv"

        return False, None

    def _is_json(self, sample: str, full_content: bytes) -> bool:
        """Check if content is JSON.

        Args:
            sample: Text sample for quick check
            full_content: Full content for validation

        Returns:
            True if valid JSON
        """
        # Quick heuristic: starts with { or [
        first_char = sample.lstrip()[:1]
        if first_char not in ("{", "["):
            return False

        # Try to parse full content as JSON
        try:
            json.loads(full_content.decode("utf-8"))
            return True
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False

    def _is_xml(self, sample: str) -> bool:
        """Check if content is XML.

        Args:
            sample: Text sample

        Returns:
            True if appears to be XML
        """
        # Check for XML declaration or root element
        stripped = sample.lstrip()
        if stripped.startswith("<?xml"):
            return True

        # Check for HTML-style tags (but not HTML itself)
        if stripped.startswith(("<!DOCTYPE html", "<html")):
            return False

        # Check for opening tag pattern
        if re.match(r"<[a-zA-Z_][a-zA-Z0-9_:-]*(?:\s|>|/>)", stripped):
            return True

        return False

    def _is_yaml(self, sample: str) -> bool:
        """Check if content is YAML.

        Args:
            sample: Text sample

        Returns:
            True if appears to be YAML
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("PyYAML not installed, YAML detection disabled")
            return False

        try:
            # Try to parse as YAML
            # Use safe_load to avoid security issues
            result = yaml.safe_load(sample)

            # YAML parses plain strings too, so check if result is structured
            if isinstance(result, dict | list) and result:
                # Additional heuristic: check for YAML-specific patterns
                # to avoid false positives on plain text
                lines = sample.split("\n")
                yaml_indicators = 0

                for line in lines[:20]:  # Check first 20 lines
                    stripped = line.strip()
                    # Check for YAML patterns
                    if stripped.startswith(("---", "- ")) or re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*:\s*", stripped):
                        yaml_indicators += 1

                # Require at least 2 YAML indicators
                return yaml_indicators >= 2
        except yaml.YAMLError:
            return False
        except Exception as e:
            logger.debug("Unexpected error in YAML detection: %s", type(e).__name__)
            return False

        return False

    def _is_csv(self, sample: str) -> bool:
        """Check if content is CSV.

        Args:
            sample: Text sample

        Returns:
            True if appears to be CSV
        """
        try:
            # Use csv.Sniffer to detect CSV
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=",;\t|")

            # Additional validation: check for consistent column count
            lines = sample.split("\n")
            if len(lines) < 2:
                return False

            reader = csv.reader(lines, dialect)
            rows = list(reader)

            if len(rows) < 2:
                return False

            # Check that most rows have the same number of columns
            col_counts = [len(row) for row in rows if row]
            if not col_counts:
                return False

            # Most common column count
            from collections import Counter

            common_count = Counter(col_counts).most_common(1)[0][0]

            # At least 2 columns and 80% of rows match
            matching = sum(1 for c in col_counts if c == common_count)
            return common_count >= 2 and matching / len(col_counts) >= 0.8

        except csv.Error:
            # Expected: not valid CSV format
            return False
        except Exception as e:
            logger.debug("Unexpected error in CSV detection: %s: %s", type(e).__name__, e)
            return False


__all__ = [
    "SniffConfig",
    "SniffResult",
    "SniffCache",
    "ContentSniffer",
]
