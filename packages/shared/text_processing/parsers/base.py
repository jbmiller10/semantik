from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from .exceptions import ParserConfigError


@dataclass(frozen=True)
class ParsedElement:
    """A single parsed element from a document."""

    text: str
    """The text content of this element."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Element metadata (page_number, element_type, etc.)."""


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a document."""

    text: str
    """Concatenated text content."""

    elements: list[ParsedElement] = field(default_factory=list)
    """Individual parsed elements with metadata."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Document-level metadata."""


class BaseParser(ABC):
    """Base class for document parsers.

    Parsers extract text from documents. They must implement sync methods
    for compatibility with billiard.Pool parallel processing.

    Config is immutable after initialization for thread safety.

    Example:
        parser = UnstructuredParser({"strategy": "fast"})
        result = parser.parse_file("/path/to/doc.pdf")
        print(result.text)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize parser with configuration.

        Args:
            config: Parser-specific options.

        Raises:
            ParserConfigError: If config contains invalid options.
        """
        validated = self._validate_config(config or {})
        # Immutable config for thread safety
        self._config: Mapping[str, Any] = MappingProxyType(validated)

    @property
    def config(self) -> Mapping[str, Any]:
        """Read-only access to parser configuration."""
        return self._config

    def _validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate config against get_config_options().

        Override for custom validation. Default implementation checks:
        - All keys are recognized options
        - Values match expected types (boolean, select, text, number)
        - Select values are in the allowed options

        Args:
            config: Raw config dict.

        Returns:
            Validated config (may include defaults).

        Raises:
            ParserConfigError: If validation fails.
        """
        options = {opt["name"]: opt for opt in self.get_config_options()}
        result = {}

        # Apply defaults
        for name, opt in options.items():
            if "default" in opt:
                result[name] = opt["default"]

        # Validate provided config
        for key, value in config.items():
            if key not in options:
                valid_keys = ", ".join(sorted(options.keys())) or "(none)"
                raise ParserConfigError(f"Unknown config option '{key}'. Valid options: {valid_keys}")

            opt = options[key]
            opt_type = opt.get("type")

            # Type validation
            if opt_type == "boolean":
                if not isinstance(value, bool):
                    raise ParserConfigError(f"Config option '{key}' must be a boolean, got {type(value).__name__}")

            elif opt_type == "text":
                if not isinstance(value, str):
                    raise ParserConfigError(f"Config option '{key}' must be a string, got {type(value).__name__}")

            elif opt_type == "number":
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ParserConfigError(f"Config option '{key}' must be a number, got {type(value).__name__}")

            elif opt_type == "select":
                if not isinstance(value, str):
                    raise ParserConfigError(f"Config option '{key}' must be a string, got {type(value).__name__}")
                allowed_values = [o["value"] for o in opt.get("options", [])]
                if value not in allowed_values:
                    raise ParserConfigError(f"Config option '{key}' must be one of {allowed_values}, got '{value}'")

            result[key] = value

        return result

    @abstractmethod
    def parse_file(
        self,
        file_path: str,
        metadata: dict[str, Any] | None = None,
        *,
        include_elements: bool = False,
    ) -> ParseResult:
        """Parse document from file path.

        Must be sync for billiard.Pool compatibility.

        Note: This reads the entire file into memory. For very large files,
        consider streaming approaches or monitor memory usage.

        Args:
            file_path: Path to document file.
            metadata: Optional metadata to include in result.
            include_elements: Whether to populate ParseResult.elements.

        Returns:
            ParseResult with extracted text and metadata.

        Raises:
            UnsupportedFormatError: If file format is not supported.
            ExtractionFailedError: If extraction fails for other reasons.
        """
        ...

    @abstractmethod
    def parse_bytes(
        self,
        content: bytes,
        *,
        filename: str | None = None,
        file_extension: str | None = None,
        mime_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        include_elements: bool = False,
    ) -> ParseResult:
        """Parse document from bytes.

        Args:
            content: Raw document bytes.
            filename: Optional filename hint (used by some parsers for format detection).
            file_extension: Optional extension hint (e.g. ".pdf").
            mime_type: Optional MIME type hint (used by some parsers for format detection).
            metadata: Optional metadata to include in result.
            include_elements: Whether to populate ParseResult.elements.

        Returns:
            ParseResult with extracted text and metadata.

        Raises:
            UnsupportedFormatError: If format is not supported.
            ExtractionFailedError: If extraction fails.
        """
        ...

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> frozenset[str]:
        """File extensions this parser can handle (e.g., frozenset({".pdf", ".docx"}))."""
        ...

    @classmethod
    def supported_mime_types(cls) -> frozenset[str]:
        """MIME types this parser can handle. Optional."""
        return frozenset()

    @classmethod
    def get_config_options(cls) -> list[dict[str, Any]]:
        """Document config options this parser accepts.

        Connector authors can use this to build their get_config_fields().
        Each option dict should have: name, type, label, and optionally
        description, default, options (for select type).

        Returns:
            List of config field definitions.
        """
        return []
