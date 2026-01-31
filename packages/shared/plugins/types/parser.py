"""Parser plugin base class for document text extraction.

Parsers extract text from documents of various formats (PDF, DOCX, plain text, etc.)
and return structured results with metadata.

Key design decisions:
- parse_file/parse_bytes methods are SYNC for billiard.Pool compatibility
- Only lifecycle methods (initialize, cleanup) are async
- Config validation via get_config_options() for UI, get_config_schema() for JSON Schema
"""

from __future__ import annotations

import logging
import mimetypes
from abc import abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, NotRequired, TypedDict

from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import AgentHints, PluginManifest

logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================


class ParserError(Exception):
    """Base exception for all parser errors."""


class ParserConfigError(ParserError):
    """Raised when parser config validation fails.

    Example:
        raise ParserConfigError("Unknown config option 'stratgy'. Valid: strategy, include_page_breaks")
    """


class UnsupportedFormatError(ParserError):
    """Raised when a file format is not supported by the parser.

    This is an "expected" error - the file simply can't be handled.
    Pipeline may try a different parser or skip the file.

    Example:
        raise UnsupportedFormatError("TextParser cannot handle .pdf files")
    """


class ExtractionFailedError(ParserError):
    """Raised when extraction fails for reasons other than format support.

    This indicates something went wrong during parsing - corrupt file,
    missing dependencies, resource exhaustion, etc.

    Example:
        raise ExtractionFailedError("PDF parsing failed: file appears corrupted")
    """

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause


# ============================================================================
# Data Classes
# ============================================================================


@dataclass(frozen=True)
class ParsedElement:
    """A single parsed element from a document.

    Represents a logical unit of content (paragraph, table cell, etc.)
    with associated metadata like page number or element type.
    """

    text: str
    """The text content of this element."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Element metadata (page_number, element_type, etc.)."""


@dataclass(frozen=True)
class ParserOutput:
    """Result of parsing a document.

    Contains the concatenated text, optional individual elements,
    and document-level metadata.
    """

    text: str
    """Concatenated text content."""

    elements: list[ParsedElement] = field(default_factory=list)
    """Individual parsed elements with metadata (populated if include_elements=True)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Document-level metadata (parser, filename, mime_type, etc.)."""


class ParsedMetadata(TypedDict, total=False):
    """Standardized parsed metadata fields.

    These fields are written to FileReference.metadata["parsed"] and can be
    used for mid-pipeline routing decisions.

    All fields are optional (NotRequired) as parsers emit only fields they can compute.
    """

    # Document structure
    page_count: NotRequired[int]
    """Number of pages (PDF, DOCX)"""

    line_count: NotRequired[int]
    """Number of text lines"""

    # Content characteristics
    has_tables: NotRequired[bool]
    """Contains table elements"""

    has_images: NotRequired[bool]
    """Contains images"""

    has_code_blocks: NotRequired[bool]
    """Contains markdown code fences"""

    # Language and tokens
    detected_language: NotRequired[str | None]
    """ISO 639-1 code (en, zh, etc.) or None if detection fails"""

    approx_token_count: NotRequired[int]
    """Approximate token count"""

    # Element metadata (unstructured parser)
    element_types: NotRequired[list[str]]
    """Unique element types found"""

    # Quality metrics (future OCR)
    text_quality: NotRequired[float]
    """0.0-1.0 OCR confidence score"""


# ============================================================================
# MIME Type Helpers
# ============================================================================

# Fallback MIME types for extensions not in Python's mimetypes module
# Covers common code file extensions and modern formats
EXTENSION_MIME_FALLBACKS: dict[str, str] = {
    # TypeScript/JSX
    ".ts": "text/typescript",
    ".tsx": "text/typescript-jsx",
    ".jsx": "text/javascript-jsx",
    # Modern languages
    ".rs": "text/rust",
    ".go": "text/x-go",
    ".kt": "text/x-kotlin",
    ".swift": "text/x-swift",
    ".scala": "text/x-scala",
    # Data/config
    ".toml": "application/toml",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".graphql": "application/graphql",
    ".gql": "application/graphql",
    # Web
    ".vue": "text/x-vue",
    ".svelte": "text/x-svelte",
    ".scss": "text/x-scss",
    ".sass": "text/x-sass",
    ".less": "text/x-less",
    # Documentation
    ".mdx": "text/mdx",
    ".adoc": "text/asciidoc",
    ".rst": "text/x-rst",
    # Shell
    ".zsh": "text/x-shellscript",
    ".bash": "text/x-shellscript",
    ".fish": "text/x-shellscript",
    # Containers
    ".dockerfile": "text/x-dockerfile",
    # SQL
    ".sql": "application/sql",
    ".psql": "application/sql",
}


def derive_input_types_from_extensions(extensions: frozenset[str]) -> list[str]:
    """Derive MIME types from file extensions for AgentHints.input_types.

    Uses Python's mimetypes module with fallbacks for common code extensions
    that aren't in the standard database.

    Args:
        extensions: Set of file extensions (with leading dots, e.g., {".py", ".js"})

    Returns:
        Sorted list of unique MIME types
    """
    mime_types: set[str] = set()

    for ext in extensions:
        ext_lower = ext.lower()

        # Check our fallbacks FIRST - they handle edge cases where
        # mimetypes returns incorrect values (e.g., .ts = Qt translation, not TypeScript)
        if ext_lower in EXTENSION_MIME_FALLBACKS:
            mime_types.add(EXTENSION_MIME_FALLBACKS[ext_lower])
        else:
            # Try standard mimetypes
            mime, _ = mimetypes.guess_type(f"file{ext_lower}")
            if mime:
                mime_types.add(mime)
            else:
                # Generic fallback for unknown extensions
                # Use text/* for likely text files
                mime_types.add(f"text/x-{ext_lower.lstrip('.')}")

    return sorted(mime_types)


def convert_config_options_to_schema(options: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert get_config_options() list format to JSON Schema.

    Bridges the UI-focused config options format to standard JSON Schema
    for programmatic validation.

    Args:
        options: List of config option dicts from get_config_options()
                 Each dict has: name, type, label, and optionally default, options

    Returns:
        JSON Schema dict with properties, required, and type fields
    """
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for opt in options:
        name = opt["name"]
        opt_type = opt.get("type", "text")
        prop: dict[str, Any] = {}

        # Map UI types to JSON Schema types
        if opt_type == "boolean":
            prop["type"] = "boolean"
        elif opt_type == "number":
            prop["type"] = "number"
        elif opt_type == "select":
            # Enum from options list
            prop["type"] = "string"
            allowed_values = [o["value"] for o in opt.get("options", [])]
            if allowed_values:
                prop["enum"] = allowed_values
        else:  # text or default
            prop["type"] = "string"

        # Add description from label
        if "label" in opt:
            prop["description"] = opt["label"]

        # Add default if present
        if "default" in opt:
            prop["default"] = opt["default"]

        properties[name] = prop

        # Mark as required if no default (unless explicitly optional)
        if "default" not in opt and not opt.get("optional", False):
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


# ============================================================================
# Parser Plugin Base Class
# ============================================================================


class ParserPlugin(SemanticPlugin):
    """Base class for document parser plugins.

    Parsers extract text from documents. Methods are SYNC for compatibility
    with billiard.Pool parallel processing in Celery workers.

    Config is immutable after initialization for thread safety.

    Example:
        parser = MyParserPlugin({"strategy": "fast"})
        result = parser.parse_file("/path/to/doc.pdf")
        print(result.text)
    """

    PLUGIN_TYPE: ClassVar[str] = "parser"

    # Subclasses must define
    PLUGIN_ID: ClassVar[str]
    PLUGIN_VERSION: ClassVar[str]

    # Optional metadata for manifest generation
    METADATA: ClassVar[dict[str, Any]] = {}

    # Optional AgentHints for agent-driven selection
    AGENT_HINTS: ClassVar[AgentHints | None] = None

    # Parsers declare which parsed.* fields they emit for UI field discovery.
    # Used by the pipeline editor to show only relevant fields for routing predicates.
    EMITTED_FIELDS: ClassVar[list[str]] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize parser with configuration.

        Args:
            config: Parser-specific options.

        Raises:
            ParserConfigError: If config contains invalid options.
        """
        super().__init__(config)
        # Validate and make immutable for thread safety
        validated = self._validate_config(dict(self._config))
        self._immutable_config: MappingProxyType[str, Any] = MappingProxyType(validated)

    @property
    def config(self) -> dict[str, Any]:
        """Read-only access to parser configuration.

        Returns a dict copy for compatibility with SemanticPlugin base.
        """
        return dict(self._immutable_config)

    @config.setter
    def config(self, value: Any) -> None:
        """Disabled - parser config is immutable after __init__."""
        _ = value  # Unused but required for setter signature
        raise AttributeError("ParserPlugin config is immutable after initialization")

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
        result: dict[str, Any] = {}

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
    ) -> ParserOutput:
        """Parse document from file path.

        Must be SYNC for billiard.Pool compatibility.

        Note: This reads the entire file into memory. For very large files,
        consider streaming approaches or monitor memory usage.

        Args:
            file_path: Path to document file.
            metadata: Optional metadata to include in result.
            include_elements: Whether to populate ParserOutput.elements.

        Returns:
            ParserOutput with extracted text and metadata.

        Raises:
            UnsupportedFormatError: If file format is not supported.
            ExtractionFailedError: If extraction fails for other reasons.
        """

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
    ) -> ParserOutput:
        """Parse document from bytes.

        Must be SYNC for billiard.Pool compatibility.

        Args:
            content: Raw document bytes.
            filename: Optional filename hint (used for format detection).
            file_extension: Optional extension hint (e.g., ".pdf").
            mime_type: Optional MIME type hint.
            metadata: Optional metadata to include in result.
            include_elements: Whether to populate ParserOutput.elements.

        Returns:
            ParserOutput with extracted text and metadata.

        Raises:
            UnsupportedFormatError: If format is not supported.
            ExtractionFailedError: If extraction fails.
        """

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> frozenset[str]:
        """File extensions this parser can handle.

        Returns:
            Frozenset of extensions with leading dots (e.g., frozenset({".pdf", ".docx"}))
        """

    @classmethod
    def supported_mime_types(cls) -> frozenset[str]:
        """MIME types this parser can handle.

        Optional - defaults to deriving from supported_extensions().

        Returns:
            Frozenset of MIME type strings.
        """
        return frozenset()

    @classmethod
    def get_config_options(cls) -> list[dict[str, Any]]:
        """Document config options this parser accepts.

        Used by UI to build configuration forms. Each option dict should have:
        - name: Config key
        - type: "boolean", "text", "number", or "select"
        - label: Human-readable label
        - default: Default value (optional)
        - options: For select type, list of {value, label} dicts

        Returns:
            List of config field definitions.
        """
        return []

    @classmethod
    def get_config_schema(cls) -> dict[str, Any] | None:
        """Return JSON Schema for parser configuration.

        Defaults to converting get_config_options() to JSON Schema.
        Override for custom schema definitions.

        Returns:
            JSON Schema dict or None if no config options.
        """
        options = cls.get_config_options()
        if not options:
            return None
        return convert_config_options_to_schema(options)

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Return plugin manifest for discovery and UI.

        Builds a PluginManifest from the plugin's class variables and capabilities.
        Uses AGENT_HINTS if defined, otherwise generates from supported extensions.

        Returns:
            PluginManifest with parser metadata.
        """
        metadata = cls.METADATA or {}

        # Collect supported extensions
        extensions: frozenset[str] = frozenset()
        try:
            extensions = cls.supported_extensions()
        except (TypeError, NotImplementedError) as e:
            logger.debug("Could not get supported_extensions for %s: %s", cls.PLUGIN_ID, e)

        # Collect supported MIME types
        mime_types: frozenset[str] = frozenset()
        try:
            mime_types = cls.supported_mime_types()
        except (TypeError, NotImplementedError) as e:
            logger.debug("Could not get supported_mime_types for %s: %s", cls.PLUGIN_ID, e)

        # Build capabilities
        capabilities: dict[str, Any] = {}
        if extensions:
            capabilities["supported_extensions"] = sorted(extensions)
        if mime_types:
            capabilities["supported_mime_types"] = sorted(mime_types)

        # Get config options
        config_options = cls.get_config_options()
        if config_options:
            capabilities["config_options"] = config_options

        # Build or use AgentHints
        agent_hints = cls.AGENT_HINTS
        if agent_hints is None and extensions:
            # Auto-generate minimal AgentHints from extensions
            input_types = derive_input_types_from_extensions(extensions)
            agent_hints = AgentHints(
                purpose=metadata.get("description", f"Parse {cls.PLUGIN_ID} format documents"),
                best_for=[f"Files with extensions: {', '.join(sorted(extensions)[:5])}"],
                not_recommended_for=[],
                input_types=input_types if input_types else None,
                output_type="text",
            )

        return PluginManifest(
            id=cls.PLUGIN_ID,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=metadata.get("display_name", cls.PLUGIN_ID.title()),
            description=metadata.get("description", ""),
            author=metadata.get("author"),
            license=metadata.get("license"),
            homepage=metadata.get("homepage"),
            requires=list(metadata.get("requires", [])),
            semantik_version=metadata.get("semantik_version"),
            capabilities=capabilities,
            agent_hints=agent_hints,
        )


__all__ = [
    # Exceptions
    "ParserError",
    "ParserConfigError",
    "UnsupportedFormatError",
    "ExtractionFailedError",
    # Data classes
    "ParsedElement",
    "ParserOutput",
    "ParsedMetadata",
    # Helper functions
    "EXTENSION_MIME_FALLBACKS",
    "derive_input_types_from_extensions",
    "convert_config_options_to_schema",
    # Plugin base
    "ParserPlugin",
]
