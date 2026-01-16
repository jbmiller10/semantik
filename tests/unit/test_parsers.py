"""Unit tests for the parser system (Phases 1, 2, 7 & 8).

Tests cover:
1. parse_content() works for both bytes and str
2. Registry functions work with direct import
3. Selection rules and fallback behavior
4. Config validation (including type/value validation - Phase 8)
5. TextParser strictness (binary detection)
6. UnstructuredParser (mocked partition, config)
7. include_elements behavior
8. Metadata normalization (required keys always present)
9. Guardrails: selection rules stability, DEFAULT_PARSER_MAP validity,
   helper set consistency, and registry stability (Phase 7)
10. Phase 8 Contract Hardening:
    - Centralized normalization helpers
    - Protected metadata keys enforcement
    - Config validation tightened (types, values)
    - Registry API hygiene (immutable views)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.text_processing.parsers import (
    DEFAULT_PARSER_MAP,
    PROTECTED_METADATA_KEYS,
    ExtractionFailedError,
    ParsedElement,
    ParserConfigError,
    ParseResult,
    TextParser,
    UnsupportedFormatError,
    build_parser_metadata,
    get_default_parser_map,
    get_parser,
    get_parser_registry,
    list_parsers,
    normalize_extension,
    normalize_file_type,
    normalize_mime_type,
    parse_content,
    parser_candidates_for_extension,
    registry as registry_module,
)


class TestParseContent:
    """Tests for parse_content() convenience function."""

    def test_parse_content_with_str_input(self) -> None:
        """String input returns TextParser-style result immediately."""
        result = parse_content("Hello, world!", file_extension=".txt")

        assert result.text == "Hello, world!"
        assert result.metadata["parser"] == "text"
        assert result.metadata["file_extension"] == ".txt"
        assert result.metadata["file_type"] == "txt"
        assert result.elements == []  # include_elements defaults to False

    def test_parse_content_with_str_includes_elements(self) -> None:
        """String input with include_elements=True populates elements."""
        result = parse_content("Hello, world!", file_extension=".txt", include_elements=True)

        assert result.text == "Hello, world!"
        assert len(result.elements) == 1
        assert result.elements[0].text == "Hello, world!"

    def test_parse_content_with_str_empty_strips_elements(self) -> None:
        """Empty/whitespace-only string doesn't populate elements."""
        result = parse_content("   \n  ", file_extension=".txt", include_elements=True)

        assert result.text == "   \n  "
        assert result.elements == []

    def test_parse_content_with_bytes_text_extension(self) -> None:
        """Bytes input with text extension uses TextParser."""
        result = parse_content(b"Hello, world!", file_extension=".txt")

        assert result.text == "Hello, world!"
        assert result.metadata["parser"] == "text"

    def test_parse_content_with_bytes_code_extension(self) -> None:
        """Bytes input with code extension uses TextParser."""
        result = parse_content(b"def hello(): pass", file_extension=".py")

        assert result.text == "def hello(): pass"
        assert result.metadata["parser"] == "text"
        assert result.metadata["file_extension"] == ".py"
        assert result.metadata["file_type"] == "py"

    def test_parse_content_with_bytes_binary_extension_fallback(self) -> None:
        """Bytes input with binary extension falls back from unstructured to text on UnsupportedFormatError."""
        from shared.text_processing.parsers import UnstructuredParser

        # Mock UnstructuredParser.parse_bytes to raise UnsupportedFormatError
        # This simulates the case where unstructured can't handle this specific content
        def mock_parse_bytes(_self: Any, _content: bytes, **_kwargs: Any) -> ParseResult:
            raise UnsupportedFormatError("UnstructuredParser cannot handle this")

        with patch.object(UnstructuredParser, "parse_bytes", mock_parse_bytes):
            # This should first try unstructured (which raises UnsupportedFormatError), then text
            result = parse_content(b"<html><body>Hello</body></html>", file_extension=".html")

            # Should fall back to text parser
            assert result.text == "<html><body>Hello</body></html>"
            assert result.metadata["parser"] == "text"

    def test_parse_content_preserves_metadata(self) -> None:
        """Custom metadata is preserved in result."""
        result = parse_content(
            "test content",
            file_extension=".txt",
            filename="test.txt",
            metadata={"source_type": "test", "custom_key": "value"},
        )

        assert result.metadata["source_type"] == "test"
        assert result.metadata["custom_key"] == "value"
        assert result.metadata["filename"] == "test.txt"

    def test_parse_content_required_metadata_keys(self) -> None:
        """Required metadata keys are always present."""
        result = parse_content("test", file_extension=".txt")

        required_keys = ["filename", "file_extension", "file_type", "mime_type", "parser"]
        for key in required_keys:
            assert key in result.metadata, f"Missing required key: {key}"


class TestRegistry:
    """Tests for registry initialization and functions."""

    def test_list_parsers_works_after_clearing_registered(self) -> None:
        """list_parsers() works even after clearing _REGISTERED flag."""
        # Clear the flag to simulate fresh import
        original_registered = registry_module._REGISTERED
        original_registry = registry_module.PARSER_REGISTRY.copy()

        try:
            registry_module._REGISTERED = False
            registry_module.PARSER_REGISTRY.clear()

            # Should trigger ensure_registered() and return parsers
            parsers = list_parsers()
            assert "text" in parsers
            assert "unstructured" in parsers
        finally:
            # Restore original state
            registry_module._REGISTERED = original_registered
            registry_module.PARSER_REGISTRY.clear()
            registry_module.PARSER_REGISTRY.update(original_registry)

    def test_get_parser_triggers_registration(self) -> None:
        """get_parser() triggers registration automatically."""
        # Clear the flag to simulate fresh import
        original_registered = registry_module._REGISTERED
        original_registry = registry_module.PARSER_REGISTRY.copy()

        try:
            registry_module._REGISTERED = False
            registry_module.PARSER_REGISTRY.clear()

            # Should trigger ensure_registered() and return parser
            parser = get_parser("text")
            assert parser is not None
            assert isinstance(parser, TextParser)
        finally:
            # Restore original state
            registry_module._REGISTERED = original_registered
            registry_module.PARSER_REGISTRY.clear()
            registry_module.PARSER_REGISTRY.update(original_registry)

    def test_get_parser_unknown_raises_value_error(self) -> None:
        """get_parser() raises ValueError for unknown parser."""
        with pytest.raises(ValueError, match="Unknown parser: nonexistent"):
            get_parser("nonexistent")


class TestSelectionRulesAndFallback:
    """Tests for parser selection rules and fallback behavior."""

    def test_html_is_unstructured_first(self) -> None:
        """HTML extension defaults to unstructured-first."""
        candidates = parser_candidates_for_extension(".html")
        assert candidates[0] == "unstructured"
        assert "text" in candidates

    def test_txt_is_text_first(self) -> None:
        """Text extension defaults to text-first."""
        candidates = parser_candidates_for_extension(".txt")
        assert candidates[0] == "text"
        assert "unstructured" in candidates

    def test_py_is_text_first(self) -> None:
        """Python extension defaults to text-first."""
        candidates = parser_candidates_for_extension(".py")
        assert candidates[0] == "text"

    def test_pdf_is_unstructured_first(self) -> None:
        """PDF extension defaults to unstructured-first."""
        candidates = parser_candidates_for_extension(".pdf")
        assert candidates[0] == "unstructured"

    def test_unknown_extension_is_unstructured_first(self) -> None:
        """Unknown extension defaults to unstructured-first."""
        candidates = parser_candidates_for_extension(".xyz123")
        assert candidates[0] == "unstructured"
        assert "text" in candidates

    def test_no_extension_is_unstructured_first(self) -> None:
        """No extension defaults to unstructured-first."""
        candidates = parser_candidates_for_extension("")
        assert candidates[0] == "unstructured"
        assert "text" in candidates

    def test_overrides_take_precedence(self) -> None:
        """Overrides appear first in candidates."""
        candidates = parser_candidates_for_extension(".html", overrides={".html": "text"})
        assert candidates[0] == "text"

    def test_fallback_only_on_unsupported_format_error(self) -> None:
        """Fallback occurs only on UnsupportedFormatError, not ExtractionFailedError."""
        # Create binary content that will make TextParser fail
        binary_content = b"\x00\x01\x02\x03"

        # TextParser should raise UnsupportedFormatError for binary
        parser = TextParser()
        with pytest.raises(UnsupportedFormatError):
            parser.parse_bytes(binary_content, file_extension=".txt")

    def test_default_parser_map_coverage(self) -> None:
        """DEFAULT_PARSER_MAP has entries for common extensions."""
        # Text extensions
        assert DEFAULT_PARSER_MAP[".txt"] == "text"
        assert DEFAULT_PARSER_MAP[".md"] == "text"
        assert DEFAULT_PARSER_MAP[".py"] == "text"
        assert DEFAULT_PARSER_MAP[".json"] == "text"

        # Binary extensions
        assert DEFAULT_PARSER_MAP[".pdf"] == "unstructured"
        assert DEFAULT_PARSER_MAP[".docx"] == "unstructured"
        assert DEFAULT_PARSER_MAP[".html"] == "unstructured"


class TestConfigValidation:
    """Tests for parser config validation."""

    def test_unknown_config_key_raises_error(self) -> None:
        """Unknown config keys raise ParserConfigError."""
        with pytest.raises(ParserConfigError, match="Unknown config option 'unknown_key'"):
            TextParser({"unknown_key": "value"})

    def test_valid_config_keys_accepted(self) -> None:
        """Valid config keys are accepted."""
        parser = TextParser({"encoding": "utf-8", "errors": "replace"})
        assert parser.config["encoding"] == "utf-8"
        assert parser.config["errors"] == "replace"

    def test_defaults_applied_automatically(self) -> None:
        """Default values are applied when not provided."""
        parser = TextParser()
        assert parser.config["encoding"] == "utf-8"
        assert parser.config["errors"] == "replace"

    def test_config_is_immutable(self) -> None:
        """Config cannot be modified after initialization."""
        parser = TextParser()
        with pytest.raises(TypeError):
            parser.config["encoding"] = "latin-1"  # type: ignore[index]


class TestTextParserStrictness:
    """Tests for TextParser binary detection."""

    def test_nul_bytes_raise_unsupported_format_error(self) -> None:
        """NUL bytes in content raise UnsupportedFormatError."""
        parser = TextParser()
        content = b"Hello\x00World"

        with pytest.raises(UnsupportedFormatError, match="cannot decode binary"):
            parser.parse_bytes(content, file_extension=".txt")

    def test_high_non_printable_ratio_raises_unsupported_format_error(self) -> None:
        """High non-printable ratio raises UnsupportedFormatError."""
        parser = TextParser()
        # Create content with >30% non-printable bytes (excluding tab/newline)
        content = bytes([0x01] * 40 + [0x41] * 60)  # 40% control chars, 60% 'A'

        with pytest.raises(UnsupportedFormatError, match="cannot decode binary"):
            parser.parse_bytes(content, file_extension=".txt")

    def test_valid_text_parses_successfully(self) -> None:
        """Valid text content parses without error."""
        parser = TextParser()
        content = b"Hello, world!\nThis is valid text.\tWith tabs."

        result = parser.parse_bytes(content, file_extension=".txt")
        assert "Hello, world!" in result.text

    def test_text_with_allowed_control_chars(self) -> None:
        """Tab (0x09) and line feed/carriage return don't trigger binary detection."""
        parser = TextParser()
        content = b"Hello\tWorld\nNew line\r\nWindows line"

        result = parser.parse_bytes(content, file_extension=".txt")
        assert "Hello" in result.text


class TestUnstructuredParser:
    """Tests for UnstructuredParser (with mocked unstructured library)."""

    def test_unsupported_extension_raises_error(self) -> None:
        """Unsupported extension raises UnsupportedFormatError."""
        from shared.text_processing.parsers import UnstructuredParser

        parser = UnstructuredParser()
        # .xyz is not a supported extension
        with pytest.raises(UnsupportedFormatError, match="does not support"):
            parser.parse_bytes(b"content", file_extension=".xyz")

    def test_missing_unstructured_library_raises_extraction_failed(self) -> None:
        """Missing unstructured library raises ExtractionFailedError."""
        from shared.text_processing.parsers import UnstructuredParser

        parser = UnstructuredParser()

        # Mock the import to fail
        with patch.dict("sys.modules", {"unstructured.partition.auto": None}):
            # Force reimport by patching __import__
            original_import = __builtins__["__import__"]

            def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if "unstructured" in name:
                    raise ImportError("unstructured not installed")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(ExtractionFailedError, match="unstructured is not installed"):
                    parser.parse_bytes(b"content", file_extension=".pdf")

    def test_unstructured_parser_with_mocked_partition(self) -> None:
        """UnstructuredParser works with mocked partition function."""
        from shared.text_processing.parsers import UnstructuredParser

        parser = UnstructuredParser()

        # Create mock elements
        mock_element = MagicMock()
        mock_element.__str__ = MagicMock(return_value="Parsed text content")
        mock_element.metadata = MagicMock()
        mock_element.metadata.page_number = 1
        mock_element.metadata.category = "NarrativeText"

        # Patch the partition function at the import location (lazy imported inside parse_bytes)
        with patch("unstructured.partition.auto.partition") as mock_partition:
            mock_partition.return_value = [mock_element]

            result = parser.parse_bytes(b"PDF content", file_extension=".pdf")

            assert "Parsed text content" in result.text
            assert result.metadata["parser"] == "unstructured"

    def test_unstructured_parser_config_options(self) -> None:
        """UnstructuredParser accepts valid config options."""
        from shared.text_processing.parsers import UnstructuredParser

        parser = UnstructuredParser(
            {
                "strategy": "fast",
                "include_page_breaks": False,
                "infer_table_structure": False,
            }
        )

        assert parser.config["strategy"] == "fast"
        assert parser.config["include_page_breaks"] is False
        assert parser.config["infer_table_structure"] is False


class TestParsedDataclasses:
    """Tests for ParsedElement and ParseResult dataclasses."""

    def test_parsed_element_creation(self) -> None:
        """ParsedElement can be created with text and metadata."""
        element = ParsedElement(text="Hello", metadata={"page": 1})
        assert element.text == "Hello"
        assert element.metadata["page"] == 1

    def test_parsed_element_immutable(self) -> None:
        """ParsedElement is immutable (frozen)."""
        element = ParsedElement(text="Hello")
        with pytest.raises(AttributeError):
            element.text = "World"  # type: ignore[misc]

    def test_parse_result_creation(self) -> None:
        """ParseResult can be created with text, elements, and metadata."""
        result = ParseResult(
            text="Combined text",
            elements=[ParsedElement(text="Part 1")],
            metadata={"filename": "test.txt"},
        )
        assert result.text == "Combined text"
        assert len(result.elements) == 1
        assert result.metadata["filename"] == "test.txt"

    def test_parse_result_immutable(self) -> None:
        """ParseResult is immutable (frozen)."""
        result = ParseResult(text="Hello")
        with pytest.raises(AttributeError):
            result.text = "World"  # type: ignore[misc]


class TestIncludeElements:
    """Tests for include_elements behavior."""

    def test_include_elements_false_yields_empty_list(self) -> None:
        """include_elements=False yields empty elements list."""
        result = parse_content("Hello", file_extension=".txt", include_elements=False)
        assert result.elements == []

    def test_include_elements_true_populates_list(self) -> None:
        """include_elements=True populates elements list."""
        result = parse_content("Hello", file_extension=".txt", include_elements=True)
        assert len(result.elements) == 1
        assert result.elements[0].text == "Hello"

    def test_text_parser_include_elements(self) -> None:
        """TextParser respects include_elements parameter."""
        parser = TextParser()

        result_without = parser.parse_bytes(b"Hello", file_extension=".txt", include_elements=False)
        assert result_without.elements == []

        result_with = parser.parse_bytes(b"Hello", file_extension=".txt", include_elements=True)
        assert len(result_with.elements) == 1


class TestGuardrails:
    """Guardrail tests to prevent drift in parser selection rules and defaults.

    These tests lock down behavior that should NOT change without careful review.
    If any of these tests fail, it indicates a potentially breaking change to
    the parser system's selection logic or configuration.
    """

    # --- Selection Rules Stability Tests ---

    def test_html_extensions_always_unstructured_first(self) -> None:
        """HTML extensions must always prefer unstructured (guardrail)."""
        for ext in [".html", ".htm"]:
            candidates = parser_candidates_for_extension(ext)
            assert candidates[0] == "unstructured", f"{ext} should be unstructured-first"

    def test_unknown_extension_always_unstructured_first(self) -> None:
        """Unknown extensions must default to unstructured-first (guardrail)."""
        for ext in [".xyz", ".unknown123", ".foo"]:
            candidates = parser_candidates_for_extension(ext)
            assert candidates[0] == "unstructured", f"{ext} should default to unstructured"

    def test_no_extension_always_unstructured_first(self) -> None:
        """No extension must default to unstructured-first (guardrail)."""
        candidates = parser_candidates_for_extension("")
        assert candidates[0] == "unstructured"

    def test_text_extensions_always_text_first(self) -> None:
        """Known text extensions must always prefer text parser (guardrail)."""
        text_exts = [".txt", ".md", ".py", ".js", ".json", ".yaml"]
        for ext in text_exts:
            candidates = parser_candidates_for_extension(ext)
            assert candidates[0] == "text", f"{ext} should be text-first"

    # --- DEFAULT_PARSER_MAP Validity Tests ---

    def test_default_parser_map_references_registered_parsers(self) -> None:
        """All parsers in DEFAULT_PARSER_MAP must be registered (guardrail)."""
        registered = set(list_parsers())
        for ext, parser_name in DEFAULT_PARSER_MAP.items():
            assert parser_name in registered, f"{ext} maps to unregistered parser {parser_name}"

    def test_default_parser_map_text_entries_supported_by_text_parser(self) -> None:
        """Extensions mapped to 'text' must be in TextParser.supported_extensions() (guardrail)."""
        text_supported = TextParser.supported_extensions()
        for ext, parser_name in DEFAULT_PARSER_MAP.items():
            if parser_name == "text":
                assert ext in text_supported, f"{ext} mapped to text but not in TextParser.supported_extensions()"

    def test_default_parser_map_unstructured_entries_supported(self) -> None:
        """Extensions mapped to 'unstructured' must be in UnstructuredParser.supported_extensions() (guardrail)."""
        from shared.text_processing.parsers import UnstructuredParser

        unstructured_supported = UnstructuredParser.supported_extensions()
        for ext, parser_name in DEFAULT_PARSER_MAP.items():
            if parser_name == "unstructured":
                assert (
                    ext in unstructured_supported
                ), f"{ext} mapped to unstructured but not in UnstructuredParser.supported_extensions()"

    # --- Selection Helper Set Consistency Tests ---

    def test_text_first_extensions_derived_from_default_parser_map(self) -> None:
        """TEXT_FIRST_EXTENSIONS must match extensions mapped to 'text' in DEFAULT_PARSER_MAP (guardrail)."""
        from shared.text_processing.parsers.registry import TEXT_FIRST_EXTENSIONS

        expected = frozenset(ext for ext, parser in DEFAULT_PARSER_MAP.items() if parser == "text")
        assert expected == TEXT_FIRST_EXTENSIONS, "TEXT_FIRST_EXTENSIONS drifted from DEFAULT_PARSER_MAP"

    def test_unstructured_first_extensions_supported_by_unstructured(self) -> None:
        """All UNSTRUCTURED_FIRST_EXTENSIONS must be supported by UnstructuredParser (guardrail)."""
        from shared.text_processing.parsers import UnstructuredParser
        from shared.text_processing.parsers.registry import UNSTRUCTURED_FIRST_EXTENSIONS

        supported = UnstructuredParser.supported_extensions()
        for ext in UNSTRUCTURED_FIRST_EXTENSIONS:
            assert ext in supported, f"{ext} in UNSTRUCTURED_FIRST_EXTENSIONS but not supported by UnstructuredParser"

    # --- Registry Stability Tests ---

    def test_ensure_registered_is_idempotent(self) -> None:
        """Multiple calls to ensure_registered() must be idempotent (guardrail)."""
        from shared.text_processing.parsers.registry import PARSER_REGISTRY, ensure_registered

        ensure_registered()
        first_state = set(PARSER_REGISTRY.keys())

        ensure_registered()
        ensure_registered()
        second_state = set(PARSER_REGISTRY.keys())

        assert first_state == second_state, "ensure_registered() is not idempotent"

    def test_list_parsers_always_returns_builtins(self) -> None:
        """list_parsers() must always include built-in parsers (guardrail)."""
        parsers = list_parsers()
        assert "text" in parsers, "text parser missing from list_parsers()"
        assert "unstructured" in parsers, "unstructured parser missing from list_parsers()"

    def test_public_api_never_exposes_empty_registry(self) -> None:
        """Public registry APIs must never return empty results (guardrail)."""
        # These all call ensure_registered() internally
        assert len(list_parsers()) >= 2
        assert get_parser("text") is not None
        assert get_parser("unstructured") is not None


# =============================================================================
# Phase 8: Contract Hardening Tests
# =============================================================================


class TestNormalization:
    """Tests for centralized normalization helpers (Phase 8)."""

    # --- Extension Normalization ---

    def test_normalize_extension_with_dot(self) -> None:
        """Extension with dot is lowercased."""
        assert normalize_extension(".TXT") == ".txt"
        assert normalize_extension(".PDF") == ".pdf"
        assert normalize_extension(".Md") == ".md"

    def test_normalize_extension_without_dot(self) -> None:
        """Extension without dot gets one added."""
        assert normalize_extension("txt") == ".txt"
        assert normalize_extension("PDF") == ".pdf"
        assert normalize_extension("md") == ".md"

    def test_normalize_extension_empty_and_none(self) -> None:
        """Empty and None return empty string."""
        assert normalize_extension(None) == ""
        assert normalize_extension("") == ""
        assert normalize_extension("   ") == ""

    def test_normalize_extension_with_whitespace(self) -> None:
        """Whitespace is stripped."""
        assert normalize_extension("  .txt  ") == ".txt"
        assert normalize_extension("  pdf  ") == ".pdf"

    # --- File Type Normalization ---

    def test_normalize_file_type_removes_dot(self) -> None:
        """File type removes leading dot and lowercases."""
        assert normalize_file_type(".txt") == "txt"
        assert normalize_file_type(".PDF") == "pdf"
        assert normalize_file_type("md") == "md"

    def test_normalize_file_type_empty(self) -> None:
        """Empty string returns empty."""
        assert normalize_file_type("") == ""

    # --- MIME Type Normalization ---

    def test_normalize_mime_type_lowercase(self) -> None:
        """MIME type is lowercased."""
        assert normalize_mime_type("Application/PDF") == "application/pdf"
        assert normalize_mime_type("TEXT/HTML") == "text/html"

    def test_normalize_mime_type_strips_whitespace(self) -> None:
        """Whitespace is stripped."""
        assert normalize_mime_type("  text/plain  ") == "text/plain"

    def test_normalize_mime_type_none_and_empty(self) -> None:
        """None and empty return None."""
        assert normalize_mime_type(None) is None
        assert normalize_mime_type("") is None
        assert normalize_mime_type("   ") is None


class TestMetadataContract:
    """Tests for protected metadata key enforcement (Phase 8)."""

    def test_protected_keys_exist(self) -> None:
        """PROTECTED_METADATA_KEYS contains expected keys."""
        assert "parser" in PROTECTED_METADATA_KEYS
        assert "file_extension" in PROTECTED_METADATA_KEYS
        assert "file_type" in PROTECTED_METADATA_KEYS

    def test_build_parser_metadata_sets_required_keys(self) -> None:
        """build_parser_metadata sets all required keys."""
        result = build_parser_metadata(
            parser_name="text",
            filename="test.txt",
            file_extension=".txt",
            mime_type="text/plain",
        )

        assert result["parser"] == "text"
        assert result["filename"] == "test.txt"
        assert result["file_extension"] == ".txt"
        assert result["file_type"] == "txt"
        assert result["mime_type"] == "text/plain"

    def test_build_parser_metadata_cannot_override_parser(self) -> None:
        """Caller cannot override 'parser' key."""
        result = build_parser_metadata(
            parser_name="text",
            file_extension=".txt",
            caller_metadata={"parser": "malicious"},
        )

        assert result["parser"] == "text"

    def test_build_parser_metadata_cannot_override_file_extension(self) -> None:
        """Caller cannot override 'file_extension' key."""
        result = build_parser_metadata(
            parser_name="text",
            file_extension=".txt",
            caller_metadata={"file_extension": ".exe"},
        )

        assert result["file_extension"] == ".txt"

    def test_build_parser_metadata_cannot_override_file_type(self) -> None:
        """Caller cannot override 'file_type' key."""
        result = build_parser_metadata(
            parser_name="text",
            file_extension=".txt",
            caller_metadata={"file_type": "exe"},
        )

        assert result["file_type"] == "txt"

    def test_build_parser_metadata_preserves_custom_keys(self) -> None:
        """Custom metadata keys are preserved."""
        result = build_parser_metadata(
            parser_name="text",
            file_extension=".txt",
            caller_metadata={"source_type": "git", "custom_key": "value"},
        )

        assert result["source_type"] == "git"
        assert result["custom_key"] == "value"

    def test_parse_content_str_respects_protected_keys(self) -> None:
        """parse_content with str input cannot have protected keys overridden."""
        result = parse_content(
            "test content",
            file_extension=".txt",
            metadata={"parser": "malicious", "file_extension": ".exe"},
        )

        assert result.metadata["parser"] == "text"
        assert result.metadata["file_extension"] == ".txt"

    def test_text_parser_respects_protected_keys(self) -> None:
        """TextParser cannot have protected keys overridden."""
        parser = TextParser()
        result = parser.parse_bytes(
            b"test content",
            file_extension=".txt",
            metadata={"parser": "malicious", "file_extension": ".exe"},
        )

        assert result.metadata["parser"] == "text"
        assert result.metadata["file_extension"] == ".txt"


class TestConfigValidationTightened:
    """Tests for tightened config validation (Phase 8)."""

    def test_boolean_type_rejects_string_true(self) -> None:
        """Boolean config rejects string 'true'."""
        from shared.text_processing.parsers import UnstructuredParser

        with pytest.raises(ParserConfigError, match="must be a boolean"):
            UnstructuredParser({"include_page_breaks": "true"})

    def test_boolean_type_rejects_int_one(self) -> None:
        """Boolean config rejects integer 1."""
        from shared.text_processing.parsers import UnstructuredParser

        with pytest.raises(ParserConfigError, match="must be a boolean"):
            UnstructuredParser({"include_page_breaks": 1})

    def test_boolean_type_accepts_true_false(self) -> None:
        """Boolean config accepts True/False."""
        from shared.text_processing.parsers import UnstructuredParser

        parser = UnstructuredParser({"include_page_breaks": True})
        assert parser.config["include_page_breaks"] is True

        parser = UnstructuredParser({"include_page_breaks": False})
        assert parser.config["include_page_breaks"] is False

    def test_select_type_rejects_invalid_value(self) -> None:
        """Select config rejects values not in options."""
        from shared.text_processing.parsers import UnstructuredParser

        with pytest.raises(ParserConfigError, match="must be one of"):
            UnstructuredParser({"strategy": "invalid_strategy"})

    def test_select_type_rejects_non_string(self) -> None:
        """Select config rejects non-string values."""
        from shared.text_processing.parsers import UnstructuredParser

        with pytest.raises(ParserConfigError, match="must be a string"):
            UnstructuredParser({"strategy": 123})

    def test_select_type_accepts_valid_value(self) -> None:
        """Select config accepts valid option values."""
        from shared.text_processing.parsers import UnstructuredParser

        for strategy in ["auto", "fast", "hi_res", "ocr_only"]:
            parser = UnstructuredParser({"strategy": strategy})
            assert parser.config["strategy"] == strategy

    def test_text_type_rejects_non_string(self) -> None:
        """Text config rejects non-string values."""
        with pytest.raises(ParserConfigError, match="must be a string"):
            TextParser({"encoding": 123})

    def test_text_type_accepts_string(self) -> None:
        """Text config accepts string values."""
        parser = TextParser({"encoding": "utf-8"})
        assert parser.config["encoding"] == "utf-8"


class TestRegistryHygiene:
    """Tests for registry API hygiene (Phase 8)."""

    def test_get_parser_registry_returns_immutable(self) -> None:
        """get_parser_registry() returns immutable mapping."""
        registry = get_parser_registry()

        # Should not be able to modify
        with pytest.raises(TypeError):
            registry["new_parser"] = TextParser  # type: ignore[index]

    def test_get_parser_registry_always_initialized(self) -> None:
        """get_parser_registry() always returns initialized registry."""
        registry = get_parser_registry()

        assert "text" in registry
        assert "unstructured" in registry
        assert len(registry) >= 2

    def test_get_default_parser_map_returns_immutable(self) -> None:
        """get_default_parser_map() returns immutable mapping."""
        parser_map = get_default_parser_map()

        # Should not be able to modify
        with pytest.raises(TypeError):
            parser_map[".xyz"] = "text"  # type: ignore[index]

    def test_get_default_parser_map_contains_expected_entries(self) -> None:
        """get_default_parser_map() contains expected extensions."""
        parser_map = get_default_parser_map()

        assert parser_map[".txt"] == "text"
        assert parser_map[".pdf"] == "unstructured"
        assert parser_map[".md"] == "text"

    def test_get_parser_registry_after_clearing_flag(self) -> None:
        """get_parser_registry() works even after clearing _REGISTERED flag."""
        # Save original state
        original_registered = registry_module._REGISTERED
        original_registry = registry_module.PARSER_REGISTRY.copy()

        try:
            registry_module._REGISTERED = False
            registry_module.PARSER_REGISTRY.clear()

            # Should trigger ensure_registered() and return populated registry
            registry = get_parser_registry()
            assert "text" in registry
            assert "unstructured" in registry
        finally:
            # Restore original state
            registry_module._REGISTERED = original_registered
            registry_module.PARSER_REGISTRY.clear()
            registry_module.PARSER_REGISTRY.update(original_registry)

    def test_safe_accessors_match_raw_dicts(self) -> None:
        """Safe accessors return same data as raw dicts (but immutable)."""
        # Registry
        safe_registry = get_parser_registry()
        assert dict(safe_registry) == registry_module.PARSER_REGISTRY

        # Default parser map
        safe_map = get_default_parser_map()
        assert dict(safe_map) == registry_module.DEFAULT_PARSER_MAP


# =============================================================================
# Phase 9: BOM Encoding Support Tests
# =============================================================================


class TestBOMEncodingSupport:
    """Tests for BOM-marked text encoding support (Phase 9)."""

    def test_utf8_bom_detected_and_stripped(self) -> None:
        """UTF-8 BOM is detected and stripped from output."""
        # UTF-8 BOM: EF BB BF
        content = b"\xef\xbb\xbfHello"
        parser = TextParser()

        result = parser.parse_bytes(content, file_extension=".txt")

        assert result.text == "Hello"
        assert "Hello" in result.text
        # BOM should be stripped - no EF BB BF in output
        assert "\ufeff" not in result.text

    def test_utf16_le_bom_detected(self) -> None:
        """UTF-16-LE BOM is detected and decoded correctly."""
        # UTF-16-LE BOM: FF FE, then "Hello" in UTF-16-LE
        content = b"\xff\xfeH\x00e\x00l\x00l\x00o\x00"
        parser = TextParser()

        result = parser.parse_bytes(content, file_extension=".txt")

        assert result.text == "Hello"

    def test_utf16_be_bom_detected(self) -> None:
        """UTF-16-BE BOM is detected and decoded correctly."""
        # UTF-16-BE BOM: FE FF, then "Hello" in UTF-16-BE
        content = b"\xfe\xff\x00H\x00e\x00l\x00l\x00o"
        parser = TextParser()

        result = parser.parse_bytes(content, file_extension=".txt")

        assert result.text == "Hello"

    def test_utf32_le_bom_detected(self) -> None:
        """UTF-32-LE BOM is detected and decoded correctly."""
        # UTF-32-LE BOM: FF FE 00 00, then "Hi" in UTF-32-LE
        content = b"\xff\xfe\x00\x00H\x00\x00\x00i\x00\x00\x00"
        parser = TextParser()

        result = parser.parse_bytes(content, file_extension=".txt")

        assert result.text == "Hi"

    def test_utf32_be_bom_detected(self) -> None:
        """UTF-32-BE BOM is detected and decoded correctly."""
        # UTF-32-BE BOM: 00 00 FE FF, then "Hi" in UTF-32-BE
        content = b"\x00\x00\xfe\xff\x00\x00\x00H\x00\x00\x00i"
        parser = TextParser()

        result = parser.parse_bytes(content, file_extension=".txt")

        assert result.text == "Hi"

    def test_utf32_le_not_confused_with_utf16_le(self) -> None:
        """UTF-32-LE BOM (FF FE 00 00) not confused with UTF-16-LE (FF FE)."""
        # UTF-32-LE BOM starts with UTF-16-LE BOM, but we must detect UTF-32-LE
        # Content: UTF-32-LE BOM + "A" in UTF-32-LE
        content = b"\xff\xfe\x00\x00A\x00\x00\x00"
        parser = TextParser()

        result = parser.parse_bytes(content, file_extension=".txt")

        # Should decode as UTF-32-LE, not UTF-16-LE
        # If it wrongly used UTF-16-LE, we'd get garbage or error
        assert result.text == "A"

    def test_binary_without_bom_still_rejected(self) -> None:
        """Binary content without BOM is still rejected."""
        # Pure binary with NUL bytes (no BOM)
        content = b"Some text\x00binary\x00data"
        parser = TextParser()

        with pytest.raises(UnsupportedFormatError, match="cannot decode binary"):
            parser.parse_bytes(content, file_extension=".txt")

    def test_high_non_printable_ratio_still_rejected(self) -> None:
        """High non-printable ratio (no BOM) still rejected."""
        # Create content with >30% non-printable bytes (excluding tab/newline)
        content = bytes([0x01] * 40 + [0x41] * 60)  # 40% control chars, 60% 'A'
        parser = TextParser()

        with pytest.raises(UnsupportedFormatError, match="cannot decode binary"):
            parser.parse_bytes(content, file_extension=".txt")

    def test_parse_file_and_bytes_same_result(self, tmp_path: Any) -> None:
        """parse_file() and parse_bytes() produce identical results."""
        content = b"\xef\xbb\xbfHello World"  # UTF-8 BOM + text
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(content)

        parser = TextParser()
        result_bytes = parser.parse_bytes(content, filename="test.txt", file_extension=".txt")
        result_file = parser.parse_file(str(test_file))

        assert result_bytes.text == result_file.text
        assert result_bytes.text == "Hello World"

    def test_parse_file_rejects_binary(self, tmp_path: Any) -> None:
        """parse_file() rejects binary content just like parse_bytes()."""
        content = b"Some\x00binary\x00content"
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(content)

        parser = TextParser()
        with pytest.raises(UnsupportedFormatError, match="cannot decode binary"):
            parser.parse_file(str(test_file))


class TestBOMGuardrails:
    """Guardrail tests for BOM handling stability (Phase 9)."""

    def test_bom_detection_order_is_longest_first(self) -> None:
        """BOM detection must check longer BOMs before shorter ones."""
        from shared.text_processing.parsers.text import _BOM_ENCODINGS

        # Verify UTF-32 comes before UTF-16 (4 bytes before 2 bytes)
        encoding_order = [enc for _, enc in _BOM_ENCODINGS]

        utf32_le_idx = encoding_order.index("utf-32-le")
        utf32_be_idx = encoding_order.index("utf-32-be")
        utf16_le_idx = encoding_order.index("utf-16-le")
        utf16_be_idx = encoding_order.index("utf-16-be")

        # UTF-32 variants must come before UTF-16 variants
        assert utf32_le_idx < utf16_le_idx, "UTF-32-LE must be checked before UTF-16-LE"
        assert utf32_be_idx < utf16_be_idx, "UTF-32-BE must be checked before UTF-16-BE"

    def test_all_standard_boms_supported(self) -> None:
        """All standard Unicode BOMs are in detection list."""
        import codecs

        from shared.text_processing.parsers.text import _BOM_ENCODINGS

        boms_in_list = {bom for bom, _ in _BOM_ENCODINGS}

        # All standard BOMs should be present
        assert codecs.BOM_UTF8 in boms_in_list
        assert codecs.BOM_UTF16_LE in boms_in_list
        assert codecs.BOM_UTF16_BE in boms_in_list
        assert codecs.BOM_UTF32_LE in boms_in_list
        assert codecs.BOM_UTF32_BE in boms_in_list

    def test_bom_detect_function_returns_correct_format(self) -> None:
        """_detect_bom returns (encoding, length) tuple or None."""
        from shared.text_processing.parsers.text import _detect_bom

        # With BOM
        result = _detect_bom(b"\xef\xbb\xbfHello")
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "utf-8-sig"
        assert result[1] == 3  # UTF-8 BOM is 3 bytes

        # Without BOM
        result = _detect_bom(b"Hello")
        assert result is None

    def test_binary_check_function_is_bom_aware(self) -> None:
        """_is_binary_content correctly identifies BOM-marked files as text."""
        from shared.text_processing.parsers.text import _is_binary_content

        # UTF-16-LE with BOM - has NUL bytes but should NOT be binary
        utf16_content = b"\xff\xfeH\x00e\x00l\x00l\x00o\x00"
        assert _is_binary_content(utf16_content) is False

        # True binary with NUL bytes - should be binary
        binary_content = b"Hello\x00World"
        assert _is_binary_content(binary_content) is True

        # Pure text - should not be binary
        text_content = b"Hello World"
        assert _is_binary_content(text_content) is False


class TestZeroLegacyReferences:
    """Guardrail tests for verifying no legacy extraction API references (Phase 9)."""

    def test_no_legacy_extraction_module_references(self) -> None:
        """Verify no references to legacy extraction module in codebase.

        The extraction module was removed in favor of the parser system.
        This test ensures no code accidentally references the old APIs.
        """
        import subprocess

        # Module-qualified patterns that would indicate legacy usage
        patterns = [
            r"shared\.text_processing\.extraction",
            r"from\s+shared\.text_processing\.extraction",
        ]

        # Directories to search
        search_paths = ["packages/", "tests/", "scripts/"]

        for pattern in patterns:
            for search_path in search_paths:
                result = subprocess.run(
                    ["grep", "-rE", "--include=*.py", pattern, search_path],
                    capture_output=True,
                    text=True,
                    cwd="/home/john/semantik",
                )
                # grep returns 0 if matches found, 1 if no matches
                assert result.returncode == 1, (
                    f"Found legacy extraction references with pattern '{pattern}' in {search_path}:\n"
                    f"{result.stdout}"
                )

    def test_no_legacy_function_references(self) -> None:
        """Verify no references to removed legacy functions.

        Functions like extract_and_serialize and parse_document_content
        were removed. This test ensures no code calls them.
        """
        import subprocess

        # Legacy function names that should not exist
        patterns = [
            r"extract_and_serialize",
            r"parse_document_content",
        ]

        search_paths = ["packages/", "tests/", "scripts/"]

        for pattern in patterns:
            for search_path in search_paths:
                result = subprocess.run(
                    ["grep", "-rE", "--include=*.py", pattern, search_path],
                    capture_output=True,
                    text=True,
                    cwd="/home/john/semantik",
                )
                # grep returns 0 if matches found, 1 if no matches
                assert result.returncode == 1, (
                    f"Found legacy function reference '{pattern}' in {search_path}:\n" f"{result.stdout}"
                )
