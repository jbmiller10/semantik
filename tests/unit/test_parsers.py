"""Unit tests for the parser system (Phases 1 & 2).

Tests cover:
1. parse_content() works for both bytes and str
2. Registry functions work with direct import
3. Selection rules and fallback behavior
4. Config validation
5. TextParser strictness (binary detection)
6. UnstructuredParser (mocked partition, config)
7. include_elements behavior
8. Metadata normalization (required keys always present)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shared.text_processing.parsers import (
    DEFAULT_PARSER_MAP,
    ExtractionFailedError,
    ParsedElement,
    ParserConfigError,
    ParseResult,
    TextParser,
    UnsupportedFormatError,
    get_parser,
    list_parsers,
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

        with pytest.raises(UnsupportedFormatError, match="NUL byte found"):
            parser.parse_bytes(content, file_extension=".txt")

    def test_high_non_printable_ratio_raises_unsupported_format_error(self) -> None:
        """High non-printable ratio raises UnsupportedFormatError."""
        parser = TextParser()
        # Create content with >30% non-printable bytes (excluding tab/newline)
        content = bytes([0x01] * 40 + [0x41] * 60)  # 40% control chars, 60% 'A'

        with pytest.raises(UnsupportedFormatError, match="non-printable ratio"):
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

        parser = UnstructuredParser({
            "strategy": "fast",
            "include_page_breaks": False,
            "infer_table_structure": False,
        })

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
