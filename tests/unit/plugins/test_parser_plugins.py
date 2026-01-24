"""Tests for parser plugins.

Tests the ParserPlugin base class, TextParserPlugin, UnstructuredParserPlugin,
plugin registration, manifest generation, and equivalence with legacy parsers.
"""

from __future__ import annotations

import pytest

from shared.plugins.builtins.text_parser import (
    TextParserPlugin,
    _detect_bom,
    _is_binary_content,
)
from shared.plugins.types.parser import (
    ExtractionFailedError,
    ParsedElement,
    ParserConfigError,
    ParserOutput,
    UnsupportedFormatError,
    convert_config_options_to_schema,
    derive_input_types_from_extensions,
)

# ============================================================================
# Helper Function Tests
# ============================================================================


class TestDeriveInputTypesFromExtensions:
    """Tests for derive_input_types_from_extensions helper."""

    def test_common_text_extensions(self) -> None:
        extensions = frozenset({".txt", ".md", ".json"})
        mime_types = derive_input_types_from_extensions(extensions)
        assert "text/plain" in mime_types
        assert "text/markdown" in mime_types
        assert "application/json" in mime_types

    def test_code_extensions(self) -> None:
        extensions = frozenset({".py", ".js", ".ts"})
        mime_types = derive_input_types_from_extensions(extensions)
        # Python should be recognized
        assert any("python" in m for m in mime_types)
        # TypeScript uses fallback
        assert "text/typescript" in mime_types

    def test_empty_extensions(self) -> None:
        mime_types = derive_input_types_from_extensions(frozenset())
        assert mime_types == []

    def test_returns_sorted_unique(self) -> None:
        extensions = frozenset({".txt", ".text", ".md"})
        mime_types = derive_input_types_from_extensions(extensions)
        assert mime_types == sorted(set(mime_types))


class TestConvertConfigOptionsToSchema:
    """Tests for convert_config_options_to_schema helper."""

    def test_boolean_option(self) -> None:
        options = [{"name": "enabled", "type": "boolean", "label": "Enable feature", "default": True}]
        schema = convert_config_options_to_schema(options)
        assert schema["type"] == "object"
        assert schema["properties"]["enabled"]["type"] == "boolean"
        assert schema["properties"]["enabled"]["default"] is True
        assert "required" not in schema  # has default, so not required

    def test_select_option(self) -> None:
        options = [
            {
                "name": "strategy",
                "type": "select",
                "label": "Strategy",
                "options": [
                    {"value": "fast", "label": "Fast"},
                    {"value": "accurate", "label": "Accurate"},
                ],
                "default": "fast",
            }
        ]
        schema = convert_config_options_to_schema(options)
        assert schema["properties"]["strategy"]["type"] == "string"
        assert schema["properties"]["strategy"]["enum"] == ["fast", "accurate"]

    def test_required_fields(self) -> None:
        options = [{"name": "required_field", "type": "text", "label": "Required"}]
        schema = convert_config_options_to_schema(options)
        assert "required" in schema
        assert "required_field" in schema["required"]

    def test_number_option(self) -> None:
        options = [{"name": "count", "type": "number", "label": "Count", "default": 10}]
        schema = convert_config_options_to_schema(options)
        assert schema["properties"]["count"]["type"] == "number"


# ============================================================================
# BOM Detection Tests
# ============================================================================


class TestBOMDetection:
    """Tests for BOM detection in text parser."""

    def test_detect_utf8_bom(self) -> None:
        content = b"\xef\xbb\xbfHello"
        result = _detect_bom(content)
        assert result is not None
        encoding, length = result
        assert encoding == "utf-8-sig"
        assert length == 3

    def test_detect_utf16_le_bom(self) -> None:
        content = b"\xff\xfeH\x00e\x00l\x00l\x00o\x00"
        result = _detect_bom(content)
        assert result is not None
        encoding, length = result
        assert encoding == "utf-16-le"
        assert length == 2

    def test_detect_utf16_be_bom(self) -> None:
        content = b"\xfe\xff\x00H\x00e\x00l\x00l\x00o"
        result = _detect_bom(content)
        assert result is not None
        encoding, length = result
        assert encoding == "utf-16-be"
        assert length == 2

    def test_detect_utf32_le_bom(self) -> None:
        content = b"\xff\xfe\x00\x00H\x00\x00\x00"
        result = _detect_bom(content)
        assert result is not None
        encoding, length = result
        assert encoding == "utf-32-le"
        assert length == 4

    def test_no_bom(self) -> None:
        content = b"Hello, world!"
        result = _detect_bom(content)
        assert result is None


class TestBinaryDetection:
    """Tests for binary content detection."""

    def test_text_is_not_binary(self) -> None:
        content = b"Hello, world!\nThis is plain text."
        assert _is_binary_content(content) is False

    def test_binary_with_null_bytes(self) -> None:
        content = b"\x00\x01\x02\x03"
        assert _is_binary_content(content) is True

    def test_utf16_with_bom_is_not_binary(self) -> None:
        # UTF-16 has null bytes but BOM indicates it's text
        content = b"\xff\xfeH\x00e\x00l\x00l\x00o\x00"
        assert _is_binary_content(content) is False

    def test_high_non_printable_ratio(self) -> None:
        # Content with >30% non-printable bytes
        content = bytes([1] * 40 + [65] * 60)  # 40% non-printable
        assert _is_binary_content(content) is True

    def test_empty_content(self) -> None:
        assert _is_binary_content(b"") is False


# ============================================================================
# TextParserPlugin Tests
# ============================================================================


class TestTextParserPlugin:
    """Tests for TextParserPlugin."""

    def test_parse_bytes_utf8(self) -> None:
        parser = TextParserPlugin()
        content = b"Hello, world!"
        result = parser.parse_bytes(content, filename="test.txt")
        assert result.text == "Hello, world!"
        assert result.metadata["parser"] == "text"

    def test_parse_bytes_with_bom(self) -> None:
        parser = TextParserPlugin()
        content = b"\xef\xbb\xbfHello with BOM"
        result = parser.parse_bytes(content, filename="test.txt")
        assert "Hello with BOM" in result.text
        # BOM should be stripped
        assert not result.text.startswith("\ufeff")

    def test_parse_bytes_binary_rejection(self) -> None:
        parser = TextParserPlugin()
        content = b"\x00\x01\x02\x03"
        with pytest.raises(UnsupportedFormatError, match="binary"):
            parser.parse_bytes(content)

    def test_parse_bytes_with_encoding_errors_replace(self) -> None:
        parser = TextParserPlugin({"errors": "replace"})
        content = b"Hello \xff\xfe world"  # Invalid UTF-8
        result = parser.parse_bytes(content)
        # Should replace invalid bytes
        assert "Hello" in result.text
        assert "world" in result.text

    def test_parse_bytes_with_encoding_errors_strict(self) -> None:
        parser = TextParserPlugin({"errors": "strict"})
        content = b"Hello \xff\xfe world"  # Invalid UTF-8
        with pytest.raises(ExtractionFailedError, match="decode"):
            parser.parse_bytes(content)

    def test_parse_bytes_include_elements(self) -> None:
        parser = TextParserPlugin()
        content = b"Test content"
        result = parser.parse_bytes(content, include_elements=True)
        assert len(result.elements) == 1
        assert result.elements[0].text == "Test content"

    def test_parse_bytes_empty_content_no_elements(self) -> None:
        parser = TextParserPlugin()
        content = b"   "  # whitespace only
        result = parser.parse_bytes(content, include_elements=True)
        assert result.elements == []  # Empty/whitespace content has no elements

    def test_supported_extensions(self) -> None:
        extensions = TextParserPlugin.supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".py" in extensions
        assert ".json" in extensions

    def test_config_validation_unknown_option(self) -> None:
        with pytest.raises(ParserConfigError, match="Unknown config option"):
            TextParserPlugin({"unknown_option": "value"})

    def test_config_validation_invalid_type(self) -> None:
        with pytest.raises(ParserConfigError, match="must be a string"):
            TextParserPlugin({"encoding": 123})

    def test_config_validation_invalid_select(self) -> None:
        with pytest.raises(ParserConfigError, match="must be one of"):
            TextParserPlugin({"errors": "invalid"})

    def test_get_manifest(self) -> None:
        manifest = TextParserPlugin.get_manifest()
        assert manifest.id == "text"
        assert manifest.type == "parser"
        assert manifest.version == "1.0.0"
        assert "supported_extensions" in manifest.capabilities
        assert manifest.agent_hints is not None

    def test_agent_hints(self) -> None:
        hints = TextParserPlugin.AGENT_HINTS
        assert hints is not None
        assert "text" in hints.purpose.lower()
        assert len(hints.best_for) > 0
        assert len(hints.not_recommended_for) > 0


# ============================================================================
# Parser Plugin Protocol Tests
# ============================================================================


class TestParserPluginProtocol:
    """Tests for ParserPlugin protocol compliance."""

    def test_text_parser_has_required_attributes(self) -> None:
        assert hasattr(TextParserPlugin, "PLUGIN_ID")
        assert hasattr(TextParserPlugin, "PLUGIN_TYPE")
        assert hasattr(TextParserPlugin, "PLUGIN_VERSION")
        assert TextParserPlugin.PLUGIN_TYPE == "parser"

    def test_text_parser_has_required_methods(self) -> None:
        parser = TextParserPlugin()
        assert callable(parser.parse_file)
        assert callable(parser.parse_bytes)
        assert callable(parser.supported_extensions)
        assert callable(parser.get_manifest)

    def test_config_is_immutable(self) -> None:
        parser = TextParserPlugin({"encoding": "utf-8"})
        with pytest.raises(AttributeError):
            parser.config = {"encoding": "latin-1"}  # type: ignore[misc]


# ============================================================================
# Parser Output Tests
# ============================================================================


class TestParserOutput:
    """Tests for ParserOutput dataclass."""

    def test_parser_output_creation(self) -> None:
        output = ParserOutput(
            text="Test content",
            elements=[ParsedElement(text="Test", metadata={"page": 1})],
            metadata={"parser": "test"},
        )
        assert output.text == "Test content"
        assert len(output.elements) == 1
        assert output.metadata["parser"] == "test"

    def test_parser_output_frozen(self) -> None:
        output = ParserOutput(text="Test")
        with pytest.raises(AttributeError):
            output.text = "Modified"  # type: ignore[misc]

    def test_parsed_element_frozen(self) -> None:
        element = ParsedElement(text="Test", metadata={})
        with pytest.raises(AttributeError):
            element.text = "Modified"  # type: ignore[misc]


# ============================================================================
# Plugin Registration Tests
# ============================================================================


class TestParserPluginRegistration:
    """Tests for parser plugin registration."""

    def test_text_parser_can_be_loaded(self) -> None:
        """Verify TextParserPlugin can be imported and instantiated."""
        from shared.plugins.builtins.text_parser import TextParserPlugin

        parser = TextParserPlugin()
        assert parser.PLUGIN_ID == "text"

    def test_unstructured_parser_can_be_loaded(self) -> None:
        """Verify UnstructuredParserPlugin can be imported."""
        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        assert UnstructuredParserPlugin.PLUGIN_ID == "unstructured"
        assert UnstructuredParserPlugin.PLUGIN_TYPE == "parser"


class TestProtocolMapping:
    """Tests for parser protocol in PROTOCOL_BY_TYPE."""

    def test_parser_in_protocol_map(self) -> None:
        from shared.plugins.protocols import PROTOCOL_BY_TYPE, ParserProtocol

        assert "parser" in PROTOCOL_BY_TYPE
        assert PROTOCOL_BY_TYPE["parser"] is ParserProtocol


# ============================================================================
# Equivalence Tests (Legacy vs Plugin)
# ============================================================================


class TestLegacyEquivalence:
    """Tests comparing legacy TextParser with TextParserPlugin."""

    def test_same_supported_extensions(self) -> None:
        """Verify plugin supports same extensions as legacy parser."""
        from shared.text_processing.parsers.text import TextParser

        legacy_extensions = TextParser.supported_extensions()
        plugin_extensions = TextParserPlugin.supported_extensions()
        assert legacy_extensions == plugin_extensions

    def test_same_output_structure(self) -> None:
        """Verify plugin produces equivalent output to legacy parser."""
        from shared.text_processing.parsers.text import TextParser

        content = b"Hello, world!"
        filename = "test.txt"

        legacy_parser = TextParser()
        legacy_result = legacy_parser.parse_bytes(content, filename=filename)

        plugin_parser = TextParserPlugin()
        plugin_result = plugin_parser.parse_bytes(content, filename=filename)

        # Same text content
        assert legacy_result.text == plugin_result.text

        # Same metadata keys
        assert legacy_result.metadata.keys() == plugin_result.metadata.keys()
        assert legacy_result.metadata["parser"] == plugin_result.metadata["parser"]

    def test_same_bom_handling(self) -> None:
        """Verify plugin handles BOM same as legacy parser."""
        from shared.text_processing.parsers.text import TextParser

        content = b"\xef\xbb\xbfHello with BOM"

        legacy_parser = TextParser()
        legacy_result = legacy_parser.parse_bytes(content)

        plugin_parser = TextParserPlugin()
        plugin_result = plugin_parser.parse_bytes(content)

        assert legacy_result.text == plugin_result.text

    def test_same_binary_rejection(self) -> None:
        """Verify plugin rejects binary content same as legacy parser."""
        from shared.text_processing.parsers.exceptions import (
            UnsupportedFormatError as LegacyUnsupportedFormatError,
        )
        from shared.text_processing.parsers.text import TextParser

        content = b"\x00\x01\x02\x03"

        legacy_parser = TextParser()
        with pytest.raises(LegacyUnsupportedFormatError):
            legacy_parser.parse_bytes(content)

        plugin_parser = TextParserPlugin()
        with pytest.raises(UnsupportedFormatError):
            plugin_parser.parse_bytes(content)


# ============================================================================
# Config Schema Tests
# ============================================================================


class TestConfigSchema:
    """Tests for parser config schema generation."""

    def test_text_parser_config_schema(self) -> None:
        schema = TextParserPlugin.get_config_schema()
        assert schema is not None
        assert "properties" in schema
        assert "encoding" in schema["properties"]
        assert "errors" in schema["properties"]

    def test_config_options_to_schema_roundtrip(self) -> None:
        """Verify config options can be converted to JSON Schema."""
        options = TextParserPlugin.get_config_options()
        schema = convert_config_options_to_schema(options)

        # Schema should have all options as properties
        for opt in options:
            assert opt["name"] in schema["properties"]
