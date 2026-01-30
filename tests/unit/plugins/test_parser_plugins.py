"""Tests for parser plugins.

Tests the ParserPlugin base class, TextParserPlugin, UnstructuredParserPlugin,
plugin registration, manifest generation, and equivalence with legacy parsers.
"""

from __future__ import annotations

import pytest

from shared.plugins.builtins.text_parser import TextParserPlugin, _detect_bom, _is_binary_content
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
        """Verify plugin produces equivalent output to legacy parser.

        Note: Plugin emits additional parsed metadata fields (detected_language,
        approx_token_count, line_count, has_code_blocks) that legacy parser doesn't.
        """
        from shared.text_processing.parsers.text import TextParser

        content = b"Hello, world!"
        filename = "test.txt"

        legacy_parser = TextParser()
        legacy_result = legacy_parser.parse_bytes(content, filename=filename)

        plugin_parser = TextParserPlugin()
        plugin_result = plugin_parser.parse_bytes(content, filename=filename)

        # Same text content
        assert legacy_result.text == plugin_result.text

        # Legacy metadata keys are present in plugin result
        for key in legacy_result.metadata:
            assert key in plugin_result.metadata
            assert legacy_result.metadata[key] == plugin_result.metadata[key]

        # Plugin has additional parsed metadata fields
        assert "detected_language" in plugin_result.metadata
        assert "approx_token_count" in plugin_result.metadata
        assert "line_count" in plugin_result.metadata
        assert "has_code_blocks" in plugin_result.metadata

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
        from shared.text_processing.parsers.exceptions import UnsupportedFormatError as LegacyUnsupportedFormatError
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


# ============================================================================
# Parser Metadata Emission Tests (Phase 3)
# ============================================================================


class TestTextParserMetadataEmission:
    """Tests for TextParserPlugin parsed metadata emission."""

    def test_emits_approx_token_count(self) -> None:
        parser = TextParserPlugin()
        result = parser.parse_bytes(b"Hello world this is a test")
        assert "approx_token_count" in result.metadata
        assert result.metadata["approx_token_count"] == 6

    def test_emits_line_count(self) -> None:
        parser = TextParserPlugin()
        result = parser.parse_bytes(b"Line 1\nLine 2\nLine 3")
        assert result.metadata["line_count"] == 3

    def test_emits_line_count_single_line(self) -> None:
        parser = TextParserPlugin()
        result = parser.parse_bytes(b"Single line")
        assert result.metadata["line_count"] == 1

    def test_detects_code_blocks_markdown(self) -> None:
        content = b"# Title\n```python\nprint('hello')\n```\nMore text"
        parser = TextParserPlugin()
        result = parser.parse_bytes(content)
        assert result.metadata["has_code_blocks"] is True

    def test_detects_code_blocks_multiple(self) -> None:
        content = b"```js\ncode1\n```\n\ntext\n\n```python\ncode2\n```"
        parser = TextParserPlugin()
        result = parser.parse_bytes(content)
        assert result.metadata["has_code_blocks"] is True

    def test_no_code_blocks_for_plain_text(self) -> None:
        parser = TextParserPlugin()
        result = parser.parse_bytes(b"Just plain text with no code fences")
        assert result.metadata["has_code_blocks"] is False

    def test_language_detection_english(self) -> None:
        content = b"This is a longer English text that should be detected as English language."
        parser = TextParserPlugin()
        result = parser.parse_bytes(content)
        # Language detection may return 'en' or None if langdetect not installed
        if result.metadata.get("detected_language"):
            assert result.metadata["detected_language"] == "en"

    def test_language_detection_short_text_returns_none(self) -> None:
        content = b"Hi"  # Too short for detection
        parser = TextParserPlugin()
        result = parser.parse_bytes(content)
        # Short text should return None
        assert result.metadata.get("detected_language") is None

    def test_all_metadata_fields_present(self) -> None:
        """Verify all expected parsed metadata fields are emitted."""
        content = b"# Test\n```python\ncode\n```\nSome text here"
        parser = TextParserPlugin()
        result = parser.parse_bytes(content)

        # Check all expected fields exist
        assert "approx_token_count" in result.metadata
        assert "line_count" in result.metadata
        assert "has_code_blocks" in result.metadata
        assert "detected_language" in result.metadata  # May be None


class TestUnstructuredParserMetadataEmission:
    """Tests for UnstructuredParserPlugin parsed metadata emission."""

    def test_emits_approx_token_count(self) -> None:
        """Test that unstructured parser emits token count."""
        try:
            from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        except ImportError:
            pytest.skip("unstructured not installed")

        content = b"This is test content with multiple words"
        parser = UnstructuredParserPlugin()
        result = parser.parse_bytes(content, mime_type="text/plain")

        # Should have approx_token_count
        assert "approx_token_count" in result.metadata
        assert result.metadata["approx_token_count"] > 0

    def test_has_tables_false_for_plain_text(self) -> None:
        """Test that has_tables is False for plain text."""
        try:
            from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        except ImportError:
            pytest.skip("unstructured not installed")

        content = b"Just plain text with no tables"
        parser = UnstructuredParserPlugin()
        result = parser.parse_bytes(content, mime_type="text/plain")
        assert result.metadata.get("has_tables") is False

    def test_has_images_false_for_plain_text(self) -> None:
        """Test that has_images is False for plain text."""
        try:
            from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        except ImportError:
            pytest.skip("unstructured not installed")

        content = b"Just plain text with no images"
        parser = UnstructuredParserPlugin()
        result = parser.parse_bytes(content, mime_type="text/plain")
        assert result.metadata.get("has_images") is False

    def test_element_types_collected(self) -> None:
        """Test that element_types list is populated."""
        try:
            from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        except ImportError:
            pytest.skip("unstructured not installed")

        content = b"Title\n\nParagraph text"
        parser = UnstructuredParserPlugin()
        result = parser.parse_bytes(content, mime_type="text/plain")
        assert "element_types" in result.metadata
        assert isinstance(result.metadata["element_types"], list)

    def test_page_count_emitted(self) -> None:
        """Test that page_count is emitted."""
        try:
            from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        except ImportError:
            pytest.skip("unstructured not installed")

        content = b"Page 1 content\n\nPage 2 content"
        parser = UnstructuredParserPlugin()
        result = parser.parse_bytes(content, mime_type="text/plain")

        # Should have page_count (at least 1)
        assert "page_count" in result.metadata
        assert result.metadata["page_count"] >= 1


# ============================================================================
# UnstructuredParser Import Failure Tests
# ============================================================================


class TestUnstructuredParserImportFailure:
    """Tests for UnstructuredParserPlugin when unstructured library is missing."""

    def test_parse_bytes_raises_extraction_failed_when_unstructured_missing(self) -> None:
        """Test that parse_bytes raises ExtractionFailedError when unstructured import fails.

        This tests the lazy import error handling in parse_bytes. The unstructured
        library is imported inside parse_bytes, and if it's not installed,
        ExtractionFailedError should be raised (not UnsupportedFormatError).
        """
        import sys
        from unittest.mock import patch

        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        parser = UnstructuredParserPlugin()

        # Remove unstructured from sys.modules to force fresh import
        modules_to_remove = [k for k in list(sys.modules.keys()) if "unstructured" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        # Create a fake module that raises on attribute access
        class FailingModule:
            def __getattr__(self, name: str) -> None:
                raise ModuleNotFoundError("No module named 'unstructured'")

        try:
            # Patch sys.modules so the import inside parse_bytes fails
            with patch.dict(
                sys.modules,
                {
                    "unstructured": FailingModule(),
                    "unstructured.partition": FailingModule(),
                    "unstructured.partition.auto": FailingModule(),
                },
            ):
                with pytest.raises(ExtractionFailedError, match="unstructured is not installed"):
                    parser.parse_bytes(b"test content", mime_type="text/plain")
        finally:
            # Restore original modules
            sys.modules.update(saved_modules)


# ============================================================================
# UnstructuredParser Element Tracking Tests
# ============================================================================


class TestUnstructuredParserElementTracking:
    """Tests for UnstructuredParserPlugin element type and page tracking."""

    def test_element_types_tracked_correctly(self) -> None:
        """Test that element categories are properly collected in element_types."""
        from unittest.mock import MagicMock, patch

        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        parser = UnstructuredParserPlugin()

        # Create mock elements with different categories
        mock_element1 = MagicMock()
        mock_element1.__str__ = MagicMock(return_value="Title text")
        mock_element1.metadata = MagicMock()
        mock_element1.metadata.page_number = 1
        mock_element1.metadata.category = "Title"

        mock_element2 = MagicMock()
        mock_element2.__str__ = MagicMock(return_value="Paragraph text")
        mock_element2.metadata = MagicMock()
        mock_element2.metadata.page_number = 1
        mock_element2.metadata.category = "NarrativeText"

        mock_element3 = MagicMock()
        mock_element3.__str__ = MagicMock(return_value="| col1 | col2 |")
        mock_element3.metadata = MagicMock()
        mock_element3.metadata.page_number = 2
        mock_element3.metadata.category = "Table"

        mock_element4 = MagicMock()
        mock_element4.__str__ = MagicMock(return_value="[image]")
        mock_element4.metadata = MagicMock()
        mock_element4.metadata.page_number = 2
        mock_element4.metadata.category = "Image"

        mock_elements = [mock_element1, mock_element2, mock_element3, mock_element4]

        # Mock the partition function at the import location
        with patch(
            "shared.plugins.builtins.unstructured_parser.partition",
            return_value=mock_elements,
            create=True,
        ):
            # Need to patch where partition is used - it's imported lazily
            # So we need to patch the actual import
            import sys

            mock_partition_module = MagicMock()
            mock_partition_module.partition = MagicMock(return_value=mock_elements)

            with patch.dict(sys.modules, {"unstructured.partition.auto": mock_partition_module}):
                result = parser.parse_bytes(b"content", mime_type="text/plain")

        # Check element_types contains all categories
        assert "element_types" in result.metadata
        element_types = result.metadata["element_types"]
        assert "Title" in element_types
        assert "NarrativeText" in element_types
        assert "Table" in element_types
        assert "Image" in element_types

        # Check has_tables and has_images flags
        assert result.metadata["has_tables"] is True
        assert result.metadata["has_images"] is True

    def test_page_count_tracks_max_page_number(self) -> None:
        """Test that page_count equals the max page_number across all elements.

        Elements with page_numbers [1, 3, 2, 5] should result in page_count == 5.
        """
        from unittest.mock import MagicMock, patch

        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        parser = UnstructuredParserPlugin()

        # Create mock elements with non-sequential page numbers
        def make_element(text: str, page: int) -> MagicMock:
            element = MagicMock()
            element.__str__ = MagicMock(return_value=text)
            element.metadata = MagicMock()
            element.metadata.page_number = page
            element.metadata.category = "NarrativeText"
            return element

        mock_elements = [
            make_element("Page 1 content", 1),
            make_element("Page 3 content", 3),  # Skip page 2
            make_element("Page 2 content", 2),  # Out of order
            make_element("Page 5 content", 5),  # Max page
        ]

        import sys

        mock_partition_module = MagicMock()
        mock_partition_module.partition = MagicMock(return_value=mock_elements)

        with patch.dict(sys.modules, {"unstructured.partition.auto": mock_partition_module}):
            result = parser.parse_bytes(b"content", mime_type="text/plain")

        # page_count should be the maximum page number encountered
        assert result.metadata["page_count"] == 5

    def test_elements_without_page_number_use_current_page(self) -> None:
        """Test that elements without page_number metadata use the current page."""
        from unittest.mock import MagicMock, patch

        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        parser = UnstructuredParserPlugin()

        # First element has page 3, second has no page_number
        mock_element1 = MagicMock()
        mock_element1.__str__ = MagicMock(return_value="Text on page 3")
        mock_element1.metadata = MagicMock()
        mock_element1.metadata.page_number = 3
        mock_element1.metadata.category = "NarrativeText"

        mock_element2 = MagicMock()
        mock_element2.__str__ = MagicMock(return_value="Text without page")
        mock_element2.metadata = MagicMock()
        mock_element2.metadata.page_number = None  # No page number
        mock_element2.metadata.category = "NarrativeText"

        mock_elements = [mock_element1, mock_element2]

        import sys

        mock_partition_module = MagicMock()
        mock_partition_module.partition = MagicMock(return_value=mock_elements)

        with patch.dict(sys.modules, {"unstructured.partition.auto": mock_partition_module}):
            result = parser.parse_bytes(b"content", mime_type="text/plain", include_elements=True)

        # Second element should have inherited page 3 from first
        assert len(result.elements) == 2
        assert result.elements[0].metadata.get("page_number") == 3
        assert result.elements[1].metadata.get("page_number") == 3


# ============================================================================
# EMITTED_FIELDS Tests (Phase 5)
# ============================================================================


class TestParserEmittedFields:
    """Tests for EMITTED_FIELDS class variable on parser plugins."""

    def test_parser_base_class_has_empty_emitted_fields(self) -> None:
        """Verify ParserPlugin base class defaults to empty EMITTED_FIELDS."""
        from shared.plugins.types.parser import ParserPlugin

        assert hasattr(ParserPlugin, "EMITTED_FIELDS")
        assert [] == ParserPlugin.EMITTED_FIELDS

    def test_text_parser_declares_emitted_fields(self) -> None:
        """Verify TextParserPlugin declares its emitted fields."""
        from shared.plugins.builtins.text_parser import TextParserPlugin

        assert hasattr(TextParserPlugin, "EMITTED_FIELDS")
        assert len(TextParserPlugin.EMITTED_FIELDS) > 0

        # Verify expected fields are declared
        expected_fields = {"detected_language", "approx_token_count", "line_count", "has_code_blocks"}
        declared_fields = set(TextParserPlugin.EMITTED_FIELDS)
        assert expected_fields == declared_fields

    def test_text_parser_emitted_fields_matches_output(self) -> None:
        """Verify TextParserPlugin EMITTED_FIELDS matches fields actually emitted.

        Each field in EMITTED_FIELDS should be present in the parser output metadata.
        """
        from shared.plugins.builtins.text_parser import TextParserPlugin

        parser = TextParserPlugin()
        # Use content that exercises all metadata fields
        content = b"# Title\n```python\ncode\n```\nSome text here with enough words for detection"
        result = parser.parse_bytes(content)

        # Every declared field should be in the output
        for field_name in TextParserPlugin.EMITTED_FIELDS:
            assert field_name in result.metadata, f"Field '{field_name}' declared in EMITTED_FIELDS but not in output"

    def test_unstructured_parser_declares_emitted_fields(self) -> None:
        """Verify UnstructuredParserPlugin declares its emitted fields."""
        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        assert hasattr(UnstructuredParserPlugin, "EMITTED_FIELDS")
        assert len(UnstructuredParserPlugin.EMITTED_FIELDS) > 0

        # Verify expected fields are declared
        expected_fields = {"page_count", "has_tables", "has_images", "element_types", "approx_token_count"}
        declared_fields = set(UnstructuredParserPlugin.EMITTED_FIELDS)
        assert expected_fields == declared_fields

    def test_unstructured_parser_emitted_fields_matches_output(self) -> None:
        """Verify UnstructuredParserPlugin EMITTED_FIELDS matches fields actually emitted.

        Uses mocking to avoid requiring unstructured library.
        """
        from unittest.mock import MagicMock, patch

        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin

        parser = UnstructuredParserPlugin()

        # Create mock element
        mock_element = MagicMock()
        mock_element.__str__ = MagicMock(return_value="Test content")
        mock_element.metadata = MagicMock()
        mock_element.metadata.page_number = 1
        mock_element.metadata.category = "NarrativeText"

        import sys

        mock_partition_module = MagicMock()
        mock_partition_module.partition = MagicMock(return_value=[mock_element])

        with patch.dict(sys.modules, {"unstructured.partition.auto": mock_partition_module}):
            result = parser.parse_bytes(b"content", mime_type="text/plain")

        # Every declared field should be in the output
        for field_name in UnstructuredParserPlugin.EMITTED_FIELDS:
            assert field_name in result.metadata, f"Field '{field_name}' declared in EMITTED_FIELDS but not in output"


class TestPluginRegistryEmittedFields:
    """Tests for PluginRegistry.get_parser_emitted_fields method."""

    def test_get_parser_emitted_fields_returns_text_parser_fields(self) -> None:
        """Verify registry returns emitted fields for text parser."""
        from shared.plugins import load_plugins
        from shared.plugins.builtins.text_parser import TextParserPlugin
        from shared.plugins.registry import plugin_registry

        # Ensure parser plugins are loaded
        load_plugins(plugin_types=["parser"])

        fields = plugin_registry.get_parser_emitted_fields("text")
        assert fields == list(TextParserPlugin.EMITTED_FIELDS)

    def test_get_parser_emitted_fields_returns_unstructured_parser_fields(self) -> None:
        """Verify registry returns emitted fields for unstructured parser."""
        from shared.plugins import load_plugins
        from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        from shared.plugins.registry import plugin_registry

        # Ensure parser plugins are loaded
        load_plugins(plugin_types=["parser"])

        fields = plugin_registry.get_parser_emitted_fields("unstructured")
        assert fields == list(UnstructuredParserPlugin.EMITTED_FIELDS)

    def test_get_parser_emitted_fields_returns_empty_for_unknown_parser(self) -> None:
        """Verify registry returns empty list for unknown parser."""
        from shared.plugins.registry import plugin_registry

        fields = plugin_registry.get_parser_emitted_fields("nonexistent_parser")
        assert fields == []

    def test_get_parser_emitted_fields_returns_copy_not_original(self) -> None:
        """Verify registry returns a copy, not the original list."""
        from shared.plugins import load_plugins
        from shared.plugins.builtins.text_parser import TextParserPlugin
        from shared.plugins.registry import plugin_registry

        # Ensure parser plugins are loaded
        load_plugins(plugin_types=["parser"])

        fields = plugin_registry.get_parser_emitted_fields("text")
        original = TextParserPlugin.EMITTED_FIELDS

        # Should be equal but not the same object
        assert fields == list(original)
        fields.append("test_field")
        assert "test_field" not in original
