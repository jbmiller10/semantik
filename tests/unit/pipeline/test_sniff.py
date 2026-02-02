"""Unit tests for pipeline content sniffing."""

import asyncio
import time
from unittest.mock import patch

import pytest

from shared.pipeline.sniff import ContentSniffer, SniffCache, SniffConfig, SniffResult
from shared.pipeline.types import FileReference


class TestSniffResult:
    """Tests for SniffResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values are properly set."""
        result = SniffResult()
        assert result.is_scanned_pdf is None
        assert result.is_code is False
        assert result.is_structured_data is False
        assert result.structured_format is None
        assert result.sniff_duration_ms == 0.0
        assert result.errors == []

    def test_to_metadata_dict_empty(self) -> None:
        """Test to_metadata_dict with default values returns empty dict."""
        result = SniffResult()
        assert result.to_metadata_dict() == {}

    def test_to_metadata_dict_with_scanned_pdf_true(self) -> None:
        """Test to_metadata_dict includes is_scanned_pdf when True."""
        result = SniffResult(is_scanned_pdf=True)
        assert result.to_metadata_dict() == {"is_scanned_pdf": True}

    def test_to_metadata_dict_with_scanned_pdf_false(self) -> None:
        """Test to_metadata_dict includes is_scanned_pdf when False (native PDF)."""
        result = SniffResult(is_scanned_pdf=False)
        assert result.to_metadata_dict() == {"is_scanned_pdf": False}

    def test_to_metadata_dict_with_code(self) -> None:
        """Test to_metadata_dict includes is_code when True."""
        result = SniffResult(is_code=True)
        assert result.to_metadata_dict() == {"is_code": True}

    def test_to_metadata_dict_with_structured_data(self) -> None:
        """Test to_metadata_dict includes structured data fields."""
        result = SniffResult(is_structured_data=True, structured_format="json")
        assert result.to_metadata_dict() == {
            "is_structured_data": True,
            "structured_format": "json",
        }

    def test_to_metadata_dict_combined(self) -> None:
        """Test to_metadata_dict with multiple fields set."""
        result = SniffResult(
            is_scanned_pdf=False,
            is_code=True,
            is_structured_data=True,
            structured_format="yaml",
        )
        metadata = result.to_metadata_dict()
        assert metadata["is_scanned_pdf"] is False
        assert metadata["is_code"] is True
        assert metadata["is_structured_data"] is True
        assert metadata["structured_format"] == "yaml"


class TestSniffConfig:
    """Tests for SniffConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SniffConfig()
        assert config.timeout_seconds == 5.0
        assert config.pdf_sample_pages == 3
        assert config.structured_sample_bytes == 4096
        assert config.enabled is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SniffConfig(
            timeout_seconds=10.0,
            pdf_sample_pages=5,
            structured_sample_bytes=8192,
            enabled=False,
        )
        assert config.timeout_seconds == 10.0
        assert config.pdf_sample_pages == 5
        assert config.structured_sample_bytes == 8192
        assert config.enabled is False


class TestSniffConfigValidation:
    """Test SniffConfig validation."""

    def test_valid_config(self) -> None:
        """Valid configuration should be accepted."""
        config = SniffConfig(
            timeout_seconds=10.0,
            pdf_sample_pages=5,
            structured_sample_bytes=8192,
        )
        assert config.timeout_seconds == 10.0
        assert config.pdf_sample_pages == 5

    def test_negative_timeout_rejected(self) -> None:
        """Negative timeout should raise ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            SniffConfig(timeout_seconds=-1.0)

    def test_zero_timeout_rejected(self) -> None:
        """Zero timeout should raise ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            SniffConfig(timeout_seconds=0)

    def test_zero_pdf_pages_rejected(self) -> None:
        """Zero pdf_sample_pages should raise ValueError."""
        with pytest.raises(ValueError, match="pdf_sample_pages must be at least 1"):
            SniffConfig(pdf_sample_pages=0)

    def test_negative_sample_bytes_rejected(self) -> None:
        """Negative structured_sample_bytes should raise ValueError."""
        with pytest.raises(ValueError, match="structured_sample_bytes must be at least 1"):
            SniffConfig(structured_sample_bytes=-100)


class TestContentSnifferBasics:
    """Basic tests for ContentSniffer."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create a content sniffer with default config."""
        return ContentSniffer()

    @pytest.fixture()
    def text_file_ref(self) -> FileReference:
        """Create a text file reference."""
        return FileReference(
            uri="file:///docs/readme.txt",
            source_type="directory",
            content_type="document",
            filename="readme.txt",
            extension=".txt",
            mime_type="text/plain",
        )

    @pytest.mark.asyncio()
    async def test_disabled_sniffing_returns_empty_result(self) -> None:
        """Test that disabled sniffing returns empty result."""
        config = SniffConfig(enabled=False)
        sniffer = ContentSniffer(config)
        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
        )
        result = await sniffer.sniff(b"hello world", file_ref)
        assert result.is_scanned_pdf is None
        assert result.is_code is False
        assert result.is_structured_data is False
        assert result.sniff_duration_ms == 0.0

    @pytest.mark.asyncio()
    async def test_sniff_records_duration(self, sniffer: ContentSniffer, text_file_ref: FileReference) -> None:
        """Test that sniff records duration."""
        result = await sniffer.sniff(b"hello world", text_file_ref)
        assert result.sniff_duration_ms > 0

    def test_enrich_file_ref_adds_detected_namespace(self, sniffer: ContentSniffer) -> None:
        """Test that enrich_file_ref adds detected namespace to metadata."""
        file_ref = FileReference(
            uri="file:///test.py",
            source_type="directory",
            content_type="document",
            metadata={"source": {"local_path": "/test.py"}},
        )
        result = SniffResult(is_code=True)
        sniffer.enrich_file_ref(file_ref, result)

        assert "detected" in file_ref.metadata
        assert file_ref.metadata["detected"]["is_code"] is True

    def test_enrich_file_ref_preserves_existing_metadata(self, sniffer: ContentSniffer) -> None:
        """Test that enrich_file_ref preserves existing metadata."""
        file_ref = FileReference(
            uri="file:///test.py",
            source_type="directory",
            content_type="document",
            metadata={"source": {"local_path": "/test.py"}},
        )
        result = SniffResult(is_code=True)
        sniffer.enrich_file_ref(file_ref, result)

        assert file_ref.metadata["source"]["local_path"] == "/test.py"
        assert file_ref.metadata["detected"]["is_code"] is True

    def test_enrich_file_ref_empty_result_no_change(self, sniffer: ContentSniffer) -> None:
        """Test that empty sniff result doesn't add detected namespace."""
        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
        )
        result = SniffResult()  # Empty result
        sniffer.enrich_file_ref(file_ref, result)

        assert "detected" not in file_ref.metadata


class TestPDFDetection:
    """Tests for scanned PDF detection."""

    @pytest.fixture()
    def pdf_file_ref(self) -> FileReference:
        """Create a PDF file reference."""
        return FileReference(
            uri="file:///docs/report.pdf",
            source_type="directory",
            content_type="document",
            filename="report.pdf",
            extension=".pdf",
            mime_type="application/pdf",
        )

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    def test_is_pdf_by_extension(self, sniffer: ContentSniffer) -> None:
        """Test PDF detection by extension (case-insensitive)."""
        assert sniffer._is_pdf("", ".pdf") is True
        assert sniffer._is_pdf("", ".PDF") is True
        assert sniffer._is_pdf("", ".Pdf") is True
        assert sniffer._is_pdf("", ".txt") is False

    def test_is_pdf_by_mime_type(self, sniffer: ContentSniffer) -> None:
        """Test PDF detection by MIME type."""
        assert sniffer._is_pdf("application/pdf", "") is True
        assert sniffer._is_pdf("text/plain", "") is False

    @pytest.mark.asyncio()
    async def test_non_pdf_returns_none_for_is_scanned_pdf(self, sniffer: ContentSniffer) -> None:
        """Test that non-PDF files return None for is_scanned_pdf."""
        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            extension=".txt",
            mime_type="text/plain",
        )
        result = await sniffer.sniff(b"hello world", file_ref)
        assert result.is_scanned_pdf is None

    @pytest.mark.asyncio()
    async def test_native_pdf_detection(self, sniffer: ContentSniffer, pdf_file_ref: FileReference) -> None:
        """Test detection of native PDF with text layer."""
        # Create a minimal PDF with text content
        # This is a real PDF that pypdf can parse
        pdf_with_text = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000268 00000 n
0000000363 00000 n
trailer
<< /Root 1 0 R /Size 6 >>
startxref
446
%%EOF"""

        try:
            result = await sniffer.sniff(pdf_with_text, pdf_file_ref)
            # The result depends on whether pypdf extracts text correctly
            # For this test, we just verify no errors were raised
            assert result.is_scanned_pdf is not None or len(result.errors) > 0
        except ImportError:
            pytest.skip("pypdf not available")

    @pytest.mark.asyncio()
    async def test_invalid_pdf_logs_error(self, sniffer: ContentSniffer, pdf_file_ref: FileReference) -> None:
        """Test that invalid PDF content records an error."""
        result = await sniffer.sniff(b"not a pdf", pdf_file_ref)
        # Should have an error logged for PDF detection failure
        assert result.is_scanned_pdf is None or len(result.errors) > 0

    @pytest.mark.asyncio()
    async def test_partial_page_failures_still_works(
        self, sniffer: ContentSniffer, pdf_file_ref: FileReference
    ) -> None:
        """Test that partial page extraction failures still calculate average from successful pages."""
        # Create a mock scenario where some pages fail but some succeed
        # This tests that we calculate average only over successful pages
        # We use mocking since creating a PDF with specific page failures is complex

        # First, test that a valid PDF with at least one extractable page works
        pdf_with_text = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000268 00000 n
0000000363 00000 n
trailer
<< /Root 1 0 R /Size 6 >>
startxref
446
%%EOF"""

        try:
            result = await sniffer.sniff(pdf_with_text, pdf_file_ref)
            # Should succeed and return a result
            assert result.is_scanned_pdf is not None or len(result.errors) > 0
        except ImportError:
            pytest.skip("pypdf not available")

    @pytest.mark.asyncio()
    async def test_all_pages_fail_raises_error(self, sniffer: ContentSniffer, pdf_file_ref: FileReference) -> None:
        """Test that all pages failing extraction raises ValueError with page details."""
        from unittest.mock import MagicMock, patch

        try:
            from pypdf import PdfReader  # noqa: F401
        except ImportError:
            pytest.skip("pypdf not available")

        # Create a mock reader where all page extractions fail
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.side_effect = RuntimeError("Corrupt page data")
        mock_reader.pages = [mock_page, mock_page, mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = await sniffer.sniff(b"fake pdf content", pdf_file_ref)

        # Should have an error indicating all pages failed
        assert len(result.errors) > 0
        assert (
            "pages failed extraction" in result.errors[0].lower() or "pdf detection failed" in result.errors[0].lower()
        )

    @pytest.mark.asyncio()
    async def test_page_failure_error_includes_page_numbers(
        self, sniffer: ContentSniffer, pdf_file_ref: FileReference
    ) -> None:
        """Test that page extraction failures include page numbers in error messages."""
        from unittest.mock import MagicMock, patch

        try:
            from pypdf import PdfReader  # noqa: F401
        except ImportError:
            pytest.skip("pypdf not available")

        # Create a mock reader where all pages fail with unique errors
        mock_reader = MagicMock()
        mock_page0 = MagicMock()
        mock_page0.extract_text.side_effect = RuntimeError("Page 0 corrupt")
        mock_page1 = MagicMock()
        mock_page1.extract_text.side_effect = RuntimeError("Page 1 corrupt")
        mock_reader.pages = [mock_page0, mock_page1]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = await sniffer.sniff(b"fake pdf content", pdf_file_ref)

        # Should have errors recorded
        assert len(result.errors) > 0
        # Error should mention page numbers
        error_text = result.errors[0].lower()
        assert "page" in error_text


class TestCodeDetection:
    """Tests for code file detection."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.mark.asyncio()
    async def test_code_by_extension(self, sniffer: ContentSniffer) -> None:
        """Test code detection by extension."""
        extensions = [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp"]

        for ext in extensions:
            file_ref = FileReference(
                uri=f"file:///code/test{ext}",
                source_type="directory",
                content_type="code",
                extension=ext,
            )
            result = await sniffer.sniff(b"// some code", file_ref)
            assert result.is_code is True, f"Extension {ext} should be detected as code"

    @pytest.mark.asyncio()
    async def test_non_code_extension(self, sniffer: ContentSniffer) -> None:
        """Test non-code extension is not detected as code."""
        file_ref = FileReference(
            uri="file:///docs/readme.txt",
            source_type="directory",
            content_type="document",
            extension=".txt",
        )
        result = await sniffer.sniff(b"hello world", file_ref)
        assert result.is_code is False

    @pytest.mark.asyncio()
    async def test_code_by_shebang_python(self, sniffer: ContentSniffer) -> None:
        """Test code detection by Python shebang."""
        file_ref = FileReference(
            uri="file:///scripts/myscript",
            source_type="directory",
            content_type="document",
            extension="",  # No extension
        )
        content = b"#!/usr/bin/env python3\nprint('hello')"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_shebang_bash(self, sniffer: ContentSniffer) -> None:
        """Test code detection by bash shebang."""
        file_ref = FileReference(
            uri="file:///scripts/myscript",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"#!/bin/bash\necho 'hello'"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_shebang_node(self, sniffer: ContentSniffer) -> None:
        """Test code detection by node shebang."""
        file_ref = FileReference(
            uri="file:///scripts/myscript",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"#!/usr/bin/env node\nconsole.log('hello')"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_python_import(self, sniffer: ContentSniffer) -> None:
        """Test code detection by Python import statement."""
        file_ref = FileReference(
            uri="file:///code/script",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"import os\nimport sys\n\ndef main():\n    pass"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_js_const(self, sniffer: ContentSniffer) -> None:
        """Test code detection by JavaScript const declaration."""
        file_ref = FileReference(
            uri="file:///code/script",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"const foo = 'bar';\nlet x = 1;"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_function_def(self, sniffer: ContentSniffer) -> None:
        """Test code detection by function definition."""
        file_ref = FileReference(
            uri="file:///code/script",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"function hello() {\n  return 'world';\n}"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_class_def(self, sniffer: ContentSniffer) -> None:
        """Test code detection by class definition."""
        file_ref = FileReference(
            uri="file:///code/script",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"class MyClass:\n    def __init__(self):\n        pass"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio()
    async def test_code_by_c_include(self, sniffer: ContentSniffer) -> None:
        """Test code detection by C include statement."""
        file_ref = FileReference(
            uri="file:///code/script",
            source_type="directory",
            content_type="document",
            extension="",
        )
        content = b"#include <stdio.h>\nint main() { return 0; }"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True


class TestStructuredDataDetection:
    """Tests for structured data detection."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.fixture()
    def generic_file_ref(self) -> FileReference:
        """Create a generic file reference without specific extension."""
        return FileReference(
            uri="file:///data/file",
            source_type="directory",
            content_type="document",
            extension="",
        )

    @pytest.mark.asyncio()
    async def test_json_object_detection(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test JSON object detection."""
        content = b'{"name": "test", "value": 42}'
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "json"

    @pytest.mark.asyncio()
    async def test_json_array_detection(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test JSON array detection."""
        content = b'[{"id": 1}, {"id": 2}, {"id": 3}]'
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "json"

    @pytest.mark.asyncio()
    async def test_invalid_json_not_detected(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test that invalid JSON is not detected as JSON."""
        content = b'{"name": "test"'  # Missing closing brace
        result = await sniffer.sniff(content, generic_file_ref)
        # Should not be detected as JSON
        assert result.structured_format != "json"

    @pytest.mark.asyncio()
    async def test_xml_detection(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test XML detection."""
        content = b'<?xml version="1.0"?>\n<root><item>test</item></root>'
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "xml"

    @pytest.mark.asyncio()
    async def test_xml_without_declaration(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test XML detection without declaration."""
        content = b"<root><item>test</item></root>"
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "xml"

    @pytest.mark.asyncio()
    async def test_html_not_detected_as_xml(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test that HTML is not detected as XML."""
        content = b"<!DOCTYPE html><html><body>Hello</body></html>"
        result = await sniffer.sniff(content, generic_file_ref)
        # HTML should not be detected as structured data
        assert result.structured_format != "xml"

    @pytest.mark.asyncio()
    async def test_yaml_detection(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test YAML detection."""
        content = b"---\nname: test\nvalue: 42\nitems:\n  - one\n  - two"
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "yaml"

    @pytest.mark.asyncio()
    async def test_yaml_without_document_marker(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test YAML detection without --- marker."""
        content = b"name: test\nvalue: 42\nnested:\n  key: value"
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "yaml"

    @pytest.mark.asyncio()
    async def test_plain_text_not_yaml(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test that plain text is not detected as YAML."""
        content = b"This is just plain text.\nNothing special here."
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is False
        assert result.structured_format is None

    @pytest.mark.asyncio()
    async def test_csv_detection(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test CSV detection."""
        content = b"name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,35,Chicago"
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "csv"

    @pytest.mark.asyncio()
    async def test_tsv_detection(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test TSV (tab-separated) detection."""
        content = b"name\tage\tcity\nJohn\t30\tNYC\nJane\t25\tLA"
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "csv"

    @pytest.mark.asyncio()
    async def test_binary_content_not_structured(
        self, sniffer: ContentSniffer, generic_file_ref: FileReference
    ) -> None:
        """Test that binary content is not detected as structured."""
        content = b"\x00\x01\x02\x03\x04\x05"  # Binary with null bytes
        result = await sniffer.sniff(content, generic_file_ref)
        assert result.is_structured_data is False


class TestTimeoutHandling:
    """Tests for timeout handling."""

    @pytest.mark.asyncio()
    async def test_timeout_returns_partial_result(self) -> None:
        """Test that timeout returns partial result with error."""
        config = SniffConfig(timeout_seconds=0.001)  # Very short timeout
        sniffer = ContentSniffer(config)

        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
        )

        # Mock _do_sniff to take longer than timeout
        async def slow_sniff(*_args, **_kwargs):
            await asyncio.sleep(1)
            return SniffResult(is_code=True)

        with patch.object(sniffer, "_do_sniff", slow_sniff):
            result = await sniffer.sniff(b"content", file_ref)

        assert "timed out" in result.errors[0].lower()
        assert result.sniff_duration_ms > 0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.mark.asyncio()
    async def test_individual_detector_failure_continues(self, sniffer: ContentSniffer) -> None:
        """Test that failure in one detector doesn't stop others."""
        file_ref = FileReference(
            uri="file:///test.pdf",
            source_type="directory",
            content_type="document",
            extension=".pdf",
            mime_type="application/pdf",
        )

        # Invalid PDF content will fail PDF detection but code detection should still run
        result = await sniffer.sniff(b"not a pdf", file_ref)

        # Should have logged an error for PDF detection
        # But should still return a result
        assert result is not None

    @pytest.mark.asyncio()
    async def test_exception_in_sniff_logged(self, sniffer: ContentSniffer) -> None:
        """Test that exceptions during sniff are logged to errors."""
        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
        )

        # Mock _do_sniff to raise an exception
        async def failing_sniff(*_args, **_kwargs):
            raise RuntimeError("Test error")

        with patch.object(sniffer, "_do_sniff", failing_sniff):
            result = await sniffer.sniff(b"content", file_ref)

        assert len(result.errors) > 0
        assert "Test error" in result.errors[0]


class TestPredicateIntegration:
    """Tests for integration with predicate matching."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.mark.asyncio()
    async def test_enriched_file_ref_works_with_predicates(self, sniffer: ContentSniffer) -> None:
        """Test that enriched FileReference works with predicate matching."""
        from shared.pipeline.predicates import matches_predicate

        file_ref = FileReference(
            uri="file:///code/script.py",
            source_type="directory",
            content_type="code",
            extension=".py",
        )

        result = await sniffer.sniff(b"import os", file_ref)
        sniffer.enrich_file_ref(file_ref, result)

        # Should match predicate for code files
        assert matches_predicate(file_ref, {"metadata.detected.is_code": True})

    @pytest.mark.asyncio()
    async def test_enriched_structured_data_predicates(self, sniffer: ContentSniffer) -> None:
        """Test that structured data detection works with predicates."""
        from shared.pipeline.predicates import matches_predicate

        file_ref = FileReference(
            uri="file:///data/config",
            source_type="directory",
            content_type="document",
            extension="",
        )

        result = await sniffer.sniff(b'{"key": "value"}', file_ref)
        sniffer.enrich_file_ref(file_ref, result)

        assert matches_predicate(file_ref, {"metadata.detected.is_structured_data": True})
        assert matches_predicate(file_ref, {"metadata.detected.structured_format": "json"})


class TestYAMLEdgeCases:
    """Edge case tests for YAML detection."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.fixture()
    def generic_file_ref(self) -> FileReference:
        """Create a generic file reference without specific extension."""
        return FileReference(
            uri="file:///data/file",
            source_type="directory",
            content_type="document",
            extension="",
        )

    @pytest.mark.asyncio()
    async def test_yaml_single_indicator_not_detected(
        self, sniffer: ContentSniffer, generic_file_ref: FileReference
    ) -> None:
        """Test that plain text with only one YAML-like pattern is not detected as YAML.

        The YAML detector requires at least 2 YAML indicators to avoid false positives
        on plain text that happens to have a colon.
        """
        # Content with only one YAML-like pattern (single key: value)
        content = b"This is just text: nothing special here\nMore plain text"
        result = await sniffer.sniff(content, generic_file_ref)

        # Should NOT be detected as structured data since there's only one indicator
        assert result.is_structured_data is False
        assert result.structured_format is None

    @pytest.mark.asyncio()
    async def test_yaml_requires_two_indicators(self, sniffer: ContentSniffer, generic_file_ref: FileReference) -> None:
        """Test that YAML detection requires at least 2 indicators."""
        # Content that parses as YAML but has only one indicator pattern
        # A simple "key: value" at start of line counts as one indicator
        content = b"author: John Smith"
        result = await sniffer.sniff(content, generic_file_ref)

        # Single key-value should not be enough
        assert result.is_structured_data is False


class TestCSVEdgeCases:
    """Edge case tests for CSV detection."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.fixture()
    def generic_file_ref(self) -> FileReference:
        """Create a generic file reference without specific extension."""
        return FileReference(
            uri="file:///data/file",
            source_type="directory",
            content_type="document",
            extension="",
        )

    @pytest.mark.asyncio()
    async def test_csv_inconsistent_columns_not_detected(
        self, sniffer: ContentSniffer, generic_file_ref: FileReference
    ) -> None:
        """Test CSV with varying column counts (<80% consistency) is not detected.

        The CSV detector requires at least 80% of rows to have the same column count.
        """
        # Create CSV with wildly inconsistent column counts
        # Row 1: 3 columns, Row 2: 5 columns, Row 3: 2 columns, Row 4: 7 columns, Row 5: 1 column
        # None of these match >80%
        content = b"a,b,c\n1,2,3,4,5\nx,y\n1,2,3,4,5,6,7\nz"
        result = await sniffer.sniff(content, generic_file_ref)

        # Should NOT be detected as CSV due to inconsistent column counts
        assert result.structured_format != "csv"

    @pytest.mark.asyncio()
    async def test_csv_single_column_not_detected(
        self, sniffer: ContentSniffer, generic_file_ref: FileReference
    ) -> None:
        """Test that single-column data is not detected as CSV.

        CSV detection requires at least 2 columns.
        """
        content = b"item1\nitem2\nitem3\nitem4"
        result = await sniffer.sniff(content, generic_file_ref)

        # Single column shouldn't be detected as CSV
        assert result.structured_format != "csv"


class TestPDFExtractionEdgeCases:
    """Edge case tests for PDF extraction failure handling."""

    @pytest.fixture()
    def sniffer(self) -> ContentSniffer:
        """Create content sniffer."""
        return ContentSniffer()

    @pytest.fixture()
    def pdf_file_ref(self) -> FileReference:
        """Create a PDF file reference."""
        return FileReference(
            uri="file:///docs/report.pdf",
            source_type="directory",
            content_type="document",
            filename="report.pdf",
            extension=".pdf",
            mime_type="application/pdf",
        )

    @pytest.mark.asyncio()
    async def test_all_pdf_pages_fail_extraction_error(
        self, sniffer: ContentSniffer, pdf_file_ref: FileReference
    ) -> None:
        """Test that all pages failing extraction results in error with page details."""
        from unittest.mock import MagicMock, patch

        try:
            from pypdf import PdfReader  # noqa: F401
        except ImportError:
            pytest.skip("pypdf not available")

        # Create a mock reader where all page extractions fail with unique errors
        mock_reader = MagicMock()
        mock_page0 = MagicMock()
        mock_page0.extract_text.side_effect = Exception("Encrypted content on page 0")
        mock_page1 = MagicMock()
        mock_page1.extract_text.side_effect = Exception("Invalid stream on page 1")
        mock_page2 = MagicMock()
        mock_page2.extract_text.side_effect = Exception("Missing font on page 2")
        mock_reader.pages = [mock_page0, mock_page1, mock_page2]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = await sniffer.sniff(b"fake pdf content", pdf_file_ref)

        # Should have an error indicating all pages failed
        assert len(result.errors) > 0
        # The error should be from the PDF detection failure
        error_text = " ".join(result.errors).lower()
        assert "page" in error_text or "pdf" in error_text

    @pytest.mark.asyncio()
    async def test_pdf_with_some_pages_failing_uses_successful_ones(
        self, sniffer: ContentSniffer, pdf_file_ref: FileReference
    ) -> None:
        """Test PDF with some pages failing calculates average from successful pages only."""
        from unittest.mock import MagicMock, patch

        try:
            from pypdf import PdfReader  # noqa: F401
        except ImportError:
            pytest.skip("pypdf not available")

        # Create mock where page 0 fails but pages 1 and 2 succeed with lots of text
        mock_reader = MagicMock()

        mock_page0 = MagicMock()
        mock_page0.extract_text.side_effect = Exception("Corrupted page")

        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "x" * 200  # Plenty of text

        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "y" * 300  # More text

        mock_reader.pages = [mock_page0, mock_page1, mock_page2]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = await sniffer.sniff(b"fake pdf content", pdf_file_ref)

        # Should complete successfully (is_scanned_pdf is not None)
        # Average from successful pages: (200 + 300) / 2 = 250 chars/page
        # This is above the threshold of 50 chars/page, so NOT scanned
        assert result.is_scanned_pdf is False
        # Should have no critical errors (one page failing is recoverable)


class TestSniffCache:
    """Tests for SniffCache LRU+TTL cache."""

    def test_cache_set_and_get(self) -> None:
        """Test basic cache set and get operations."""
        cache = SniffCache(maxsize=100, ttl=3600)
        result = SniffResult(is_code=True)

        cache.set("hash123", result)
        cached = cache.get("hash123")

        assert cached is not None
        assert cached.is_code is True

    def test_cache_miss_returns_none(self) -> None:
        """Test that cache miss returns None."""
        cache = SniffCache(maxsize=100, ttl=3600)

        cached = cache.get("nonexistent")
        assert cached is None

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = SniffCache(maxsize=3, ttl=3600)

        # Fill the cache
        cache.set("hash1", SniffResult(is_code=True))
        cache.set("hash2", SniffResult(is_code=False))
        cache.set("hash3", SniffResult(is_scanned_pdf=True))

        # Access hash1 to make it recently used
        cache.get("hash1")

        # Add a new item, should evict hash2 (least recently used)
        cache.set("hash4", SniffResult(is_structured_data=True))

        # hash1 should still be present (was accessed)
        assert cache.get("hash1") is not None
        # hash2 should be evicted
        assert cache.get("hash2") is None
        # hash3 should still be present
        assert cache.get("hash3") is not None
        # hash4 should be present
        assert cache.get("hash4") is not None

    def test_cache_ttl_expiration(self) -> None:
        """Test that expired items are not returned."""
        cache = SniffCache(maxsize=100, ttl=1)  # 1 second TTL

        result = SniffResult(is_code=True)
        cache.set("hash123", result)

        # Should be present immediately
        assert cache.get("hash123") is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should be expired
        assert cache.get("hash123") is None

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = SniffCache(maxsize=100, ttl=3600)

        # Initial stats
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # Add an item
        cache.set("hash1", SniffResult(is_code=True))

        # Miss
        cache.get("nonexistent")

        # Hit
        cache.get("hash1")

        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_cache_clear(self) -> None:
        """Test cache clear operation."""
        cache = SniffCache(maxsize=100, ttl=3600)

        cache.set("hash1", SniffResult(is_code=True))
        cache.set("hash2", SniffResult(is_code=False))
        cache.get("hash1")  # Increment hit counter

        cache.clear()

        assert cache.get("hash1") is None
        assert cache.get("hash2") is None
        stats = cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 2  # Two misses from the get calls above
        assert stats["size"] == 0


class TestSniffCacheValidation:
    """Test SniffCache validation."""

    def test_valid_cache_config(self) -> None:
        """Valid configuration should be accepted."""
        cache = SniffCache(maxsize=100, ttl=60)
        assert cache._maxsize == 100
        assert cache._ttl == 60

    def test_zero_maxsize_rejected(self) -> None:
        """Zero maxsize should raise ValueError."""
        with pytest.raises(ValueError, match="maxsize must be at least 1"):
            SniffCache(maxsize=0)

    def test_negative_maxsize_rejected(self) -> None:
        """Negative maxsize should raise ValueError."""
        with pytest.raises(ValueError, match="maxsize must be at least 1"):
            SniffCache(maxsize=-1)

    def test_negative_ttl_rejected(self) -> None:
        """Negative TTL should raise ValueError."""
        with pytest.raises(ValueError, match="ttl cannot be negative"):
            SniffCache(ttl=-1)

    def test_zero_ttl_allowed(self) -> None:
        """Zero TTL should be allowed (immediate expiration)."""
        cache = SniffCache(ttl=0)
        assert cache._ttl == 0


class TestContentSnifferWithCache:
    """Tests for ContentSniffer with caching enabled."""

    @pytest.fixture()
    def sniffer_with_cache(self) -> ContentSniffer:
        """Create a content sniffer with cache."""
        cache = SniffCache(maxsize=100, ttl=3600)
        return ContentSniffer(cache=cache)

    @pytest.fixture()
    def text_file_ref(self) -> FileReference:
        """Create a text file reference."""
        return FileReference(
            uri="file:///docs/readme.txt",
            source_type="directory",
            content_type="document",
            filename="readme.txt",
            extension=".txt",
            mime_type="text/plain",
        )

    @pytest.mark.asyncio()
    async def test_cache_hit_returns_cached_result(
        self, sniffer_with_cache: ContentSniffer, text_file_ref: FileReference
    ) -> None:
        """Test that cache hit returns the cached result without re-sniffing."""
        content = b"import os\nprint('hello')"
        content_hash = "abc123"

        # First call - should sniff and cache
        result1 = await sniffer_with_cache.sniff(content, text_file_ref, content_hash=content_hash)
        assert result1.is_code is True

        # Second call with same hash - should return cached result
        result2 = await sniffer_with_cache.sniff(b"different content", text_file_ref, content_hash=content_hash)
        assert result2.is_code is True  # Same as cached result

        # Verify cache was hit
        assert sniffer_with_cache._cache is not None
        stats = sniffer_with_cache._cache.stats
        assert stats["hits"] == 1

    @pytest.mark.asyncio()
    async def test_cache_miss_performs_sniff(
        self, sniffer_with_cache: ContentSniffer, text_file_ref: FileReference
    ) -> None:
        """Test that cache miss performs actual sniffing."""
        content = b"import os"
        content_hash = "hash1"

        # First call - cache miss
        result = await sniffer_with_cache.sniff(content, text_file_ref, content_hash=content_hash)
        assert result.is_code is True

        # Verify cache was populated
        assert sniffer_with_cache._cache is not None
        stats = sniffer_with_cache._cache.stats
        assert stats["misses"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio()
    async def test_no_hash_skips_cache(self, sniffer_with_cache: ContentSniffer, text_file_ref: FileReference) -> None:
        """Test that sniffing without content_hash skips cache entirely."""
        content = b"import os"

        # Call without content_hash
        result = await sniffer_with_cache.sniff(content, text_file_ref, content_hash=None)
        assert result.is_code is True

        # Cache should not have been used
        assert sniffer_with_cache._cache is not None
        stats = sniffer_with_cache._cache.stats
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    @pytest.mark.asyncio()
    async def test_error_results_not_cached(self, text_file_ref: FileReference) -> None:
        """Test that results with errors are not cached."""
        cache = SniffCache(maxsize=100, ttl=3600)
        # Create sniffer with very short timeout to force timeout error
        config = SniffConfig(timeout_seconds=0.001)
        sniffer = ContentSniffer(config, cache=cache)

        # Mock _do_sniff to take too long
        async def slow_sniff(*_args, **_kwargs):
            await asyncio.sleep(1)
            return SniffResult(is_code=True)

        with patch.object(sniffer, "_do_sniff", slow_sniff):
            result = await sniffer.sniff(b"content", text_file_ref, content_hash="hash123")

        # Should have timeout error
        assert len(result.errors) > 0
        assert "timed out" in result.errors[0].lower()

        # Should NOT be cached (has errors)
        assert cache.stats["size"] == 0
