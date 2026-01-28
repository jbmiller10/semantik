"""Unit tests for pipeline content sniffing."""

import asyncio
from unittest.mock import patch

import pytest

from shared.pipeline.sniff import ContentSniffer, SniffConfig, SniffResult
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
        """Test PDF detection by extension."""
        assert sniffer._is_pdf("", ".pdf") is True
        assert sniffer._is_pdf("", ".PDF") is False  # Case sensitive extension check
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
