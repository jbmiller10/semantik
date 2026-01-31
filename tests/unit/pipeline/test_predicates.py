"""Unit tests for pipeline predicate matching."""

import pytest

from shared.pipeline.predicates import _translate_legacy_path, get_nested_value, match_value, matches_predicate
from shared.pipeline.types import FileReference


class TestGetNestedValue:
    """Tests for get_nested_value helper function."""

    def test_simple_dict_key(self) -> None:
        """Test getting a simple dict key."""
        obj = {"name": "test", "value": 42}
        assert get_nested_value(obj, "name") == "test"
        assert get_nested_value(obj, "value") == 42

    def test_nested_dict_path(self) -> None:
        """Test getting nested dict value with dot notation."""
        obj = {"metadata": {"author": "John", "tags": ["a", "b"]}}
        assert get_nested_value(obj, "metadata.author") == "John"
        assert get_nested_value(obj, "metadata.tags") == ["a", "b"]

    def test_deeply_nested_path(self) -> None:
        """Test deeply nested paths."""
        obj = {"a": {"b": {"c": {"d": "deep"}}}}
        assert get_nested_value(obj, "a.b.c.d") == "deep"

    def test_missing_key_returns_none(self) -> None:
        """Test that missing key returns None."""
        obj = {"name": "test"}
        assert get_nested_value(obj, "missing") is None

    def test_missing_nested_key_returns_none(self) -> None:
        """Test that missing nested key returns None."""
        obj = {"metadata": {"author": "John"}}
        assert get_nested_value(obj, "metadata.missing") is None
        assert get_nested_value(obj, "missing.key") is None

    def test_none_in_path_returns_none(self) -> None:
        """Test that None value in path returns None."""
        obj = {"metadata": None}
        assert get_nested_value(obj, "metadata.author") is None

    def test_dataclass_attribute(self) -> None:
        """Test getting attribute from dataclass."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            mime_type="application/pdf",
        )
        assert get_nested_value(ref, "uri") == "file:///doc.pdf"
        assert get_nested_value(ref, "mime_type") == "application/pdf"

    def test_dataclass_nested_dict(self) -> None:
        """Test getting nested dict from dataclass with new metadata format."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            metadata={"source": {"language": "zh", "author": "Test"}},
        )
        assert get_nested_value(ref, "metadata.source.language") == "zh"
        assert get_nested_value(ref, "metadata.source.author") == "Test"


class TestTranslateLegacyPath:
    """Tests for _translate_legacy_path helper function."""

    def test_translate_source_metadata_dot_path(self) -> None:
        """Test translating source_metadata.X to metadata.source.X."""
        assert _translate_legacy_path("source_metadata.language") == "metadata.source.language"
        assert _translate_legacy_path("source_metadata.author") == "metadata.source.author"
        assert _translate_legacy_path("source_metadata.deep.nested") == "metadata.source.deep.nested"

    def test_translate_source_metadata_root(self) -> None:
        """Test translating bare source_metadata to metadata.source."""
        assert _translate_legacy_path("source_metadata") == "metadata.source"

    def test_no_translation_for_new_paths(self) -> None:
        """Test that new metadata paths are not modified."""
        assert _translate_legacy_path("metadata.source.language") == "metadata.source.language"
        assert _translate_legacy_path("metadata.detected.type") == "metadata.detected.type"

    def test_no_translation_for_other_fields(self) -> None:
        """Test that non-metadata fields are not modified."""
        assert _translate_legacy_path("mime_type") == "mime_type"
        assert _translate_legacy_path("extension") == "extension"
        assert _translate_legacy_path("size_bytes") == "size_bytes"
        assert _translate_legacy_path("uri") == "uri"

    def test_no_translation_for_partial_match(self) -> None:
        """Test that partial matches are not translated."""
        # Should not translate source_metadata_extra (doesn't start with source_metadata.)
        assert _translate_legacy_path("source_metadata_extra") == "source_metadata_extra"


class TestMatchValue:
    """Tests for match_value helper function."""

    def test_none_pattern_always_matches(self) -> None:
        """Test that None pattern matches any value."""
        assert match_value(None, "anything") is True
        assert match_value(None, 42) is True
        assert match_value(None, None) is True
        assert match_value(None, []) is True

    def test_none_value_not_matched_by_non_none(self) -> None:
        """Test that None value is not matched by non-None pattern."""
        assert match_value("test", None) is False
        assert match_value(42, None) is False
        assert match_value(True, None) is False

    def test_exact_string_match(self) -> None:
        """Test exact string matching."""
        assert match_value("application/pdf", "application/pdf") is True
        assert match_value("application/pdf", "text/plain") is False
        assert match_value("test", "test") is True
        assert match_value("test", "Test") is False  # Case sensitive

    def test_glob_pattern_asterisk(self) -> None:
        """Test glob pattern with asterisk."""
        assert match_value("application/*", "application/pdf") is True
        assert match_value("application/*", "application/json") is True
        assert match_value("application/*", "text/plain") is False
        assert match_value("*.pdf", "document.pdf") is True
        assert match_value("*.pdf", "document.txt") is False

    def test_glob_pattern_question_mark(self) -> None:
        """Test glob pattern with question mark."""
        assert match_value("file?.txt", "file1.txt") is True
        assert match_value("file?.txt", "file2.txt") is True
        assert match_value("file?.txt", "file12.txt") is False

    def test_glob_pattern_brackets(self) -> None:
        """Test glob pattern with character brackets."""
        assert match_value("file[123].txt", "file1.txt") is True
        assert match_value("file[123].txt", "file2.txt") is True
        assert match_value("file[123].txt", "file4.txt") is False
        assert match_value("file[a-z].txt", "filea.txt") is True

    def test_negation_pattern(self) -> None:
        """Test negation with ! prefix."""
        assert match_value("!image/*", "application/pdf") is True
        assert match_value("!image/*", "image/png") is False
        assert match_value("!test", "test") is False
        assert match_value("!test", "other") is True

    def test_string_boolean_patterns_match_boolean_values(self) -> None:
        """Test that string 'true'/'false' patterns work against boolean values (including negation)."""
        assert match_value("true", True) is True
        assert match_value("true", False) is False
        assert match_value("false", False) is True
        assert match_value("false", True) is False

        assert match_value("!true", True) is False
        assert match_value("!true", False) is True
        assert match_value("!false", False) is False
        assert match_value("!false", True) is True

    def test_numeric_greater_than(self) -> None:
        """Test numeric > comparison."""
        assert match_value(">100", 150) is True
        assert match_value(">100", 100) is False
        assert match_value(">100", 50) is False
        assert match_value(">100", "150") is True  # String converted to number

    def test_numeric_greater_equal(self) -> None:
        """Test numeric >= comparison."""
        assert match_value(">=100", 150) is True
        assert match_value(">=100", 100) is True
        assert match_value(">=100", 50) is False

    def test_numeric_less_than(self) -> None:
        """Test numeric < comparison."""
        assert match_value("<100", 50) is True
        assert match_value("<100", 100) is False
        assert match_value("<100", 150) is False

    def test_numeric_less_equal(self) -> None:
        """Test numeric <= comparison."""
        assert match_value("<=100", 50) is True
        assert match_value("<=100", 100) is True
        assert match_value("<=100", 150) is False

    def test_numeric_equals(self) -> None:
        """Test numeric == comparison."""
        assert match_value("==100", 100) is True
        assert match_value("==100", 100.0) is True
        assert match_value("==100", 50) is False

    def test_numeric_not_equals(self) -> None:
        """Test numeric != comparison."""
        assert match_value("!=100", 50) is True
        assert match_value("!=100", 100) is False

    def test_numeric_with_spaces(self) -> None:
        """Test numeric comparison with spaces."""
        assert match_value("> 100", 150) is True
        assert match_value("<= 50", 50) is True

    def test_numeric_with_negative(self) -> None:
        """Test numeric comparison with negative numbers."""
        assert match_value(">-50", 0) is True
        assert match_value(">-50", -100) is False

    def test_numeric_with_decimal(self) -> None:
        """Test numeric comparison with decimal numbers."""
        assert match_value(">1.5", 2.0) is True
        assert match_value(">1.5", 1.0) is False

    def test_numeric_invalid_value(self) -> None:
        """Test numeric comparison with non-numeric value."""
        assert match_value(">100", "not a number") is False
        assert match_value(">100", None) is False

    def test_array_or_matching(self) -> None:
        """Test array pattern (OR logic)."""
        assert match_value([".md", ".txt"], ".md") is True
        assert match_value([".md", ".txt"], ".txt") is True
        assert match_value([".md", ".txt"], ".pdf") is False

    def test_array_with_patterns(self) -> None:
        """Test array with glob patterns."""
        assert match_value(["text/*", "application/json"], "text/plain") is True
        assert match_value(["text/*", "application/json"], "application/json") is True
        assert match_value(["text/*", "application/json"], "image/png") is False

    def test_array_with_negation(self) -> None:
        """Test array with negation patterns."""
        # Matches if any pattern matches
        assert match_value(["!image/*", "!video/*"], "application/pdf") is True
        assert match_value(["image/*", "!text/*"], "image/png") is True

    def test_boolean_pattern_true(self) -> None:
        """Test boolean pattern matching True."""
        assert match_value(True, True) is True
        assert match_value(True, False) is False
        assert match_value(True, "true") is True
        assert match_value(True, "yes") is True
        assert match_value(True, "1") is True

    def test_boolean_pattern_false(self) -> None:
        """Test boolean pattern matching False."""
        assert match_value(False, False) is True
        assert match_value(False, True) is False
        assert match_value(False, "false") is True
        assert match_value(False, "no") is True
        assert match_value(False, "0") is True

    def test_numeric_pattern_exact(self) -> None:
        """Test numeric pattern (exact match)."""
        assert match_value(100, 100) is True
        assert match_value(100, 100.0) is True
        assert match_value(100, "100") is True
        assert match_value(100, 50) is False

    def test_numeric_pattern_float(self) -> None:
        """Test float pattern."""
        assert match_value(3.14, 3.14) is True
        assert match_value(3.14, "3.14") is True
        assert match_value(3.14, 3.15) is False


class TestMatchesPredicate:
    """Tests for matches_predicate function."""

    @pytest.fixture()
    def pdf_file_ref(self) -> FileReference:
        """Create a PDF file reference for testing."""
        return FileReference(
            uri="file:///docs/report.pdf",
            source_type="directory",
            content_type="document",
            filename="report.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=1024000,  # 1MB
            metadata={"source": {"language": "en", "pages": 10}},
        )

    @pytest.fixture()
    def markdown_file_ref(self) -> FileReference:
        """Create a Markdown file reference for testing."""
        return FileReference(
            uri="file:///docs/readme.md",
            source_type="directory",
            content_type="document",
            filename="readme.md",
            extension=".md",
            mime_type="text/markdown",
            size_bytes=5000,
            metadata={"source": {"language": "zh", "encoding": "utf-8"}},
        )

    def test_catch_all_none(self, pdf_file_ref: FileReference) -> None:
        """Test that None predicate always matches (catch-all)."""
        assert matches_predicate(pdf_file_ref, None) is True

    def test_catch_all_empty_dict(self, pdf_file_ref: FileReference) -> None:
        """Test that empty dict predicate always matches (catch-all)."""
        assert matches_predicate(pdf_file_ref, {}) is True

    def test_exact_mime_type_match(self, pdf_file_ref: FileReference) -> None:
        """Test exact mime_type matching."""
        assert matches_predicate(pdf_file_ref, {"mime_type": "application/pdf"}) is True
        assert matches_predicate(pdf_file_ref, {"mime_type": "text/plain"}) is False

    def test_exact_extension_match(self, pdf_file_ref: FileReference) -> None:
        """Test exact extension matching."""
        assert matches_predicate(pdf_file_ref, {"extension": ".pdf"}) is True
        assert matches_predicate(pdf_file_ref, {"extension": ".txt"}) is False

    def test_glob_mime_type(self, pdf_file_ref: FileReference, markdown_file_ref: FileReference) -> None:
        """Test glob pattern for mime_type."""
        assert matches_predicate(pdf_file_ref, {"mime_type": "application/*"}) is True
        assert matches_predicate(markdown_file_ref, {"mime_type": "application/*"}) is False
        assert matches_predicate(markdown_file_ref, {"mime_type": "text/*"}) is True

    def test_negation_mime_type(self, pdf_file_ref: FileReference) -> None:
        """Test negation pattern for mime_type."""
        assert matches_predicate(pdf_file_ref, {"mime_type": "!image/*"}) is True
        assert matches_predicate(pdf_file_ref, {"mime_type": "!application/*"}) is False

    def test_numeric_size_comparison(self, pdf_file_ref: FileReference, markdown_file_ref: FileReference) -> None:
        """Test numeric comparison for size_bytes."""
        # PDF is 1MB (1024000 bytes)
        assert matches_predicate(pdf_file_ref, {"size_bytes": ">100000"}) is True
        assert matches_predicate(pdf_file_ref, {"size_bytes": "<100000"}) is False
        # Markdown is 5000 bytes
        assert matches_predicate(markdown_file_ref, {"size_bytes": "<10000"}) is True
        assert matches_predicate(markdown_file_ref, {"size_bytes": ">=5000"}) is True
        assert matches_predicate(markdown_file_ref, {"size_bytes": ">5000"}) is False

    def test_array_extension_match(self, pdf_file_ref: FileReference, markdown_file_ref: FileReference) -> None:
        """Test array OR matching for extension."""
        predicate = {"extension": [".md", ".txt", ".rst"]}
        assert matches_predicate(markdown_file_ref, predicate) is True
        assert matches_predicate(pdf_file_ref, predicate) is False

    def test_nested_field_match_new_format(self, pdf_file_ref: FileReference, markdown_file_ref: FileReference) -> None:
        """Test nested field access with new metadata.source path."""
        assert matches_predicate(pdf_file_ref, {"metadata.source.language": "en"}) is True
        assert matches_predicate(pdf_file_ref, {"metadata.source.language": "zh"}) is False
        assert matches_predicate(markdown_file_ref, {"metadata.source.language": "zh"}) is True

    def test_nested_field_match_legacy_format(
        self, pdf_file_ref: FileReference, markdown_file_ref: FileReference
    ) -> None:
        """Test nested field access with legacy source_metadata path (auto-translated)."""
        # Legacy paths should be automatically translated to metadata.source
        assert matches_predicate(pdf_file_ref, {"source_metadata.language": "en"}) is True
        assert matches_predicate(pdf_file_ref, {"source_metadata.language": "zh"}) is False
        assert matches_predicate(markdown_file_ref, {"source_metadata.language": "zh"}) is True

    def test_combined_and_predicates(self, pdf_file_ref: FileReference) -> None:
        """Test multiple fields (AND logic)."""
        # Both conditions must match
        predicate = {"mime_type": "application/pdf", "size_bytes": ">500000"}
        assert matches_predicate(pdf_file_ref, predicate) is True

        # One condition fails
        predicate = {"mime_type": "application/pdf", "size_bytes": ">2000000"}
        assert matches_predicate(pdf_file_ref, predicate) is False

        # Both conditions fail
        predicate = {"mime_type": "text/plain", "size_bytes": ">2000000"}
        assert matches_predicate(pdf_file_ref, predicate) is False

    def test_complex_predicate(self, pdf_file_ref: FileReference) -> None:
        """Test complex predicate with multiple conditions using new format."""
        predicate = {
            "mime_type": "application/*",
            "extension": [".pdf", ".docx"],
            "size_bytes": "<=2000000",
            "metadata.source.language": "en",
        }
        assert matches_predicate(pdf_file_ref, predicate) is True

    def test_complex_predicate_legacy_format(self, pdf_file_ref: FileReference) -> None:
        """Test complex predicate with legacy source_metadata path."""
        predicate = {
            "mime_type": "application/*",
            "extension": [".pdf", ".docx"],
            "size_bytes": "<=2000000",
            "source_metadata.language": "en",  # Legacy path - auto translated
        }
        assert matches_predicate(pdf_file_ref, predicate) is True

    def test_none_field_value(self) -> None:
        """Test matching when field value is None."""
        ref = FileReference(
            uri="file:///doc",
            source_type="directory",
            content_type="document",
            mime_type=None,  # None value
        )
        # Non-None pattern should not match None value
        assert matches_predicate(ref, {"mime_type": "application/pdf"}) is False
        # None pattern should match (catch-all behavior)
        assert matches_predicate(ref, {}) is True

    def test_empty_string_field(self) -> None:
        """Test matching empty string field."""
        ref = FileReference(
            uri="file:///doc",
            source_type="directory",
            content_type="document",
            extension="",  # Empty string
        )
        assert matches_predicate(ref, {"extension": ""}) is True
        assert matches_predicate(ref, {"extension": ".pdf"}) is False

    def test_missing_nested_field(self, pdf_file_ref: FileReference) -> None:
        """Test matching when nested field doesn't exist."""
        # metadata.source doesn't have "author" key
        assert matches_predicate(pdf_file_ref, {"metadata.source.author": "John"}) is False
        # Legacy path should also work
        assert matches_predicate(pdf_file_ref, {"source_metadata.author": "John"}) is False

    def test_source_type_match(self, pdf_file_ref: FileReference) -> None:
        """Test matching source_type field."""
        assert matches_predicate(pdf_file_ref, {"source_type": "directory"}) is True
        assert matches_predicate(pdf_file_ref, {"source_type": "web"}) is False

    def test_content_type_match(self, pdf_file_ref: FileReference) -> None:
        """Test matching content_type field."""
        assert matches_predicate(pdf_file_ref, {"content_type": "document"}) is True
        assert matches_predicate(pdf_file_ref, {"content_type": "message"}) is False


class TestPredicateEdgeCases:
    """Tests for edge cases in predicate matching."""

    def test_unicode_values(self) -> None:
        """Test matching Unicode values."""
        ref = FileReference(
            uri="file:///文档.pdf",
            source_type="directory",
            content_type="document",
            filename="文档.pdf",
            metadata={"source": {"author": "张三"}},
        )
        assert matches_predicate(ref, {"filename": "文档.pdf"}) is True
        assert matches_predicate(ref, {"metadata.source.author": "张三"}) is True
        # Legacy path still works
        assert matches_predicate(ref, {"source_metadata.author": "张三"}) is True

    def test_special_characters_in_values(self) -> None:
        """Test matching values with special characters."""
        ref = FileReference(
            uri="file:///path/to/file (1).pdf",
            source_type="directory",
            content_type="document",
            filename="file (1).pdf",
        )
        assert matches_predicate(ref, {"filename": "file (1).pdf"}) is True

    def test_numeric_string_field(self) -> None:
        """Test numeric comparison on string field."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            metadata={"source": {"version": "2"}},
        )
        # String "2" should be comparable
        assert matches_predicate(ref, {"metadata.source.version": ">1"}) is True
        assert matches_predicate(ref, {"metadata.source.version": "<1"}) is False
        # Legacy paths
        assert matches_predicate(ref, {"source_metadata.version": ">1"}) is True

    def test_very_large_size(self) -> None:
        """Test matching very large file sizes."""
        ref = FileReference(
            uri="file:///large.bin",
            source_type="directory",
            content_type="binary",
            size_bytes=10_000_000_000,  # 10GB
        )
        assert matches_predicate(ref, {"size_bytes": ">1000000000"}) is True
        assert matches_predicate(ref, {"size_bytes": ">=10000000000"}) is True

    def test_zero_size(self) -> None:
        """Test matching zero-byte file."""
        ref = FileReference(
            uri="file:///empty.txt",
            source_type="directory",
            content_type="document",
            size_bytes=0,
        )
        assert matches_predicate(ref, {"size_bytes": "==0"}) is True
        assert matches_predicate(ref, {"size_bytes": ">0"}) is False

    def test_array_with_single_element(self) -> None:
        """Test array pattern with single element."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            extension=".pdf",
        )
        assert matches_predicate(ref, {"extension": [".pdf"]}) is True

    def test_empty_array_pattern(self) -> None:
        """Test empty array pattern (no elements match)."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            extension=".pdf",
        )
        # Empty array means no patterns to match against
        assert matches_predicate(ref, {"extension": []}) is False

    def test_deeply_nested_metadata(self) -> None:
        """Test deeply nested metadata.source access."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            metadata={
                "source": {
                    "custom": {
                        "deep": {
                            "value": "found",
                        },
                    },
                },
            },
        )
        # New format
        assert matches_predicate(ref, {"metadata.source.custom.deep.value": "found"}) is True
        assert matches_predicate(ref, {"metadata.source.custom.deep.missing": "x"}) is False
        # Legacy format (translated)
        assert matches_predicate(ref, {"source_metadata.custom.deep.value": "found"}) is True
        assert matches_predicate(ref, {"source_metadata.custom.deep.missing": "x"}) is False
