#!/usr/bin/env python3

"""
Unit tests for ChunkingSecurityValidator.

This module tests the security validation for chunking operations.
"""

import pytest

from packages.webui.services.chunking_security import ChunkingSecurityValidator, ValidationError
from packages.webui.services.chunking_validation import ChunkingInputValidator


class TestChunkingSecurityValidator:
    """Tests for ChunkingSecurityValidator."""

    def test_validate_chunk_params_valid(self) -> None:
        """Test validation of valid chunk parameters."""
        # Valid parameters
        params = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
        }

        # Should not raise
        ChunkingSecurityValidator.validate_chunk_params(params)

    def test_validate_chunk_params_invalid_size(self) -> None:
        """Test validation rejects invalid chunk size."""
        # Too small
        with pytest.raises(ValidationError, match="chunk_size must be between"):
            ChunkingSecurityValidator.validate_chunk_params({"chunk_size": 10})

        # Too large
        with pytest.raises(ValidationError, match="chunk_size must be between"):
            ChunkingSecurityValidator.validate_chunk_params({"chunk_size": 20000})

        # Not an integer
        with pytest.raises(ValidationError, match="chunk_size must be an integer"):
            ChunkingSecurityValidator.validate_chunk_params({"chunk_size": "1000"})

        # Negative
        with pytest.raises(ValidationError, match="chunk_size must be between"):
            ChunkingSecurityValidator.validate_chunk_params({"chunk_size": -100})

    def test_validate_chunk_params_invalid_overlap(self) -> None:
        """Test validation rejects invalid chunk overlap."""
        # Negative overlap
        with pytest.raises(ValidationError, match="chunk_overlap must be non-negative"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "chunk_size": 1000,
                    "chunk_overlap": -10,
                }
            )

        # Overlap greater than chunk size
        with pytest.raises(ValidationError, match="chunk_overlap .* must be less than"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "chunk_size": 100,
                    "chunk_overlap": 200,
                }
            )

        # Not an integer
        with pytest.raises(ValidationError, match="chunk_overlap must be an integer"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "chunk_overlap": "100",
                }
            )

    def test_validate_chunk_params_semantic(self) -> None:
        """Test validation of semantic chunking parameters."""
        # Valid
        params = {
            "breakpoint_percentile_threshold": 95,
        }
        ChunkingSecurityValidator.validate_chunk_params(params)

        # Invalid type
        with pytest.raises(ValidationError, match="must be a number"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "breakpoint_percentile_threshold": "95",
                }
            )

        # Out of range
        with pytest.raises(ValidationError, match="must be between 0 and 100"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "breakpoint_percentile_threshold": 150,
                }
            )

    def test_validate_chunk_params_hierarchical(self) -> None:
        """Test validation of hierarchical chunking parameters."""
        # Valid
        params = {
            "chunk_sizes": [2048, 512, 128],
        }
        ChunkingSecurityValidator.validate_chunk_params(params)

        # Not a list
        with pytest.raises(ValidationError, match="chunk_sizes must be a list"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "chunk_sizes": "2048,512,128",
                }
            )

        # Invalid values
        with pytest.raises(ValidationError, match="must be positive integers"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "chunk_sizes": [2048, -512, 128],
                }
            )

        # Too many levels
        with pytest.raises(ValidationError, match="Maximum 5 hierarchical levels"):
            ChunkingSecurityValidator.validate_chunk_params(
                {
                    "chunk_sizes": [2048, 1024, 512, 256, 128, 64],
                }
            )

    def test_validate_document_size(self) -> None:
        """Test document size validation."""
        # Valid size
        ChunkingSecurityValidator.validate_document_size(1000)

        # Max allowed size
        max_size = ChunkingSecurityValidator.MAX_DOCUMENT_SIZE
        ChunkingSecurityValidator.validate_document_size(max_size)

        # Too large
        with pytest.raises(ValidationError, match="Document too large"):
            ChunkingSecurityValidator.validate_document_size(max_size + 1)

    def test_validate_document_size_preview(self) -> None:
        """Test document size validation for preview."""
        # Valid preview size
        ChunkingSecurityValidator.validate_document_size(100000, is_preview=True)

        # Max preview size
        max_preview = ChunkingSecurityValidator.MAX_PREVIEW_SIZE
        ChunkingSecurityValidator.validate_document_size(max_preview, is_preview=True)

        # Too large for preview
        with pytest.raises(ValidationError, match="Document too large"):
            ChunkingSecurityValidator.validate_document_size(
                max_preview + 1,
                is_preview=True,
            )

    def test_validate_strategy_name(self) -> None:
        """Test strategy name validation."""
        # Valid names
        ChunkingSecurityValidator.validate_strategy_name("recursive")
        ChunkingSecurityValidator.validate_strategy_name("character_based")
        ChunkingSecurityValidator.validate_strategy_name("semantic123")

        # Invalid characters
        with pytest.raises(ValidationError, match="Only alphanumeric"):
            ChunkingSecurityValidator.validate_strategy_name("recursive-chunker")

        with pytest.raises(ValidationError, match="Only alphanumeric"):
            ChunkingSecurityValidator.validate_strategy_name("recursive.chunker")

        with pytest.raises(ValidationError, match="Only alphanumeric"):
            ChunkingSecurityValidator.validate_strategy_name("recursive@chunker")

        # Too long
        long_name = "a" * 60
        with pytest.raises(ValidationError, match="Strategy name too long"):
            ChunkingSecurityValidator.validate_strategy_name(long_name)

    def test_validate_file_paths(self) -> None:
        """Test file path validation."""
        # Valid paths
        ChunkingSecurityValidator.validate_file_paths(
            [
                "document.txt",
                "folder/file.pdf",
                "deep/nested/path/file.md",
            ]
        )

        # Not a list
        with pytest.raises(ValidationError, match="must be a list"):
            ChunkingSecurityValidator.validate_file_paths("single_file.txt")  # type: ignore

        # Too many paths
        many_paths = ["file.txt"] * 1500
        with pytest.raises(ValidationError, match="Too many file paths"):
            ChunkingSecurityValidator.validate_file_paths(many_paths)

        # Invalid path type
        with pytest.raises(ValidationError, match="must be string"):
            ChunkingSecurityValidator.validate_file_paths([123, 456])  # type: ignore

        # Directory traversal attempt
        with pytest.raises(ValidationError, match="Invalid file path"):
            ChunkingSecurityValidator.validate_file_paths(["../../../etc/passwd"])

        # Absolute path
        with pytest.raises(ValidationError, match="Invalid file path"):
            ChunkingSecurityValidator.validate_file_paths(["/etc/passwd"])

        # Path too long
        long_path = "a" * 1500
        with pytest.raises(ValidationError, match="File path too long"):
            ChunkingSecurityValidator.validate_file_paths([long_path])

    def test_sanitize_text_for_preview(self) -> None:
        """Test text sanitization for preview."""
        # HTML tags
        text = "<script>alert('xss')</script>Hello"
        sanitized = ChunkingSecurityValidator.sanitize_text_for_preview(text)
        assert "<script>" not in sanitized
        assert "alert" in sanitized  # Content preserved
        assert "Hello" in sanitized

        # Length limiting
        long_text = "x" * 300
        sanitized = ChunkingSecurityValidator.sanitize_text_for_preview(long_text)
        assert len(sanitized) <= 203  # 200 + "..."
        assert sanitized.endswith("...")

        # Special characters
        text = 'Test "quoted" text\nNew line\tTab'
        sanitized = ChunkingSecurityValidator.sanitize_text_for_preview(text)
        assert '\\"' in sanitized
        assert "\\n" in sanitized
        assert "\\t" in sanitized

        # Backslashes
        text = "Path: C:\\Users\\test"
        sanitized = ChunkingSecurityValidator.sanitize_text_for_preview(text)
        assert "\\\\" in sanitized

    def test_xss_prevention_in_metadata(self) -> None:
        """Test XSS prevention in metadata sanitization."""
        # Test various XSS vectors
        xss_vectors = [
            '<script>alert("XSS")</script>',
            '"><script>alert("XSS")</script>',
            '<img src=x onerror=alert("XSS")>',
            'javascript:alert("XSS")',
            '<svg onload=alert("XSS")>',
            "<iframe src=\"javascript:alert('XSS')\">",
            '<body onload=alert("XSS")>',
        ]

        for vector in xss_vectors:
            metadata = {"user_input": vector, "safe_key": "safe_value"}
            sanitized = ChunkingInputValidator.sanitize_metadata(metadata)

            # XSS content should be sanitized
            assert sanitized["user_input"] == "[Content removed for security]"
            # Safe content should be preserved
            assert sanitized["safe_key"] == "safe_value"

    def test_comprehensive_html_escaping(self) -> None:
        """Test that all HTML special characters are escaped."""
        metadata = {
            "quotes": "Test \"double\" and 'single' quotes",
            "ampersand": "Test & ampersand",
            "less_than": "Test < less than",
            "greater_than": "Test > greater than",
        }

        sanitized = ChunkingInputValidator.sanitize_metadata(metadata)

        # All special characters should be escaped
        assert "&quot;" in sanitized["quotes"] or "&#x27;" in sanitized["quotes"]
        assert "&amp;" in sanitized["ampersand"]
        # Note: < and > in isolation might be allowed, but in dangerous contexts should be blocked

    def test_nested_xss_prevention(self) -> None:
        """Test XSS prevention in nested metadata structures."""
        metadata = {
            "level1": {
                "safe": "clean data",
                "xss": "<script>alert('nested')</script>",
                "level2": {"another_xss": "javascript:void(0)"},
            }
        }

        sanitized = ChunkingInputValidator.sanitize_metadata(metadata)

        # Nested XSS should be sanitized
        assert sanitized["level1"]["xss"] == "[Content removed for security]"
        assert sanitized["level1"]["level2"]["another_xss"] == "[Content removed for security]"
        # Safe data should be preserved
        assert sanitized["level1"]["safe"] == "clean data"

    def test_null_byte_injection(self) -> None:
        """Test null byte injection prevention."""
        metadata = {"null_byte": "test\x00data", "normal": "clean data"}

        sanitized = ChunkingInputValidator.sanitize_metadata(metadata)

        # Null bytes should be removed
        assert "\x00" not in sanitized["null_byte"]
        assert sanitized["normal"] == "clean data"

    def test_validate_collection_config(self) -> None:
        """Test complete collection configuration validation."""
        # Valid config
        config = {
            "strategy": "recursive",
            "params": {
                "chunk_size": 600,
                "chunk_overlap": 100,
            },
            "metadata": {"source": "test"},
        }
        ChunkingSecurityValidator.validate_collection_config(config)

        # Not a dict
        with pytest.raises(ValidationError, match="Config must be a dictionary"):
            ChunkingSecurityValidator.validate_collection_config("not_a_dict")  # type: ignore

        # Missing strategy
        with pytest.raises(ValidationError, match="must include 'strategy'"):
            ChunkingSecurityValidator.validate_collection_config({})

        # Invalid strategy name
        with pytest.raises(ValidationError, match="Invalid strategy name"):
            ChunkingSecurityValidator.validate_collection_config(
                {
                    "strategy": "../../etc/passwd",
                }
            )

        # Invalid params type
        with pytest.raises(ValidationError, match="params must be a dictionary"):
            ChunkingSecurityValidator.validate_collection_config(
                {
                    "strategy": "recursive",
                    "params": "chunk_size=100",
                }
            )

        # Unknown fields (should log warning but not fail)
        config_with_unknown = {
            "strategy": "recursive",
            "params": {},
            "unknown_field": "value",
            "another_unknown": 123,
        }
        # Should not raise
        ChunkingSecurityValidator.validate_collection_config(config_with_unknown)

    def test_estimate_memory_usage(self) -> None:
        """Test memory usage estimation."""
        # Small document
        memory = ChunkingSecurityValidator.estimate_memory_usage(
            text_length=1000,
            chunk_size=100,
            strategy="character",
        )
        assert memory > 0
        assert memory < 1024 * 1024  # Less than 1MB

        # Large document with semantic strategy
        memory = ChunkingSecurityValidator.estimate_memory_usage(
            text_length=1_000_000,  # 1M chars
            chunk_size=1000,
            strategy="semantic",
        )
        assert memory > 1024 * 1024  # More than 1MB
        assert memory < 100 * 1024 * 1024  # Less than 100MB

        # Different strategies should have different multipliers
        memory_char = ChunkingSecurityValidator.estimate_memory_usage(
            text_length=100_000,
            chunk_size=1000,
            strategy="character",
        )

        memory_semantic = ChunkingSecurityValidator.estimate_memory_usage(
            text_length=100_000,
            chunk_size=1000,
            strategy="semantic",
        )

        assert memory_semantic > memory_char  # Semantic uses more memory

        # Unknown strategy should use default multiplier
        memory_unknown = ChunkingSecurityValidator.estimate_memory_usage(
            text_length=100_000,
            chunk_size=1000,
            strategy="unknown_strategy",
        )
        assert memory_unknown > 0
