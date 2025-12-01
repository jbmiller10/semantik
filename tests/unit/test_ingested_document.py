"""Unit tests for IngestedDocument DTO."""

import pytest

from shared.dtos.ingestion import IngestedDocument
from shared.utils.hashing import compute_content_hash


class TestIngestedDocument:
    """Tests for IngestedDocument dataclass."""

    def test_valid_construction_all_fields(self) -> None:
        """Test creating IngestedDocument with all fields."""
        doc = IngestedDocument(
            content="Test document content",
            unique_id="file:///path/to/doc.txt",
            source_type="directory",
            metadata={"author": "test", "created": "2025-01-01"},
            content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            file_path="/path/to/doc.txt",
        )

        assert doc.content == "Test document content"
        assert doc.unique_id == "file:///path/to/doc.txt"
        assert doc.source_type == "directory"
        assert doc.metadata == {"author": "test", "created": "2025-01-01"}
        assert doc.content_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert doc.file_path == "/path/to/doc.txt"

    def test_file_path_optional(self) -> None:
        """Test that file_path can be None (default)."""
        doc = IngestedDocument(
            content="Web content",
            unique_id="https://example.com/page",
            source_type="web",
            metadata={"url": "https://example.com/page"},
            content_hash="a" * 64,  # Valid 64-char hex
        )

        assert doc.file_path is None

    def test_valid_hash_accepted(self) -> None:
        """Test that valid 64-char lowercase hex hash is accepted."""
        valid_hash = compute_content_hash("test content")
        doc = IngestedDocument(
            content="test content",
            unique_id="test-id",
            source_type="test",
            metadata={},
            content_hash=valid_hash,
        )
        assert doc.content_hash == valid_hash

    def test_invalid_hash_too_short(self) -> None:
        """Test that hash shorter than 64 chars raises ValueError."""
        with pytest.raises(ValueError, match="64-character lowercase hex string"):
            IngestedDocument(
                content="content",
                unique_id="id",
                source_type="test",
                metadata={},
                content_hash="abc123",  # Too short
            )

    def test_invalid_hash_too_long(self) -> None:
        """Test that hash longer than 64 chars raises ValueError."""
        with pytest.raises(ValueError, match="64-character lowercase hex string"):
            IngestedDocument(
                content="content",
                unique_id="id",
                source_type="test",
                metadata={},
                content_hash="a" * 65,  # Too long
            )

    def test_invalid_hash_uppercase(self) -> None:
        """Test that uppercase hex chars in hash raise ValueError."""
        with pytest.raises(ValueError, match="64-character lowercase hex string"):
            IngestedDocument(
                content="content",
                unique_id="id",
                source_type="test",
                metadata={},
                content_hash="A" * 64,  # Uppercase not allowed
            )

    def test_invalid_hash_non_hex(self) -> None:
        """Test that non-hex chars in hash raise ValueError."""
        with pytest.raises(ValueError, match="64-character lowercase hex string"):
            IngestedDocument(
                content="content",
                unique_id="id",
                source_type="test",
                metadata={},
                content_hash="g" * 64,  # 'g' is not valid hex
            )

    def test_empty_metadata_allowed(self) -> None:
        """Test that empty metadata dict is allowed."""
        doc = IngestedDocument(
            content="content",
            unique_id="id",
            source_type="test",
            metadata={},
            content_hash="a" * 64,
        )
        assert doc.metadata == {}

    def test_web_source_example(self) -> None:
        """Test typical web source document."""
        content = "<html>Page content</html>"
        doc = IngestedDocument(
            content=content,
            unique_id="https://example.com/page",
            source_type="web",
            metadata={
                "url": "https://example.com/page",
                "status_code": 200,
                "content_type": "text/html",
            },
            content_hash=compute_content_hash(content),
        )

        assert doc.source_type == "web"
        assert doc.unique_id == "https://example.com/page"
        assert doc.file_path is None

    def test_slack_source_example(self) -> None:
        """Test typical Slack source document."""
        content = "Hello team, here's an update..."
        doc = IngestedDocument(
            content=content,
            unique_id="slack://C12345/p1234567890",
            source_type="slack",
            metadata={
                "channel_id": "C12345",
                "message_ts": "1234567890.123456",
                "user_id": "U12345",
            },
            content_hash=compute_content_hash(content),
        )

        assert doc.source_type == "slack"
        assert "channel_id" in doc.metadata

    def test_directory_source_example(self) -> None:
        """Test typical directory/file source document."""
        content = "File content here"
        doc = IngestedDocument(
            content=content,
            unique_id="file:///home/user/docs/readme.md",
            source_type="directory",
            metadata={
                "file_size": 1234,
                "mime_type": "text/markdown",
                "modified_at": "2025-01-01T00:00:00Z",
            },
            content_hash=compute_content_hash(content),
            file_path="/home/user/docs/readme.md",
        )

        assert doc.source_type == "directory"
        assert doc.file_path == "/home/user/docs/readme.md"
