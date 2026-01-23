"""Unit tests for pipeline loader."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.pipeline.loader import LoadError, PipelineLoader
from shared.pipeline.types import FileReference


class TestPipelineLoader:
    """Tests for PipelineLoader class."""

    @pytest.fixture()
    def temp_file(self) -> Path:
        """Create a temporary file with content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Hello, World!")
            return Path(f.name)

    @pytest.fixture()
    def loader(self) -> PipelineLoader:
        """Create a loader without connector."""
        return PipelineLoader()

    def test_init_without_connector(self) -> None:
        """Test initializing loader without connector."""
        loader = PipelineLoader()
        assert loader.connector is None

    def test_init_with_connector(self) -> None:
        """Test initializing loader with connector."""
        mock_connector = MagicMock()
        loader = PipelineLoader(connector=mock_connector)
        assert loader.connector is mock_connector

    @pytest.mark.asyncio()
    async def test_load_from_local_path(self, temp_file: Path) -> None:
        """Test loading content via local_path in source_metadata."""
        loader = PipelineLoader()
        file_ref = FileReference(
            uri=f"file://{temp_file}",
            source_type="directory",
            content_type="document",
            size_bytes=13,
            source_metadata={"local_path": str(temp_file)},
        )

        result = await loader.load(file_ref)

        assert result.content == b"Hello, World!"
        assert result.file_ref == file_ref
        assert len(result.content_hash) == 64
        assert result.retention == "ephemeral"
        assert result.local_path == str(temp_file)

        # Verify hash is correct
        expected_hash = hashlib.sha256(b"Hello, World!").hexdigest()
        assert result.content_hash == expected_hash

    @pytest.mark.asyncio()
    async def test_load_from_file_uri(self, temp_file: Path) -> None:
        """Test loading content from file:// URI."""
        loader = PipelineLoader()
        file_ref = FileReference(
            uri=f"file://{temp_file}",
            source_type="directory",
            content_type="document",
            size_bytes=13,
        )

        result = await loader.load(file_ref)

        assert result.content == b"Hello, World!"

    @pytest.mark.asyncio()
    async def test_load_file_not_found(self) -> None:
        """Test loading non-existent file raises LoadError."""
        loader = PipelineLoader()
        file_ref = FileReference(
            uri="file:///nonexistent/path/file.txt",
            source_type="directory",
            content_type="document",
            size_bytes=0,
            source_metadata={"local_path": "/nonexistent/path/file.txt"},
        )

        with pytest.raises(LoadError) as exc_info:
            await loader.load(file_ref)

        assert "nonexistent" in exc_info.value.file_uri
        assert "not found" in exc_info.value.reason.lower() or "File not found" in exc_info.value.reason

    @pytest.mark.asyncio()
    async def test_load_directory_raises_error(self, tmp_path: Path) -> None:
        """Test loading a directory raises LoadError."""
        loader = PipelineLoader()
        file_ref = FileReference(
            uri=f"file://{tmp_path}",
            source_type="directory",
            content_type="document",
            size_bytes=0,
            source_metadata={"local_path": str(tmp_path)},
        )

        with pytest.raises(LoadError) as exc_info:
            await loader.load(file_ref)

        assert "Not a file" in exc_info.value.reason

    @pytest.mark.asyncio()
    async def test_load_via_connector(self) -> None:
        """Test loading content via connector's load_content method."""
        mock_connector = MagicMock()
        mock_connector.load_content = AsyncMock(return_value=b"Connector content")

        loader = PipelineLoader(connector=mock_connector)
        file_ref = FileReference(
            uri="imap://test@example.com/INBOX/123",
            source_type="imap",
            content_type="message",
            size_bytes=17,
        )

        result = await loader.load(file_ref)

        assert result.content == b"Connector content"
        mock_connector.load_content.assert_called_once_with(file_ref)

    @pytest.mark.asyncio()
    async def test_load_no_connector_for_non_file_uri(self) -> None:
        """Test loading non-file:// URI without connector raises error."""
        loader = PipelineLoader()  # No connector
        file_ref = FileReference(
            uri="https://example.com/doc.pdf",
            source_type="web",
            content_type="document",
            size_bytes=1000,
        )

        with pytest.raises(LoadError) as exc_info:
            await loader.load(file_ref)

        assert "No connector configured" in exc_info.value.reason

    @pytest.mark.asyncio()
    async def test_hash_computation_is_sha256(self, temp_file: Path) -> None:
        """Test that hash is computed using SHA-256."""
        content = temp_file.read_bytes()
        expected_hash = hashlib.sha256(content).hexdigest()

        loader = PipelineLoader()
        file_ref = FileReference(
            uri=f"file://{temp_file}",
            source_type="directory",
            content_type="document",
            size_bytes=len(content),
            source_metadata={"local_path": str(temp_file)},
        )

        result = await loader.load(file_ref)

        assert result.content_hash == expected_hash
        assert len(result.content_hash) == 64  # SHA-256 produces 64 hex chars

    @pytest.mark.asyncio()
    async def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading empty file."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        loader = PipelineLoader()
        file_ref = FileReference(
            uri=f"file://{empty_file}",
            source_type="directory",
            content_type="document",
            size_bytes=0,
            source_metadata={"local_path": str(empty_file)},
        )

        result = await loader.load(file_ref)

        assert result.content == b""
        # SHA-256 of empty string
        assert result.content_hash == hashlib.sha256(b"").hexdigest()

    @pytest.mark.asyncio()
    async def test_load_binary_file(self, tmp_path: Path) -> None:
        """Test loading binary file."""
        binary_file = tmp_path / "binary.bin"
        binary_content = bytes(range(256))
        binary_file.write_bytes(binary_content)

        loader = PipelineLoader()
        file_ref = FileReference(
            uri=f"file://{binary_file}",
            source_type="directory",
            content_type="document",
            size_bytes=256,
            source_metadata={"local_path": str(binary_file)},
        )

        result = await loader.load(file_ref)

        assert result.content == binary_content
        assert result.content_hash == hashlib.sha256(binary_content).hexdigest()


    @pytest.mark.asyncio()
    async def test_load_permission_denied(self, tmp_path: Path) -> None:
        """Test loading file with no read permissions raises LoadError."""
        import sys

        # Skip on Windows - permission model is different
        if sys.platform == "win32":
            pytest.skip("Permission test not applicable on Windows")

        restricted_file = tmp_path / "restricted.txt"
        restricted_file.write_text("secret content")
        original_mode = restricted_file.stat().st_mode

        try:
            # Remove read permissions
            restricted_file.chmod(0o000)

            loader = PipelineLoader()
            file_ref = FileReference(
                uri=f"file://{restricted_file}",
                source_type="directory",
                content_type="document",
                size_bytes=14,
                source_metadata={"local_path": str(restricted_file)},
            )

            with pytest.raises(LoadError) as exc_info:
                await loader.load(file_ref)

            assert "Permission denied" in exc_info.value.reason

        finally:
            # Restore permissions for cleanup
            restricted_file.chmod(original_mode)


class TestLoadError:
    """Tests for LoadError exception."""

    def test_error_attributes(self) -> None:
        """Test LoadError has correct attributes."""
        error = LoadError("file:///test.pdf", "File not found")
        assert error.file_uri == "file:///test.pdf"
        assert error.reason == "File not found"

    def test_error_message(self) -> None:
        """Test LoadError message format."""
        error = LoadError("file:///test.pdf", "Permission denied")
        assert str(error) == "Failed to load file:///test.pdf: Permission denied"
