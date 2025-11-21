"""
Comprehensive tests for DirectoryScanService covering all methods and edge cases.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from webui.api.schemas import DirectoryScanResponse
from webui.services.directory_scan_service import (
    HASH_CHUNK_SIZE,
    MAX_FILE_SIZE,
    PROGRESS_UPDATE_INTERVAL,
    DirectoryScanService,
)


@pytest.fixture()
def directory_scan_service() -> DirectoryScanService:
    """Create a DirectoryScanService instance."""
    return DirectoryScanService()


@pytest.fixture()
def mock_ws_manager() -> AsyncMock:
    """Mock websocket manager."""
    with patch("webui.services.directory_scan_service.ws_manager") as mock:
        mock.send_to_operation = AsyncMock()
        yield mock


@pytest.fixture()
def temp_scan_directory(tmp_path: Path) -> Path:
    """Create a temporary directory structure for testing."""
    # Create main directory
    scan_dir = tmp_path / "scan_test"
    scan_dir.mkdir()

    # Create files in root
    (scan_dir / "document.pdf").write_text("PDF content")
    (scan_dir / "text.txt").write_text("Text content")
    (scan_dir / "image.jpg").write_text("Image content")  # Should be ignored
    (scan_dir / "large_file.pdf").write_text("x" * (MAX_FILE_SIZE + 1))  # Too large

    # Create subdirectory with files
    sub_dir = scan_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "nested.docx").write_text("Word content")
    (sub_dir / "hidden.txt").write_text("Hidden content")

    # Create deeply nested directory
    deep_dir = sub_dir / "deep" / "deeper"
    deep_dir.mkdir(parents=True)
    (deep_dir / "very_nested.md").write_text("Markdown content")

    # Create directory with permission issues (will be mocked)
    restricted_dir = scan_dir / "restricted"
    restricted_dir.mkdir()
    (restricted_dir / "secret.pdf").write_text("Secret content")

    return scan_dir


class TestDirectoryScanService:
    """Test suite for DirectoryScanService."""

    async def test_scan_directory_preview_success(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test successful directory scan with recursive option."""
        scan_id = "test-scan-123"
        user_id = 1

        result = await directory_scan_service.scan_directory_preview(
            path=str(temp_scan_directory), scan_id=scan_id, user_id=user_id, recursive=True
        )

        # Verify response
        assert isinstance(result, DirectoryScanResponse)
        assert result.scan_id == scan_id
        assert result.path == str(temp_scan_directory)
        # Files: document.pdf, text.txt, nested.docx, hidden.txt, very_nested.md, secret.pdf
        # Not included: image.jpg (unsupported), large_file.pdf (too large)
        assert result.total_files == 6  # pdf, txt, docx, txt, md, secret.pdf
        assert len(result.files) == 6
        assert result.total_size > 0
        assert len(result.warnings) == 1  # Large file warning

        # Verify file types
        file_paths = {f.file_path for f in result.files}
        assert str(temp_scan_directory / "document.pdf") in file_paths
        assert str(temp_scan_directory / "text.txt") in file_paths
        assert str(temp_scan_directory / "subdir" / "nested.docx") in file_paths
        assert str(temp_scan_directory / "subdir" / "hidden.txt") in file_paths
        assert str(temp_scan_directory / "subdir" / "deep" / "deeper" / "very_nested.md") in file_paths
        assert str(temp_scan_directory / "restricted" / "secret.pdf") in file_paths

        # Verify WebSocket calls
        assert mock_ws_manager.send_to_operation.called

        # Check for counting message
        counting_calls = [
            call
            for call in mock_ws_manager.send_to_operation.call_args_list
            if call[0][0] == scan_id and call[0][1]["type"] == "counting"
        ]
        assert len(counting_calls) > 0

        # Check for completion message
        completion_calls = [
            call
            for call in mock_ws_manager.send_to_operation.call_args_list
            if call[0][0] == scan_id and call[0][1]["type"] == "completed"
        ]
        assert len(completion_calls) == 1

    async def test_scan_directory_preview_non_recursive(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test non-recursive directory scan."""
        scan_id = "test-scan-456"
        user_id = 1

        result = await directory_scan_service.scan_directory_preview(
            path=str(temp_scan_directory), scan_id=scan_id, user_id=user_id, recursive=False
        )

        # Should only find files in root directory
        assert result.total_files == 2  # pdf and txt (not jpg or large file)
        assert len(result.files) == 2

        file_names = {f.file_name for f in result.files}
        assert "document.pdf" in file_names
        assert "text.txt" in file_names
        assert "nested.docx" not in file_names  # In subdirectory

    async def test_scan_directory_preview_with_include_patterns(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test directory scan with include patterns."""
        result = await directory_scan_service.scan_directory_preview(
            path=str(temp_scan_directory),
            scan_id="test-scan-789",
            user_id=1,
            recursive=True,
            include_patterns=["*.pdf", "*.md"],
        )

        # Should only find PDF and MD files
        # Files: document.pdf, very_nested.md, secret.pdf (large_file.pdf is excluded due to size)
        assert result.total_files == 3
        file_extensions = {Path(f.file_path).suffix for f in result.files}
        assert file_extensions == {".pdf", ".md"}

    async def test_scan_directory_preview_with_exclude_patterns(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test directory scan with exclude patterns."""
        result = await directory_scan_service.scan_directory_preview(
            path=str(temp_scan_directory),
            scan_id="test-scan-101",
            user_id=1,
            recursive=True,
            exclude_patterns=["*/subdir/*", "*.txt"],
        )

        # Should exclude all files in subdir and txt files
        # Note: The pattern "*/subdir/*" only matches files directly in subdir, not in deeper subdirs
        # Files excluded: text.txt, nested.docx, hidden.txt
        # Files NOT excluded: document.pdf, very_nested.md (in subdir/deep/deeper), secret.pdf
        assert result.total_files == 3
        file_names = {f.file_name for f in result.files}
        assert "document.pdf" in file_names
        assert "very_nested.md" in file_names
        assert "secret.pdf" in file_names

    async def test_scan_directory_preview_path_not_exists(
        self, directory_scan_service: DirectoryScanService, mock_ws_manager: AsyncMock
    ) -> None:
        """Test scanning non-existent path."""
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            await directory_scan_service.scan_directory_preview(
                path="/non/existent/path", scan_id="test-scan-404", user_id=1
            )

    async def test_scan_directory_preview_path_not_directory(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test scanning a file instead of directory."""
        file_path = temp_scan_directory / "document.pdf"

        with pytest.raises(ValueError, match="Path is not a directory"):
            await directory_scan_service.scan_directory_preview(
                path=str(file_path), scan_id="test-scan-file", user_id=1
            )

    async def test_scan_directory_preview_permission_denied(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test scanning directory with permission denied."""
        with (
            patch("os.listdir", side_effect=PermissionError("Access denied")),
            pytest.raises(PermissionError, match="Access denied to directory"),
        ):
            await directory_scan_service.scan_directory_preview(
                path=str(temp_scan_directory), scan_id="test-scan-perm", user_id=1
            )

    async def test_scan_file_permission_denied(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test scanning files with permission issues."""
        # Mock stat to raise PermissionError for specific file
        original_stat = Path.stat

        def mock_stat(self, *args, **kwargs) -> None:
            if "restricted" in str(self):
                raise PermissionError("Access denied")
            return original_stat(self, *args, **kwargs)

        with patch.object(Path, "stat", mock_stat):
            result = await directory_scan_service.scan_directory_preview(
                path=str(temp_scan_directory), scan_id="test-scan-file-perm", user_id=1, recursive=True
            )

            # Should have warning about permission denied file
            assert any("Permission denied" in w for w in result.warnings)

    async def test_scan_directory_with_symlinks(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test scanning directory with symbolic links."""
        # Create a symlink
        symlink_path = temp_scan_directory / "symlink.pdf"
        symlink_path.symlink_to(temp_scan_directory / "document.pdf")

        result = await directory_scan_service.scan_directory_preview(
            path=str(temp_scan_directory), scan_id="test-scan-symlink", user_id=1, recursive=False
        )

        # Should include both original and symlink
        # document.pdf, text.txt, symlink.pdf (large_file.pdf excluded due to size, image.jpg unsupported)
        assert result.total_files == 3  # document.pdf, text.txt, symlink.pdf

    async def test_count_files(self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path) -> None:
        """Test file counting functionality."""
        # Test recursive count
        count = await directory_scan_service._count_files(
            path=temp_scan_directory, recursive=True, include_patterns=None, exclude_patterns=None
        )
        # Count includes all files with supported extensions, regardless of size
        # Files: document.pdf, text.txt, large_file.pdf, nested.docx, hidden.txt, very_nested.md, secret.pdf
        # Not counted: image.jpg (unsupported extension)
        assert count == 7  # All supported files including large_file.pdf

        # Test non-recursive count
        count = await directory_scan_service._count_files(
            path=temp_scan_directory, recursive=False, include_patterns=None, exclude_patterns=None
        )
        # Root files: document.pdf, text.txt, large_file.pdf (image.jpg is unsupported)
        assert count == 3  # Only root directory files

        # Test with patterns
        count = await directory_scan_service._count_files(
            path=temp_scan_directory, recursive=True, include_patterns=["*.pdf"], exclude_patterns=None
        )
        # _count_files doesn't check file size, only extension and patterns
        assert count == 3  # document.pdf, large_file.pdf, secret.pdf

    async def test_count_files_with_error(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test file counting with errors."""
        with patch("os.walk", side_effect=OSError("Walk error")):
            count = await directory_scan_service._count_files(
                path=temp_scan_directory, recursive=True, include_patterns=None, exclude_patterns=None
            )
            assert count == 0  # Should return 0 on error

    async def test_scan_file_success(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test scanning individual file."""
        file_path = temp_scan_directory / "document.pdf"

        file_info, warning = await directory_scan_service._scan_file(
            file_path=file_path, include_patterns=None, exclude_patterns=None
        )

        assert file_info is not None
        assert warning is None
        assert file_info.file_name == "document.pdf"
        assert file_info.file_size > 0
        assert file_info.mime_type == "application/pdf"
        assert file_info.content_hash is not None
        assert isinstance(file_info.modified_at, datetime)

    async def test_scan_file_too_large(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test scanning file that exceeds size limit."""
        file_path = temp_scan_directory / "large_file.pdf"

        file_info, warning = await directory_scan_service._scan_file(
            file_path=file_path, include_patterns=None, exclude_patterns=None
        )

        assert file_info is None
        assert warning is not None
        assert "File too large" in warning
        assert str(file_path) in warning

    async def test_scan_file_unsupported_type(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test scanning unsupported file type."""
        file_path = temp_scan_directory / "image.jpg"

        file_info, warning = await directory_scan_service._scan_file(
            file_path=file_path, include_patterns=None, exclude_patterns=None
        )

        assert file_info is None
        assert warning is None  # Silently skipped

    async def test_scan_file_with_error(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test scanning file with general error."""
        file_path = temp_scan_directory / "document.pdf"

        with patch.object(Path, "stat", side_effect=Exception("Stat error")):
            file_info, warning = await directory_scan_service._scan_file(
                file_path=file_path, include_patterns=None, exclude_patterns=None
            )

            assert file_info is None
            assert warning is not None
            assert "Error scanning file" in warning

    def test_should_include_file(self, directory_scan_service: DirectoryScanService) -> None:
        """Test file inclusion logic."""
        # Test supported extension
        assert directory_scan_service._should_include_file(Path("test.pdf"), None, None) is True

        # Test unsupported extension
        assert directory_scan_service._should_include_file(Path("test.jpg"), None, None) is False

        # Test include patterns
        assert directory_scan_service._should_include_file(Path("test.pdf"), ["*.pdf"], None) is True
        assert directory_scan_service._should_include_file(Path("test.txt"), ["*.pdf"], None) is False

        # Test exclude patterns
        assert directory_scan_service._should_include_file(Path("test.pdf"), None, ["*.pdf"]) is False
        assert directory_scan_service._should_include_file(Path("test.txt"), None, ["*.pdf"]) is True

        # Test both include and exclude patterns
        assert (
            directory_scan_service._should_include_file(Path("test.pdf"), ["*.pdf", "*.txt"], ["test.*"]) is False
        )  # Excluded takes precedence

    async def test_calculate_file_hash(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test file hash calculation."""
        file_path = temp_scan_directory / "document.pdf"
        content = b"PDF content"
        file_path.write_bytes(content)

        # Calculate expected hash
        expected_hash = hashlib.sha256(content).hexdigest()

        # Calculate actual hash
        actual_hash = await directory_scan_service._calculate_file_hash(file_path)

        assert actual_hash == expected_hash

    async def test_calculate_file_hash_large_file(
        self, directory_scan_service: DirectoryScanService, tmp_path: Path
    ) -> None:
        """Test file hash calculation for large file with chunked reading."""
        file_path = tmp_path / "large.pdf"

        # Create file larger than chunk size
        content = b"x" * (HASH_CHUNK_SIZE * 3 + 1000)
        file_path.write_bytes(content)

        # Calculate expected hash
        expected_hash = hashlib.sha256(content).hexdigest()

        # Calculate actual hash (should use chunked reading)
        actual_hash = await directory_scan_service._calculate_file_hash(file_path)

        assert actual_hash == expected_hash

    async def test_calculate_file_hash_error(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test file hash calculation with error."""
        file_path = temp_scan_directory / "nonexistent.pdf"

        with pytest.raises(OSError, match="Failed to calculate hash"):
            await directory_scan_service._calculate_file_hash(file_path)

    def test_get_mime_type(self, directory_scan_service: DirectoryScanService) -> None:
        """Test MIME type detection."""
        # Test standard types
        assert directory_scan_service._get_mime_type(Path("test.pdf")) == "application/pdf"
        assert directory_scan_service._get_mime_type(Path("test.txt")) == "text/plain"
        assert directory_scan_service._get_mime_type(Path("test.html")) == "text/html"
        assert directory_scan_service._get_mime_type(Path("test.md")) == "text/markdown"

        # Test Word documents
        assert (
            directory_scan_service._get_mime_type(Path("test.docx"))
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert directory_scan_service._get_mime_type(Path("test.doc")) == "application/msword"

        # Test PowerPoint
        assert (
            directory_scan_service._get_mime_type(Path("test.pptx"))
            == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        )

        # Test email
        assert directory_scan_service._get_mime_type(Path("test.eml")) == "message/rfc822"

        # Test case insensitive
        assert directory_scan_service._get_mime_type(Path("TEST.PDF")) == "application/pdf"

    def test_format_size(self, directory_scan_service: DirectoryScanService) -> None:
        """Test file size formatting."""
        assert directory_scan_service._format_size(0) == "0.0 B"
        assert directory_scan_service._format_size(512) == "512.0 B"
        assert directory_scan_service._format_size(1024) == "1.0 KB"
        assert directory_scan_service._format_size(1536) == "1.5 KB"
        assert directory_scan_service._format_size(1048576) == "1.0 MB"
        assert directory_scan_service._format_size(1073741824) == "1.0 GB"
        assert directory_scan_service._format_size(1099511627776) == "1.0 TB"
        assert directory_scan_service._format_size(1125899906842624) == "1.0 PB"

    async def test_send_progress_success(
        self, directory_scan_service: DirectoryScanService, mock_ws_manager: AsyncMock
    ) -> None:
        """Test sending progress update via WebSocket."""
        channel_id = "test-channel"
        scan_id = "test-scan"
        data = {"message": "Test progress"}

        await directory_scan_service._send_progress(
            channel_id=channel_id, scan_id=scan_id, msg_type="progress", data=data
        )

        mock_ws_manager.send_to_operation.assert_called_once()
        call_args = mock_ws_manager.send_to_operation.call_args[0]
        assert call_args[0] == scan_id  # First arg is scan_id, not channel_id

        broadcast_data = call_args[1]
        assert broadcast_data["type"] == "progress"
        assert broadcast_data["scan_id"] == scan_id
        assert broadcast_data["data"] == data

    async def test_send_progress_error(
        self, directory_scan_service: DirectoryScanService, mock_ws_manager: AsyncMock
    ) -> None:
        """Test sending progress update with WebSocket error."""
        mock_ws_manager.send_to_operation.side_effect = Exception("WebSocket error")

        # Should not raise exception
        await directory_scan_service._send_progress(
            channel_id="test-channel", scan_id="test-scan", msg_type="error", data={"message": "Error"}
        )

        # Verify broadcast was attempted
        assert mock_ws_manager.send_to_operation.called

    async def test_scan_recursive_error_handling(
        self, directory_scan_service: DirectoryScanService, temp_scan_directory: Path
    ) -> None:
        """Test error handling in recursive scan."""
        with patch("os.walk", side_effect=Exception("Walk error")):
            files = []
            warnings = []

            async for file_info, warning in directory_scan_service._scan_recursive(
                path=temp_scan_directory, include_patterns=None, exclude_patterns=None
            ):
                if file_info:
                    files.append(file_info)
                if warning:
                    warnings.append(warning)

            assert len(files) == 0
            assert len(warnings) == 1
            assert "Error scanning directory" in warnings[0]

    async def test_progress_updates_during_scan(
        self, directory_scan_service: DirectoryScanService, tmp_path: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test that progress updates are sent at correct intervals."""
        # Create many files to trigger progress updates
        scan_dir = tmp_path / "many_files"
        scan_dir.mkdir()

        # Create more files than PROGRESS_UPDATE_INTERVAL
        num_files = PROGRESS_UPDATE_INTERVAL + 10
        for i in range(num_files):
            (scan_dir / f"file{i}.txt").write_text(f"Content {i}")

        scan_id = "test-progress"
        await directory_scan_service.scan_directory_preview(
            path=str(scan_dir), scan_id=scan_id, user_id=1, recursive=False
        )

        # Check that progress updates were sent
        progress_calls = [
            call
            for call in mock_ws_manager.send_to_operation.call_args_list
            if call[0][0] == scan_id and call[0][1]["type"] == "progress"
        ]

        # Should have at least one progress update due to interval
        assert len(progress_calls) >= 1

        # Check if we have progress updates
        # With 60 files and PROGRESS_UPDATE_INTERVAL=50, we should get updates at:
        # - 0 files (0%)
        # - 50 files (83.3%)
        # Note: Non-recursive scan doesn't send a final 100% update in the current implementation
        percentages = [
            call[0][1]["data"].get("percentage", 0) for call in progress_calls if "percentage" in call[0][1]["data"]
        ]
        assert len(percentages) >= 2, f"Expected at least 2 progress updates, got {len(percentages)}"
        assert 0.0 in percentages, "Should have initial 0% progress"
        assert any(p > 80.0 for p in percentages), f"Should have progress > 80%, got: {percentages}"

    async def test_scan_with_all_edge_cases(
        self, directory_scan_service: DirectoryScanService, tmp_path: Path, mock_ws_manager: AsyncMock
    ) -> None:
        """Test scanning with multiple edge cases combined."""
        # Create complex directory structure
        scan_dir = tmp_path / "complex"
        scan_dir.mkdir()

        # Hidden file (Unix-style)
        (scan_dir / ".hidden.pdf").write_text("Hidden PDF")

        # File with spaces in name
        (scan_dir / "file with spaces.txt").write_text("Spaced content")

        # File with special characters
        (scan_dir / "special-chars_#1.docx").write_text("Special content")

        # Empty file
        (scan_dir / "empty.pdf").write_text("")

        # File with no extension
        (scan_dir / "noextension").write_text("No extension")

        # File with multiple dots
        (scan_dir / "file.backup.txt").write_text("Backup content")

        # Very long filename
        long_name = "a" * 200 + ".pdf"
        (scan_dir / long_name).write_text("Long name content")

        result = await directory_scan_service.scan_directory_preview(
            path=str(scan_dir), scan_id="test-edge-cases", user_id=1, recursive=False
        )

        # Should handle all valid files
        assert result.total_files == 6  # All except noextension

        # Verify specific files
        file_names = {f.file_name for f in result.files}
        assert ".hidden.pdf" in file_names  # Hidden files are included
        assert "file with spaces.txt" in file_names
        assert "special-chars_#1.docx" in file_names
        assert "empty.pdf" in file_names
        assert "file.backup.txt" in file_names
        assert long_name in file_names

        # Verify empty file has hash
        empty_file = next(f for f in result.files if f.file_name == "empty.pdf")
        assert empty_file.content_hash == hashlib.sha256(b"").hexdigest()
        assert empty_file.file_size == 0
