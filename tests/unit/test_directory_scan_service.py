#!/usr/bin/env python3

"""
Comprehensive test suite for webui/services/directory_scan_service.py
Tests path validation, permission checking, file discovery logic, and error scenarios
"""

import asyncio
import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pyfakefs.fake_filesystem_unittest import Patcher

from packages.webui.api.schemas import DirectoryScanResponse
from packages.webui.services.directory_scan_service import (
    MAX_FILE_SIZE,
    PROGRESS_UPDATE_INTERVAL,
    SUPPORTED_EXTENSIONS,
    DirectoryScanService,
)


class TestDirectoryScanService:
    """Test DirectoryScanService implementation"""

    @pytest.fixture()
    def service(self) -> None:
        """Create DirectoryScanService instance"""
        return DirectoryScanService()

    @pytest.fixture()
    def fake_fs(self) -> Generator[Any, None, None]:
        """Create fake filesystem for testing"""
        with Patcher() as patcher:
            # Create test directory structure
            patcher.fs.create_dir("/test_dir")
            patcher.fs.create_file("/test_dir/document1.pdf", contents=b"PDF content" * 100)
            patcher.fs.create_file("/test_dir/document2.docx", contents=b"DOCX content" * 150)
            patcher.fs.create_file("/test_dir/text_file.txt", contents=b"Text content" * 40)
            patcher.fs.create_file("/test_dir/not_supported.xyz", contents=b"XYZ content" * 20)
            patcher.fs.create_dir("/test_dir/subdir")
            patcher.fs.create_file("/test_dir/subdir/nested.md", contents=b"MD content" * 70)
            patcher.fs.create_file("/test_dir/subdir/large_file.pdf", st_size=MAX_FILE_SIZE + 1)
            yield patcher.fs

    @pytest.fixture()
    def mock_ws_manager(self) -> Generator[Any, None, None]:
        """Mock WebSocket manager"""
        with patch("packages.webui.services.directory_scan_service.ws_manager") as mock:
            mock._broadcast = AsyncMock()
            yield mock

    @pytest.mark.asyncio()
    async def test_scan_directory_preview_success(self, service, fake_fs, mock_ws_manager) -> None:
        """Test successful directory scan with recursive option"""
        # Test scan
        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-123",
            user_id=123,
            recursive=True,
        )

        # Verify result structure
        assert isinstance(result, DirectoryScanResponse)
        assert result.scan_id == "scan-123"
        assert result.path == "/test_dir"
        assert result.total_files == 4  # Excludes .xyz and large file
        # Total size should be sum of all valid files
        assert result.total_size == 1100 + 1800 + 480 + 700  # Actual sizes from content

        # Verify files
        file_paths = {f.file_path for f in result.files}
        expected_paths = {
            "/test_dir/document1.pdf",
            "/test_dir/document2.docx",
            "/test_dir/text_file.txt",
            "/test_dir/subdir/nested.md",
        }
        assert file_paths == expected_paths

        # Verify WebSocket messages were sent
        assert mock_ws_manager._broadcast.call_count > 0

    @pytest.mark.asyncio()
    async def test_scan_directory_non_recursive(self, service, fake_fs, mock_ws_manager) -> None:
        """Test non-recursive directory scan"""
        # Test scan
        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-456",
            user_id=123,
            recursive=False,
        )

        # Should only get files in root directory
        assert result.total_files == 3
        file_paths = {f.file_path for f in result.files}
        expected_paths = {
            "/test_dir/document1.pdf",
            "/test_dir/document2.docx",
            "/test_dir/text_file.txt",
        }
        assert file_paths == expected_paths

    @pytest.mark.asyncio()
    async def test_scan_directory_with_patterns(self, service, fake_fs, mock_ws_manager) -> None:
        """Test directory scan with include/exclude patterns"""
        # Test with include pattern
        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-789",
            user_id=123,
            recursive=True,
            include_patterns=["*.pdf"],
        )

        # Should only include PDF files
        assert result.total_files == 1
        assert result.files[0].file_path == "/test_dir/document1.pdf"

        # Test with exclude pattern
        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-790",
            user_id=123,
            recursive=True,
            exclude_patterns=["subdir/*"],
        )

        # Should exclude files in subdir
        assert result.total_files == 3
        file_paths = {f.file_path for f in result.files}
        assert "/test_dir/subdir/nested.md" not in file_paths

    @pytest.mark.asyncio()
    async def test_scan_directory_path_not_exists(self, service) -> None:
        """Test scanning non-existent path"""
        with pytest.raises(FileNotFoundError, match="Path does not exist"):
            await service.scan_directory_preview(
                path="/non_existent",
                scan_id="scan-error",
                user_id=123,
            )

    @pytest.mark.asyncio()
    async def test_scan_directory_not_a_directory(self, service, fake_fs) -> None:
        """Test scanning a file instead of directory"""
        with pytest.raises(ValueError, match="Path is not a directory"):
            await service.scan_directory_preview(
                path="/test_dir/document1.pdf",
                scan_id="scan-error",
                user_id=123,
            )

    @pytest.mark.asyncio()
    async def test_scan_directory_permission_denied(self, service, fake_fs) -> None:
        """Test scanning directory without permissions"""
        # Create directory without read permissions
        fake_fs.create_dir("/restricted", perm_bits=0o000)

        with pytest.raises(PermissionError, match="Access denied to directory"):
            await service.scan_directory_preview(
                path="/restricted",
                scan_id="scan-error",
                user_id=123,
            )

    @pytest.mark.asyncio()
    async def test_scan_file_size_limit(self, service, fake_fs, mock_ws_manager) -> None:
        """Test that files exceeding size limit are warned about"""
        # Scan should complete but large file should generate warning
        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-size",
            user_id=123,
            recursive=True,
        )

        # Large file should not be in results
        file_paths = {f.file_path for f in result.files}
        assert "/test_dir/subdir/large_file.pdf" not in file_paths

        # Should have warning about large file
        assert any("File too large" in warning for warning in result.warnings)

    @pytest.mark.asyncio()
    async def test_file_hash_calculation(self, service, fake_fs) -> None:
        """Test SHA-256 hash calculation for files"""
        # Create file with known content
        test_content = b"test content for hash"
        fake_fs.create_file("/test_hash.txt", contents=test_content)

        # Calculate expected hash
        expected_hash = hashlib.sha256(test_content).hexdigest()

        # Test hash calculation
        file_path = Path("/test_hash.txt")
        calculated_hash = await service._calculate_file_hash(file_path)

        assert calculated_hash == expected_hash

    @pytest.mark.asyncio()
    async def test_mime_type_detection(self, service) -> None:
        """Test MIME type detection for various file types"""
        test_cases = [
            ("document.pdf", "application/pdf"),
            ("document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("document.doc", "application/msword"),
            ("file.txt", "text/plain"),
            ("file.text", "text/plain"),
            ("presentation.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
            ("email.eml", "message/rfc822"),
            ("markdown.md", "text/markdown"),
            ("webpage.html", "text/html"),
        ]

        for filename, expected_mime in test_cases:
            file_path = Path(f"/test/{filename}")
            mime_type = service._get_mime_type(file_path)
            assert mime_type == expected_mime

    @pytest.mark.asyncio()
    async def test_progress_updates(self, service, fake_fs, mock_ws_manager) -> None:
        """Test WebSocket progress updates during scan"""
        # Create more files to trigger progress updates
        for i in range(PROGRESS_UPDATE_INTERVAL + 5):
            fake_fs.create_file(f"/test_dir/file_{i}.txt", contents=b"x" * 100)

        await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-progress",
            user_id=123,
            recursive=True,
        )

        # Verify progress messages were sent
        broadcast_calls = mock_ws_manager._broadcast.call_args_list
        progress_messages = []

        for call in broadcast_calls:
            channel_id, message = call[0]
            if isinstance(message, dict) and message.get("type") == "progress":
                progress_messages.append(message)

        # Should have multiple progress updates
        assert len(progress_messages) > 0

        # Verify progress message structure
        for msg in progress_messages:
            assert "data" in msg
            assert "files_scanned" in msg["data"]
            assert "total_files" in msg["data"]
            assert "percentage" in msg["data"]

    @pytest.mark.asyncio()
    async def test_scan_error_handling(self, service, fake_fs, mock_ws_manager) -> None:
        """Test error handling during scan"""
        # Mock an error during file scanning
        with patch.object(service, "_scan_file", side_effect=Exception("Scan error")):
            with pytest.raises(Exception, match="Scan error"):
                await service.scan_directory_preview(
                    path="/test_dir",
                    scan_id="scan-error",
                    user_id=123,
                    recursive=False,
                )

            # Verify error message was sent via WebSocket before raising
            error_calls = [
                call for call in mock_ws_manager._broadcast.call_args_list if call[0][1].get("type") == "error"
            ]
            assert len(error_calls) > 0

    @pytest.mark.asyncio()
    async def test_format_size(self, service) -> None:
        """Test human-readable size formatting"""
        test_cases = [
            (0, "0.0 B"),
            (512, "512.0 B"),
            (1024, "1.0 KB"),
            (1024 * 1024, "1.0 MB"),
            (1024 * 1024 * 1024, "1.0 GB"),
            (1024 * 1024 * 1024 * 1024, "1.0 TB"),
            (1536, "1.5 KB"),
            (1024 * 1024 * 1.5, "1.5 MB"),
        ]

        for size_bytes, expected in test_cases:
            formatted = service._format_size(size_bytes)
            assert formatted == expected

    @pytest.mark.asyncio()
    async def test_should_include_file(self, service) -> None:
        """Test file inclusion logic"""
        # Test supported extensions
        for ext in SUPPORTED_EXTENSIONS:
            file_path = Path(f"/test/file{ext}")
            assert service._should_include_file(file_path, None, None) is True

        # Test unsupported extension
        assert service._should_include_file(Path("/test/file.xyz"), None, None) is False

        # Test include patterns
        file_path = Path("/test/document.pdf")
        assert service._should_include_file(file_path, ["*.pdf"], None) is True
        assert service._should_include_file(file_path, ["*.docx"], None) is False

        # Test exclude patterns
        assert service._should_include_file(file_path, None, ["*.pdf"]) is False
        assert service._should_include_file(file_path, None, ["*.docx"]) is True

        # Test combined patterns
        assert service._should_include_file(file_path, ["*.pdf"], ["test/*"]) is False

    @pytest.mark.asyncio()
    async def test_websocket_message_format(self, service, fake_fs, mock_ws_manager) -> None:
        """Test WebSocket message format and types"""
        await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-ws-test",
            user_id=123,
            recursive=False,
        )

        # Collect all broadcast calls
        broadcast_calls = mock_ws_manager._broadcast.call_args_list

        # Verify different message types
        message_types = set()
        for call in broadcast_calls:
            channel_id, message = call[0]
            assert channel_id == "directory-scan:scan-ws-test"
            assert "type" in message
            assert "scan_id" in message
            assert "data" in message
            message_types.add(message["type"])

        # Should have at least these message types
        expected_types = {"counting", "progress", "completed"}
        assert expected_types.issubset(message_types)

    @pytest.mark.asyncio()
    async def test_scan_empty_directory(self, service, fake_fs, mock_ws_manager) -> None:
        """Test scanning empty directory"""
        fake_fs.create_dir("/empty_dir")

        result = await service.scan_directory_preview(
            path="/empty_dir",
            scan_id="scan-empty",
            user_id=123,
        )

        assert result.total_files == 0
        assert result.total_size == 0
        assert len(result.files) == 0
        assert len(result.warnings) == 0

    @pytest.mark.asyncio()
    async def test_file_permission_errors(self, service, fake_fs, mock_ws_manager) -> None:
        """Test handling files with permission errors"""
        # Create file without read permissions
        fake_fs.create_file("/test_dir/restricted.pdf", contents=b"restricted" * 100, st_mode=0o000)

        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-perm",
            user_id=123,
            recursive=False,
        )

        # Should have warning about permission denied
        assert any("Permission denied" in warning for warning in result.warnings)

        # File should not be in results
        file_paths = {f.file_path for f in result.files}
        assert "/test_dir/restricted.pdf" not in file_paths

    @pytest.mark.asyncio()
    async def test_concurrent_scans(self, service, fake_fs, mock_ws_manager) -> None:
        """Test multiple concurrent scans"""
        # Create multiple scan tasks
        scan_tasks = []
        for i in range(3):
            task = service.scan_directory_preview(
                path="/test_dir",
                scan_id=f"scan-concurrent-{i}",
                user_id=123 + i,
                recursive=True,
            )
            scan_tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*scan_tasks)

        # All scans should succeed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, DirectoryScanResponse)
            assert result.total_files == 4

    @pytest.mark.asyncio()
    async def test_scan_with_symlinks(self, service, fake_fs, mock_ws_manager) -> None:
        """Test scanning directories with symbolic links"""
        # Create symlink
        fake_fs.create_symlink("/test_dir/link_to_doc", "/test_dir/document1.pdf")

        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-symlink",
            user_id=123,
            recursive=False,
        )

        # Symlink should be followed and file counted once
        # (os.walk follows symlinks by default)
        assert result.total_files >= 3  # At least the original files


class TestDirectoryScanServiceEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.fixture()
    def service(self) -> None:
        return DirectoryScanService()

    @pytest.mark.asyncio()
    async def test_scan_file_with_invalid_hash(self, service, fs) -> None:
        """Test handling files that fail hash calculation"""
        # Create a file that will fail to hash
        with patch.object(service, "_calculate_file_hash", side_effect=OSError("Hash calculation failed")):
            # This should raise an OSError wrapped in the scan_file method
            file_path = Path("/test.pdf")
            fs.create_file(str(file_path), contents=b"test" * 256)

            file_info, warning = await service._scan_file(file_path, None, None)

            assert file_info is None
            assert warning is not None
            assert "Error scanning file" in warning

    @pytest.mark.asyncio()
    async def test_count_files_error_handling(self, service, fs) -> None:
        """Test file counting with errors"""
        fs.create_dir("/test_count")
        fs.create_file("/test_count/file1.pdf", contents=b"x" * 100)

        # Mock os.walk to raise an error
        with patch("os.walk", side_effect=Exception("Walk error")):
            count = await service._count_files(
                Path("/test_count"),
                recursive=True,
                include_patterns=None,
                exclude_patterns=None,
            )

            # Should return 0 on error
            assert count == 0

    @pytest.mark.asyncio()
    @patch("packages.webui.services.directory_scan_service.ws_manager")
    async def test_websocket_broadcast_failure(self, mock_ws_manager, service, fs) -> None:
        """Test handling WebSocket broadcast failures"""
        # Create test directory
        fs.create_dir("/test_dir")
        fs.create_file("/test_dir/test.pdf", contents=b"test" * 100)

        # Make broadcast fail
        mock_ws_manager._broadcast.side_effect = Exception("WebSocket error")

        # Should still complete scan despite WebSocket errors
        result = await service.scan_directory_preview(
            path="/test_dir",
            scan_id="scan-ws-fail",
            user_id=123,
            recursive=False,
        )

        assert isinstance(result, DirectoryScanResponse)

    @pytest.mark.asyncio()
    async def test_scan_recursive_error_handling(self, service, fs) -> Generator[Any, None, None]:
        """Test error handling in recursive scan"""
        fs.create_dir("/test_recursive")
        fs.create_file("/test_recursive/file.pdf", contents=b"x" * 100)

        # Mock os.walk to raise an error
        with patch("os.walk", side_effect=Exception("Walk error")):
            # Use the generator method directly
            result_list = []
            async for file_info, warning in service._scan_recursive(Path("/test_recursive"), None, None):
                result_list.append((file_info, warning))

            # Should yield error warning
            assert len(result_list) == 1
            assert result_list[0][0] is None
            assert "Error scanning directory" in result_list[0][1]


class TestDirectoryScanProgress:
    """Test progress tracking and reporting"""

    @pytest.fixture()
    def service(self) -> None:
        return DirectoryScanService()

    @pytest.mark.asyncio()
    @patch("packages.webui.services.directory_scan_service.ws_manager")
    async def test_progress_message_structure(self, mock_ws_manager, service) -> None:
        """Test progress message data structure"""
        await service._send_progress(
            channel_id="test-channel",
            scan_id="test-scan",
            msg_type="progress",
            data={
                "files_scanned": 50,
                "total_files": 100,
                "percentage": 50.0,
                "current_path": "/test/file.pdf",
            },
        )

        # Verify broadcast was called
        mock_ws_manager._broadcast.assert_called_once()

        # Verify message structure
        channel_id, message = mock_ws_manager._broadcast.call_args[0]
        assert channel_id == "test-channel"
        assert message["type"] == "progress"
        assert message["scan_id"] == "test-scan"
        assert message["data"]["files_scanned"] == 50
        assert message["data"]["percentage"] == 50.0

    @pytest.mark.asyncio()
    async def test_progress_update_intervals(self, service, fs) -> None:
        """Test that progress updates happen at correct intervals"""
        # Create test directory first
        fs.create_dir("/test_progress")
        # Create exactly PROGRESS_UPDATE_INTERVAL + 1 files
        num_files = PROGRESS_UPDATE_INTERVAL + 1
        for i in range(num_files):
            fs.create_file(f"/test_progress/file_{i}.txt", contents=b"x" * 100)

        with patch.object(service, "_send_progress", new_callable=AsyncMock) as mock_send:
            await service.scan_directory_preview(
                path="/test_progress",
                scan_id="scan-interval",
                user_id=123,
                recursive=False,
            )

            # Count progress update calls
            progress_calls = [call for call in mock_send.call_args_list if call[1]["msg_type"] == "progress"]

            # Should have at least 2 progress updates
            # (one at PROGRESS_UPDATE_INTERVAL and one at completion)
            assert len(progress_calls) >= 2
