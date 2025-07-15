"""
Unit tests for file scanning and hashing functionality
"""

import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pyfakefs.fake_filesystem_unittest import Patcher
from webui.api.files import compute_file_content_hash_async, scan_directory_async


class TestScanDirectoryAsync:
    """Test suite for scan_directory_async function"""

    @pytest.fixture()
    def fake_fs(self) -> Generator[Patcher, None, None]:
        """Create a fake filesystem"""
        with Patcher() as patcher:
            fs = patcher.fs
            assert fs is not None
            # Add supported extensions
            fs.create_file("/test_dir/doc1.pdf", contents=b"PDF content")
            fs.create_file("/test_dir/doc2.docx", contents=b"DOCX content")
            fs.create_file("/test_dir/doc3.txt", contents=b"Text content")
            fs.create_file("/test_dir/doc4.md", contents=b"Markdown content")
            fs.create_file("/test_dir/doc5.html", contents=b"HTML content")

            # Add unsupported file types (should be ignored)
            fs.create_file("/test_dir/image.jpg", contents=b"Image content")
            fs.create_file("/test_dir/script.py", contents=b"Python content")

            # Add files in subdirectory for recursive testing
            fs.create_file("/test_dir/subdir/subdoc1.pdf", contents=b"Sub PDF content")
            fs.create_file("/test_dir/subdir/subdoc2.txt", contents=b"Sub text content")

            # Create a symlink for testing
            fs.create_symlink("/test_dir/link_to_doc1.pdf", "/test_dir/doc1.pdf")

            yield patcher

    @pytest.mark.asyncio()
    async def test_scan_directory_recursive(self, fake_fs: Patcher) -> None:
        """Test recursive directory scanning"""
        result = await scan_directory_async("/test_dir", recursive=True)

        # Should find 7 supported files (5 in root + 2 in subdir) + 1 symlink
        assert result["total_files"] == 8
        assert len(result["files"]) == 8

        # Check that subdirectory files are included
        file_paths = [f.path for f in result["files"]]
        assert "/test_dir/subdir/subdoc1.pdf" in file_paths
        assert "/test_dir/subdir/subdoc2.txt" in file_paths

        # Check that unsupported files are not included
        assert "/test_dir/image.jpg" not in file_paths
        assert "/test_dir/script.py" not in file_paths

    @pytest.mark.asyncio()
    async def test_scan_directory_non_recursive(self, fake_fs: Patcher) -> None:
        """Test non-recursive directory scanning"""
        result = await scan_directory_async("/test_dir", recursive=False)

        # Should find only 5 supported files in root directory + 1 symlink
        assert result["total_files"] == 6
        assert len(result["files"]) == 6

        # Check that subdirectory files are NOT included
        file_paths = [f.path for f in result["files"]]
        assert "/test_dir/subdir/subdoc1.pdf" not in file_paths
        assert "/test_dir/subdir/subdoc2.txt" not in file_paths

        # Check that root files are included
        assert "/test_dir/doc1.pdf" in file_paths
        assert "/test_dir/doc2.docx" in file_paths

    @pytest.mark.asyncio()
    async def test_scan_directory_supported_extensions(self, fake_fs: Patcher) -> None:
        """Test that only files with supported extensions are included"""
        with patch(
            "webui.api.files.SUPPORTED_EXTENSIONS",
            [".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"],
        ):
            result = await scan_directory_async("/test_dir", recursive=False)

            # Verify all returned files have supported extensions
            for file_info in result["files"]:
                assert file_info.extension in [
                    ".pdf",
                    ".docx",
                    ".doc",
                    ".txt",
                    ".text",
                    ".pptx",
                    ".eml",
                    ".md",
                    ".html",
                ]

    @pytest.mark.asyncio()
    async def test_scan_directory_high_file_count_warning(self, fake_fs: Patcher) -> None:
        """Test warning generation for high file count"""
        # Create many files to trigger warning
        for i in range(10001):
            fs = fake_fs.fs
            assert fs is not None
            fs.create_file(f"/test_dir/file_{i}.txt", contents=b"content")

        result = await scan_directory_async("/test_dir", recursive=False)

        # Check that warning was generated
        assert len(result["warnings"]) > 0
        assert any(w["type"] == "high_file_count" for w in result["warnings"])
        assert result["total_files"] > 10000

    @pytest.mark.asyncio()
    async def test_scan_directory_high_total_size_warning(self, fake_fs: Patcher) -> None:
        """Test warning generation for high total size"""
        # Create large files to trigger size warning (>50GB)
        large_content = b"x" * (1024 * 1024 * 1024)  # 1GB
        for i in range(51):
            fs = fake_fs.fs
            assert fs is not None
            fs.create_file(f"/test_dir/large_{i}.pdf", contents=large_content)

        result = await scan_directory_async("/test_dir", recursive=False)

        # Check that warning was generated
        assert len(result["warnings"]) > 0
        assert any(w["type"] == "high_total_size" for w in result["warnings"])
        assert result["total_size"] > 50 * 1024 * 1024 * 1024

    @pytest.mark.asyncio()
    async def test_scan_directory_path_not_exists(self, fake_fs: Patcher) -> None:
        """Test error handling when path doesn't exist"""
        with pytest.raises(ValueError, match="Path does not exist"):
            await scan_directory_async("/nonexistent_dir")

    @pytest.mark.asyncio()
    async def test_scan_directory_path_not_directory(self, fake_fs: Patcher) -> None:
        """Test error handling when path is not a directory"""
        fs = fake_fs.fs
        assert fs is not None
        fs.create_file("/test_file.txt", contents=b"content")

        with pytest.raises(ValueError, match="Path is not a directory"):
            await scan_directory_async("/test_file.txt")

    @pytest.mark.asyncio()
    async def test_scan_directory_with_permission_error(self, fake_fs: Patcher) -> None:
        """Test handling of permission errors during scanning"""
        # Create a file that will raise permission error on stat
        fs = fake_fs.fs
        assert fs is not None
        fs.create_file("/test_dir/restricted.pdf", contents=b"content")

        # Mock stat to raise OSError
        original_stat = Path.stat

        def mock_stat(self: Path) -> Any:
            if str(self).endswith("restricted.pdf"):
                raise OSError("Permission denied")
            return original_stat(self)

        with patch.object(Path, "stat", mock_stat):
            result = await scan_directory_async("/test_dir", recursive=False)

            # Should continue scanning other files
            assert result["total_files"] >= 5  # Other files should still be found

    @pytest.mark.asyncio()
    async def test_scan_directory_with_websocket_updates(self, fake_fs: Patcher) -> None:
        """Test that scan sends progress updates when scan_id is provided"""
        # Mock the manager to capture updates
        mock_manager = MagicMock()
        mock_manager.send_job_update = AsyncMock()

        with patch("webui.api.files.manager", mock_manager):
            await scan_directory_async("/test_dir", recursive=True, scan_id="test_scan_123")

            # Verify that updates were sent
            mock_manager.send_job_update.assert_called()

            # Check for different types of updates
            update_types = []
            for call in mock_manager.send_job_update.call_args_list:
                channel, update_type, data = call[0]
                assert channel == "scan_test_scan_123"
                update_types.append(update_type)

            # Should have counting and progress updates
            assert "counting" in update_types or "progress" in update_types


class TestComputeFileContentHashAsync:
    """Test suite for compute_file_content_hash_async function"""

    @pytest.fixture()
    def fake_fs(self) -> Generator[Patcher, None, None]:
        """Create a fake filesystem for hash testing"""
        with Patcher() as patcher:
            fs = patcher.fs
            assert fs is not None
            # Regular file
            fs.create_file("/test_file.txt", contents=b"Hello, World!")

            # Large file (>10MB to test async handling)
            large_content = b"x" * (11 * 1024 * 1024)  # 11MB
            fs.create_file("/large_file.txt", contents=large_content)

            # Symlink
            fs.create_symlink("/link_to_test.txt", "/test_file.txt")

            # Permission denied file
            fs.create_file("/restricted_file.txt", contents=b"Secret")

            yield patcher

    @pytest.mark.asyncio()
    async def test_hash_regular_file(self, fake_fs: Patcher) -> None:
        """Test hashing a regular file"""
        result = await compute_file_content_hash_async(Path("/test_file.txt"))

        # Compute expected hash
        expected_hash = hashlib.sha256(b"Hello, World!").hexdigest()
        assert result == expected_hash

    @pytest.mark.asyncio()
    async def test_hash_symlink(self, fake_fs: Patcher) -> None:
        """Test hashing a symbolic link"""
        result = await compute_file_content_hash_async(Path("/link_to_test.txt"))

        # Should return symlink: prefix with hash of target path
        assert result.startswith("symlink:")

        # Verify the hash is of the resolved path
        target_path = str(Path("/link_to_test.txt").resolve())
        expected_hash = hashlib.sha256(target_path.encode("utf-8")).hexdigest()
        assert result == f"symlink:{expected_hash}"

    @pytest.mark.asyncio()
    async def test_hash_large_file(self, fake_fs: Patcher) -> None:
        """Test hashing a large file (>10MB)"""
        result = await compute_file_content_hash_async(Path("/large_file.txt"))

        # Compute expected hash
        content = b"x" * (11 * 1024 * 1024)
        expected_hash = hashlib.sha256(content).hexdigest()
        assert result == expected_hash

    @pytest.mark.asyncio()
    async def test_hash_permission_error(self) -> None:
        """Test handling permission errors"""
        # Mock compute_file_content_hash to return None (simulating permission error)
        with patch("webui.api.files.compute_file_content_hash", return_value=None):
            # Create a path that exists and is not a symlink but small (<10MB)
            test_path = MagicMock(spec=Path)
            test_path.is_symlink.return_value = False
            test_path.stat.return_value.st_size = 100  # Small file

            result = await compute_file_content_hash_async(test_path)
            assert result is None

    @pytest.mark.asyncio()
    async def test_hash_io_error(self) -> None:
        """Test handling IO errors for large files"""
        # Create a mock path for a large file
        test_path = MagicMock(spec=Path)
        test_path.is_symlink.return_value = False
        test_path.stat.return_value.st_size = 11 * 1024 * 1024  # 11MB - large file
        test_path.__str__ = lambda self: "/large_file.txt"  # type: ignore[method-assign]  # noqa: ARG005

        # Mock the Path constructor to raise OSError when opening
        def mock_path_init(path_str: str) -> MagicMock:  # noqa: ARG001
            mock = MagicMock(spec=Path)
            mock.open.side_effect = OSError("IO error")
            return mock

        with patch("webui.api.files.Path", side_effect=mock_path_init):
            result = await compute_file_content_hash_async(test_path)
            assert result is None

    @pytest.mark.asyncio()
    async def test_hash_unexpected_error(self) -> None:
        """Test handling unexpected errors"""
        # Create a mock path that raises RuntimeError on stat
        test_path = MagicMock(spec=Path)
        test_path.is_symlink.return_value = False
        test_path.stat.side_effect = RuntimeError("Unexpected error")

        result = await compute_file_content_hash_async(test_path)
        assert result is None

    @pytest.mark.asyncio()
    async def test_hash_file_chunks(self, fake_fs: Patcher) -> None:
        """Test that file is read in chunks"""
        # Create a file with specific content to verify chunked reading
        content = b"a" * 65536 + b"b" * 65536 + b"c" * 1000  # > 2 chunks
        fs = fake_fs.fs
        assert fs is not None
        fs.create_file("/chunked_file.txt", contents=content)

        result = await compute_file_content_hash_async(Path("/chunked_file.txt"))

        # Verify hash is correct
        expected_hash = hashlib.sha256(content).hexdigest()
        assert result == expected_hash
