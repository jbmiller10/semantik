"""Unit tests for FileScanningService."""
# mypy: ignore-errors

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from shared.database.models import Document
from webui.services.file_scanning_service import SUPPORTED_EXTENSIONS, FileScanningService


class TestFileScanningService:
    """Test cases for FileScanningService."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock async session."""
        return AsyncMock()

    @pytest.fixture()
    def mock_document_repo(self):
        """Create a mock DocumentRepository."""
        mock = AsyncMock()

        # Default behavior - create returns a new document
        def default_create(**kwargs):
            doc = Document(
                id=str(uuid4()),
                collection_id=kwargs.get("collection_id", str(uuid4())),
                file_path=kwargs.get("file_path", "/test/file.txt"),
                file_name=kwargs.get("file_name", "file.txt"),
                file_size=kwargs.get("file_size", 1024),
                content_hash=kwargs.get("content_hash", "a" * 64),
            )
            # Set created_at timestamp to current time
            doc.created_at = datetime.now(UTC)
            return doc

        mock.create.side_effect = default_create
        return mock

    @pytest.fixture()
    def service(self, mock_session, mock_document_repo):
        """Create a FileScanningService instance with mocked dependencies."""
        return FileScanningService(db_session=mock_session, document_repo=mock_document_repo)

    def create_test_file(self, temp_dir: Path, filename: str, content: str = "test content") -> Path:
        """Helper to create a test file."""
        file_path = temp_dir / filename
        file_path.write_text(content)
        return file_path

    @pytest.mark.asyncio()
    async def test_scan_directory_nonexistent_path(self, service):
        """Test scanning a non-existent directory raises ValueError."""
        with pytest.raises(ValueError, match="Source path does not exist"):
            await service.scan_directory_and_register_documents(
                collection_id=str(uuid4()), source_path="/nonexistent/path"
            )

    @pytest.mark.asyncio()
    async def test_scan_directory_not_a_directory(self, service):
        """Test scanning a file instead of directory raises ValueError."""
        with (
            tempfile.NamedTemporaryFile() as tmp_file,
            pytest.raises(ValueError, match="Source path is not a directory"),
        ):
            await service.scan_directory_and_register_documents(collection_id=str(uuid4()), source_path=tmp_file.name)

    @pytest.mark.asyncio()
    async def test_scan_empty_directory(self, service):
        """Test scanning an empty directory returns zero stats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            stats = await service.scan_directory_and_register_documents(
                collection_id=str(uuid4()), source_path=temp_dir
            )

            assert stats["total_files_found"] == 0
            assert stats["new_files_registered"] == 0
            assert stats["duplicate_files_skipped"] == 0
            assert stats["errors"] == []
            assert stats["total_size_bytes"] == 0

    @pytest.mark.asyncio()
    async def test_scan_directory_with_supported_files(self, service, mock_document_repo):
        """Test scanning directory with supported file types."""
        collection_id = str(uuid4())

        # Setup mock to return documents with created_at timestamps
        def create_mock_document(**kwargs):
            doc = Document(
                id=str(uuid4()),
                collection_id=collection_id,
                file_path=kwargs.get("file_path", "/test/file"),
                file_name=kwargs.get("file_name", "file"),
                file_size=kwargs.get("file_size", 100),
                content_hash=kwargs.get("content_hash", "a" * 64),
            )
            doc.created_at = datetime.now(UTC)
            return doc

        mock_document_repo.create.side_effect = create_mock_document

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            self.create_test_file(temp_path, "test.pdf", "PDF content")
            self.create_test_file(temp_path, "test.txt", "Text content")
            self.create_test_file(temp_path, "test.docx", "DOCX content")

            # Create unsupported file that should be ignored
            self.create_test_file(temp_path, "test.jpg", "Image content")

            stats = await service.scan_directory_and_register_documents(
                collection_id=collection_id, source_path=temp_dir, recursive=False
            )

            # Verify results
            assert stats["total_files_found"] == 3  # Only supported files
            assert stats["new_files_registered"] == 3
            assert stats["duplicate_files_skipped"] == 0
            assert len(stats["errors"]) == 0
            assert stats["total_size_bytes"] > 0

            # Verify document repo was called for each supported file
            assert mock_document_repo.create.call_count == 3

    @pytest.mark.asyncio()
    async def test_scan_directory_with_duplicates(self, service, mock_document_repo):
        """Test that duplicate files are properly detected."""
        collection_id = str(uuid4())

        # Create documents with different created_at times
        existing_doc = Document(
            id="doc1",
            collection_id=collection_id,
            file_path="/test1.txt",
            file_name="test1.txt",
            file_size=100,
            content_hash="a" * 64,
        )
        # Simulate existing document created 1 hour ago
        existing_doc.created_at = datetime.now(UTC) - timedelta(hours=1)

        # Configure mock to return existing doc for first file,
        # and create a new doc dynamically for second file
        def create_side_effect(**kwargs):
            if kwargs.get("file_name") == "test1.txt":
                return existing_doc
            # Create new document with current timestamp
            new_doc = Document(
                id="doc2",
                collection_id=collection_id,
                file_path="/test2.txt",
                file_name="test2.txt",
                file_size=100,
                content_hash="b" * 64,
            )
            # This will be created after scan starts
            new_doc.created_at = datetime.now(UTC)
            return new_doc

        mock_document_repo.create.side_effect = create_side_effect

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create two files
            self.create_test_file(temp_path, "test1.txt", "Content 1")
            self.create_test_file(temp_path, "test2.txt", "Content 2")

            stats = await service.scan_directory_and_register_documents(
                collection_id=collection_id, source_path=temp_dir
            )

            # Both files should be found
            assert stats["total_files_found"] == 2
            assert stats["new_files_registered"] == 1  # Only second file is new
            assert stats["duplicate_files_skipped"] == 1  # First file is duplicate
            assert mock_document_repo.create.call_count == 2

    @pytest.mark.asyncio()
    async def test_scan_directory_recursive(self, service, mock_document_repo):
        """Test recursive directory scanning."""
        collection_id = str(uuid4())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested directory structure
            subdir = temp_path / "subdir"
            subdir.mkdir()
            nested_dir = subdir / "nested"
            nested_dir.mkdir()

            # Create files at different levels
            self.create_test_file(temp_path, "root.txt")
            self.create_test_file(subdir, "sub.pdf")
            self.create_test_file(nested_dir, "nested.docx")

            stats = await service.scan_directory_and_register_documents(
                collection_id=collection_id, source_path=temp_dir, recursive=True
            )

            assert stats["total_files_found"] == 3
            assert mock_document_repo.create.call_count == 3

    @pytest.mark.asyncio()
    async def test_scan_directory_non_recursive(self, service, mock_document_repo):
        """Test non-recursive directory scanning only gets root files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested directory structure
            subdir = temp_path / "subdir"
            subdir.mkdir()

            # Create files at different levels
            self.create_test_file(temp_path, "root.txt")
            self.create_test_file(subdir, "sub.pdf")

            stats = await service.scan_directory_and_register_documents(
                collection_id=str(uuid4()), source_path=temp_dir, recursive=False
            )

            assert stats["total_files_found"] == 1  # Only root file
            assert mock_document_repo.create.call_count == 1

    @pytest.mark.asyncio()
    async def test_scan_directory_with_file_errors(self, service, mock_document_repo):
        """Test handling of file processing errors."""
        # Make document creation fail for testing
        mock_document_repo.create.side_effect = Exception("Database error")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.create_test_file(temp_path, "test.txt")

            stats = await service.scan_directory_and_register_documents(
                collection_id=str(uuid4()), source_path=temp_dir
            )

            assert stats["total_files_found"] == 1
            assert stats["new_files_registered"] == 0
            assert stats["duplicate_files_skipped"] == 0
            assert len(stats["errors"]) == 1
            assert "test.txt" in stats["errors"][0]["file"]
            assert "Database error" in stats["errors"][0]["error"]

    @pytest.mark.asyncio()
    async def test_scan_file_single(self, service, mock_document_repo):
        """Test scanning a single file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"Test content")
            tmp_file.flush()

            try:
                result = await service.scan_file(collection_id=str(uuid4()), file_path=tmp_file.name)

                assert "document_id" in result
                assert result["is_new"] is True
                assert result["file_size"] > 0
                assert result["file_name"] == Path(tmp_file.name).name
                assert result["mime_type"] == "text/plain"

            finally:
                Path(tmp_file.name).unlink()

    @pytest.mark.asyncio()
    async def test_scan_file_unsupported_type(self, service):
        """Test scanning unsupported file type raises ValueError."""
        with (
            tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file,
            pytest.raises(ValueError, match="Unsupported file type"),
        ):
            await service.scan_file(collection_id=str(uuid4()), file_path=tmp_file.name)

    @pytest.mark.asyncio()
    async def test_calculate_file_hash(self, service):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            content = b"Test content for hashing"
            tmp_file.write(content)
            tmp_file.flush()

            try:
                hash_result = await service._calculate_file_hash(Path(tmp_file.name))

                # Verify it's a valid SHA-256 hash (64 hex characters)
                assert len(hash_result) == 64
                assert all(c in "0123456789abcdef" for c in hash_result)

                # Verify hash is consistent
                hash_result2 = await service._calculate_file_hash(Path(tmp_file.name))
                assert hash_result == hash_result2

            finally:
                Path(tmp_file.name).unlink()

    def test_get_mime_type(self, service):
        """Test MIME type detection."""
        # Test known extensions
        assert service._get_mime_type(Path("test.pdf")) == "application/pdf"
        assert service._get_mime_type(Path("test.txt")) == "text/plain"
        assert (
            service._get_mime_type(Path("test.docx"))
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert service._get_mime_type(Path("test.html")) == "text/html"
        assert service._get_mime_type(Path("test.md")) == "text/markdown"

        # Test case insensitive
        assert service._get_mime_type(Path("TEST.PDF")) == "application/pdf"

    def test_supported_extensions(self):
        """Test that all supported extensions are defined correctly."""
        # Verify expected extensions are supported
        expected_extensions = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}
        assert expected_extensions == SUPPORTED_EXTENSIONS

    @pytest.mark.asyncio()
    async def test_large_file_rejection(self, service):
        """Test that files exceeding max size are rejected."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            # Create a file that appears to be over the size limit
            # We'll mock the stat call to return a large size
            tmp_file.write(b"Small content")
            tmp_file.flush()

            try:
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value = MagicMock(st_size=600 * 1024 * 1024)  # 600MB

                    with pytest.raises(ValueError, match="File too large"):
                        await service._register_file(collection_id=str(uuid4()), file_path=Path(tmp_file.name))

            finally:
                Path(tmp_file.name).unlink()

    @pytest.mark.asyncio()
    async def test_batch_processing(self, service, mock_document_repo, mock_session):
        """Test batch processing with commit after batch_size files."""
        collection_id = str(uuid4())

        # Setup mock to return documents with created_at
        def create_mock_document(**kwargs):
            doc = Document(
                id=str(uuid4()),
                collection_id=collection_id,
                file_path=kwargs.get("file_path", "/test/file"),
                file_name=kwargs.get("file_name", "file"),
                file_size=kwargs.get("file_size", 100),
                content_hash=kwargs.get("content_hash", "a" * 64),
            )
            doc.created_at = datetime.now(UTC)
            return doc

        mock_document_repo.create.side_effect = create_mock_document

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create 5 files
            for i in range(5):
                self.create_test_file(temp_path, f"test{i}.txt", f"Content {i}")

            # Set batch size to 2
            stats = await service.scan_directory_and_register_documents(
                collection_id=collection_id, source_path=temp_dir, batch_size=2
            )

            assert stats["total_files_found"] == 5
            # Session commit should be called 3 times (2 + 2 + 1)
            assert mock_session.commit.call_count == 3

    @pytest.mark.asyncio()
    async def test_progress_callback(self, service, mock_document_repo):
        """Test progress callback functionality."""
        collection_id = str(uuid4())
        progress_calls = []

        # Setup mock to return documents with created_at
        def create_mock_document(**kwargs):
            doc = Document(
                id=str(uuid4()),
                collection_id=collection_id,
                file_path=kwargs.get("file_path", "/test/file"),
                file_name=kwargs.get("file_name", "file"),
                file_size=kwargs.get("file_size", 100),
                content_hash=kwargs.get("content_hash", "a" * 64),
            )
            doc.created_at = datetime.now(UTC)
            return doc

        mock_document_repo.create.side_effect = create_mock_document

        async def progress_callback(processed: int, total: int) -> None:
            progress_calls.append((processed, total))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create 3 files
            for i in range(3):
                self.create_test_file(temp_path, f"test{i}.txt", f"Content {i}")

            stats = await service.scan_directory_and_register_documents(
                collection_id=collection_id, source_path=temp_dir, progress_callback=progress_callback
            )

            assert stats["total_files_found"] == 3
            # Progress callback should be called 3 times
            assert len(progress_calls) == 3
            # In non-recursive mode, total is updated incrementally
            assert progress_calls[0][0] == 1  # First file processed
            assert progress_calls[1][0] == 2  # Second file processed
            assert progress_calls[2][0] == 3  # Third file processed
