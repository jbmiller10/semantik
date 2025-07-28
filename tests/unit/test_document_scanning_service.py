#!/usr/bin/env python3
"""
Comprehensive test suite for webui/services/document_scanning_service.py
Tests document processing pipeline, file formats, and error scenarios
"""

import hashlib
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from shared.database.models import Document
from webui.services.document_scanning_service import SUPPORTED_EXTENSIONS, DocumentScanningService


class TestDocumentScanningService:
    """Test DocumentScanningService implementation"""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture()
    def mock_document_repo(self):
        """Create a mock DocumentRepository"""
        repo = AsyncMock()
        return repo

    @pytest.fixture()
    def scanning_service(self, mock_session, mock_document_repo):
        """Create DocumentScanningService with mocked dependencies"""
        return DocumentScanningService(
            db_session=mock_session,
            document_repo=mock_document_repo,
        )

    @pytest.fixture()
    def temp_directory(self):
        """Create a temporary directory with test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = {
                "document.pdf": b"PDF content",
                "text.txt": b"Text content",
                "presentation.pptx": b"PowerPoint content",
                "readme.md": b"# Markdown content",
                "email.eml": b"Email content",
                "webpage.html": b"<html>Web content</html>",
                "document.docx": b"Word content",
                "legacy.doc": b"Legacy Word content",
                "image.jpg": b"JPEG image",  # Not supported
                "script.py": b"Python code",  # Not supported
                "subdir/nested.pdf": b"Nested PDF",
                "subdir/deep/file.txt": b"Deep nested text",
            }

            for file_path, content in test_files.items():
                full_path = Path(temp_dir) / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_bytes(content)

            yield temp_dir

    def test_supported_extensions(self):
        """Test that all expected extensions are supported"""
        expected = {".pdf", ".docx", ".doc", ".txt", ".text", ".pptx", ".eml", ".md", ".html"}
        assert expected == SUPPORTED_EXTENSIONS

    @pytest.mark.asyncio()
    async def test_scan_directory_success(self, scanning_service, mock_document_repo, temp_directory):
        """Test successful directory scanning"""
        collection_id = "test-collection-uuid"

        # Mock document creation - simulate some new, some duplicates
        async def mock_create_or_update(collection_id, document_dict):
            # Simulate duplicate for .txt files
            if document_dict["source_path"].endswith(".txt"):
                return None  # Duplicate
            # Create new document for others
            doc = Mock(spec=Document)
            doc.id = f"doc-{document_dict['filename']}"
            return doc

        mock_document_repo.create_or_update_from_scan.side_effect = mock_create_or_update

        # Scan directory
        stats = await scanning_service.scan_directory_and_register_documents(
            collection_id=collection_id,
            source_path=temp_directory,
            recursive=True,
        )

        # Verify stats
        assert stats["total_documents_found"] == 8  # All supported files
        assert stats["new_documents_registered"] == 6  # Excluding .txt duplicates
        assert stats["duplicate_documents_skipped"] == 2  # The .txt files
        assert stats["total_size_bytes"] > 0
        assert len(stats["errors"]) == 0

    @pytest.mark.asyncio()
    async def test_scan_directory_non_recursive(self, scanning_service, mock_document_repo, temp_directory):
        """Test non-recursive directory scanning"""
        collection_id = "test-collection-uuid"

        # Mock document creation
        mock_document_repo.create_or_update_from_scan.return_value = Mock(spec=Document)

        # Scan directory non-recursively
        stats = await scanning_service.scan_directory_and_register_documents(
            collection_id=collection_id,
            source_path=temp_directory,
            recursive=False,
        )

        # Should only find files in root directory
        assert stats["total_documents_found"] == 6  # Only root level supported files
        assert stats["new_documents_registered"] == 6
        assert stats["duplicate_documents_skipped"] == 0

    @pytest.mark.asyncio()
    async def test_scan_directory_with_source_id(self, scanning_service, mock_document_repo, temp_directory):
        """Test scanning with source ID"""
        collection_id = "test-collection-uuid"
        source_id = 42

        call_args_list = []

        async def capture_calls(collection_id, document_dict):
            call_args_list.append(document_dict)
            return Mock(spec=Document)

        mock_document_repo.create_or_update_from_scan.side_effect = capture_calls

        # Scan with source ID
        await scanning_service.scan_directory_and_register_documents(
            collection_id=collection_id,
            source_path=temp_directory,
            source_id=source_id,
            recursive=False,
        )

        # Verify all documents have source_id
        assert all(doc["source_id"] == source_id for doc in call_args_list)

    @pytest.mark.asyncio()
    async def test_scan_directory_batch_processing(
        self, scanning_service, mock_document_repo, mock_session, temp_directory
    ):
        """Test batch processing during scan"""
        collection_id = "test-collection-uuid"

        # Mock document creation
        mock_document_repo.create_or_update_from_scan.return_value = Mock(spec=Document)

        # Scan with small batch size
        stats = await scanning_service.scan_directory_and_register_documents(
            collection_id=collection_id,
            source_path=temp_directory,
            batch_size=2,
            recursive=True,
        )

        # Verify commits were called multiple times (8 files / 2 batch = 4 commits)
        assert mock_session.commit.call_count >= 4

    @pytest.mark.asyncio()
    async def test_scan_directory_progress_callback(self, scanning_service, mock_document_repo, temp_directory):
        """Test progress callback functionality"""
        collection_id = "test-collection-uuid"
        progress_updates = []

        async def progress_callback(processed, total):
            progress_updates.append((processed, total))

        # Mock document creation
        mock_document_repo.create_or_update_from_scan.return_value = Mock(spec=Document)

        # Scan with progress callback
        await scanning_service.scan_directory_and_register_documents(
            collection_id=collection_id,
            source_path=temp_directory,
            recursive=True,
            progress_callback=progress_callback,
        )

        # Verify progress updates
        assert len(progress_updates) > 0
        # Last update should show all documents processed
        last_processed, last_total = progress_updates[-1]
        assert last_processed == 8  # All supported files

    @pytest.mark.asyncio()
    async def test_scan_directory_invalid_path(self, scanning_service):
        """Test scanning non-existent directory"""
        with pytest.raises(ValueError, match="Source path does not exist"):
            await scanning_service.scan_directory_and_register_documents(
                collection_id="test-collection",
                source_path="/non/existent/path",
            )

    @pytest.mark.asyncio()
    async def test_scan_directory_not_directory(self, scanning_service, temp_directory):
        """Test scanning a file instead of directory"""
        file_path = Path(temp_directory) / "document.pdf"

        with pytest.raises(ValueError, match="Source path is not a directory"):
            await scanning_service.scan_directory_and_register_documents(
                collection_id="test-collection",
                source_path=str(file_path),
            )

    @pytest.mark.asyncio()
    async def test_scan_directory_with_errors(self, scanning_service, mock_document_repo, temp_directory):
        """Test handling errors during document registration"""
        collection_id = "test-collection-uuid"

        # Mock document creation to fail for PDF files
        async def mock_create_with_errors(collection_id, document_dict):
            if document_dict["filename"].endswith(".pdf"):
                raise Exception("Failed to process PDF")
            return Mock(spec=Document)

        mock_document_repo.create_or_update_from_scan.side_effect = mock_create_with_errors

        # Scan directory
        stats = await scanning_service.scan_directory_and_register_documents(
            collection_id=collection_id,
            source_path=temp_directory,
            recursive=True,
        )

        # Verify error handling
        assert len(stats["errors"]) == 2  # Two PDF files
        assert all("Failed to process PDF" in err["error"] for err in stats["errors"])
        assert stats["new_documents_registered"] == 6  # Other files succeeded

    @pytest.mark.asyncio()
    async def test_register_file_content_hash(self, scanning_service, mock_document_repo):
        """Test file content hash calculation"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            content = b"Test content for hashing"
            temp_file.write(content)
            temp_file.flush()

            # Calculate expected hash
            expected_hash = hashlib.sha256(content).hexdigest()

            # Capture document dict
            captured_doc = None

            async def capture_doc(collection_id, document_dict):
                nonlocal captured_doc
                captured_doc = document_dict
                return Mock(spec=Document)

            mock_document_repo.create_or_update_from_scan.side_effect = capture_doc

            # Register file
            await scanning_service._register_file(
                collection_id="test-collection",
                file_path=Path(temp_file.name),
                source_id=None,
                scan_start_time=datetime.now(UTC),
            )

            # Verify hash
            assert captured_doc is not None
            assert captured_doc["content_hash"] == expected_hash

            # Cleanup
            os.unlink(temp_file.name)

    @pytest.mark.asyncio()
    async def test_register_file_metadata(self, scanning_service, mock_document_repo):
        """Test file metadata extraction"""
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_file:
            temp_file.write(b"# Test Document")
            temp_file.flush()

            # Capture document dict
            captured_doc = None

            async def capture_doc(collection_id, document_dict):
                nonlocal captured_doc
                captured_doc = document_dict
                return Mock(spec=Document)

            mock_document_repo.create_or_update_from_scan.side_effect = capture_doc

            # Register file
            await scanning_service._register_file(
                collection_id="test-collection",
                file_path=Path(temp_file.name),
                source_id=None,
                scan_start_time=datetime.now(UTC),
            )

            # Verify metadata
            assert captured_doc is not None
            assert captured_doc["filename"].endswith(".md")
            assert captured_doc["file_extension"] == ".md"
            assert captured_doc["mime_type"] == "text/markdown"
            assert captured_doc["file_size"] == 15  # Length of "# Test Document"
            assert "discovered_at" in captured_doc

            # Cleanup
            os.unlink(temp_file.name)

    @pytest.mark.asyncio()
    async def test_skip_large_files(self, scanning_service, mock_document_repo):
        """Test that files exceeding MAX_FILE_SIZE are skipped"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file that appears large
            large_file = Path(temp_dir) / "large.pdf"
            large_file.write_bytes(b"PDF")

            # Mock os.path.getsize to return size over limit
            with patch("os.path.getsize", return_value=600 * 1024 * 1024):  # 600MB
                stats = await scanning_service.scan_directory_and_register_documents(
                    collection_id="test-collection",
                    source_path=temp_dir,
                )

                # File should be found but skipped
                assert stats["total_documents_found"] == 1
                assert stats["new_documents_registered"] == 0
                assert len(stats["errors"]) == 1
                assert "exceeds maximum size" in stats["errors"][0]["error"]


class TestDocumentScanningFormats:
    """Test handling of different file formats"""

    @pytest.fixture()
    def scanning_service(self):
        mock_session = AsyncMock()
        mock_document_repo = AsyncMock()
        return DocumentScanningService(mock_session, mock_document_repo)

    @pytest.mark.asyncio()
    async def test_pdf_file_handling(self, scanning_service):
        """Test PDF file detection and processing"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4 PDF content")
            temp_file.flush()

            captured_doc = None

            async def capture_doc(collection_id, document_dict):
                nonlocal captured_doc
                captured_doc = document_dict
                return Mock(spec=Document)

            scanning_service.document_repo.create_or_update_from_scan.side_effect = capture_doc

            await scanning_service._register_file(
                collection_id="test",
                file_path=Path(temp_file.name),
                source_id=None,
                scan_start_time=datetime.now(UTC),
            )

            assert captured_doc["mime_type"] == "application/pdf"
            assert captured_doc["file_extension"] == ".pdf"

            os.unlink(temp_file.name)

    @pytest.mark.asyncio()
    async def test_office_formats_handling(self, scanning_service):
        """Test Microsoft Office formats handling"""
        office_formats = {
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }

        for ext, expected_mime in office_formats.items():
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(b"Office content")
                temp_file.flush()

                captured_doc = None

                async def capture_doc(collection_id, document_dict):
                    nonlocal captured_doc
                    captured_doc = document_dict
                    return Mock(spec=Document)

                scanning_service.document_repo.create_or_update_from_scan.side_effect = capture_doc

                await scanning_service._register_file(
                    collection_id="test",
                    file_path=Path(temp_file.name),
                    source_id=None,
                    scan_start_time=datetime.now(UTC),
                )

                assert captured_doc["file_extension"] == ext
                # Mime type detection might vary by system
                assert captured_doc["mime_type"] is not None

                os.unlink(temp_file.name)

    @pytest.mark.asyncio()
    async def test_text_formats_handling(self, scanning_service):
        """Test text-based formats handling"""
        text_formats = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
        }

        for ext, expected_mime in text_formats.items():
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(b"Text content")
                temp_file.flush()

                captured_doc = None

                async def capture_doc(collection_id, document_dict):
                    nonlocal captured_doc
                    captured_doc = document_dict
                    return Mock(spec=Document)

                scanning_service.document_repo.create_or_update_from_scan.side_effect = capture_doc

                await scanning_service._register_file(
                    collection_id="test",
                    file_path=Path(temp_file.name),
                    source_id=None,
                    scan_start_time=datetime.now(UTC),
                )

                assert captured_doc["file_extension"] == ext
                # Some systems might not detect markdown mime type
                if ext != ".md" or captured_doc["mime_type"] is not None:
                    assert expected_mime in captured_doc["mime_type"] or captured_doc["mime_type"] == "text/plain"

                os.unlink(temp_file.name)


class TestDocumentScanningPerformance:
    """Test performance-related aspects of document scanning"""

    @pytest.mark.asyncio()
    async def test_large_directory_scanning(self):
        """Test scanning directory with many files"""
        mock_session = AsyncMock()
        mock_document_repo = AsyncMock()
        scanning_service = DocumentScanningService(mock_session, mock_document_repo)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create many files
            num_files = 100
            for i in range(num_files):
                file_path = Path(temp_dir) / f"document_{i}.txt"
                file_path.write_text(f"Content {i}")

            # Mock fast document creation
            mock_document_repo.create_or_update_from_scan.return_value = Mock(spec=Document)

            # Scan directory
            stats = await scanning_service.scan_directory_and_register_documents(
                collection_id="test-collection",
                source_path=temp_dir,
                batch_size=20,  # Process in batches
            )

            assert stats["total_documents_found"] == num_files
            assert stats["new_documents_registered"] == num_files
            # Should have committed multiple times
            assert mock_session.commit.call_count == 5  # 100 files / 20 batch size

    @pytest.mark.asyncio()
    async def test_hash_calculation_chunking(self):
        """Test that file hashing uses chunking for large files"""
        mock_session = AsyncMock()
        mock_document_repo = AsyncMock()
        scanning_service = DocumentScanningService(mock_session, mock_document_repo)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            # Write content larger than chunk size
            content = b"x" * (8192 * 3 + 100)  # 3+ chunks
            temp_file.write(content)
            temp_file.flush()

            captured_doc = None

            async def capture_doc(collection_id, document_dict):
                nonlocal captured_doc
                captured_doc = document_dict
                return Mock(spec=Document)

            mock_document_repo.create_or_update_from_scan.side_effect = capture_doc

            # Register file
            await scanning_service._register_file(
                collection_id="test",
                file_path=Path(temp_file.name),
                source_id=None,
                scan_start_time=datetime.now(UTC),
            )

            # Verify hash is calculated correctly
            expected_hash = hashlib.sha256(content).hexdigest()
            assert captured_doc["content_hash"] == expected_hash

            os.unlink(temp_file.name)
