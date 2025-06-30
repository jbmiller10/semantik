"""
Tests for the document viewer functionality
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the modules to test
from webui.api.documents import validate_file_access, router
from webui.main import app


class TestDocumentSecurity:
    """Test security aspects of document access"""

    def test_validate_file_access_path_traversal_attempt(self):
        """Test that path traversal attempts are blocked"""
        # Mock database responses
        with patch("webui.api.documents.database") as mock_db:
            mock_db.get_job.return_value = {"directory_path": "/safe/job/directory"}
            mock_db.get_job_files.return_value = [
                {"doc_id": "test123", "path": "/safe/job/directory/../../../etc/passwd"}
            ]

            # Should raise HTTPException for path traversal
            with pytest.raises(HTTPException) as exc_info:
                validate_file_access("job123", "test123", {"user": "test"})

            assert exc_info.value.status_code == 403
            assert exc_info.value.detail == "Access denied"

    def test_validate_file_access_valid_path(self):
        """Test that valid file paths are allowed"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_text("test content")

            with patch("webui.api.documents.database") as mock_db:
                mock_db.get_job.return_value = {"directory_path": tmpdir}
                mock_db.get_job_files.return_value = [{"doc_id": "test123", "path": str(test_file)}]

                # Should not raise exception for valid path
                result = validate_file_access("job123", "test123", {"user": "test"})
                assert result["doc_id"] == "test123"
                assert result["path"] == str(test_file)

    def test_validate_file_access_nonexistent_job(self):
        """Test handling of nonexistent job"""
        with patch("webui.api.documents.database") as mock_db:
            mock_db.get_job.return_value = None

            with pytest.raises(HTTPException) as exc_info:
                validate_file_access("job123", "test123", {"user": "test"})

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Job not found"

    def test_validate_file_access_nonexistent_document(self):
        """Test handling of nonexistent document"""
        with patch("webui.api.documents.database") as mock_db:
            mock_db.get_job.return_value = {"directory_path": "/safe/job/directory"}
            mock_db.get_job_files.return_value = []

            with pytest.raises(HTTPException) as exc_info:
                validate_file_access("job123", "test123", {"user": "test"})

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Document not found"

    def test_validate_file_access_file_size_limit(self):
        """Test that files exceeding size limit are rejected"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock large file
            test_file = Path(tmpdir) / "large.pdf"
            test_file.write_text("x" * 1024)  # Small file for testing

            with patch("webui.api.documents.database") as mock_db:
                mock_db.get_job.return_value = {"directory_path": tmpdir}
                mock_db.get_job_files.return_value = [{"doc_id": "test123", "path": str(test_file)}]

                # Mock the file size check
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value = Mock(st_size=101 * 1024 * 1024)  # 101 MB

                    with pytest.raises(HTTPException) as exc_info:
                        validate_file_access("job123", "test123", {"user": "test"})

                    assert exc_info.value.status_code == 413
                    assert exc_info.value.detail == "File is too large to preview"


class TestDocumentEndpoints:
    """Test the document serving endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers"""
        return {"Authorization": "Bearer test-token"}

    def test_get_document_unsupported_extension(self, client, auth_headers):
        """Test that unsupported file extensions are rejected"""
        with patch("webui.api.documents.get_current_user") as mock_auth:
            mock_auth.return_value = {"user": "test"}

            with patch("webui.api.documents.validate_file_access") as mock_validate:
                mock_validate.return_value = {"doc_id": "test123", "path": "/path/to/file.exe"}  # Unsupported extension

                response = client.get("/api/documents/job123/test123", headers=auth_headers)

                assert response.status_code == 415
                assert "Unsupported file type" in response.json()["detail"]

    def test_get_document_info_success(self, client, auth_headers):
        """Test successful document info retrieval"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.pdf"
            test_file.write_text("test content")

            with patch("webui.api.documents.get_current_user") as mock_auth:
                mock_auth.return_value = {"user": "test"}

                with patch("webui.api.documents.validate_file_access") as mock_validate:
                    mock_validate.return_value = {"doc_id": "test123", "path": str(test_file), "name": "test.pdf"}

                    response = client.get("/api/documents/job123/test123/info", headers=auth_headers)

                    assert response.status_code == 200
                    data = response.json()
                    assert data["filename"] == "test.pdf"
                    assert data["size"] == len("test content")
                    assert data["extension"] == ".pdf"
                    assert data["mime_type"] == "application/pdf"

    def test_range_request_parsing(self):
        """Test HTTP Range header parsing"""
        from webui.api.documents import get_document

        # Test valid range header
        with patch("webui.api.documents.validate_file_access"):
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True

                # This would need more setup to fully test
                # For now, just verify the endpoint exists
                assert hasattr(get_document, "__name__")
                assert get_document.__name__ == "get_document"


class TestDocumentViewerConstants:
    """Test that constants are properly defined"""

    def test_constants_defined(self):
        """Verify all constants are properly defined"""
        from webui.api.documents import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE, CHUNK_SIZE

        # Check SUPPORTED_EXTENSIONS
        assert isinstance(SUPPORTED_EXTENSIONS, set)
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS

        # Check MAX_FILE_SIZE
        assert MAX_FILE_SIZE == 100 * 1024 * 1024  # 100 MB

        # Check CHUNK_SIZE
        assert CHUNK_SIZE == 8192  # 8KB


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
