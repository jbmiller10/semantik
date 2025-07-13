"""
Unit tests for Document API endpoints
"""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def temp_test_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestDocumentAPI:
    """Test document serving API endpoints"""

    def test_get_document_success(self, test_client_with_mocks, test_user, temp_test_dir, 
                                  mock_job_repository, mock_file_repository):
        """Test successful document retrieval"""
        # Setup
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        # Mock async methods
        from unittest.mock import AsyncMock
        mock_job_repository.get_job = AsyncMock(return_value={"id": "test-job", "user_id": test_user["id"], "directory_path": str(temp_test_dir)})
        mock_file_repository.get_job_files = AsyncMock(return_value=[
            {"id": 1, "job_id": "test-job", "doc_id": "test-doc", "path": str(test_file), "filename": "test.pdf"}
        ])

        # Test
        response = test_client_with_mocks.get("/api/documents/test-job/test-doc")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "cache-control" in response.headers
        assert "etag" in response.headers

    def test_path_traversal_prevention(self, test_client_with_mocks):
        """Test that path traversal attacks are prevented"""
        # Attempt path traversal
        response = test_client_with_mocks.get("/api/documents/test-job/../../../etc/passwd")

        assert response.status_code == 404

    def test_file_size_limit(self, test_client_with_mocks, test_user, temp_test_dir,
                             mock_job_repository, mock_file_repository):
        """Test file size limit enforcement"""

        # Create a mock file that exceeds size limit
        test_file = temp_test_dir / "large.pdf"
        test_file.write_text("x" * 1000)

        # Mock async methods
        from unittest.mock import AsyncMock
        mock_job_repository.get_job = AsyncMock(return_value={"id": "test-job", "user_id": test_user["id"], "directory_path": str(temp_test_dir)})
        mock_file_repository.get_job_files = AsyncMock(return_value=[
            {"id": 1, "job_id": "test-job", "doc_id": "test-doc", "path": str(test_file), "filename": "large.pdf"}
        ])

        # Mock file size to exceed limit
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=600 * 1024 * 1024)  # 600MB

            response = test_client_with_mocks.get("/api/documents/test-job/test-doc")

            assert response.status_code == 413

    def test_temp_image_cleanup(self):
        """Test that temporary images are cleaned up"""
        # Test skipped - requires importing internal module constants
        # This functionality is tested as part of integration tests
        pytest.skip("Test requires importing internal module constants")


class TestDocumentViewer:
    """Test document viewer security"""

    def test_authentication_required(self, unauthenticated_test_client):
        """Test that authentication is required"""
        response = unauthenticated_test_client.get("/api/documents/test-job/test-doc")
        assert response.status_code == 403

    def test_authorization_check(self, test_client_with_mocks, temp_test_dir, 
                                 mock_job_repository, mock_file_repository):
        """Test that user authorization is checked"""

        # Create test file
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        # Mock async methods - job belongs to different user
        from unittest.mock import AsyncMock
        mock_job_repository.get_job = AsyncMock(return_value={
            "id": "test-job",
            "user_id": 999,  # Different user
            "directory_path": str(temp_test_dir),
        })
        mock_file_repository.get_job_files = AsyncMock(return_value=[{"job_id": "test-job", "doc_id": "test-doc", "path": str(test_file)}])

        response = test_client_with_mocks.get("/api/documents/test-job/test-doc")

        assert response.status_code == 403


class TestPPTXConversion:
    """Test PPTX conversion functionality"""

    @patch("webui.api.documents.PPTX2MD_AVAILABLE", True)
    @patch("subprocess.run")
    def test_pptx_conversion_success(
        self, mock_run, test_client_with_mocks, test_user, temp_test_dir,
        mock_job_repository, mock_file_repository
    ):
        """Test successful PPTX to Markdown conversion"""

        # Setup test files
        test_pptx = temp_test_dir / "test.pptx"
        test_pptx.write_bytes(b"PPTX content")

        # Mock async methods
        from unittest.mock import AsyncMock
        mock_job_repository.get_job = AsyncMock(return_value={"id": "test-job", "user_id": test_user["id"], "directory_path": str(temp_test_dir)})
        mock_file_repository.get_job_files = AsyncMock(return_value=[
            {"id": 1, "job_id": "test-job", "doc_id": "test-doc", "path": str(test_pptx), "filename": "test.pptx"}
        ])

        # Mock conversion output
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create expected output file
        with patch("tempfile.TemporaryDirectory") as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(temp_test_dir)
            output_file = temp_test_dir / "test.md"
            output_file.write_text("# Slide 1\\n\\nContent")

            response = test_client_with_mocks.get("/api/documents/test-job/test-doc")

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/markdown; charset=utf-8"
            assert response.headers["x-converted-from"] == "pptx"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
