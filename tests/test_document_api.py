"""
Unit tests for Document API endpoints
"""

import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from webui.api.documents import IMAGE_SESSIONS, TEMP_IMAGE_DIR


@pytest.fixture()
def temp_test_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestDocumentAPI:
    """Test document serving API endpoints"""

    @patch("webui.database.get_job_files")
    @patch("webui.database.get_job")
    def test_get_document_success(self, mock_get_job, mock_get_files, test_client, test_user, temp_test_dir):
        """Test successful document retrieval"""
        # Setup
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        mock_get_job.return_value = {"id": "test-job", "user_id": test_user["id"], "directory_path": str(temp_test_dir)}

        mock_get_files.return_value = [
            {"id": 1, "job_id": "test-job", "doc_id": "test-doc", "path": str(test_file), "filename": "test.pdf"}
        ]

        # Test
        response = test_client.get("/api/documents/test-job/test-doc")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "cache-control" in response.headers
        assert "etag" in response.headers

    def test_path_traversal_prevention(self, test_client):
        """Test that path traversal attacks are prevented"""
        # Attempt path traversal
        response = test_client.get("/api/documents/test-job/../../../etc/passwd")

        assert response.status_code == 404

    @patch("webui.database.get_job_files")
    @patch("webui.database.get_job")
    def test_file_size_limit(self, mock_get_job, mock_get_files, test_client, test_user, temp_test_dir):
        """Test file size limit enforcement"""

        # Create a mock file that exceeds size limit
        test_file = temp_test_dir / "large.pdf"
        test_file.write_text("x" * 1000)

        mock_get_job.return_value = {"id": "test-job", "user_id": test_user["id"], "directory_path": str(temp_test_dir)}

        mock_get_files.return_value = [
            {"id": 1, "job_id": "test-job", "doc_id": "test-doc", "path": str(test_file), "filename": "large.pdf"}
        ]

        # Mock file size to exceed limit
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=600 * 1024 * 1024)  # 600MB

            response = test_client.get("/api/documents/test-job/test-doc")

            assert response.status_code == 413

    def test_temp_image_cleanup(self):
        """Test that temporary images are cleaned up"""
        # Create test session
        session_id = "test-session"
        test_dir = TEMP_IMAGE_DIR / session_id
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create a test file in the directory
        test_file = test_dir / "test.png"
        test_file.write_text("test image")

        # Add to sessions with expired time
        IMAGE_SESSIONS[session_id] = (1, time.time() - 7200, test_dir)  # user_id  # 2 hours ago

        # Manually trigger cleanup logic (since cleanup_temp_images runs in a thread)
        from webui.api.documents import TEMP_IMAGE_TTL, cleanup_lock

        with cleanup_lock:
            current_time = time.time()
            expired_sessions = []

            for sid, (user_id, created_time, image_dir) in IMAGE_SESSIONS.items():
                if current_time - created_time > TEMP_IMAGE_TTL:
                    expired_sessions.append(sid)
                    if image_dir.exists():
                        shutil.rmtree(image_dir)

            for sid in expired_sessions:
                del IMAGE_SESSIONS[sid]

        # Check cleanup
        assert session_id not in IMAGE_SESSIONS
        assert not test_dir.exists()


class TestDocumentViewer:
    """Test document viewer security"""

    def test_authentication_required(self, unauthenticated_test_client):
        """Test that authentication is required"""
        response = unauthenticated_test_client.get("/api/documents/test-job/test-doc")
        assert response.status_code == 403

    @patch("webui.database.get_job_files")
    @patch("webui.database.get_job")
    def test_authorization_check(self, mock_get_job, mock_get_files, test_client, test_user, temp_test_dir):
        """Test that user authorization is checked"""

        # Create test file
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        # Return job with different user_id
        mock_get_job.return_value = {
            "id": "test-job",
            "user_id": 999,  # Different user
            "directory_path": str(temp_test_dir),
        }

        mock_get_files.return_value = [{"job_id": "test-job", "doc_id": "test-doc", "path": str(test_file)}]

        response = test_client.get("/api/documents/test-job/test-doc")

        assert response.status_code == 403


class TestPPTXConversion:
    """Test PPTX conversion functionality"""

    @patch("webui.api.documents.PPTX2MD_AVAILABLE", True)
    @patch("subprocess.run")
    @patch("webui.database.get_job_files")
    @patch("webui.database.get_job")
    def test_pptx_conversion_success(
        self, mock_get_job, mock_get_files, mock_run, test_client, test_user, temp_test_dir
    ):
        """Test successful PPTX to Markdown conversion"""

        # Setup test files
        test_pptx = temp_test_dir / "test.pptx"
        test_pptx.write_bytes(b"PPTX content")

        mock_get_job.return_value = {"id": "test-job", "user_id": test_user["id"], "directory_path": str(temp_test_dir)}

        mock_get_files.return_value = [
            {"id": 1, "job_id": "test-job", "doc_id": "test-doc", "path": str(test_pptx), "filename": "test.pptx"}
        ]

        # Mock conversion output
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create expected output file
        with patch("tempfile.TemporaryDirectory") as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(temp_test_dir)
            output_file = temp_test_dir / "test.md"
            output_file.write_text("# Slide 1\\n\\nContent")

            response = test_client.get("/api/documents/test-job/test-doc")

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/markdown; charset=utf-8"
            assert response.headers["x-converted-from"] == "pptx"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
