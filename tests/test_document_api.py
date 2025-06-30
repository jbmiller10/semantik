"""
Unit tests for Document API endpoints
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import time

# Mock dependencies before imports
with patch('webui.auth.get_current_user'):
    from webui.api.documents import router, IMAGE_SESSIONS, TEMP_IMAGE_DIR
    from webui.app import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock authenticated user"""
    return {
        'id': 1,
        'username': 'testuser',
        'email': 'test@example.com'
    }


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestDocumentAPI:
    """Test document serving API endpoints"""
    
    @patch('webui.auth.get_current_user')
    @patch('webui.database.get_file_by_doc_id')
    def test_get_document_success(self, mock_get_file, mock_auth, client, mock_user, temp_test_dir):
        """Test successful document retrieval"""
        # Setup
        mock_auth.return_value = mock_user
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")
        
        mock_get_file.return_value = {
            'id': 1,
            'job_id': 'test-job',
            'doc_id': 'test-doc',
            'path': str(test_file),
            'filename': 'test.pdf'
        }
        
        # Test
        response = client.get(
            "/api/documents/test-job/test-doc",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/pdf'
        assert 'cache-control' in response.headers
        assert 'etag' in response.headers
    
    @patch('webui.auth.get_current_user')
    def test_path_traversal_prevention(self, mock_auth, client, mock_user):
        """Test that path traversal attacks are prevented"""
        mock_auth.return_value = mock_user
        
        # Attempt path traversal
        response = client.get(
            "/api/documents/test-job/../../../etc/passwd",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 404
    
    @patch('webui.auth.get_current_user')
    @patch('webui.database.get_file_by_doc_id')
    def test_file_size_limit(self, mock_get_file, mock_auth, client, mock_user, temp_test_dir):
        """Test file size limit enforcement"""
        mock_auth.return_value = mock_user
        
        # Create a mock file that exceeds size limit
        test_file = temp_test_dir / "large.pdf"
        test_file.write_text("x" * 1000)
        
        mock_get_file.return_value = {
            'id': 1,
            'job_id': 'test-job',
            'doc_id': 'test-doc',
            'path': str(test_file),
            'filename': 'large.pdf'
        }
        
        # Mock file size to exceed limit
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value = MagicMock(st_size=600*1024*1024)  # 600MB
            
            response = client.get(
                "/api/documents/test-job/test-doc",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 413
    
    def test_temp_image_cleanup(self):
        """Test that temporary images are cleaned up"""
        # Create test session
        session_id = "test-session"
        test_dir = TEMP_IMAGE_DIR / session_id
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Add to sessions with expired time
        IMAGE_SESSIONS[session_id] = (
            1,  # user_id
            time.time() - 7200,  # 2 hours ago
            test_dir
        )
        
        # Import and run cleanup
        from webui.api.documents import cleanup_temp_images
        cleanup_temp_images()
        
        # Check cleanup
        assert session_id not in IMAGE_SESSIONS
        assert not test_dir.exists()


class TestDocumentViewer:
    """Test document viewer security"""
    
    @patch('webui.auth.get_current_user')
    def test_authentication_required(self, mock_auth, client):
        """Test that authentication is required"""
        mock_auth.side_effect = Exception("Not authenticated")
        
        response = client.get("/api/documents/test-job/test-doc")
        assert response.status_code == 403
    
    @patch('webui.auth.get_current_user')
    @patch('webui.database.get_file_by_doc_id')
    @patch('webui.database.user_has_access_to_job')
    def test_authorization_check(self, mock_access, mock_get_file, mock_auth, client, mock_user):
        """Test that user authorization is checked"""
        mock_auth.return_value = mock_user
        mock_get_file.return_value = {'job_id': 'test-job'}
        mock_access.return_value = False
        
        response = client.get(
            "/api/documents/test-job/test-doc",
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 403


class TestPPTXConversion:
    """Test PPTX conversion functionality"""
    
    @patch('webui.api.documents.PPTX2MD_AVAILABLE', True)
    @patch('subprocess.run')
    @patch('webui.auth.get_current_user')
    @patch('webui.database.get_file_by_doc_id')
    def test_pptx_conversion_success(self, mock_get_file, mock_auth, mock_run, client, mock_user, temp_test_dir):
        """Test successful PPTX to Markdown conversion"""
        mock_auth.return_value = mock_user
        
        # Setup test files
        test_pptx = temp_test_dir / "test.pptx"
        test_pptx.write_bytes(b"PPTX content")
        
        mock_get_file.return_value = {
            'id': 1,
            'job_id': 'test-job',
            'doc_id': 'test-doc',
            'path': str(test_pptx),
            'filename': 'test.pptx'
        }
        
        # Mock conversion output
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr=""
        )
        
        # Create expected output file
        with patch('tempfile.TemporaryDirectory') as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(temp_test_dir)
            output_file = temp_test_dir / "test.md"
            output_file.write_text("# Slide 1\\n\\nContent")
            
            response = client.get(
                "/api/documents/test-job/test-doc",
                headers={"Authorization": "Bearer test-token"}
            )
            
            assert response.status_code == 200
            assert response.headers['content-type'] == 'text/markdown; charset=utf-8'
            assert response.headers['x-converted-from'] == 'pptx'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])