"""
Unit tests for Document API endpoints
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def temp_test_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture()
def mock_operation_repository():
    """Create a mock OperationRepository for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture()
def mock_document_repository():
    """Create a mock DocumentRepository for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture()
def test_client_with_document_mocks(
    test_user,
    mock_operation_repository,
    mock_document_repository,
):
    """Create a test client with mocked document-related repositories."""
    from shared.database.factory import create_operation_repository, create_document_repository

    from packages.webui.auth import get_current_user
    from packages.webui.main import app

    # Override the authentication dependency
    async def override_get_current_user():
        return test_user

    # Override repository dependencies
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[create_operation_repository] = lambda: mock_operation_repository
    app.dependency_overrides[create_document_repository] = lambda: mock_document_repository

    from fastapi.testclient import TestClient
    client = TestClient(app)

    # Ensure we clean up after the test
    yield client

    app.dependency_overrides.clear()


class TestDocumentAPI:
    """Test document serving API endpoints"""

    def test_get_document_success(
        self, test_client_with_document_mocks, test_user, temp_test_dir, mock_operation_repository, mock_document_repository
    ):
        """Test successful document retrieval"""
        # Setup
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        # Mock async methods
        from unittest.mock import AsyncMock, MagicMock

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.id = 1
        mock_operation_repository.get_by_uuid_with_permission_check = AsyncMock(
            return_value=mock_operation
        )

        # Create mock document
        mock_document = MagicMock()
        mock_document.doc_id = "test-doc"
        mock_document.path = str(test_file)
        mock_document.size = 1000
        mock_document.extension = ".pdf"
        mock_document.modified = "2024-01-01"
        mock_document.operation_id = 1
        mock_document_repository.get_by_doc_id = AsyncMock(
            return_value=mock_document
        )

        # Test
        response = test_client_with_document_mocks.get("/api/documents/test-operation/test-doc")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "cache-control" in response.headers
        assert "etag" in response.headers

    def test_path_traversal_prevention(self, test_client_with_document_mocks):
        """Test that path traversal attacks are prevented"""
        # Attempt path traversal
        response = test_client_with_document_mocks.get("/api/documents/test-operation/../../../etc/passwd")

        assert response.status_code == 404

    def test_file_size_limit(
        self, test_client_with_document_mocks, test_user, temp_test_dir, mock_operation_repository, mock_document_repository
    ):
        """Test file size limit enforcement"""

        # Create a mock file that exceeds size limit
        test_file = temp_test_dir / "large.pdf"
        test_file.write_text("x" * 1000)

        # Mock async methods
        from unittest.mock import AsyncMock, MagicMock

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.id = 1
        mock_operation_repository.get_by_uuid_with_permission_check = AsyncMock(
            return_value=mock_operation
        )

        # Create mock document
        mock_document = MagicMock()
        mock_document.doc_id = "test-doc"
        mock_document.path = str(test_file)
        mock_document.size = 600 * 1024 * 1024  # 600MB
        mock_document.extension = ".pdf"
        mock_document.modified = "2024-01-01"
        mock_document.operation_id = 1
        mock_document_repository.get_by_doc_id = AsyncMock(
            return_value=mock_document
        )

        # Mock file size to exceed limit
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=600 * 1024 * 1024)  # 600MB

            response = test_client_with_document_mocks.get("/api/documents/test-operation/test-doc")

            assert response.status_code == 413

    def test_temp_image_cleanup(self):
        """Test that temporary images are cleaned up"""
        # Test skipped - requires importing internal module constants
        # This functionality is tested as part of integration tests
        pytest.skip("Test requires importing internal module constants")


class TestDocumentViewer:
    """Test document viewer security"""

    def test_authentication_required(self, unauthenticated_test_client, monkeypatch):
        """Test that authentication is required"""
        # Temporarily disable DISABLE_AUTH for this test
        monkeypatch.setattr("packages.webui.auth.settings.DISABLE_AUTH", False)
        response = unauthenticated_test_client.get("/api/documents/test-operation/test-doc")
        assert response.status_code == 401  # Should be 401 Unauthorized, not 403

    def test_authorization_check(
        self, test_client_with_document_mocks, temp_test_dir, mock_operation_repository, mock_document_repository
    ):
        """Test that user authorization is checked"""

        # Create test file
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        # Mock async methods - simulate permission denied
        from unittest.mock import AsyncMock

        # Mock permission check to raise exception
        mock_operation_repository.get_by_uuid_with_permission_check = AsyncMock(
            side_effect=Exception("Permission denied")
        )

        response = test_client_with_document_mocks.get("/api/documents/test-operation/test-doc")

        # Should return 404 (generic error for security)
        assert response.status_code == 404


class TestPPTXConversion:
    """Test PPTX conversion functionality"""

    @patch("packages.webui.api.documents.PPTX2MD_AVAILABLE", True)
    @patch("subprocess.run")
    def test_pptx_conversion_success(
        self, mock_run, test_client_with_document_mocks, test_user, temp_test_dir, 
        mock_operation_repository, mock_document_repository
    ):
        """Test successful PPTX to Markdown conversion"""

        # Setup test files
        test_pptx = temp_test_dir / "test.pptx"
        test_pptx.write_bytes(b"PPTX content")

        # Mock async methods
        from unittest.mock import AsyncMock, MagicMock

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.id = 1
        mock_operation_repository.get_by_uuid_with_permission_check = AsyncMock(
            return_value=mock_operation
        )

        # Create mock document
        mock_document = MagicMock()
        mock_document.doc_id = "test-doc"
        mock_document.path = str(test_pptx)
        mock_document.size = 1000
        mock_document.extension = ".pptx"
        mock_document.modified = "2024-01-01"
        mock_document.operation_id = 1
        mock_document_repository.get_by_doc_id = AsyncMock(
            return_value=mock_document
        )

        # Mock conversion output
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        # Create expected output file
        with patch("tempfile.TemporaryDirectory") as mock_temp:
            mock_temp.return_value.__enter__.return_value = str(temp_test_dir)
            output_file = temp_test_dir / "output.md"
            output_file.write_text("# Slide 1\\n\\nContent")

            response = test_client_with_document_mocks.get("/api/documents/test-operation/test-doc")

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/markdown"
            assert response.headers["x-converted-from"] == "pptx"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])