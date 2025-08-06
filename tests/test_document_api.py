"""
Unit tests for Document API endpoints
"""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def temp_test_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture()
def mock_operation_repository() -> MagicMock:
    """Create a mock OperationRepository for testing."""
    return MagicMock()


@pytest.fixture()
def mock_document_repository() -> AsyncMock:
    """Create a mock DocumentRepository for testing."""
    return AsyncMock()


@pytest.fixture()
def test_client_with_document_mocks(
    test_user,
    mock_operation_repository,  # noqa: ARG001
    mock_document_repository,  # noqa: ARG001
) -> None:
    """Create a test client with mocked document-related repositories."""
    from packages.webui.auth import get_current_user
    from packages.webui.dependencies import get_collection_for_user
    from packages.webui.main import app

    # Override the authentication dependency
    async def override_get_current_user():
        return test_user

    # Mock collection for access control
    mock_collection = MagicMock()
    mock_collection.uuid = "test-operation"
    mock_collection.user_id = test_user["id"]

    async def override_get_collection_for_user(collection_uuid: str, current_user=None, db=None):  # noqa: ARG001
        return mock_collection

    # Mock database session
    from unittest.mock import AsyncMock

    from packages.shared.database import get_db

    mock_db = AsyncMock()

    async def override_get_db():
        yield mock_db

    # Override dependencies
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_collection_for_user] = override_get_collection_for_user
    app.dependency_overrides[get_db] = override_get_db

    from fastapi.testclient import TestClient

    client = TestClient(app)

    # Ensure we clean up after the test
    yield client

    app.dependency_overrides.clear()


class TestDocumentAPI:
    """Test document serving API endpoints"""

    def test_get_document_success(
        self,
        test_client_with_document_mocks,
        test_user,
        temp_test_dir,
        mock_operation_repository,
        mock_document_repository,
    ) -> None:
        """Test successful document retrieval"""
        # Setup
        test_file = temp_test_dir / "test.pdf"
        test_file.write_text("PDF content")

        # Mock async methods
        from unittest.mock import AsyncMock, MagicMock

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.id = 1
        mock_operation_repository.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)

        # Create mock document
        mock_document = MagicMock()
        mock_document.id = "test-doc"
        mock_document.collection_id = "test-operation"
        mock_document.file_path = str(test_file)
        mock_document.file_name = "test.pdf"
        mock_document.mime_type = "application/pdf"
        mock_document.size = 1000
        mock_document_repository.get_by_id = AsyncMock(return_value=mock_document)

        # Patch create_document_repository to return our mock
        with patch("packages.webui.api.v2.documents.create_document_repository", return_value=mock_document_repository):
            # Test
            response = test_client_with_document_mocks.get(
                "/api/v2/collections/test-operation/documents/test-doc/content"
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert "test.pdf" in response.headers["content-disposition"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
