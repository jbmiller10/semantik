"""Shared test configuration and fixtures."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set required environment variables for tests
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")
os.environ.setdefault("DEFAULT_COLLECTION", "test_collection")
os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from webui.main import app
    return TestClient(app)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.search.return_value = []
    return mock


@pytest.fixture
def test_user():
    """Test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "full_name": "Test User",
        "disabled": False
    }


@pytest.fixture
def auth_headers(test_user):
    """Create authorization headers with a test JWT token."""
    from webui.auth import create_access_token
    token = create_access_token(data={"sub": test_user["username"]})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def temp_test_file(tmp_path):
    """Create a temporary test file."""
    test_file = tmp_path / "test_document.txt"
    test_file.write_text("This is a test document.")
    return test_file


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    mock = MagicMock()
    mock.embed_texts.return_value = [[0.1] * 384]  # Mock embedding vector
    mock.embed_documents.return_value = [[0.1] * 384]
    return mock


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # This helps ensure test isolation
    yield
    # Cleanup code here if needed