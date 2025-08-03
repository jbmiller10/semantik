"""Shared fixtures for API endpoint tests."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.webui.api.v2.collections import router as collections_router
from packages.webui.dependencies import get_db
from packages.webui.services.factory import get_collection_service


@pytest.fixture()
def test_client():
    """Create test client with collections router and mocked dependencies."""
    app = FastAPI()
    app.include_router(collections_router)
    
    # Mock database session
    mock_db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Note: Individual tests will override get_collection_service with their own mocks
    
    client = TestClient(app)
    yield client
    
    # Clean up dependency overrides
    app.dependency_overrides.clear()