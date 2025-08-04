"""Shared fixtures for API endpoint tests."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from packages.webui.api.v2.collections import router as collections_router
from packages.webui.auth import get_current_user
from packages.webui.dependencies import get_db
from packages.webui.services.factory import get_collection_service


@pytest.fixture()
def test_app():
    """Create test FastAPI app with collections router."""
    app = FastAPI()
    app.include_router(collections_router)
    
    # Mock database session
    mock_db = AsyncMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    
    # Mock authentication - provide a default test user
    mock_user = {"id": "1", "username": "testuser"}
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    yield app
    
    # Clean up dependency overrides
    app.dependency_overrides.clear()


@pytest.fixture()
def test_client(test_app):
    """Create test client from test app."""
    return TestClient(test_app)