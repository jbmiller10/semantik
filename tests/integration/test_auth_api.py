"""Integration tests for authentication API endpoints."""

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


class TempDatabase:
    """Context manager for temporary database."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def __enter__(self):
        return self.db_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


@pytest.fixture()
def test_db():
    """Create a temporary database for each test."""
    with TempDatabase() as db_path:
        # Patch the DB_PATH at the module level
        with patch("webui.database.DB_PATH", db_path):
            # Import and initialize after patching
            from webui.database import init_db

            init_db()
            yield db_path


@pytest.fixture()
def client(test_db):
    """Create a test client with the isolated database."""
    from webui.main import app

    # Clear any dependency overrides
    app.dependency_overrides.clear()

    return TestClient(app)


def test_user_registration_success(client):
    """Test successful user registration."""
    # Prepare registration data
    registration_data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "securepassword123",
        "full_name": "New Test User",
    }

    # Send registration request
    response = client.post("/api/auth/register", json=registration_data)

    # Assert successful registration
    assert response.status_code == 200

    # Verify response data
    data = response.json()
    assert data["username"] == registration_data["username"]
    assert data["email"] == registration_data["email"]
    assert data["full_name"] == registration_data["full_name"]
    assert "id" in data
    assert "created_at" in data
    assert "is_active" in data
    assert data["is_active"] is True
    # Password should not be returned
    assert "password" not in data
    assert "hashed_password" not in data


def test_user_registration_duplicate(client):
    """Test that duplicate registration fails."""
    # Register first user
    registration_data = {
        "username": "duplicateuser",
        "email": "duplicate@example.com",
        "password": "password123",
        "full_name": "Duplicate User",
    }

    # First registration should succeed
    response = client.post("/api/auth/register", json=registration_data)
    assert response.status_code == 200

    # Attempt to register with same username
    duplicate_username_data = {
        "username": "duplicateuser",  # Same username
        "email": "different@example.com",
        "password": "password123",
        "full_name": "Different User",
    }

    response = client.post("/api/auth/register", json=duplicate_username_data)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()

    # Attempt to register with same email
    duplicate_email_data = {
        "username": "differentuser",
        "email": "duplicate@example.com",  # Same email
        "password": "password123",
        "full_name": "Different User",
    }

    response = client.post("/api/auth/register", json=duplicate_email_data)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_user_login_success(client):
    """Test successful user login."""
    # First register a user
    registration_data = {
        "username": "loginuser",
        "email": "login@example.com",
        "password": "loginpassword123",
        "full_name": "Login Test User",
    }

    reg_response = client.post("/api/auth/register", json=registration_data)
    assert reg_response.status_code == 200

    # Now test login
    login_data = {"username": "loginuser", "password": "loginpassword123"}

    response = client.post("/api/auth/login", json=login_data)

    # Assert successful login
    assert response.status_code == 200

    # Verify tokens are returned
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"

    # Tokens should be non-empty strings
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0
    assert isinstance(data["refresh_token"], str)
    assert len(data["refresh_token"]) > 0


def test_user_login_failure(client):
    """Test login with incorrect password."""
    # First register a user
    registration_data = {
        "username": "testfailure",
        "email": "failure@example.com",
        "password": "correctpassword123",
        "full_name": "Failure Test User",
    }

    reg_response = client.post("/api/auth/register", json=registration_data)
    assert reg_response.status_code == 200

    # Try to login with wrong password
    login_data = {"username": "testfailure", "password": "wrongpassword123"}

    response = client.post("/api/auth/login", json=login_data)

    # Assert unauthorized
    assert response.status_code == 401
    assert "incorrect username or password" in response.json()["detail"].lower()

    # Also test with non-existent username
    login_data = {"username": "nonexistentuser", "password": "anypassword"}

    response = client.post("/api/auth/login", json=login_data)
    assert response.status_code == 401
    assert "incorrect username or password" in response.json()["detail"].lower()


def test_get_me_protected(client):
    """Test that /me endpoint requires authentication."""
    # Test without token - should fail
    response = client.get("/api/auth/me")
    assert response.status_code == 403
    assert "not authenticated" in response.json()["detail"].lower()

    # Test with invalid token - should fail
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 401

    # Register and login to get valid token
    registration_data = {
        "username": "protecteduser",
        "email": "protected@example.com",
        "password": "protectedpass123",
        "full_name": "Protected Test User",
    }

    reg_response = client.post("/api/auth/register", json=registration_data)
    assert reg_response.status_code == 200

    login_data = {"username": "protecteduser", "password": "protectedpass123"}

    login_response = client.post("/api/auth/login", json=login_data)
    assert login_response.status_code == 200
    access_token = login_response.json()["access_token"]

    # Test with valid token - should succeed
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/auth/me", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "protecteduser"
    assert data["email"] == "protected@example.com"
    assert data["full_name"] == "Protected Test User"
    assert "id" in data
    assert "created_at" in data
    assert "is_active" in data
    assert data["is_active"] is True
