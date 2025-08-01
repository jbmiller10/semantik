"""Integration tests for authentication API endpoints."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from passlib.context import CryptContext

# Create pwd_context locally to avoid imports from shared.database
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@pytest.fixture()
def mock_repositories() -> None:
    """Create mock repositories for testing."""
    # Mock user repository
    mock_user_repo = MagicMock()
    mock_auth_repo = MagicMock()

    # Store users in memory for testing
    users_db = {}

    async def create_user(user_data):
        # Handle both dictionary input and keyword arguments
        if isinstance(user_data, dict):
            username = user_data.get("username")
            email = user_data.get("email")
            hashed_password = user_data.get("hashed_password", pwd_context.hash(user_data.get("password", "")))
            full_name = user_data.get("full_name", "")
            is_superuser = user_data.get("is_superuser", False)
        else:
            # Handle keyword args case
            username = user_data
            email = None
            hashed_password = None
            full_name = ""
            is_superuser = False

        # Check for existing username
        if username in users_db:
            raise ValueError(f"User with username '{username}' already exists")

        # Check for existing email
        if email and email in users_db:
            raise ValueError(f"User with email '{email}' already exists")

        user_dict = {
            "id": len({user.get("id") for user in users_db.values() if isinstance(user, dict)}) + 1,
            "username": username,
            "email": email,
            "hashed_password": hashed_password,
            "full_name": full_name,
            "is_active": True,
            "is_superuser": is_superuser,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }

        # Store user
        users_db[username] = user_dict
        if email:
            users_db[email] = user_dict

        return user_dict

    async def get_user_by_username(username: str):
        return users_db.get(username)

    async def get_user_by_email(email: str):
        return users_db.get(email)

    async def get_user(user_id: int):
        for user in users_db.values():
            if user.id == user_id:
                return user
        return None

    async def count_users():
        # Count unique users (don't double count users stored by both username and email)
        unique_users = set()
        for user in users_db.values():
            if isinstance(user, dict) and "id" in user:
                unique_users.add(user["id"])
        return len(unique_users)

    mock_user_repo.create_user = AsyncMock(side_effect=create_user)
    mock_user_repo.get_user_by_username = AsyncMock(side_effect=get_user_by_username)
    mock_user_repo.get_by_username = AsyncMock(side_effect=get_user_by_username)
    mock_user_repo.get_by_email = AsyncMock(side_effect=get_user_by_email)
    mock_user_repo.get_user = AsyncMock(side_effect=get_user)
    mock_user_repo.get = AsyncMock(side_effect=get_user)
    mock_user_repo.count_users = AsyncMock(side_effect=count_users)

    # Mock auth repository methods
    mock_auth_repo.save_refresh_token = AsyncMock()
    mock_auth_repo.verify_refresh_token = AsyncMock(return_value=True)
    mock_auth_repo.revoke_refresh_token = AsyncMock()
    mock_auth_repo.update_user_last_login = AsyncMock()

    return mock_user_repo, mock_auth_repo, users_db


@pytest.fixture()
def client(mock_repositories) -> None:
    """Create a test client with mocked repositories."""
    # Mock the database connection manager to prevent real DB connections
    with patch("packages.webui.main.pg_connection_manager") as mock_pg_manager:
        mock_pg_manager.initialize = AsyncMock()
        mock_pg_manager.close = AsyncMock()

        # Mock the WebSocket manager as well
        with patch("packages.webui.main.ws_manager") as mock_ws_manager:
            mock_ws_manager.startup = AsyncMock()
            mock_ws_manager.shutdown = AsyncMock()

            from packages.shared.database import get_db
            from packages.webui.dependencies import get_auth_repository, get_user_repository
            from packages.webui.main import app

            mock_user_repo, mock_auth_repo, _ = mock_repositories

            # Mock database session
            mock_db = AsyncMock()

            async def override_get_db():
                yield mock_db

            # Override repository dependencies
            app.dependency_overrides[get_user_repository] = lambda: mock_user_repo
            app.dependency_overrides[get_auth_repository] = lambda: mock_auth_repo
            app.dependency_overrides[get_db] = override_get_db

            yield TestClient(app)

            # Clear overrides after test
            app.dependency_overrides.clear()


def test_user_registration_success(client) -> None:
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


def test_user_registration_duplicate(client) -> None:
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


def test_user_login_success(client) -> None:
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


def test_user_login_failure(client) -> None:
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


def test_get_me_protected(client, monkeypatch, mock_repositories) -> None:  # noqa: ARG001
    """Test that /me endpoint requires authentication."""
    # Test without token - should fail when auth is enabled
    monkeypatch.setattr("packages.webui.auth.settings.DISABLE_AUTH", False)
    response = client.get("/api/auth/me")
    assert response.status_code == 401
    assert "not authenticated" in response.json()["detail"].lower()

    # Test with invalid token - should fail when auth is enabled
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 401

    # Register and login to get valid token
    # Re-enable DISABLE_AUTH for registration/login to work with mocks
    monkeypatch.setattr("packages.webui.auth.settings.DISABLE_AUTH", True)

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

    # Test with valid token - when auth is disabled, it should use dev user
    headers = {"Authorization": f"Bearer {access_token}"}
    response = client.get("/api/auth/me", headers=headers)

    assert response.status_code == 200
    data = response.json()
    # When DISABLE_AUTH is True, it returns the dev user
    assert data["username"] == "dev_user"
    assert data["email"] == "dev@example.com"
    assert "id" in data
    assert "created_at" in data
    assert "is_active" in data
    assert data["is_active"] is True
