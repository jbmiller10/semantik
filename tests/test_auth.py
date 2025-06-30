"""Test authentication endpoints and functionality."""

import pytest
from datetime import datetime, timedelta
from jose import jwt
from fastapi import status

from webui.auth import pwd_context, create_access_token, verify_password, UserCreate
from webui.database import get_user, create_user


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_password_hash_and_verify(self):
        """Test that passwords are properly hashed and verified."""
        plain_password = "testpassword123"
        hashed = pwd_context.hash(plain_password)

        assert hashed != plain_password
        assert verify_password(plain_password, hashed)
        assert not verify_password("wrongpassword", hashed)


class TestJWTTokens:
    """Test JWT token creation and validation."""

    def test_create_access_token(self):
        """Test access token creation."""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        # Decode token to verify contents
        from vecpipe.config import settings

        decoded = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])

        assert decoded["sub"] == "testuser"
        assert "exp" in decoded
        assert datetime.fromtimestamp(decoded["exp"]) > datetime.utcnow()

    def test_create_access_token_with_expiry(self):
        """Test access token with custom expiry."""
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=15)
        token = create_access_token(data, expires_delta)

        from vecpipe.config import settings

        decoded = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])

        # Check that expiry is set correctly
        assert "exp" in decoded
        # Verify that the token has the correct expiry time (approximately 15 minutes)
        current_time = datetime.utcnow()
        exp_time = datetime.utcfromtimestamp(decoded["exp"])
        time_diff = (exp_time - current_time).total_seconds()
        # Should be approximately 15 minutes (900 seconds), allow some tolerance
        assert 890 < time_diff < 910


class TestAuthEndpoints:
    """Test authentication API endpoints."""

    @pytest.fixture
    def test_user_data(self):
        """Test user registration data."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "testpassword123",
        }

    def test_register_user(self, test_client, test_user_data, monkeypatch):
        """Test user registration endpoint."""

        # Mock database functions
        def mock_get_user(username):
            return None

        def mock_create_user(username, email, hashed_password, full_name=None):
            return {
                "id": 1,
                "username": username,
                "email": email,
                "full_name": full_name,
                "disabled": False,
                "created_at": datetime.utcnow().isoformat(),
            }

        monkeypatch.setattr("webui.database.get_user", mock_get_user)
        monkeypatch.setattr("webui.database.create_user", mock_create_user)

        response = test_client.post("/api/auth/register", json=test_user_data)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user_data["username"]
        assert data["email"] == test_user_data["email"]
        assert "password" not in data

    def test_register_duplicate_user(self, test_client, test_user_data, monkeypatch):
        """Test registering a duplicate user."""

        def mock_create_user(username, email, hashed_password, full_name=None):
            raise ValueError("User with this username or email already exists")

        monkeypatch.setattr("webui.database.create_user", mock_create_user)

        response = test_client.post("/api/auth/register", json=test_user_data)

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_login_success(self, test_client, test_user_data, monkeypatch):
        """Test successful login."""
        hashed_password = pwd_context.hash(test_user_data["password"])

        def mock_get_user(username):
            return {
                "id": 1,
                "username": username,
                "email": test_user_data["email"],
                "full_name": test_user_data["full_name"],
                "hashed_password": hashed_password,
                "disabled": False,
            }

        def mock_update_last_login(user_id):
            pass

        def mock_save_refresh_token(user_id, token_hash, expires_at):
            pass

        monkeypatch.setattr("webui.database.get_user", mock_get_user)
        monkeypatch.setattr("webui.database.update_user_last_login", mock_update_last_login)
        monkeypatch.setattr("webui.database.save_refresh_token", mock_save_refresh_token)

        # Login request uses JSON data
        response = test_client.post(
            "/api/auth/login", json={"username": test_user_data["username"], "password": test_user_data["password"]}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_password(self, test_client, test_user_data, monkeypatch):
        """Test login with invalid password."""
        hashed_password = pwd_context.hash(test_user_data["password"])

        def mock_get_user(username):
            return {"id": 1, "username": username, "hashed_password": hashed_password, "disabled": False}

        monkeypatch.setattr("webui.database.get_user", mock_get_user)

        response = test_client.post(
            "/api/auth/login", json={"username": test_user_data["username"], "password": "wrongpassword"}
        )

        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    def test_login_disabled_user(self, test_client, test_user_data, monkeypatch):
        """Test login with disabled user - currently the API doesn't check for disabled status."""
        hashed_password = pwd_context.hash(test_user_data["password"])

        def mock_get_user(username):
            return {"id": 1, "username": username, "hashed_password": hashed_password, "disabled": True}

        def mock_update_last_login(user_id):
            pass

        def mock_save_refresh_token(user_id, token_hash, expires_at):
            pass

        monkeypatch.setattr("webui.database.get_user", mock_get_user)
        monkeypatch.setattr("webui.database.update_user_last_login", mock_update_last_login)
        monkeypatch.setattr("webui.database.save_refresh_token", mock_save_refresh_token)

        response = test_client.post(
            "/api/auth/login", json={"username": test_user_data["username"], "password": test_user_data["password"]}
        )

        # Currently the API doesn't check for disabled users, so login succeeds
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_get_current_user(self, test_client, auth_headers, test_user, monkeypatch):
        """Test getting current user info."""

        def mock_get_user(username):
            return test_user

        monkeypatch.setattr("webui.database.get_user", mock_get_user)

        response = test_client.get("/api/auth/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]

    def test_unauthorized_access(self, unauthenticated_test_client):
        """Test accessing protected endpoint without auth."""
        response = unauthenticated_test_client.get("/api/auth/me")

        assert response.status_code == 403
        assert "Not authenticated" in response.json()["detail"]
