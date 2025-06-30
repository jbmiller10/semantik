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
        
        # Check that expiry is approximately 15 minutes from now
        exp_time = datetime.fromtimestamp(decoded["exp"])
        expected_exp = datetime.utcnow() + expires_delta
        assert abs((exp_time - expected_exp).total_seconds()) < 60  # Within 1 minute


class TestAuthEndpoints:
    """Test authentication API endpoints."""
    
    @pytest.fixture
    def test_user_data(self):
        """Test user registration data."""
        return {
            "username": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "testpassword123"
        }
    
    def test_register_user(self, test_client, test_user_data, monkeypatch):
        """Test user registration endpoint."""
        # Mock database functions
        def mock_get_user(conn, username):
            return None
        
        def mock_create_user(conn, user_data):
            return {
                "username": user_data.username,
                "email": user_data.email,
                "full_name": user_data.full_name,
                "disabled": False
            }
        
        monkeypatch.setattr("webui.api.auth.get_user", mock_get_user)
        monkeypatch.setattr("webui.api.auth.create_user", mock_create_user)
        
        response = test_client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user_data["username"]
        assert data["email"] == test_user_data["email"]
        assert "password" not in data
    
    def test_register_duplicate_user(self, test_client, test_user_data, monkeypatch):
        """Test registering a duplicate user."""
        def mock_get_user(conn, username):
            return {"username": username}  # User already exists
        
        monkeypatch.setattr("webui.api.auth.get_user", mock_get_user)
        
        response = test_client.post("/auth/register", json=test_user_data)
        
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]
    
    def test_login_success(self, test_client, test_user_data, monkeypatch):
        """Test successful login."""
        hashed_password = pwd_context.hash(test_user_data["password"])
        
        def mock_get_user(conn, username):
            return {
                "username": username,
                "email": test_user_data["email"],
                "full_name": test_user_data["full_name"],
                "hashed_password": hashed_password,
                "disabled": False
            }
        
        monkeypatch.setattr("webui.api.auth.get_user", mock_get_user)
        
        # Login request uses form data
        response = test_client.post(
            "/auth/token",
            data={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
    
    def test_login_invalid_password(self, test_client, test_user_data, monkeypatch):
        """Test login with invalid password."""
        hashed_password = pwd_context.hash(test_user_data["password"])
        
        def mock_get_user(conn, username):
            return {
                "username": username,
                "hashed_password": hashed_password,
                "disabled": False
            }
        
        monkeypatch.setattr("webui.api.auth.get_user", mock_get_user)
        
        response = test_client.post(
            "/auth/token",
            data={
                "username": test_user_data["username"],
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
    
    def test_login_disabled_user(self, test_client, test_user_data, monkeypatch):
        """Test login with disabled user."""
        hashed_password = pwd_context.hash(test_user_data["password"])
        
        def mock_get_user(conn, username):
            return {
                "username": username,
                "hashed_password": hashed_password,
                "disabled": True
            }
        
        monkeypatch.setattr("webui.api.auth.get_user", mock_get_user)
        
        response = test_client.post(
            "/auth/token",
            data={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        
        assert response.status_code == 400
        assert "Inactive user" in response.json()["detail"]
    
    def test_get_current_user(self, test_client, auth_headers, test_user, monkeypatch):
        """Test getting current user info."""
        def mock_get_user(conn, username):
            return test_user
        
        monkeypatch.setattr("webui.api.auth.get_user", mock_get_user)
        
        response = test_client.get("/auth/users/me", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == test_user["username"]
        assert data["email"] == test_user["email"]
    
    def test_unauthorized_access(self, test_client):
        """Test accessing protected endpoint without auth."""
        response = test_client.get("/auth/users/me")
        
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]