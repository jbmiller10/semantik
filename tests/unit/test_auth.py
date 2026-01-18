#!/usr/bin/env python3

"""
Comprehensive test suite for webui/auth.py
Tests JWT authentication, password hashing, and user authentication flows
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from webui.auth import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    UserCreate,
    _verify_api_key,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_current_user_websocket,
    get_password_hash,
    verify_password,
    verify_token,
)


def _make_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
        }
    )


class TestUserModels:
    """Test Pydantic model validation"""

    def test_user_create_valid(self) -> None:
        """Test creating a valid user"""
        user = UserCreate(
            username="test_user",
            email="test@example.com",
            password="password123",
            full_name="Test User",
        )
        assert user.username == "test_user"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"

    def test_user_create_short_username(self) -> None:
        """Test username validation - too short"""
        with pytest.raises(ValueError, match="Username must be at least 3 characters"):
            UserCreate(
                username="ab",
                email="test@example.com",
                password="password123",
            )

    def test_user_create_invalid_username(self) -> None:
        """Test username validation - invalid characters"""
        with pytest.raises(ValueError, match="Username must contain only alphanumeric"):
            UserCreate(
                username="test-user",
                email="test@example.com",
                password="password123",
            )

    def test_user_create_short_password(self) -> None:
        """Test password validation - too short"""
        with pytest.raises(ValueError, match="Password must be at least 8 characters"):
            UserCreate(
                username="test_user",
                email="test@example.com",
                password="pass",
            )

    def test_user_create_invalid_email(self) -> None:
        """Test email validation"""
        with pytest.raises(ValueError, match="validation error"):
            UserCreate(
                username="test_user",
                email="invalid-email",
                password="password123",
            )


class TestPasswordHashing:
    """Test password hashing functions"""

    def test_get_password_hash(self) -> None:
        """Test password hashing"""
        password = "test_password123"
        hashed = get_password_hash(password)

        # Hash should be different from original
        assert hashed != password
        # Should be a valid bcrypt hash
        assert hashed.startswith("$2b$")

    def test_verify_password_correct(self) -> None:
        """Test verifying correct password"""
        password = "test_password123"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self) -> None:
        """Test verifying incorrect password"""
        password = "test_password123"
        hashed = get_password_hash(password)

        assert verify_password("wrong_password", hashed) is False

    def test_password_hash_uniqueness(self) -> None:
        """Test that same password produces different hashes"""
        password = "test_password123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Same password should produce different hashes (due to salt)
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestTokenFunctions:
    """Test JWT token creation and verification"""

    @patch("webui.auth.settings")
    def test_create_access_token(self, mock_settings) -> None:
        """Test creating access token"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30

        data = {"sub": "test_user"}
        token = create_access_token(data)

        # Decode and verify token
        payload = jwt.decode(token, "test_secret", algorithms=["HS256"])
        assert payload["sub"] == "test_user"
        assert payload["type"] == "access"
        assert "exp" in payload

    @patch("webui.auth.settings")
    def test_create_access_token_custom_expiry(self, mock_settings) -> None:
        """Test creating access token with custom expiry"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        data = {"sub": "test_user"}
        expires_delta = timedelta(minutes=60)
        token = create_access_token(data, expires_delta)

        payload = jwt.decode(token, "test_secret", algorithms=["HS256"])
        assert payload["sub"] == "test_user"
        assert payload["type"] == "access"

    @patch("webui.auth.settings")
    def test_create_refresh_token(self, mock_settings) -> None:
        """Test creating refresh token"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        data = {"sub": "test_user"}
        token = create_refresh_token(data)

        # Decode and verify token
        payload = jwt.decode(token, "test_secret", algorithms=["HS256"])
        assert payload["sub"] == "test_user"
        assert payload["type"] == "refresh"
        assert "exp" in payload

    @patch("webui.auth.settings")
    def test_create_refresh_token_custom_expiry(self, mock_settings) -> None:
        """Test creating refresh token with custom expiry."""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        data = {"sub": "test_user"}
        expires_delta = timedelta(days=1)
        token = create_refresh_token(data, expires_delta)

        payload = jwt.decode(token, "test_secret", algorithms=["HS256"])
        assert payload["sub"] == "test_user"
        assert payload["type"] == "refresh"

    @patch("webui.auth.settings")
    def test_verify_token_valid_access(self, mock_settings) -> None:
        """Test verifying valid access token"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30

        # Create a token
        data = {"sub": "test_user"}
        token = create_access_token(data)

        # Verify it
        username = verify_token(token, "access")
        assert username == "test_user"

    @patch("webui.auth.settings")
    def test_verify_token_valid_refresh(self, mock_settings) -> None:
        """Test verifying valid refresh token"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        # Create a token
        data = {"sub": "test_user"}
        token = create_refresh_token(data)

        # Verify it
        username = verify_token(token, "refresh")
        assert username == "test_user"

    @patch("webui.auth.settings")
    def test_verify_token_wrong_type(self, mock_settings) -> None:
        """Test verifying token with wrong type"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30

        # Create access token but verify as refresh
        data = {"sub": "test_user"}
        token = create_access_token(data)

        username = verify_token(token, "refresh")
        assert username is None

    @patch("webui.auth.settings")
    def test_verify_token_expired(self, mock_settings) -> None:
        """Test verifying expired token"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        # Create expired token
        data = {"sub": "test_user"}
        token = create_access_token(data, timedelta(seconds=-1))

        username = verify_token(token, "access")
        assert username is None

    @patch("webui.auth.settings")
    def test_verify_token_invalid_signature(self, mock_settings) -> None:
        """Test verifying token with invalid signature"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        # Create token with different secret
        data = {"sub": "test_user", "type": "access", "exp": datetime.now(UTC) + timedelta(minutes=30)}
        token = jwt.encode(data, "wrong_secret", algorithm="HS256")

        username = verify_token(token, "access")
        assert username is None

    @patch("webui.auth.settings")
    def test_verify_token_missing_sub(self, mock_settings) -> None:
        """Test verifying token without sub claim"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"

        # Create token without sub
        data = {"type": "access", "exp": datetime.now(UTC) + timedelta(minutes=30)}
        token = jwt.encode(data, "test_secret", algorithm="HS256")

        username = verify_token(token, "access")
        assert username is None


class TestAuthentication:
    """Test user authentication functions"""

    @pytest.mark.asyncio()
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    @patch("webui.auth.create_auth_repository")
    async def test_authenticate_user_success(self, mock_create_auth_repo, mock_create_user_repo, mock_get_db) -> None:
        """Test successful user authentication"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_auth_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_create_auth_repo.return_value = mock_auth_repo

        # Mock user data
        hashed_password = get_password_hash("correct_password")
        mock_user_repo.get_user_by_username.return_value = {
            "id": 1,
            "username": "test_user",
            "hashed_password": hashed_password,
            "is_active": True,
        }

        # Test authentication
        result = await authenticate_user("test_user", "correct_password")

        assert result is not None
        assert result["username"] == "test_user"
        mock_auth_repo.update_user_last_login.assert_called_once_with("1")

    @pytest.mark.asyncio()
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_authenticate_user_not_found(self, mock_create_user_repo, mock_get_db) -> None:
        """Test authentication with non-existent user"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = None

        # Test authentication
        result = await authenticate_user("nonexistent", "password")
        assert result is None

    @pytest.mark.asyncio()
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_authenticate_user_wrong_password(self, mock_create_user_repo, mock_get_db) -> None:
        """Test authentication with wrong password"""
        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo

        # Mock user data
        hashed_password = get_password_hash("correct_password")
        mock_user_repo.get_user_by_username.return_value = {
            "id": 1,
            "username": "test_user",
            "hashed_password": hashed_password,
            "is_active": True,
        }

        # Test authentication
        result = await authenticate_user("test_user", "wrong_password")
        assert result is None

    @pytest.mark.asyncio()
    @patch("webui.auth.get_db_session")
    async def test_authenticate_user_db_connection_failure(self, mock_get_db) -> None:
        mock_get_db.return_value.__aiter__.return_value = []

        result = await authenticate_user("test_user", "password")
        assert result is None


class TestGetCurrentUser:
    """Test get_current_user dependency"""

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    async def test_get_current_user_auth_disabled(self, mock_settings) -> None:
        """Test get_current_user when auth is disabled"""
        mock_settings.DISABLE_AUTH = True

        result = await get_current_user(_make_request(), None)

        assert result["username"] == "dev_user"
        assert result["email"] == "dev@example.com"
        assert result["is_superuser"] is False  # Dev user should NOT be superuser for security

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    async def test_get_current_user_disallow_disable_auth_in_production(self, mock_settings) -> None:
        mock_settings.DISABLE_AUTH = True
        mock_settings.ENVIRONMENT = "production"

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), None)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "DISABLE_AUTH cannot be enabled in production" in str(exc_info.value.detail)

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    async def test_get_current_user_no_credentials(self, mock_settings) -> None:
        """Test get_current_user with no credentials"""
        mock_settings.DISABLE_AUTH = False

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Not authenticated"

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    async def test_get_current_user_invalid_token(self, mock_verify_token, mock_settings) -> None:
        """Test get_current_user with invalid token"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = None

        # Use a JWT-like token (has 2 dots) so it's routed to JWT path
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.jwt.token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid authentication credentials"

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_get_current_user_user_not_found(
        self, mock_create_user_repo, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        """Test get_current_user when user not found in database"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = None

        # Use a JWT-like token (has 2 dots) so it's routed to JWT path
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid.jwt.token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "User not found"

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_get_current_user_inactive(
        self, mock_create_user_repo, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        """Test get_current_user with inactive user"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = {
            "id": 1,
            "username": "test_user",
            "is_active": False,
        }

        # Use a JWT-like token (has 2 dots) so it's routed to JWT path
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid.jwt.token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), credentials)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert exc_info.value.detail == "Inactive user"

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_get_current_user_success(
        self, mock_create_user_repo, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        """Test successful get_current_user"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = {
            "id": 1,
            "username": "test_user",
            "email": "test@example.com",
            "is_active": True,
        }

        # Use a JWT-like token (has 2 dots) so it's routed to JWT path
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid.jwt.token")

        result = await get_current_user(_make_request(), credentials)

        assert result["username"] == "test_user"
        assert result["email"] == "test@example.com"
        assert result["is_active"] is True


class TestGetCurrentUserWebSocket:
    """Test get_current_user_websocket function"""

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    async def test_get_current_user_websocket_auth_disabled(self, mock_settings) -> None:
        """Test WebSocket auth when auth is disabled"""
        mock_settings.DISABLE_AUTH = True

        result = await get_current_user_websocket(None)

        assert result["username"] == "dev_user"
        assert result["email"] == "dev@example.com"

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    async def test_get_current_user_websocket_disallow_disable_auth_in_production(self, mock_settings) -> None:
        mock_settings.DISABLE_AUTH = True
        mock_settings.ENVIRONMENT = "production"

        with pytest.raises(ValueError, match="DISABLE_AUTH cannot be enabled in production"):
            await get_current_user_websocket(None)

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    async def test_get_current_user_websocket_no_token(self, mock_settings) -> None:
        """Test WebSocket auth with no token"""
        mock_settings.DISABLE_AUTH = False

        with pytest.raises(ValueError, match="Missing authentication token"):
            await get_current_user_websocket(None)

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    async def test_get_current_user_websocket_invalid_token(self, mock_verify_token, mock_settings) -> None:
        """Test WebSocket auth with invalid token"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = None

        with pytest.raises(ValueError, match="Invalid authentication token"):
            await get_current_user_websocket("invalid.jwt.token")

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_get_current_user_websocket_user_not_found(
        self, mock_create_user_repo, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        """Test WebSocket auth when user not found"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = None

        with pytest.raises(ValueError, match="User not found"):
            await get_current_user_websocket("valid.jwt.token")

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_get_current_user_websocket_inactive_user(
        self, mock_create_user_repo, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        """Test WebSocket auth with inactive user"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = {
            "id": 1,
            "username": "test_user",
            "is_active": False,
        }

        with pytest.raises(ValueError, match="User account is inactive"):
            await get_current_user_websocket("valid.jwt.token")

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.create_user_repository")
    async def test_get_current_user_websocket_success(
        self, mock_create_user_repo, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        """Test successful WebSocket auth"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Setup mocks
        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_user_repo = AsyncMock()
        mock_create_user_repo.return_value = mock_user_repo
        mock_user_repo.get_user_by_username.return_value = {
            "id": 1,
            "username": "test_user",
            "email": "test@example.com",
            "is_active": True,
        }

        result = await get_current_user_websocket("valid.jwt.token")

        assert result["username"] == "test_user"
        assert result["email"] == "test@example.com"
        assert result["is_active"] is True


class TestEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    async def test_get_current_user_db_connection_failure(self, mock_get_db, mock_verify_token, mock_settings) -> None:
        """Test handling database connection failure"""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"

        # Simulate database connection failure
        mock_get_db.return_value.__aiter__.return_value = []

        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid.jwt.token")

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), credentials)

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.PostgreSQLApiKeyRepository")
    async def test_get_current_user_api_key_verification_exception(
        self, mock_repo_class, mock_get_db, mock_settings
    ) -> None:
        mock_settings.DISABLE_AUTH = False

        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_repo = AsyncMock()
        mock_repo.verify_api_key.side_effect = Exception("boom")
        mock_repo_class.return_value = mock_repo

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="smtk_12345678_secret",
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(_make_request(), credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert exc_info.value.detail == "Invalid API key"

    def test_token_expiry_constants(self) -> None:
        """Test token expiry constants"""
        assert REFRESH_TOKEN_EXPIRE_DAYS == 30

    @patch("webui.auth.settings")
    def test_create_token_with_additional_claims(self, mock_settings) -> None:
        """Test creating tokens with additional claims"""
        mock_settings.JWT_SECRET_KEY = "test_secret"
        mock_settings.ALGORITHM = "HS256"
        mock_settings.ACCESS_TOKEN_EXPIRE_MINUTES = 30

        data = {"sub": "test_user", "role": "admin", "permissions": ["read", "write"]}
        token = create_access_token(data)

        payload = jwt.decode(token, "test_secret", algorithms=["HS256"])
        assert payload["sub"] == "test_user"
        assert payload["role"] == "admin"
        assert payload["permissions"] == ["read", "write"]
        assert payload["type"] == "access"


class TestApiKeyAuth:
    @pytest.mark.asyncio()
    async def test_verify_api_key_returns_none_for_empty_or_short_tokens(self) -> None:
        async def should_not_be_called():
            raise AssertionError("get_db_session should not be called for short tokens")
            if False:  # pragma: no cover
                yield

        with patch("webui.auth.get_db_session", new=should_not_be_called):
            assert await _verify_api_key("") is None
            assert await _verify_api_key("   ") is None
            assert await _verify_api_key("short-token") is None

    @pytest.mark.asyncio()
    @patch("webui.auth.get_db_session")
    async def test_verify_api_key_returns_none_when_db_yields_no_sessions(self, mock_get_db) -> None:
        mock_get_db.return_value.__aiter__.return_value = []

        token = "smtk_12345678_" + ("x" * 32)
        assert await _verify_api_key(token) is None

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.verify_token")
    @patch("webui.auth.get_db_session")
    async def test_get_current_user_websocket_db_connection_failure(
        self, mock_get_db, mock_verify_token, mock_settings
    ) -> None:
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "test_user"
        mock_get_db.return_value.__aiter__.return_value = []

        with pytest.raises(ValueError, match="Database connection failed"):
            await get_current_user_websocket("valid.jwt.token")

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.PostgreSQLApiKeyRepository")
    async def test_get_current_user_api_key_success_and_caches(
        self, mock_repo_class, mock_get_db, mock_settings
    ) -> None:
        mock_settings.DISABLE_AUTH = False

        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        api_key_data = {
            "id": "key-1",
            "name": "integration key",
            "user": {
                "id": 12,
                "username": "alice",
                "email": "alice@example.com",
                "full_name": "Alice",
                "is_active": True,
            },
        }
        mock_repo = AsyncMock()
        mock_repo.verify_api_key.return_value = api_key_data
        mock_repo_class.return_value = mock_repo

        request = _make_request()
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="smtk_12345678_" + ("x" * 32))

        first = await get_current_user(request, credentials)
        second = await get_current_user(request, credentials)

        assert first["id"] == 12
        assert first["username"] == "alice"
        assert first["_auth_method"] == "api_key"
        assert first["_api_key_id"] == "key-1"
        assert first["_api_key_name"] == "integration key"

        assert second["_api_key_id"] == "key-1"
        mock_repo.verify_api_key.assert_called_once_with(credentials.credentials, update_last_used=True)

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.PostgreSQLApiKeyRepository")
    async def test_get_current_user_websocket_api_key_success(
        self, mock_repo_class, mock_get_db, mock_settings
    ) -> None:
        mock_settings.DISABLE_AUTH = False

        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        api_key_data = {"id": "key-1", "name": "key", "user": {"id": 7, "username": "alice"}}
        mock_repo = AsyncMock()
        mock_repo.verify_api_key.return_value = api_key_data
        mock_repo_class.return_value = mock_repo

        token = "smtk_12345678_" + ("x" * 32)
        user = await get_current_user_websocket(token)

        assert user["id"] == 7
        assert user["username"] == "alice"
        assert user["_auth_method"] == "api_key"
        mock_repo.verify_api_key.assert_called_once_with(token, update_last_used=True)

    @pytest.mark.asyncio()
    @patch("webui.auth.settings")
    @patch("webui.auth.get_db_session")
    @patch("webui.auth.PostgreSQLApiKeyRepository")
    async def test_get_current_user_websocket_api_key_invalid(
        self, mock_repo_class, mock_get_db, mock_settings
    ) -> None:
        mock_settings.DISABLE_AUTH = False

        mock_session = AsyncMock()
        mock_get_db.return_value.__aiter__.return_value = [mock_session]

        mock_repo = AsyncMock()
        mock_repo.verify_api_key.return_value = None
        mock_repo_class.return_value = mock_repo

        token = "smtk_12345678_" + ("x" * 32)
        with pytest.raises(ValueError, match="Invalid API key"):
            await get_current_user_websocket(token)
