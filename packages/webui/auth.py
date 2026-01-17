#!/usr/bin/env python3
"""
Authentication system for the Document Embedding Web UI
Provides JWT-based authentication with user management
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import uuid4

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel, EmailStr, field_validator

from shared.config import settings
from shared.database import create_auth_repository, create_user_repository, get_db_session, pwd_context
from webui.repositories.postgres.api_key_repository import PostgreSQLApiKeyRepository

# Configure logging
logger = logging.getLogger(__name__)

# Constants
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing (use shared context to avoid duplication)

# Security
security = HTTPBearer(auto_error=False)


# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str | None = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long")
        # Check if username contains only alphanumeric characters and underscores
        if not all(c.isalnum() or c == "_" for c in v):
            raise ValueError("Username must contain only alphanumeric characters and underscores")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    id: int
    username: str
    email: str
    full_name: str | None = None
    is_active: bool = True
    created_at: str
    last_login: str | None = None


SENSITIVE_USER_FIELDS = {"hashed_password", "password", "password_hash"}


def sanitize_user_dict(user: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive fields from a user dict before returning it to clients."""
    return {key: value for key, value in user.items() if key not in SENSITIVE_USER_FIELDS}


# Password hashing functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bool(pwd_context.verify(plain_password, hashed_password))


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return str(pwd_context.hash(password))


# Token functions
def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    return str(jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM))


def create_refresh_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    # Add a unique identifier to avoid duplicate tokens within the same second.
    to_encode.update({"exp": expire, "type": "refresh", "jti": uuid4().hex, "iat": datetime.now(UTC)})
    return str(jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.ALGORITHM))


def verify_token(token: str, token_type: str = "access") -> str | None:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        token_type_claim: str = payload.get("type")

        if username is None or token_type_claim != token_type:
            return None
        return username
    except InvalidTokenError:
        return None


async def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    """Authenticate a user"""
    async for session in get_db_session():
        user_repo = create_user_repository(session)
        auth_repo = create_auth_repository(session)

        user = await user_repo.get_user_by_username(username)
        if not user:
            return None
        if not verify_password(password, user["hashed_password"]):
            return None

        # Update last login
        await auth_repo.update_user_last_login(str(user["id"]))

        # Type cast to satisfy mypy - we've verified user is not None
        return sanitize_user_dict(cast(dict[str, Any], user))
    return None


# API Key authentication helpers
def _looks_like_jwt(token: str) -> bool:
    """Check if token looks like a JWT (has exactly 2 dots).

    JWTs have format: header.payload.signature (3 parts separated by 2 dots).
    API keys use smtk_<prefix>_<secret> format (underscores, no dots).
    """
    return token.count(".") == 2


def _api_key_result_to_user_dict(api_key_data: dict[str, Any]) -> dict[str, Any]:
    """Convert verify_api_key() result to user dict format for auth dependencies.

    Args:
        api_key_data: Result from PostgreSQLApiKeyRepository.verify_api_key()

    Returns:
        User dict compatible with get_current_user() return format
    """
    user_info = api_key_data.get("user", {})
    now = datetime.now(UTC).isoformat()

    return {
        "id": user_info.get("id"),
        "username": user_info.get("username"),
        "email": user_info.get("email"),
        "full_name": user_info.get("full_name"),
        "is_active": user_info.get("is_active", True),
        "is_superuser": False,  # Security: API keys never grant superuser access
        "created_at": now,  # Not available from API key data
        "last_login": now,
        # Audit metadata for logging/tracking
        "_auth_method": "api_key",
        "_api_key_id": api_key_data.get("id"),
        "_api_key_name": api_key_data.get("name"),
    }


async def _verify_api_key_auth(token: str) -> dict[str, Any] | None:
    """Verify an API key and return user dict if valid.

    Args:
        token: API key string (expected format: smtk_<prefix>_<secret>)

    Returns:
        User dict if valid, None otherwise
    """
    # Quick format check - API keys must start with smtk_
    if not token.startswith("smtk_"):
        return None

    try:
        async for session in get_db_session():
            api_key_repo = PostgreSQLApiKeyRepository(session)
            api_key_data = await api_key_repo.verify_api_key(token)

            if api_key_data is None:
                return None

            return _api_key_result_to_user_dict(api_key_data)
    except Exception as e:
        logger.warning(f"API key verification failed: {e}")
        return None

    return None


# FastAPI dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict[str, Any]:
    """Get current authenticated user"""
    if settings.DISABLE_AUTH and (settings.ENVIRONMENT or "").lower() == "production":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DISABLE_AUTH cannot be enabled in production.",
        )
    # Check if auth is disabled for development
    if settings.DISABLE_AUTH and credentials is None:
        # Return a dummy user for development when auth is disabled and no credentials were provided
        now = datetime.now(UTC).isoformat()
        return {
            "id": 0,
            "username": "dev_user",
            "email": "dev@example.com",
            "full_name": "Development User",
            "is_active": True,
            "is_superuser": False,  # Security: dev user should not have admin privileges
            "created_at": now,
            "last_login": now,
        }

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Route based on token format: JWT (has dots) vs API key (smtk_ prefix)
    if _looks_like_jwt(token):
        # JWT authentication path
        username = verify_token(token, "access")

        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        async for session in get_db_session():
            user_repo = create_user_repository(session)
            user = await user_repo.get_user_by_username(username)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            if not user.get("is_active", True):
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

            # Type cast to satisfy mypy - we know user is not None here
            return sanitize_user_dict(cast(dict[str, Any], user))

        # This should never be reached, but satisfies the type checker
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database connection failed")

    # API key authentication path
    user = await _verify_api_key_auth(token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# Removed unused function: get_current_admin_user
# This function was defined for future admin functionality but is not currently used


async def get_current_user_websocket(token: str | None) -> dict[str, Any]:
    """Get current authenticated user for WebSocket connections.

    WebSocket connections can't use the standard HTTPBearer dependency,
    so they pass the token as a query parameter, subprotocol, or in the first message.

    Supports both JWT tokens and API keys:
    - JWT tokens have format: header.payload.signature (detected by 2 dots)
    - API keys have format: smtk_<prefix>_<secret> (detected by smtk_ prefix)

    Args:
        token: JWT token or API key from query parameter, subprotocol, or WebSocket message

    Returns:
        User dictionary if authenticated

    Raises:
        ValueError: If authentication fails
    """
    if settings.DISABLE_AUTH and (settings.ENVIRONMENT or "").lower() == "production":
        raise ValueError("DISABLE_AUTH cannot be enabled in production.")
    if not token:
        if settings.DISABLE_AUTH:
            # Return a dummy user for development when auth is disabled
            return {
                "id": 0,
                "username": "dev_user",
                "email": "dev@example.com",
                "full_name": "Development User",
                "is_active": True,
                "is_superuser": False,  # Security: dev user should not have admin privileges
                "created_at": datetime.now(UTC).isoformat(),
                "last_login": datetime.now(UTC).isoformat(),
            }
        raise ValueError("Missing authentication token")

    # Route based on token format: JWT (has dots) vs API key (smtk_ prefix)
    if _looks_like_jwt(token):
        # JWT authentication path
        username = verify_token(token, "access")
        if username is None:
            raise ValueError("Invalid authentication token")

        async for session in get_db_session():
            user_repo = create_user_repository(session)
            user = await user_repo.get_user_by_username(username)
            if user is None:
                raise ValueError("User not found")

            if not user.get("is_active", True):
                raise ValueError("User account is inactive")

            return sanitize_user_dict(cast(dict[str, Any], user))

        # This should never be reached, but satisfies the type checker
        raise ValueError("Database connection failed")

    # API key authentication path
    user = await _verify_api_key_auth(token)

    if user is None:
        raise ValueError("Invalid API key")

    return user
