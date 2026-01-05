#!/usr/bin/env python3
"""
Authentication system for the Document Embedding Web UI
Provides JWT-based authentication with user management
"""

import logging
from datetime import UTC, datetime, timedelta
from uuid import uuid4
from typing import Any, cast

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel, EmailStr, field_validator

from shared.config import settings
from shared.database import create_auth_repository, create_user_repository, get_db_session, pwd_context

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


# Removed unused function: get_current_admin_user
# This function was defined for future admin functionality but is not currently used


async def get_current_user_websocket(token: str | None) -> dict[str, Any]:
    """Get current authenticated user for WebSocket connections.

    WebSocket connections can't use the standard HTTPBearer dependency,
    so they pass the token as a query parameter or in the first message.

    Args:
        token: JWT token from query parameter or WebSocket message

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
