"""
Authentication routes for the Web UI
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel

from shared.database.base import AuthRepository, UserRepository
from webui.auth import (
    Token,
    User,
    UserCreate,
    UserLogin,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    pwd_context,
)
from webui.dependencies import get_auth_repository, get_user_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str | None = None


@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    user_repo: UserRepository = Depends(get_user_repository),
) -> User:
    """Register a new user"""
    try:
        hashed_password = get_password_hash(user_data.password)

        # Check if this is the first user in the system
        # If so, make them a superuser automatically
        user_count = await user_repo.count_users()
        is_first_user = user_count == 0

        user_dict = await user_repo.create_user(
            {
                "username": user_data.username,
                "email": user_data.email,
                "hashed_password": hashed_password,
                "full_name": user_data.full_name,
                "is_superuser": is_first_user,  # First user becomes superuser
            }
        )

        if is_first_user:
            logger.info(f"Created first user '{user_data.username}' as superuser")
        return User(**user_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed") from e


@router.post("/login", response_model=Token)
async def login(
    login_data: UserLogin,
    user_repo: UserRepository = Depends(get_user_repository),
    auth_repo: AuthRepository = Depends(get_auth_repository),
) -> Token:
    """Login and receive access token"""
    # Authenticate user
    user = await user_repo.get_user_by_username(login_data.username)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Verify password
    from webui.auth import verify_password

    if not verify_password(login_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Update last login
    await auth_repo.update_user_last_login(str(user["id"]))

    # Create tokens
    access_token = create_access_token(data={"sub": user["username"], "user_id": user.get("id")})
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Save refresh token
    expires_at = datetime.now(UTC) + timedelta(days=30)
    # Hash the token for storage
    token_hash = pwd_context.hash(refresh_token)
    await auth_repo.save_refresh_token(str(user["id"]), token_hash, expires_at)

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    payload: RefreshTokenRequest | None = Body(None),
    refresh_token: str | None = None,
    user_repo: UserRepository = Depends(get_user_repository),
    auth_repo: AuthRepository = Depends(get_auth_repository),
) -> Token:
    """Refresh access token using refresh token"""
    resolved_refresh_token = refresh_token or (payload.refresh_token if payload else None)
    if not resolved_refresh_token:
        raise HTTPException(status_code=400, detail="Missing refresh token")

    user_id = await auth_repo.verify_refresh_token(resolved_refresh_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Get user by ID
    user = await user_repo.get_user(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Create new tokens
    access_token = create_access_token(data={"sub": user["username"], "user_id": user.get("id")})
    new_refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Revoke old refresh token and save new one
    await auth_repo.revoke_refresh_token(resolved_refresh_token)
    expires_at = datetime.now(UTC) + timedelta(days=30)
    # Hash the new token for storage
    token_hash = pwd_context.hash(new_refresh_token)
    await auth_repo.save_refresh_token(user_id, token_hash, expires_at)

    return Token(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout")
async def logout(
    payload: LogoutRequest | None = Body(None),
    refresh_token: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    auth_repo: AuthRepository = Depends(get_auth_repository),
) -> dict[str, str]:
    """Logout and revoke refresh token"""
    resolved_refresh_token = refresh_token or (payload.refresh_token if payload else None)
    if resolved_refresh_token:
        await auth_repo.revoke_refresh_token(resolved_refresh_token)
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=User)
async def get_me(current_user: dict[str, Any] = Depends(get_current_user)) -> User:
    """Get current user info"""
    return User(**current_user)
