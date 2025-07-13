"""
Authentication routes for the Web UI
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from shared.database.base import AuthRepository, UserRepository
from shared.database.factory import create_auth_repository, create_user_repository
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=User)
async def register(
    user_data: UserCreate,
    user_repo: UserRepository = Depends(create_user_repository),
) -> User:
    """Register a new user"""
    try:
        hashed_password = get_password_hash(user_data.password)
        user_dict = await user_repo.create_user(
            {
                "username": user_data.username,
                "email": user_data.email,
                "hashed_password": hashed_password,
                "full_name": user_data.full_name,
            }
        )
        return User(**user_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed") from e


@router.post("/login", response_model=Token)
async def login(
    login_data: UserLogin,
    user_repo: UserRepository = Depends(create_user_repository),
    auth_repo: AuthRepository = Depends(create_auth_repository),
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
    access_token = create_access_token(data={"sub": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Save refresh token
    expires_at = datetime.now(UTC) + timedelta(days=30)
    # Hash the token for storage
    token_hash = pwd_context.hash(refresh_token)
    await auth_repo.save_refresh_token(str(user["id"]), token_hash, expires_at)

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    user_repo: UserRepository = Depends(create_user_repository),
    auth_repo: AuthRepository = Depends(create_auth_repository),
) -> Token:
    """Refresh access token using refresh token"""
    user_id = await auth_repo.verify_refresh_token(refresh_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Get user by ID
    user = await user_repo.get_user(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Create new tokens
    access_token = create_access_token(data={"sub": user["username"]})
    new_refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Revoke old refresh token and save new one
    await auth_repo.revoke_refresh_token(refresh_token)
    expires_at = datetime.now(UTC) + timedelta(days=30)
    # Hash the new token for storage
    token_hash = pwd_context.hash(new_refresh_token)
    await auth_repo.save_refresh_token(user_id, token_hash, expires_at)

    return Token(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout")
async def logout(
    refresh_token: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
    auth_repo: AuthRepository = Depends(create_auth_repository),
) -> dict[str, str]:
    """Logout and revoke refresh token"""
    if refresh_token:
        await auth_repo.revoke_refresh_token(refresh_token)
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=User)
async def get_me(current_user: dict[str, Any] = Depends(get_current_user)) -> User:
    """Get current user info"""
    return User(**current_user)
