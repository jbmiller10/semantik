"""
Authentication routes for the Web UI
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from webui import database
from webui.auth import (
    Token,
    User,
    UserCreate,
    UserLogin,
    authenticate_user,
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    pwd_context,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=User)
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        hashed_password = get_password_hash(user_data.password)
        return database.create_user(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed") from e


@router.post("/login", response_model=Token)
async def login(login_data: UserLogin):
    """Login and receive access token"""
    user = authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # Create tokens
    access_token = create_access_token(data={"sub": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Save refresh token
    expires_at = datetime.now(UTC) + timedelta(days=30)
    # Hash the token for storage
    token_hash = pwd_context.hash(refresh_token)
    database.save_refresh_token(user["id"], token_hash, expires_at)

    return Token(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    user_id = database.verify_refresh_token(refresh_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Get user by ID
    user = database.get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    # Create new tokens
    access_token = create_access_token(data={"sub": user["username"]})
    new_refresh_token = create_refresh_token(data={"sub": user["username"]})

    # Revoke old refresh token and save new one
    database.revoke_refresh_token(refresh_token)
    expires_at = datetime.now(UTC) + timedelta(days=30)
    # Hash the new token for storage
    token_hash = pwd_context.hash(new_refresh_token)
    database.save_refresh_token(user["id"], token_hash, expires_at)

    return Token(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/logout")
async def logout(refresh_token: str = None, current_user: dict[str, Any] = Depends(get_current_user)):  # noqa: ARG001
    """Logout and revoke refresh token"""
    if refresh_token:
        database.revoke_refresh_token(refresh_token)
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=User)
async def get_me(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get current user info"""
    return User(**current_user)
