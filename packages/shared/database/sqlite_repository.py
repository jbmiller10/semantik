"""SQLite implementation of repository interfaces.

This module provides SQLite-specific implementations of the repository interfaces.
These implementations wrap the existing database functions to provide a clean
abstraction layer that can be replaced in the future.
"""

import logging
from typing import Any

from . import sqlite_implementation as db_impl
from .base import AuthRepository, UserRepository
from .exceptions import (
    DatabaseOperationError,
    InvalidUserIdError,
)
from .utils import parse_user_id

logger = logging.getLogger(__name__)


class SQLiteUserRepository(UserRepository):
    """SQLite implementation of UserRepository.

    This is a wrapper around the existing database functions,
    providing an async interface that matches the repository pattern.
    """

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def create_user(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Create a new user.

        Args:
            user_data: Dictionary containing user fields (username, email, hashed_password, full_name)

        Returns:
            The created user object including the generated ID

        Raises:
            EntityAlreadyExistsError: If username already exists
            DatabaseOperationError: For database errors
        """
        try:
            # Extract fields with defaults
            username = user_data["username"]
            email = user_data.get("email", "")
            hashed_password = user_data["hashed_password"]
            full_name = user_data.get("full_name")

            # The create_user function returns the full user object
            return self.db.create_user(username, email, hashed_password, full_name)
        except ValueError:
            # Re-raise ValueError to maintain backward compatibility with auth API
            raise
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise DatabaseOperationError("create", "user", str(e)) from e

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        return await self.get_user_by_id(user_id)

    async def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID.

        Args:
            user_id: Numeric user ID as a string (e.g., "123")

        Returns:
            User dictionary or None if not found

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            # Convert string user_id to int for SQLite
            user_id_int = parse_user_id(user_id)

            result: dict[str, Any] | None = self.db.get_user_by_id(user_id_int)
            return result
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError directly (which is also a ValueError)
            raise
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            raise DatabaseOperationError("retrieve", "user", str(e)) from e

    async def get_user_by_username(self, username: str) -> dict[str, Any] | None:
        """Get a user by username."""
        try:
            # The sqlite implementation uses get_user for username lookup
            result: dict[str, Any] | None = self.db.get_user(username)
            return result
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            raise

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """Update a user.

        TODO: Implement in sqlite_implementation.py by Q1 2025
        Currently the database layer only supports create_user and get_user.
        This method will be implemented when user profile editing is added to the UI.

        Args:
            user_id: ID of the user to update
            updates: Dictionary of fields to update

        Returns:
            The existing user object (updates not applied)

        Raises:
            NotImplementedError: When proper implementation is needed
        """
        # For now, just return the existing user without modifications
        return await self.get_user_by_id(user_id)

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user.

        Args:
            user_id: ID of the user to delete

        Returns:
            True if user was deleted, False if not found

        Raises:
            InvalidUserIdError: If user_id is not numeric
            NotImplementedError: Always (not supported in SQLite backend)
        """
        # Validate user_id format
        parse_user_id(user_id)

        # SQLite backend doesn't support user deletion
        raise NotImplementedError("User deletion is not supported in SQLite backend")

    async def verify_password(self, username: str, password: str) -> dict[str, Any] | None:
        """Verify user password and return user data if valid.

        Args:
            username: The username to check
            password: The plain text password to verify

        Returns:
            User dictionary if credentials are valid, None otherwise

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Get user by username
            user = await self.get_user_by_username(username)
            if not user:
                return None

            # Verify password using pwd_context
            if self.db.pwd_context.verify(password, user["hashed_password"]):
                return user
            return None
        except Exception as e:
            logger.error(f"Failed to verify password for {username}: {e}")
            raise DatabaseOperationError("verify", "password", str(e)) from e

    async def list_users(
        self,
        **filters: Any,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """List all users.

        Note: SQLite implementation doesn't have a list_users method.
        This is a stub that returns an empty list.

        Args:
            **filters: Filter parameters (all ignored in SQLite implementation)

        Returns:
            Empty list (not implemented in SQLite backend)
        """
        # SQLite backend doesn't support listing users
        return []

    async def update_last_login(self, user_id: str) -> None:
        """Update the last login timestamp for a user.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)

            self.db.update_user_last_login(user_id_int)
        except InvalidUserIdError:
            # Re-raise InvalidUserIdError (which is also a ValueError)
            raise
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            raise


class SQLiteAuthRepository(AuthRepository):
    """SQLite implementation of AuthRepository.

    This handles authentication tokens and related operations.
    Note: The SQLite backend uses refresh tokens instead of regular tokens.
    """

    def __init__(self) -> None:
        """Initialize with the local database implementation."""
        self.db = db_impl

    async def save_refresh_token(self, user_id: str, token_hash: str, expires_at: Any) -> None:
        """Save a refresh token for a user.

        Args:
            user_id: ID of the user
            token_hash: Hashed token
            expires_at: Expiration datetime

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)
            self.db.save_refresh_token(user_id_int, token_hash, expires_at)
        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to save refresh token: {e}")
            raise DatabaseOperationError("save", "refresh token", str(e)) from e

    async def verify_refresh_token(self, token: str) -> str | None:
        """Verify a refresh token and return user_id if valid.

        Args:
            token: The refresh token

        Returns:
            User ID as string or None if invalid

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            user_id = self.db.verify_refresh_token(token)
            return str(user_id) if user_id else None
        except Exception as e:
            logger.error(f"Failed to verify refresh token: {e}")
            raise DatabaseOperationError("verify", "refresh token", str(e)) from e

    async def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token.

        Args:
            token: The token to revoke

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            self.db.revoke_refresh_token(token)
        except Exception as e:
            logger.error(f"Failed to revoke refresh token: {e}")
            raise DatabaseOperationError("revoke", "refresh token", str(e)) from e

    async def update_user_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            user_id_int = parse_user_id(user_id)
            self.db.update_user_last_login(user_id_int)
        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            raise DatabaseOperationError("update", "last login", str(e)) from e

    async def create_token(self, user_id: str, token: str, expires_at: str) -> None:
        """Store an authentication token.

        Note: SQLite backend doesn't support this directly.
        Tokens are managed differently in the SQLite implementation.

        Args:
            user_id: ID of the user
            token: The token string
            expires_at: ISO format expiration timestamp

        Raises:
            NotImplementedError: Always (not supported in SQLite backend)
        """
        raise NotImplementedError("Token storage is not supported in SQLite backend")

    async def get_token_user_id(self, token: str) -> str | None:
        """Get the user ID associated with a token.

        Args:
            token: The token string

        Returns:
            User ID as string or None if token not found/expired

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Try to verify as refresh token
            user_id: int | None = self.db.verify_refresh_token(token)
            return str(user_id) if user_id else None
        except Exception as e:
            logger.error(f"Failed to get token user ID: {e}")
            raise DatabaseOperationError("retrieve", "token", str(e)) from e

    async def delete_token(self, token: str) -> None:
        """Delete a token (logout).

        Args:
            token: The token to delete

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Revoke refresh token
            self.db.revoke_refresh_token(token)
        except Exception as e:
            logger.error(f"Failed to delete token: {e}")
            raise DatabaseOperationError("delete", "token", str(e)) from e

    async def delete_user_tokens(self, user_id: str) -> None:
        """Delete all tokens for a user.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            NotImplementedError: Always (not supported in SQLite backend)
        """
        # Validate user_id format
        parse_user_id(user_id)

        # SQLite backend doesn't have this method
        raise NotImplementedError("Deleting all user tokens is not supported in SQLite backend")
