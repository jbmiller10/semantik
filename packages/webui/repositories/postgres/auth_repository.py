"""PostgreSQL implementation of AuthRepository."""

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.base import AuthRepository
from shared.database.exceptions import DatabaseOperationError, InvalidUserIdError
from shared.database.models import RefreshToken, User

from .base import PostgreSQLBaseRepository

logger = logging.getLogger(__name__)


class PostgreSQLAuthRepository(PostgreSQLBaseRepository, AuthRepository):
    """PostgreSQL implementation of AuthRepository for authentication tokens."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        super().__init__(session, RefreshToken)

    async def save_refresh_token(self, user_id: str, token_hash: str, expires_at: Any) -> None:
        """Save a refresh token for a user.

        Args:
            user_id: ID of the user
            token_hash: Hashed refresh token
            expires_at: Expiration datetime

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            # Validate and convert user_id
            try:
                user_id_int = int(user_id)
            except ValueError as e:
                raise InvalidUserIdError(user_id) from e

            # Create refresh token record
            refresh_token = RefreshToken(
                user_id=user_id_int, token_hash=token_hash, expires_at=expires_at, is_revoked=False
            )

            self.session.add(refresh_token)
            await self.session.flush()

            logger.debug(f"Saved refresh token for user {user_id}")

        except InvalidUserIdError:
            raise
        except IntegrityError as e:
            # Could be foreign key violation if user doesn't exist
            logger.error(f"Integrity error saving refresh token: {e}")
            raise DatabaseOperationError("save", "refresh_token", str(e)) from e
        except Exception as e:
            logger.error(f"Failed to save refresh token: {e}")
            raise DatabaseOperationError("save", "refresh_token", str(e)) from e

    async def verify_refresh_token(self, token: str) -> str | None:
        """Verify a refresh token and return user_id if valid.

        Args:
            token: The refresh token to verify

        Returns:
            User ID as string if valid, None otherwise

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Hash the token for lookup
            token_hash = self.hash_refresh_token(token)

            # Find the token
            result = await self.session.execute(
                select(RefreshToken).where(
                    (RefreshToken.token_hash == token_hash)
                    & (~RefreshToken.is_revoked)
                    & (RefreshToken.expires_at > datetime.now(UTC))
                )
            )
            refresh_token = result.scalar_one_or_none()

            if not refresh_token:
                return None

            # Check if associated user is active
            user_active = await self.session.scalar(select(User.is_active).where(User.id == refresh_token.user_id))

            if not user_active:
                logger.info(f"User {refresh_token.user_id} is inactive")
                return None

            return str(refresh_token.user_id)

        except Exception as e:
            logger.error(f"Failed to verify refresh token: {e}")
            raise DatabaseOperationError("verify", "refresh_token", str(e)) from e

    async def revoke_refresh_token(self, token: str) -> None:
        """Revoke a refresh token.

        Args:
            token: The token to revoke

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            # Hash the token
            token_hash = self.hash_refresh_token(token)

            # Update token to revoked
            result = await self.session.execute(
                update(RefreshToken)
                .where(RefreshToken.token_hash == token_hash)
                .values(is_revoked=True)
                .returning(RefreshToken.id)
            )

            revoked_id = result.scalar_one_or_none()
            if revoked_id:
                logger.info(f"Revoked refresh token {revoked_id}")
            else:
                logger.warning("Attempted to revoke non-existent refresh token")

        except Exception as e:
            logger.error(f"Failed to revoke refresh token: {e}")
            raise DatabaseOperationError("revoke", "refresh_token", str(e)) from e

    async def update_user_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp.

        Args:
            user_id: ID of the user

        Raises:
            InvalidUserIdError: If user_id is not numeric
            DatabaseOperationError: For database errors
        """
        try:
            # Validate and convert user_id
            try:
                user_id_int = int(user_id)
            except ValueError as e:
                raise InvalidUserIdError(user_id) from e

            # Update last login
            await self.session.execute(update(User).where(User.id == user_id_int).values(last_login=datetime.now(UTC)))
            await self.session.flush()

            logger.debug(f"Updated last login for user {user_id}")

        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
            raise DatabaseOperationError("update", "last_login", str(e)) from e

    async def cleanup_expired_tokens(self) -> int:
        """Delete expired refresh tokens.

        Returns:
            Number of tokens deleted

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            result = await self.session.execute(
                delete(RefreshToken).where(RefreshToken.expires_at <= datetime.now(UTC)).returning(RefreshToken.id)
            )
            deleted_ids = result.scalars().all()

            if deleted_ids:
                logger.info(f"Cleaned up {len(deleted_ids)} expired refresh tokens")

            return len(deleted_ids)

        except Exception as e:
            logger.error(f"Failed to cleanup expired tokens: {e}")
            raise DatabaseOperationError("cleanup", "refresh_tokens", str(e)) from e

    async def get_token_user_id(self, token: str) -> str | None:
        """Get the user ID associated with a token.

        Args:
            token: The token string

        Returns:
            User ID as string or None if token not found/expired
        """
        # Use refresh token verification
        return await self.verify_refresh_token(token)

    async def delete_token(self, token: str) -> None:
        """Delete a token (logout).

        Args:
            token: The token to delete
        """
        # Use refresh token revocation
        await self.revoke_refresh_token(token)

    def _hash_token(self, token: str) -> str:
        """Hash a token using SHA-256.

        Args:
            token: The token to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(token.encode()).hexdigest()

    def hash_refresh_token(self, token: str) -> str:
        """Return the deterministic hash for refresh token storage."""
        return self._hash_token(token)
