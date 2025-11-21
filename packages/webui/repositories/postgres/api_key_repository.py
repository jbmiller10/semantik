"""PostgreSQL implementation of ApiKeyRepository."""

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.database.base import ApiKeyRepository
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError, InvalidUserIdError
from shared.database.models import ApiKey, User

from .base import PostgreSQLBaseRepository

logger = logging.getLogger(__name__)


class PostgreSQLApiKeyRepository(PostgreSQLBaseRepository, ApiKeyRepository):
    """PostgreSQL implementation of ApiKeyRepository with secure key handling."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        super().__init__(session, ApiKey)

    async def create_api_key(
        self, user_id: str, name: str, permissions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a new API key for a user.

        Args:
            user_id: ID of the user creating the key
            name: Name/description for the API key
            permissions: Optional permissions dictionary

        Returns:
            Created API key data including the actual key (only returned on creation)

        Raises:
            InvalidUserIdError: If user_id is not numeric
            EntityNotFoundError: If user doesn't exist
            DatabaseOperationError: For database errors
        """
        try:
            # Validate and convert user_id
            try:
                user_id_int = int(user_id)
            except ValueError as e:
                raise InvalidUserIdError(user_id) from e

            # Verify user exists
            user_exists = await self.session.scalar(select(User.id).where(User.id == user_id_int))
            if not user_exists:
                raise EntityNotFoundError("user", user_id)

            # Generate secure API key
            api_key = secrets.token_urlsafe(32)
            key_hash = self._hash_api_key(api_key)

            # Create API key record
            api_key_id = str(uuid4())
            api_key_record = ApiKey(
                id=api_key_id,
                user_id=user_id_int,
                name=name,
                key_hash=key_hash,
                permissions=permissions or {},
                is_active=True,
            )

            self.session.add(api_key_record)
            await self.session.flush()

            # Refresh to get database-generated defaults (like created_at)
            await self.session.refresh(api_key_record)
            # Ensure relationship is available for serialization
            await self.session.refresh(api_key_record, attribute_names=["user"])

            logger.info(f"Created API key {api_key_id} for user {user_id}")

            # Return data including the actual key (only time it's available)
            result = self._api_key_to_dict(api_key_record)
            assert result is not None  # API key was just created, can't be None
            result["api_key"] = api_key  # Include actual key only on creation
            return result

        except (InvalidUserIdError, EntityNotFoundError):
            raise
        except IntegrityError as e:
            self.handle_integrity_error(e, "create_api_key")
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise DatabaseOperationError("create", "api_key", str(e)) from e

        # This should never be reached due to exceptions, but mypy needs it
        raise RuntimeError("Unexpected code path in create_api_key")

    async def get_api_key(self, api_key_id: str) -> dict[str, Any] | None:
        """Get an API key by ID.

        Args:
            api_key_id: UUID of the API key

        Returns:
            API key data or None if not found
        """
        try:
            result = await self.session.execute(
                select(ApiKey).where(ApiKey.id == api_key_id).options(selectinload(ApiKey.user))
            )
            api_key = result.scalar_one_or_none()

            return self._api_key_to_dict(api_key) if api_key else None

        except Exception as e:
            logger.error(f"Failed to get API key {api_key_id}: {e}")
            raise DatabaseOperationError("get", "api_key", str(e)) from e

    async def get_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        """Get an API key by its hash.

        Args:
            key_hash: Hash of the API key

        Returns:
            API key data or None if not found
        """
        try:
            result = await self.session.execute(
                select(ApiKey).where(ApiKey.key_hash == key_hash).options(selectinload(ApiKey.user))
            )
            api_key = result.scalar_one_or_none()

            return self._api_key_to_dict(api_key) if api_key else None

        except Exception as e:
            logger.error(f"Failed to get API key by hash: {e}")
            raise DatabaseOperationError("get", "api_key", str(e)) from e

    async def list_user_api_keys(self, user_id: str) -> list[dict[str, Any]]:
        """List all API keys for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of API key dictionaries

        Raises:
            InvalidUserIdError: If user_id is not numeric
        """
        try:
            # Validate and convert user_id
            try:
                user_id_int = int(user_id)
            except ValueError as e:
                raise InvalidUserIdError(user_id) from e

            result = await self.session.execute(
                select(ApiKey)
                .where(ApiKey.user_id == user_id_int)
                .options(selectinload(ApiKey.user))
                .order_by(ApiKey.created_at.desc())
            )
            api_keys = result.scalars().all()

            return [d for d in (self._api_key_to_dict(key) for key in api_keys) if d is not None]

        except InvalidUserIdError:
            raise
        except Exception as e:
            logger.error(f"Failed to list API keys for user {user_id}: {e}")
            raise DatabaseOperationError("list", "api_keys", str(e)) from e

    async def update_api_key(self, api_key_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update an API key.

        Args:
            api_key_id: ID of the API key to update
            updates: Dictionary of fields to update

        Returns:
            Updated API key data or None if not found
        """
        try:
            # Get the API key
            api_key = await self.session.get(ApiKey, api_key_id)
            if not api_key:
                return None

            # Update allowed fields
            allowed_fields = {"name", "permissions", "is_active", "expires_at"}
            for field, value in updates.items():
                if field in allowed_fields:
                    setattr(api_key, field, value)

            await self.session.flush()

            await self.session.refresh(api_key, attribute_names=["user"])

            logger.info(f"Updated API key {api_key_id} with fields: {list(updates.keys())}")
            return self._api_key_to_dict(api_key)

        except Exception as e:
            logger.error(f"Failed to update API key {api_key_id}: {e}")
            raise DatabaseOperationError("update", "api_key", str(e)) from e

    async def delete_api_key(self, api_key_id: str) -> bool:
        """Delete an API key.

        Args:
            api_key_id: ID of the API key to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            # Use PostgreSQL's DELETE ... RETURNING
            result = await self.session.execute(delete(ApiKey).where(ApiKey.id == api_key_id).returning(ApiKey.id))
            deleted_id = result.scalar_one_or_none()

            if deleted_id:
                logger.info(f"Deleted API key {api_key_id}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete API key {api_key_id}: {e}")
            raise DatabaseOperationError("delete", "api_key", str(e)) from e

    async def verify_api_key(self, api_key: str) -> dict[str, Any] | None:
        """Verify an API key and return associated data if valid.

        Args:
            api_key: The actual API key string

        Returns:
            API key data with user info if valid, None otherwise
        """
        try:
            # Hash the provided key
            key_hash = self._hash_api_key(api_key)

            # Look up by hash
            result = await self.session.execute(
                select(ApiKey)
                .where((ApiKey.key_hash == key_hash) & (ApiKey.is_active))
                .options(selectinload(ApiKey.user))
            )
            api_key_record = result.scalar_one_or_none()

            if not api_key_record:
                return None

            # Check expiration
            if api_key_record.expires_at and api_key_record.expires_at < datetime.now(UTC):
                logger.info(f"API key {api_key_record.id} has expired")
                return None

            # Check if user is active
            if not api_key_record.user.is_active:
                logger.info(f"User {api_key_record.user_id} is inactive")
                return None

            # Update last used timestamp (fire-and-forget)
            await self.update_last_used(api_key_record.id)

            return self._api_key_to_dict(api_key_record)

        except Exception as e:
            logger.error(f"Failed to verify API key: {e}")
            raise DatabaseOperationError("verify", "api_key", str(e)) from e

    async def update_last_used(self, api_key_id: str) -> None:
        """Update the last used timestamp for an API key.

        Args:
            api_key_id: ID of the API key
        """
        try:
            await self.session.execute(
                update(ApiKey).where(ApiKey.id == api_key_id).values(last_used_at=datetime.now(UTC))
            )
            # Don't flush here - this is often called during request processing

        except Exception as e:
            # Log but don't raise - this is a non-critical operation
            logger.warning(f"Failed to update last used for API key {api_key_id}: {e}")

    async def cleanup_expired_keys(self) -> int:
        """Delete expired API keys.

        Returns:
            Number of keys deleted
        """
        try:
            result = await self.session.execute(
                delete(ApiKey)
                .where((ApiKey.expires_at.isnot(None)) & (ApiKey.expires_at < datetime.now(UTC)))
                .returning(ApiKey.id)
            )
            deleted_ids = result.scalars().all()

            if deleted_ids:
                logger.info(f"Cleaned up {len(deleted_ids)} expired API keys")

            return len(deleted_ids)

        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            raise DatabaseOperationError("cleanup", "api_keys", str(e)) from e

    async def create_api_key_with_expiration(
        self, user_id: str, name: str, expires_in_days: int = 365, permissions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create an API key with automatic expiration.

        Args:
            user_id: ID of the user
            name: Name for the API key
            expires_in_days: Days until expiration (default: 365)
            permissions: Optional permissions

        Returns:
            Created API key data including the actual key
        """
        # Calculate expiration
        expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        # Create the key
        result = await self.create_api_key(user_id, name, permissions)

        # Update with expiration
        await self.update_api_key(result["id"], {"expires_at": expires_at})

        # Return result with expiration info
        result["expires_at"] = expires_at.isoformat()
        return result

    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key using SHA-256.

        Args:
            api_key: The API key to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _api_key_to_dict(self, api_key: ApiKey | None) -> dict[str, Any] | None:
        """Convert ApiKey model to dictionary.

        Args:
            api_key: ApiKey model instance

        Returns:
            API key dictionary or None
        """
        if not api_key:
            return None

        # Helper function to safely convert datetime to string
        def datetime_to_str(dt: Any) -> str | None:
            if dt is None:
                return None
            if hasattr(dt, "isoformat"):
                return dt.isoformat()  # type: ignore[no-any-return]
            # If it's already a string, return it
            return str(dt)

        # Fetch relationship via getattr so lazy loading makes user data available when needed.
        user_obj = getattr(api_key, "user", None)

        return {
            "id": api_key.id,
            "user_id": api_key.user_id,
            "name": api_key.name,
            "permissions": api_key.permissions,
            "is_active": api_key.is_active,
            "last_used_at": datetime_to_str(api_key.last_used_at),
            "expires_at": datetime_to_str(api_key.expires_at),
            "created_at": datetime_to_str(api_key.created_at),
            # Include user info if loaded
            "user": (
                {
                    "id": user_obj.id,
                    "username": user_obj.username,
                    "email": user_obj.email,
                    "is_active": user_obj.is_active,
                }
                if user_obj
                else None
            ),
        }
