"""API Key Service for managing API key business logic."""

import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.config import settings
from shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, EntityNotFoundError, ValidationError
from shared.database.models import ApiKey
from webui.api.v2.api_key_schemas import ApiKeyCreate

logger = logging.getLogger(__name__)


class ApiKeyService:
    """Service for managing API key business logic."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the API key service.

        Args:
            db_session: AsyncSession instance for database operations.
        """
        self.db_session = db_session

    async def create(self, data: ApiKeyCreate, user_id: int) -> tuple[ApiKey, str]:
        """Create a new API key.

        Args:
            data: API key creation data.
            user_id: ID of the user creating the key.

        Returns:
            Tuple of (ApiKey model, raw_key string).
            The raw_key is only available at creation time.

        Raises:
            ValidationError: If user has reached the maximum key limit.
            EntityAlreadyExistsError: If a key with the same name exists for this user.
        """
        # Check key limit
        key_count = await self._count_user_keys(user_id)
        max_keys = settings.API_KEY_MAX_PER_USER
        if key_count >= max_keys:
            raise ValidationError(
                f"Maximum number of API keys ({max_keys}) reached. "
                "Please revoke an existing key before creating a new one.",
                field="name",
            )

        # Check name uniqueness (case-insensitive)
        existing = await self._get_by_name(data.name, user_id)
        if existing:
            raise EntityAlreadyExistsError("ApiKey", data.name)

        # Generate the API key
        key_id = str(uuid4())
        raw_key = self._generate_key(key_id)
        key_hash = self._hash_key(raw_key)

        # Calculate expiration
        expires_in_days = data.expires_in_days
        if expires_in_days is None:
            expires_in_days = settings.API_KEY_DEFAULT_EXPIRY_DAYS

        # Enforce max expiry if configured
        max_expiry = settings.API_KEY_MAX_EXPIRY_DAYS
        if max_expiry > 0 and expires_in_days > max_expiry:
            expires_in_days = max_expiry

        expires_at = datetime.now(UTC) + timedelta(days=expires_in_days)

        # Create the API key record
        api_key = ApiKey(
            id=key_id,
            user_id=user_id,
            name=data.name,
            key_hash=key_hash,
            permissions=None,  # Reserved for future use
            expires_at=expires_at,
            is_active=True,
        )
        self.db_session.add(api_key)
        await self.db_session.flush()

        logger.info(
            "Created API key for user %d: key_id=%s, name=%s, expires=%s",
            user_id,
            key_id,
            data.name,
            expires_at.isoformat(),
        )

        return api_key, raw_key

    async def list_for_user(self, user_id: int) -> list[ApiKey]:
        """List all API keys for a user.

        Args:
            user_id: ID of the key owner.

        Returns:
            List of ApiKey instances ordered by creation date (newest first).
        """
        stmt = select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
        result = await self.db_session.execute(stmt)
        return list(result.scalars().all())

    async def get(self, key_id: str, user_id: int) -> ApiKey:
        """Get an API key by ID with ownership check.

        Args:
            key_id: UUID of the API key.
            user_id: ID of the expected owner.

        Returns:
            ApiKey instance.

        Raises:
            EntityNotFoundError: If key doesn't exist.
            AccessDeniedError: If user doesn't own the key.
        """
        stmt = select(ApiKey).where(ApiKey.id == key_id)
        result = await self.db_session.execute(stmt)
        api_key = result.scalar_one_or_none()

        if api_key is None:
            raise EntityNotFoundError("ApiKey", key_id)

        if api_key.user_id != user_id:
            raise AccessDeniedError(str(user_id), "ApiKey", key_id)

        return api_key

    async def update_active_status(
        self,
        key_id: str,
        user_id: int,
        is_active: bool,
    ) -> ApiKey:
        """Update the active status of an API key (soft revoke/reactivate).

        Args:
            key_id: UUID of the API key.
            user_id: ID of the key owner.
            is_active: New active status.

        Returns:
            Updated ApiKey instance.

        Raises:
            EntityNotFoundError: If key doesn't exist.
            AccessDeniedError: If user doesn't own the key.
        """
        api_key = await self.get(key_id, user_id)
        api_key.is_active = is_active
        await self.db_session.flush()

        action = "reactivated" if is_active else "revoked"
        logger.info(
            "API key %s for user %d: key_id=%s, name=%s",
            action,
            user_id,
            key_id,
            api_key.name,
        )

        return api_key

    async def _count_user_keys(self, user_id: int) -> int:
        """Count total API keys for a user."""
        stmt = select(func.count(ApiKey.id)).where(ApiKey.user_id == user_id)
        result = await self.db_session.execute(stmt)
        return result.scalar_one()

    async def _get_by_name(self, name: str, user_id: int) -> ApiKey | None:
        """Get an API key by name for a user (case-insensitive)."""
        stmt = select(ApiKey).where(
            func.lower(ApiKey.name) == func.lower(name),
            ApiKey.user_id == user_id,
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    def _generate_key(key_id: str) -> str:
        """Generate a new API key string.

        Format: smtk_<first8_of_uuid>_<secret>
        Example: smtk_550e8400_Wq3xY5pZ...

        The first 8 characters of the UUID allow identifying the key
        without exposing the full secret (useful for logging).
        """
        # Get first 8 characters of the UUID (without hyphens)
        uuid_prefix = key_id.replace("-", "")[:8]

        # Generate a secure random secret (32 bytes = 43 base64 characters)
        secret = secrets.token_urlsafe(32)

        return f"smtk_{uuid_prefix}_{secret}"

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Hash an API key for secure storage.

        Uses SHA-256 for consistent, non-reversible hashing.
        """
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
