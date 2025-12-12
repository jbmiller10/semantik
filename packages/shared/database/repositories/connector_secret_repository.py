"""Repository implementation for ConnectorSecret model."""

import logging

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.db_retry import with_db_retry
from shared.database.exceptions import DatabaseOperationError
from shared.database.models import ConnectorSecret
from shared.utils.encryption import (
    DecryptionError,
    EncryptionNotConfiguredError,
    SecretEncryption,
)

logger = logging.getLogger(__name__)


class ConnectorSecretRepository:
    """Repository for ConnectorSecret model operations.

    This repository manages encrypted connector secrets (passwords, tokens,
    SSH keys) for collection sources. All secrets are encrypted at rest
    using Fernet symmetric encryption.

    IMPORTANT: SecretEncryption.initialize() must be called before using
    this repository. If encryption is not configured, operations that
    require encryption will raise EncryptionNotConfiguredError.

    Example:
        ```python
        secret_repo = ConnectorSecretRepository(session)

        # Store a secret (encrypts automatically)
        await secret_repo.set_secret(
            source_id=123,
            secret_type="password",
            plaintext="my-secret-password",
        )

        # Retrieve a secret (decrypts automatically)
        password = await secret_repo.get_secret(source_id=123, secret_type="password")

        # Check if secret exists (without decrypting)
        has_password = await secret_repo.has_secret(source_id=123, secret_type="password")
        ```
    """

    # Valid secret types
    VALID_SECRET_TYPES = frozenset({"password", "token", "ssh_key", "ssh_passphrase"})

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    def _validate_secret_type(self, secret_type: str) -> None:
        """Validate that secret_type is allowed.

        Args:
            secret_type: The secret type to validate

        Raises:
            ValueError: If secret_type is not valid
        """
        if secret_type not in self.VALID_SECRET_TYPES:
            raise ValueError(
                f"Invalid secret_type '{secret_type}'. " f"Must be one of: {', '.join(sorted(self.VALID_SECRET_TYPES))}"
            )

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def set_secret(
        self,
        source_id: int,
        secret_type: str,
        plaintext: str,
    ) -> ConnectorSecret:
        """Store or update an encrypted secret for a source.

        If a secret of the same type already exists for the source,
        it is replaced with the new encrypted value.

        Args:
            source_id: ID of the collection source
            secret_type: Type of secret ('password', 'token', 'ssh_key', 'ssh_passphrase')
            plaintext: The secret value to encrypt and store

        Returns:
            Created/updated ConnectorSecret instance

        Raises:
            ValueError: If secret_type is invalid
            EncryptionNotConfiguredError: If encryption key is not configured
            DatabaseOperationError: For database errors
        """
        self._validate_secret_type(secret_type)

        try:
            # Encrypt the secret
            ciphertext = SecretEncryption.encrypt(plaintext)
            key_id = SecretEncryption.get_key_id()

            # Delete existing secret with same type (upsert pattern)
            await self.session.execute(
                delete(ConnectorSecret).where(
                    ConnectorSecret.collection_source_id == source_id,
                    ConnectorSecret.secret_type == secret_type,
                )
            )

            # Create new secret
            secret = ConnectorSecret(
                collection_source_id=source_id,
                secret_type=secret_type,
                ciphertext=ciphertext,
                key_id=key_id,
            )

            self.session.add(secret)
            await self.session.flush()

            logger.debug(f"Stored encrypted secret type={secret_type} for source_id={source_id} " f"(key_id={key_id})")

            return secret

        except (ValueError, EncryptionNotConfiguredError):
            raise
        except Exception as e:
            logger.error(f"Failed to store secret for source {source_id}: {e}")
            raise DatabaseOperationError(f"Failed to store secret: {e}") from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_secret(
        self,
        source_id: int,
        secret_type: str,
    ) -> str | None:
        """Retrieve and decrypt a secret for a source.

        Args:
            source_id: ID of the collection source
            secret_type: Type of secret to retrieve

        Returns:
            Decrypted secret string, or None if not found

        Raises:
            ValueError: If secret_type is invalid
            EncryptionNotConfiguredError: If encryption key is not configured
            DecryptionError: If decryption fails (key may have changed)
            DatabaseOperationError: For database errors
        """
        self._validate_secret_type(secret_type)

        try:
            result = await self.session.execute(
                select(ConnectorSecret).where(
                    ConnectorSecret.collection_source_id == source_id,
                    ConnectorSecret.secret_type == secret_type,
                )
            )
            secret = result.scalar_one_or_none()

            if secret is None:
                return None

            # Decrypt and return
            plaintext = SecretEncryption.decrypt(secret.ciphertext)

            logger.debug(f"Retrieved secret type={secret_type} for source_id={source_id}")

            return plaintext

        except (ValueError, EncryptionNotConfiguredError, DecryptionError):
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve secret for source {source_id}: {e}")
            raise DatabaseOperationError(f"Failed to retrieve secret: {e}") from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def has_secret(
        self,
        source_id: int,
        secret_type: str,
    ) -> bool:
        """Check if a secret exists without decrypting it.

        This is a lightweight check that doesn't require the encryption
        key to be configured.

        Args:
            source_id: ID of the collection source
            secret_type: Type of secret to check

        Returns:
            True if secret exists, False otherwise

        Raises:
            ValueError: If secret_type is invalid
            DatabaseOperationError: For database errors
        """
        self._validate_secret_type(secret_type)

        try:
            result = await self.session.execute(
                select(ConnectorSecret.id).where(
                    ConnectorSecret.collection_source_id == source_id,
                    ConnectorSecret.secret_type == secret_type,
                )
            )
            return result.scalar_one_or_none() is not None

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to check secret existence for source {source_id}: {e}")
            raise DatabaseOperationError(f"Failed to check secret: {e}") from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_secret(
        self,
        source_id: int,
        secret_type: str,
    ) -> bool:
        """Delete a specific secret for a source.

        Args:
            source_id: ID of the collection source
            secret_type: Type of secret to delete

        Returns:
            True if a secret was deleted, False if it didn't exist

        Raises:
            ValueError: If secret_type is invalid
            DatabaseOperationError: For database errors
        """
        self._validate_secret_type(secret_type)

        try:
            result = await self.session.execute(
                delete(ConnectorSecret).where(
                    ConnectorSecret.collection_source_id == source_id,
                    ConnectorSecret.secret_type == secret_type,
                )
            )
            deleted = (result.rowcount or 0) > 0

            if deleted:
                logger.debug(f"Deleted secret type={secret_type} for source_id={source_id}")

            return deleted

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete secret for source {source_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete secret: {e}") from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def delete_for_source(self, source_id: int) -> int:
        """Delete all secrets for a source.

        This is typically called when a source is being deleted.
        Note: With CASCADE delete on the foreign key, this may not be
        necessary if the source itself is deleted, but it's useful
        for explicit cleanup.

        Args:
            source_id: ID of the collection source

        Returns:
            Number of secrets deleted
        """
        try:
            result = await self.session.execute(
                delete(ConnectorSecret).where(ConnectorSecret.collection_source_id == source_id)
            )
            count = result.rowcount or 0

            if count > 0:
                logger.debug(f"Deleted {count} secrets for source_id={source_id}")

            return count

        except Exception as e:
            logger.error(f"Failed to delete secrets for source {source_id}: {e}")
            raise DatabaseOperationError(f"Failed to delete secrets: {e}") from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def get_secret_types_for_source(self, source_id: int) -> list[str]:
        """Get list of secret types stored for a source.

        This is useful for building API responses that indicate which
        secrets are configured without revealing the values.

        Args:
            source_id: ID of the collection source

        Returns:
            List of secret type strings
        """
        try:
            result = await self.session.execute(
                select(ConnectorSecret.secret_type).where(ConnectorSecret.collection_source_id == source_id)
            )
            return [row[0] for row in result.all()]

        except Exception as e:
            logger.error(f"Failed to get secret types for source {source_id}: {e}")
            raise DatabaseOperationError(f"Failed to get secret types: {e}") from e

    @with_db_retry(retries=3, delay=0.3, backoff=2.0, max_delay=5.0)
    async def set_secrets_batch(
        self,
        source_id: int,
        secrets: dict[str, str],
    ) -> list[ConnectorSecret]:
        """Store multiple secrets for a source in a single operation.

        Args:
            source_id: ID of the collection source
            secrets: Dict mapping secret_type to plaintext value

        Returns:
            List of created ConnectorSecret instances

        Raises:
            ValueError: If any secret_type is invalid
            EncryptionNotConfiguredError: If encryption key is not configured
            DatabaseOperationError: For database errors
        """
        if not secrets:
            return []

        # Validate all secret types first
        for secret_type in secrets:
            self._validate_secret_type(secret_type)

        created = []
        for secret_type, plaintext in secrets.items():
            if plaintext:  # Only store non-empty secrets
                secret = await self.set_secret(source_id, secret_type, plaintext)
                created.append(secret)

        return created
