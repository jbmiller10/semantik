"""Fernet-based encryption utilities for connector secrets.

This module provides secure encryption/decryption of sensitive connector
credentials (passwords, tokens, SSH keys) using Fernet symmetric encryption.

Fernet uses:
- AES-128-CBC for encryption
- HMAC-SHA256 for authentication
- Base64 encoding for the ciphertext

Key Management:
- Key is provided via CONNECTOR_SECRETS_KEY environment variable
- Key ID (SHA-256 fingerprint) is stored with ciphertext for rotation support
- Key must be a valid Fernet key (32 bytes, base64-encoded = 44 chars)

Usage:
    from shared.utils.encryption import SecretEncryption

    # Initialize (typically at app startup)
    SecretEncryption.initialize("your-fernet-key")

    # Encrypt
    ciphertext = SecretEncryption.encrypt("my-secret-password")

    # Decrypt
    plaintext = SecretEncryption.decrypt(ciphertext)

    # Check if configured
    if SecretEncryption.is_configured():
        ...
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class SecretEncryptionError(Exception):
    """Base exception for encryption errors."""


class EncryptionNotConfiguredError(SecretEncryptionError):
    """Raised when encryption is used but not configured."""


class DecryptionError(SecretEncryptionError):
    """Raised when decryption fails."""


class SecretEncryption:
    """Singleton class for Fernet-based secret encryption.

    This class provides static methods for encrypting and decrypting
    sensitive connector credentials. It must be initialized with a
    Fernet key before use.

    The key_id is a truncated SHA-256 hash of the key, stored alongside
    ciphertext to support key rotation scenarios.
    """

    _fernet: ClassVar[Fernet | None] = None
    _key_id: ClassVar[str | None] = None
    _initialized: ClassVar[bool] = False

    @classmethod
    def initialize(cls, key: str | None) -> bool:
        """Initialize encryption with a Fernet key.

        Args:
            key: Fernet key string (44 chars, base64-encoded 32 bytes)
                 or None to disable encryption

        Returns:
            True if encryption is now enabled, False if disabled

        Raises:
            ValueError: If key is invalid
        """
        if not key or not key.strip():
            cls._fernet = None
            cls._key_id = None
            cls._initialized = True
            logger.warning("SecretEncryption initialized without key - encryption disabled")
            return False

        try:
            from cryptography.fernet import Fernet

            # Validate key format
            key = key.strip()
            cls._fernet = Fernet(key.encode())

            # Generate key ID (first 16 chars of SHA-256)
            cls._key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            cls._initialized = True

            logger.info(f"SecretEncryption initialized with key_id={cls._key_id}")
            return True

        except Exception as e:
            cls._fernet = None
            cls._key_id = None
            cls._initialized = True
            raise ValueError(f"Invalid Fernet key: {e}") from e

    @classmethod
    def is_configured(cls) -> bool:
        """Check if encryption is configured and enabled.

        Returns:
            True if encryption is ready to use
        """
        return cls._fernet is not None

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if initialize() has been called.

        Returns:
            True if initialize() was called (even if encryption is disabled)
        """
        return cls._initialized

    @classmethod
    def get_key_id(cls) -> str:
        """Get the current key ID.

        Returns:
            Key ID string (16 hex chars) or "none" if not configured
        """
        return cls._key_id or "none"

    @classmethod
    def encrypt(cls, plaintext: str) -> bytes:
        """Encrypt a secret value.

        Args:
            plaintext: The secret string to encrypt

        Returns:
            Encrypted ciphertext as bytes

        Raises:
            EncryptionNotConfiguredError: If encryption is not configured
            SecretEncryptionError: If encryption fails
        """
        if cls._fernet is None:
            raise EncryptionNotConfiguredError(
                "Encryption not configured - set CONNECTOR_SECRETS_KEY environment variable"
            )

        try:
            return cls._fernet.encrypt(plaintext.encode("utf-8"))
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecretEncryptionError(f"Encryption failed: {e}") from e

    @classmethod
    def decrypt(cls, ciphertext: bytes) -> str:
        """Decrypt a secret value.

        Args:
            ciphertext: The encrypted bytes to decrypt

        Returns:
            Decrypted plaintext string

        Raises:
            EncryptionNotConfiguredError: If encryption is not configured
            DecryptionError: If decryption fails (wrong key, corrupted data)
        """
        if cls._fernet is None:
            raise EncryptionNotConfiguredError(
                "Encryption not configured - set CONNECTOR_SECRETS_KEY environment variable"
            )

        try:
            from cryptography.fernet import InvalidToken

            return cls._fernet.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as e:
            logger.error("Decryption failed - invalid token (key may have changed)")
            raise DecryptionError("Failed to decrypt secret - encryption key may have changed") from e
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DecryptionError(f"Decryption failed: {e}") from e

    @classmethod
    def reset(cls) -> None:
        """Reset encryption state (primarily for testing).

        This clears the key and marks the class as uninitialized.
        """
        cls._fernet = None
        cls._key_id = None
        cls._initialized = False


def generate_fernet_key() -> str:
    """Generate a new Fernet key.

    Returns:
        A new Fernet key string (44 chars, base64-encoded)
    """
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode("utf-8")


# Convenience functions for encrypting/decrypting secrets as base64 strings.
# These are useful when secrets need to be stored in JSON fields (like inline_source_config).

import base64


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a secret value and return as base64 string.

    This convenience function encrypts a secret and encodes the ciphertext
    as a base64 string, suitable for storage in JSON fields.

    Args:
        plaintext: The secret string to encrypt

    Returns:
        Base64-encoded ciphertext string

    Raises:
        EncryptionNotConfiguredError: If encryption is not configured
        SecretEncryptionError: If encryption fails
    """
    ciphertext = SecretEncryption.encrypt(plaintext)
    return base64.b64encode(ciphertext).decode("utf-8")


def decrypt_secret(encoded_ciphertext: str) -> str:
    """Decrypt a base64-encoded secret value.

    This convenience function decodes a base64 ciphertext and decrypts it,
    suitable for retrieving secrets stored in JSON fields.

    Args:
        encoded_ciphertext: Base64-encoded ciphertext string

    Returns:
        Decrypted plaintext string

    Raises:
        EncryptionNotConfiguredError: If encryption is not configured
        DecryptionError: If decryption fails (wrong key, corrupted data)
    """
    ciphertext = base64.b64decode(encoded_ciphertext.encode("utf-8"))
    return SecretEncryption.decrypt(ciphertext)
