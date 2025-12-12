"""Unit tests for ConnectorSecretRepository."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.exceptions import DatabaseOperationError
from shared.database.repositories.connector_secret_repository import ConnectorSecretRepository
from shared.utils.encryption import EncryptionNotConfiguredError


class MockSession:
    """Mock async session for testing."""

    def __init__(self):
        self.added = []
        self.executed = []
        self.flushed = False
        self._execute_results = []

    async def execute(self, stmt):
        self.executed.append(stmt)
        if self._execute_results:
            return self._execute_results.pop(0)
        return MagicMock(scalar_one_or_none=lambda: None, rowcount=0, all=lambda: [])

    async def flush(self):
        self.flushed = True

    def add(self, obj):
        self.added.append(obj)

    def set_execute_result(self, result):
        self._execute_results.append(result)


class TestConnectorSecretRepository:
    """Test cases for ConnectorSecretRepository."""

    @pytest.fixture()
    def session(self):
        return MockSession()

    @pytest.fixture()
    def repo(self, session):
        return ConnectorSecretRepository(session)

    # --- Validation tests ---

    def test_validate_secret_type_valid(self, repo):
        """Test validation passes for valid secret types."""
        for secret_type in ["password", "token", "ssh_key", "ssh_passphrase"]:
            repo._validate_secret_type(secret_type)  # Should not raise

    def test_validate_secret_type_invalid(self, repo):
        """Test validation fails for invalid secret types."""
        with pytest.raises(ValueError) as exc_info:
            repo._validate_secret_type("invalid_type")
        assert "Invalid secret_type" in str(exc_info.value)
        assert "password" in str(exc_info.value)

    # --- set_secret tests ---

    @pytest.mark.asyncio()
    async def test_set_secret_success(self, repo, session):
        """Test successful secret storage."""
        with (
            patch("shared.database.repositories.connector_secret_repository.SecretEncryption") as mock_enc,
        ):
            mock_enc.encrypt.return_value = b"encrypted_data"
            mock_enc.get_key_id.return_value = "key123"

            secret = await repo.set_secret(
                source_id=1,
                secret_type="password",
                plaintext="my-secret",
            )

            assert secret.collection_source_id == 1
            assert secret.secret_type == "password"
            assert secret.ciphertext == b"encrypted_data"
            assert secret.key_id == "key123"
            assert len(session.added) == 1
            assert session.flushed

    @pytest.mark.asyncio()
    async def test_set_secret_invalid_type(self, repo):
        """Test set_secret fails with invalid secret type."""
        with pytest.raises(ValueError):
            await repo.set_secret(
                source_id=1,
                secret_type="invalid",
                plaintext="secret",
            )

    @pytest.mark.asyncio()
    async def test_set_secret_encryption_not_configured(self, repo):
        """Test set_secret fails when encryption not configured."""
        with patch(
            "shared.database.repositories.connector_secret_repository.SecretEncryption"
        ) as mock_enc:
            mock_enc.encrypt.side_effect = EncryptionNotConfiguredError("Not configured")

            with pytest.raises(EncryptionNotConfiguredError):
                await repo.set_secret(
                    source_id=1,
                    secret_type="password",
                    plaintext="secret",
                )

    # --- get_secret tests ---

    @pytest.mark.asyncio()
    async def test_get_secret_success(self, repo, session):
        """Test successful secret retrieval."""
        mock_secret = MagicMock()
        mock_secret.ciphertext = b"encrypted_data"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_secret
        session.set_execute_result(mock_result)

        with patch(
            "shared.database.repositories.connector_secret_repository.SecretEncryption"
        ) as mock_enc:
            mock_enc.decrypt.return_value = "decrypted_secret"

            result = await repo.get_secret(source_id=1, secret_type="password")

            assert result == "decrypted_secret"
            mock_enc.decrypt.assert_called_once_with(b"encrypted_data")

    @pytest.mark.asyncio()
    async def test_get_secret_not_found(self, repo, session):
        """Test get_secret returns None when secret doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.set_execute_result(mock_result)

        result = await repo.get_secret(source_id=1, secret_type="password")

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_secret_invalid_type(self, repo):
        """Test get_secret fails with invalid secret type."""
        with pytest.raises(ValueError):
            await repo.get_secret(source_id=1, secret_type="invalid")

    # --- has_secret tests ---

    @pytest.mark.asyncio()
    async def test_has_secret_true(self, repo, session):
        """Test has_secret returns True when secret exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 1  # Secret ID
        session.set_execute_result(mock_result)

        result = await repo.has_secret(source_id=1, secret_type="password")

        assert result is True

    @pytest.mark.asyncio()
    async def test_has_secret_false(self, repo, session):
        """Test has_secret returns False when secret doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.set_execute_result(mock_result)

        result = await repo.has_secret(source_id=1, secret_type="password")

        assert result is False

    @pytest.mark.asyncio()
    async def test_has_secret_invalid_type(self, repo):
        """Test has_secret fails with invalid secret type."""
        with pytest.raises(ValueError):
            await repo.has_secret(source_id=1, secret_type="invalid")

    # --- delete_secret tests ---

    @pytest.mark.asyncio()
    async def test_delete_secret_success(self, repo, session):
        """Test successful secret deletion."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.set_execute_result(mock_result)

        result = await repo.delete_secret(source_id=1, secret_type="password")

        assert result is True

    @pytest.mark.asyncio()
    async def test_delete_secret_not_found(self, repo, session):
        """Test delete_secret returns False when secret doesn't exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.set_execute_result(mock_result)

        result = await repo.delete_secret(source_id=1, secret_type="password")

        assert result is False

    @pytest.mark.asyncio()
    async def test_delete_secret_invalid_type(self, repo):
        """Test delete_secret fails with invalid secret type."""
        with pytest.raises(ValueError):
            await repo.delete_secret(source_id=1, secret_type="invalid")

    # --- delete_for_source tests ---

    @pytest.mark.asyncio()
    async def test_delete_for_source_success(self, repo, session):
        """Test deleting all secrets for a source."""
        mock_result = MagicMock()
        mock_result.rowcount = 3
        session.set_execute_result(mock_result)

        result = await repo.delete_for_source(source_id=1)

        assert result == 3

    @pytest.mark.asyncio()
    async def test_delete_for_source_none_exist(self, repo, session):
        """Test delete_for_source when no secrets exist."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        session.set_execute_result(mock_result)

        result = await repo.delete_for_source(source_id=1)

        assert result == 0

    # --- get_secret_types_for_source tests ---

    @pytest.mark.asyncio()
    async def test_get_secret_types_for_source(self, repo, session):
        """Test retrieving secret types for a source."""
        mock_result = MagicMock()
        mock_result.all.return_value = [("password",), ("token",)]
        session.set_execute_result(mock_result)

        result = await repo.get_secret_types_for_source(source_id=1)

        assert result == ["password", "token"]

    @pytest.mark.asyncio()
    async def test_get_secret_types_for_source_empty(self, repo, session):
        """Test get_secret_types_for_source when no secrets exist."""
        mock_result = MagicMock()
        mock_result.all.return_value = []
        session.set_execute_result(mock_result)

        result = await repo.get_secret_types_for_source(source_id=1)

        assert result == []

    # --- set_secrets_batch tests ---

    @pytest.mark.asyncio()
    async def test_set_secrets_batch_success(self, repo, session):
        """Test storing multiple secrets in batch."""
        with patch(
            "shared.database.repositories.connector_secret_repository.SecretEncryption"
        ) as mock_enc:
            mock_enc.encrypt.return_value = b"encrypted"
            mock_enc.get_key_id.return_value = "key123"

            secrets = {
                "password": "secret1",
                "token": "secret2",
            }

            result = await repo.set_secrets_batch(source_id=1, secrets=secrets)

            assert len(result) == 2

    @pytest.mark.asyncio()
    async def test_set_secrets_batch_empty(self, repo):
        """Test set_secrets_batch with empty dict."""
        result = await repo.set_secrets_batch(source_id=1, secrets={})

        assert result == []

    @pytest.mark.asyncio()
    async def test_set_secrets_batch_skips_empty_values(self, repo, session):
        """Test set_secrets_batch skips empty string values."""
        with patch(
            "shared.database.repositories.connector_secret_repository.SecretEncryption"
        ) as mock_enc:
            mock_enc.encrypt.return_value = b"encrypted"
            mock_enc.get_key_id.return_value = "key123"

            secrets = {
                "password": "secret1",
                "token": "",  # Empty - should be skipped
            }

            result = await repo.set_secrets_batch(source_id=1, secrets=secrets)

            assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_set_secrets_batch_invalid_type(self, repo):
        """Test set_secrets_batch fails with invalid secret type."""
        secrets = {"invalid_type": "secret"}

        with pytest.raises(ValueError):
            await repo.set_secrets_batch(source_id=1, secrets=secrets)

    # --- Error handling tests ---

    @pytest.mark.asyncio()
    async def test_set_secret_database_error(self, repo):
        """Test set_secret wraps database errors."""
        with patch(
            "shared.database.repositories.connector_secret_repository.SecretEncryption"
        ) as mock_enc:
            mock_enc.encrypt.side_effect = Exception("DB error")

            with pytest.raises(DatabaseOperationError):
                await repo.set_secret(
                    source_id=1,
                    secret_type="password",
                    plaintext="secret",
                )

    @pytest.mark.asyncio()
    async def test_get_secret_database_error(self, repo, session):
        """Test get_secret wraps database errors."""
        # Make execute raise an exception
        async def raise_error(stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.get_secret(source_id=1, secret_type="password")

    @pytest.mark.asyncio()
    async def test_has_secret_database_error(self, repo, session):
        """Test has_secret wraps database errors."""
        async def raise_error(stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.has_secret(source_id=1, secret_type="password")

    @pytest.mark.asyncio()
    async def test_delete_secret_database_error(self, repo, session):
        """Test delete_secret wraps database errors."""
        async def raise_error(stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.delete_secret(source_id=1, secret_type="password")

    @pytest.mark.asyncio()
    async def test_delete_for_source_database_error(self, repo, session):
        """Test delete_for_source wraps database errors."""
        async def raise_error(stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.delete_for_source(source_id=1)

    @pytest.mark.asyncio()
    async def test_get_secret_types_database_error(self, repo, session):
        """Test get_secret_types_for_source wraps database errors."""
        async def raise_error(stmt):
            raise Exception("DB error")

        session.execute = raise_error

        with pytest.raises(DatabaseOperationError):
            await repo.get_secret_types_for_source(source_id=1)
