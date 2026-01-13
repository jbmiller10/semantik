"""Tests for LLM provider config repository."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.database.exceptions import DatabaseOperationError
from shared.database.repositories.llm_provider_config_repository import LLMProviderConfigRepository
from shared.llm.types import LLMQualityTier
from shared.utils.encryption import EncryptionNotConfiguredError


class TestLLMProviderConfigRepository:
    """Tests for LLMProviderConfigRepository."""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.fixture()
    def repo(self, mock_session):
        """Create repository with mocked session."""
        return LLMProviderConfigRepository(mock_session)

    # =========================================================================
    # Provider Validation Tests
    # =========================================================================

    def test_validate_provider_valid(self, repo):
        """Valid providers pass validation."""
        repo._validate_provider("anthropic")  # No exception
        repo._validate_provider("openai")  # No exception

    def test_validate_provider_invalid(self, repo):
        """Invalid providers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid provider"):
            repo._validate_provider("unknown")

    # =========================================================================
    # Config CRUD Tests
    # =========================================================================

    async def test_get_by_user_id_found(self, repo, mock_session):
        """Returns config when found."""
        mock_config = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_config
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_user_id(123)

        assert result == mock_config

    async def test_get_by_user_id_not_found(self, repo, mock_session):
        """Returns None when not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_by_user_id(123)

        assert result is None

    async def test_get_or_create_existing(self, repo, mock_session):
        """Returns existing config without creating new one."""
        mock_config = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_config
        mock_session.execute.return_value = mock_result

        result = await repo.get_or_create(123)

        assert result == mock_config
        mock_session.add.assert_not_called()

    async def test_get_or_create_new(self, repo, mock_session):
        """Creates new config when not found."""
        # First call returns None (not found), second call for flush
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_or_create(123)

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        assert result is not None

    async def test_update_validates_temperature(self, repo):
        """Update validates temperature range."""
        with pytest.raises(ValueError, match="temperature must be between"):
            await repo.update(123, default_temperature=3.0)

        with pytest.raises(ValueError, match="temperature must be between"):
            await repo.update(123, default_temperature=-0.1)

    async def test_update_validates_provider(self, repo):
        """Update validates provider names."""
        with pytest.raises(ValueError, match="Invalid provider"):
            await repo.update(123, high_quality_provider="unknown")

    async def test_update_tier_config(self, repo):
        """update_tier_config updates correct tier."""
        with (patch.object(repo, "update", return_value=MagicMock()) as mock_update,):
            await repo.update_tier_config(
                user_id=123,
                tier=LLMQualityTier.HIGH,
                provider="anthropic",
                model="claude-opus-4-5-20251101",
            )

            mock_update.assert_called_once_with(
                123,
                high_quality_provider="anthropic",
                high_quality_model="claude-opus-4-5-20251101",
            )

    async def test_update_tier_config_low(self, repo):
        """update_tier_config updates low tier correctly."""
        with (patch.object(repo, "update", return_value=MagicMock()) as mock_update,):
            await repo.update_tier_config(
                user_id=123,
                tier=LLMQualityTier.LOW,
                provider="openai",
                model="gpt-4o-mini",
            )

            mock_update.assert_called_once_with(
                123,
                low_quality_provider="openai",
                low_quality_model="gpt-4o-mini",
            )

    async def test_delete_returns_true(self, repo, mock_session):
        """Delete returns True when config deleted."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repo.delete(123)

        assert result is True

    async def test_delete_returns_false(self, repo, mock_session):
        """Delete returns False when no config found."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await repo.delete(123)

        assert result is False

    # =========================================================================
    # API Key Tests
    # =========================================================================

    async def test_set_api_key_validates_provider(self, repo):
        """set_api_key validates provider name."""
        with pytest.raises(ValueError, match="Invalid provider"):
            await repo.set_api_key(1, "unknown", "key")

    async def test_set_api_key_encrypts(self, repo, mock_session):
        """set_api_key encrypts the key before storing."""
        with (patch("shared.database.repositories.llm_provider_config_repository.SecretEncryption") as mock_enc,):
            mock_enc.encrypt.return_value = b"encrypted"
            mock_enc.get_key_id.return_value = "key123"
            mock_session.execute.return_value = MagicMock()

            await repo.set_api_key(1, "anthropic", "sk-ant-test")

            mock_enc.encrypt.assert_called_once_with("sk-ant-test")
            mock_session.add.assert_called_once()

    async def test_set_api_key_raises_encryption_error(self, repo):
        """set_api_key raises when encryption not configured."""
        with (patch("shared.database.repositories.llm_provider_config_repository.SecretEncryption") as mock_enc,):
            mock_enc.encrypt.side_effect = EncryptionNotConfiguredError("Not configured")

            with pytest.raises(EncryptionNotConfiguredError):
                await repo.set_api_key(1, "anthropic", "key")

    async def test_get_api_key_decrypts(self, repo, mock_session):
        """get_api_key decrypts the stored key."""
        mock_key = MagicMock()
        mock_key.ciphertext = b"encrypted"
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_key
        mock_session.execute.return_value = mock_result

        with (patch("shared.database.repositories.llm_provider_config_repository.SecretEncryption") as mock_enc,):
            mock_enc.decrypt.return_value = "decrypted-key"

            result = await repo.get_api_key(1, "anthropic")

            assert result == "decrypted-key"
            mock_enc.decrypt.assert_called_once_with(b"encrypted")

    async def test_get_api_key_not_found(self, repo, mock_session):
        """get_api_key returns None when key not found."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_api_key(1, "anthropic")

        assert result is None

    async def test_has_api_key_true(self, repo, mock_session):
        """has_api_key returns True when key exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = 1  # ID exists
        mock_session.execute.return_value = mock_result

        result = await repo.has_api_key(1, "anthropic")

        assert result is True

    async def test_has_api_key_false(self, repo, mock_session):
        """has_api_key returns False when key doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.has_api_key(1, "anthropic")

        assert result is False

    async def test_delete_api_key_returns_true(self, repo, mock_session):
        """delete_api_key returns True when key deleted."""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repo.delete_api_key(1, "anthropic")

        assert result is True

    async def test_delete_api_key_returns_false(self, repo, mock_session):
        """delete_api_key returns False when no key found."""
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await repo.delete_api_key(1, "anthropic")

        assert result is False

    async def test_get_configured_providers(self, repo, mock_session):
        """get_configured_providers returns list of provider names."""
        mock_result = MagicMock()
        mock_result.all.return_value = [("anthropic",), ("openai",)]
        mock_session.execute.return_value = mock_result

        result = await repo.get_configured_providers(1)

        assert result == ["anthropic", "openai"]

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    async def test_get_by_user_id_db_error(self, repo, mock_session):
        """get_by_user_id wraps database errors."""
        mock_session.execute.side_effect = Exception("DB error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repo.get_by_user_id(123)

        assert "get" in str(exc_info.value)

    async def test_set_api_key_db_error(self, repo, mock_session):
        """set_api_key wraps database errors."""
        with (patch("shared.database.repositories.llm_provider_config_repository.SecretEncryption") as mock_enc,):
            mock_enc.encrypt.return_value = b"encrypted"
            mock_enc.get_key_id.return_value = "key123"
            mock_session.execute.side_effect = Exception("DB error")

            with pytest.raises(DatabaseOperationError) as exc_info:
                await repo.set_api_key(1, "anthropic", "key")

            assert "store" in str(exc_info.value)
