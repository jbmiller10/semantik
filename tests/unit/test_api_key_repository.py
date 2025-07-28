#!/usr/bin/env python3
"""
Comprehensive test suite for webui/repositories/postgres/api_key_repository.py
Tests API key repository with mocked database operations
"""

import hashlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError, InvalidUserIdError
from shared.database.models import ApiKey, User
from sqlalchemy.exc import IntegrityError
from webui.repositories.postgres.api_key_repository import PostgreSQLApiKeyRepository


class TestPostgreSQLApiKeyRepository:
    """Test PostgreSQL API key repository implementation"""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.add = Mock()
        session.flush = AsyncMock()
        session.refresh = AsyncMock()
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        session.get = AsyncMock()
        return session

    @pytest.fixture()
    def api_key_repo(self, mock_session):
        """Create API key repository with mocked session"""
        return PostgreSQLApiKeyRepository(mock_session)

    def test_initialization(self, mock_session):
        """Test repository initialization"""
        repo = PostgreSQLApiKeyRepository(mock_session)
        assert repo.session == mock_session
        assert repo.model_class == ApiKey

    @pytest.mark.asyncio()
    @patch("secrets.token_urlsafe")
    @patch("uuid.uuid4")
    async def test_create_api_key_success(self, mock_uuid, mock_token, api_key_repo, mock_session):
        """Test successful API key creation"""
        # Setup mocks
        mock_uuid.return_value = "test-uuid"
        mock_token.return_value = "test_api_key_token"
        mock_session.scalar.return_value = 123  # User exists

        # Mock the created API key
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test Key"
        mock_api_key.key_hash = hashlib.sha256(b"test_api_key_token").hexdigest()
        mock_api_key.permissions = {}
        mock_api_key.is_active = True
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None
        mock_api_key.expires_at = None

        # Create API key
        result = await api_key_repo.create_api_key("123", "Test Key", {"read": True})

        # Verify operations
        mock_session.scalar.assert_called_once()  # Check user exists
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

        # Check the API key object
        api_key_obj = mock_session.add.call_args[0][0]
        assert isinstance(api_key_obj, ApiKey)
        assert api_key_obj.user_id == 123
        assert api_key_obj.name == "Test Key"
        assert api_key_obj.permissions == {"read": True}
        assert api_key_obj.is_active is True

        # Result should include the actual API key
        assert "api_key" in result
        assert result["api_key"] == "test_api_key_token"

    @pytest.mark.asyncio()
    async def test_create_api_key_invalid_user_id(self, api_key_repo):
        """Test creating API key with invalid user ID"""
        with pytest.raises(InvalidUserIdError):
            await api_key_repo.create_api_key("invalid", "Test Key")

    @pytest.mark.asyncio()
    async def test_create_api_key_user_not_found(self, api_key_repo, mock_session):
        """Test creating API key for non-existent user"""
        mock_session.scalar.return_value = None  # User doesn't exist

        with pytest.raises(EntityNotFoundError) as exc_info:
            await api_key_repo.create_api_key("123", "Test Key")

        assert exc_info.value.entity_type == "user"
        assert exc_info.value.entity_id == "123"

    @pytest.mark.asyncio()
    async def test_create_api_key_integrity_error(self, api_key_repo, mock_session):
        """Test creating API key with database integrity error"""
        mock_session.scalar.return_value = 123  # User exists
        mock_session.flush.side_effect = IntegrityError("statement", "params", "orig")

        with pytest.raises(DatabaseOperationError):
            await api_key_repo.create_api_key("123", "Test Key")

    @pytest.mark.asyncio()
    async def test_get_api_key_success(self, api_key_repo, mock_session):
        """Test getting API key by ID"""
        # Mock API key with user
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.is_active = True

        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test Key"
        mock_api_key.permissions = {"read": True}
        mock_api_key.is_active = True
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None
        mock_api_key.expires_at = None
        mock_api_key.user = mock_user

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.get_api_key("test-uuid")

        assert result is not None
        assert result["id"] == "test-uuid"
        assert result["name"] == "Test Key"
        assert result["user"]["username"] == "testuser"

    @pytest.mark.asyncio()
    async def test_get_api_key_not_found(self, api_key_repo, mock_session):
        """Test getting non-existent API key"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.get_api_key("non-existent")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_api_key_by_hash_success(self, api_key_repo, mock_session):
        """Test getting API key by hash"""
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test Key"
        mock_api_key.permissions = {}
        mock_api_key.is_active = True
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None
        mock_api_key.expires_at = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        mock_session.execute.return_value = mock_result

        key_hash = "test_hash"
        result = await api_key_repo.get_api_key_by_hash(key_hash)

        assert result is not None
        assert result["id"] == "test-uuid"

    @pytest.mark.asyncio()
    async def test_list_user_api_keys_success(self, api_key_repo, mock_session):
        """Test listing user's API keys"""
        # Mock multiple API keys
        mock_keys = []
        for i in range(3):
            key = Mock(spec=ApiKey)
            key.id = f"key-{i}"
            key.user_id = 123
            key.name = f"Key {i}"
            key.permissions = {}
            key.is_active = True
            key.created_at = datetime.now(UTC) - timedelta(days=i)
            key.last_used_at = None
            key.expires_at = None
            mock_keys.append(key)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_keys
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.list_user_api_keys("123")

        assert len(result) == 3
        assert result[0]["id"] == "key-0"
        assert result[1]["id"] == "key-1"
        assert result[2]["id"] == "key-2"

    @pytest.mark.asyncio()
    async def test_list_user_api_keys_invalid_user_id(self, api_key_repo):
        """Test listing API keys with invalid user ID"""
        with pytest.raises(InvalidUserIdError):
            await api_key_repo.list_user_api_keys("invalid")

    @pytest.mark.asyncio()
    async def test_update_api_key_success(self, api_key_repo, mock_session):
        """Test updating API key"""
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.name = "Old Name"
        mock_api_key.permissions = {}
        mock_api_key.is_active = True
        mock_api_key.expires_at = None
        mock_api_key.user_id = 123
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None

        mock_session.get.return_value = mock_api_key

        updates = {"name": "New Name", "permissions": {"read": True, "write": False}, "is_active": False}

        result = await api_key_repo.update_api_key("test-uuid", updates)

        # Verify updates were applied
        assert mock_api_key.name == "New Name"
        assert mock_api_key.permissions == {"read": True, "write": False}
        assert mock_api_key.is_active is False
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_api_key_not_found(self, api_key_repo, mock_session):
        """Test updating non-existent API key"""
        mock_session.get.return_value = None

        result = await api_key_repo.update_api_key("non-existent", {"name": "New"})
        assert result is None

    @pytest.mark.asyncio()
    async def test_update_api_key_ignores_invalid_fields(self, api_key_repo, mock_session):
        """Test that update ignores non-allowed fields"""
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123  # Should not be changeable
        mock_api_key.key_hash = "original_hash"  # Should not be changeable
        mock_api_key.name = "Test"
        mock_api_key.permissions = {}
        mock_api_key.is_active = True
        mock_api_key.expires_at = None
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None

        mock_session.get.return_value = mock_api_key

        # Try to update non-allowed fields
        updates = {
            "name": "New Name",  # Allowed
            "user_id": 456,  # Not allowed
            "key_hash": "new_hash",  # Not allowed
            "created_at": datetime.now(UTC),  # Not allowed
        }

        await api_key_repo.update_api_key("test-uuid", updates)

        # Only name should be updated
        assert mock_api_key.name == "New Name"
        assert mock_api_key.user_id == 123  # Unchanged
        assert mock_api_key.key_hash == "original_hash"  # Unchanged

    @pytest.mark.asyncio()
    async def test_delete_api_key_success(self, api_key_repo, mock_session):
        """Test successful API key deletion"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = "test-uuid"
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.delete_api_key("test-uuid")

        assert result is True
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_api_key_not_found(self, api_key_repo, mock_session):
        """Test deleting non-existent API key"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.delete_api_key("non-existent")

        assert result is False

    @pytest.mark.asyncio()
    async def test_verify_api_key_success(self, api_key_repo, mock_session):
        """Test successful API key verification"""
        api_key = "test_api_key_token"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Mock active user
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.is_active = True

        # Mock API key
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test Key"
        mock_api_key.key_hash = key_hash
        mock_api_key.permissions = {"read": True}
        mock_api_key.is_active = True
        mock_api_key.expires_at = None
        mock_api_key.user = mock_user
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        mock_session.execute.return_value = mock_result

        # Mock update_last_used
        with patch.object(api_key_repo, "update_last_used", new_callable=AsyncMock):
            result = await api_key_repo.verify_api_key(api_key)

        assert result is not None
        assert result["id"] == "test-uuid"
        assert result["user"]["username"] == "testuser"

    @pytest.mark.asyncio()
    async def test_verify_api_key_not_found(self, api_key_repo, mock_session):
        """Test verifying non-existent API key"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.verify_api_key("invalid_key")
        assert result is None

    @pytest.mark.asyncio()
    async def test_verify_api_key_expired(self, api_key_repo, mock_session):
        """Test verifying expired API key"""
        api_key = "test_api_key_token"

        # Mock expired API key
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.expires_at = datetime.now(UTC) - timedelta(days=1)  # Expired

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.verify_api_key(api_key)
        assert result is None

    @pytest.mark.asyncio()
    async def test_verify_api_key_inactive_user(self, api_key_repo, mock_session):
        """Test verifying API key for inactive user"""
        # Mock inactive user
        mock_user = Mock(spec=User)
        mock_user.is_active = False

        # Mock API key
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.expires_at = None
        mock_api_key.user = mock_user

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        mock_session.execute.return_value = mock_result

        result = await api_key_repo.verify_api_key("test_key")
        assert result is None

    @pytest.mark.asyncio()
    async def test_update_last_used(self, api_key_repo, mock_session):
        """Test updating last used timestamp"""
        await api_key_repo.update_last_used("test-uuid")

        mock_session.execute.assert_called_once()
        # Should not flush (as per implementation)

    @pytest.mark.asyncio()
    async def test_update_last_used_error_handled(self, api_key_repo, mock_session):
        """Test update_last_used handles errors gracefully"""
        mock_session.execute.side_effect = Exception("Database error")

        # Should not raise, just log warning
        await api_key_repo.update_last_used("test-uuid")

    @pytest.mark.asyncio()
    async def test_cleanup_expired_keys_success(self, api_key_repo, mock_session):
        """Test successful cleanup of expired keys"""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = ["key1", "key2", "key3"]
        mock_session.execute.return_value = mock_result

        count = await api_key_repo.cleanup_expired_keys()

        assert count == 3
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cleanup_expired_keys_none_found(self, api_key_repo, mock_session):
        """Test cleanup when no expired keys exist"""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        count = await api_key_repo.cleanup_expired_keys()

        assert count == 0

    @pytest.mark.asyncio()
    @patch("secrets.token_urlsafe")
    async def test_create_api_key_with_expiration(self, mock_token, api_key_repo, mock_session):
        """Test creating API key with automatic expiration"""
        mock_token.return_value = "test_api_key_token"
        mock_session.scalar.return_value = 123  # User exists

        # Mock create_api_key
        with patch.object(api_key_repo, "create_api_key", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = {
                "id": "test-uuid",
                "user_id": 123,
                "name": "Test Key",
                "api_key": "test_api_key_token",
            }

            # Mock update_api_key
            with patch.object(api_key_repo, "update_api_key", new_callable=AsyncMock):
                result = await api_key_repo.create_api_key_with_expiration("123", "Test Key", expires_in_days=30)

        assert "expires_at" in result
        assert result["api_key"] == "test_api_key_token"

    def test_hash_api_key(self, api_key_repo):
        """Test API key hashing"""
        api_key = "test_api_key_token"
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()

        result = api_key_repo._hash_api_key(api_key)

        assert result == expected_hash
        assert len(result) == 64  # SHA-256 produces 64 hex characters

    def test_api_key_to_dict_none(self, api_key_repo):
        """Test converting None to dict"""
        result = api_key_repo._api_key_to_dict(None)
        assert result is None

    def test_api_key_to_dict_with_user(self, api_key_repo):
        """Test converting API key with user to dict"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.is_active = True

        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test Key"
        mock_api_key.permissions = {"read": True}
        mock_api_key.is_active = True
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None
        mock_api_key.expires_at = None
        mock_api_key.user = mock_user

        result = api_key_repo._api_key_to_dict(mock_api_key)

        assert result["id"] == "test-uuid"
        assert result["name"] == "Test Key"
        assert result["user"]["username"] == "testuser"

    def test_api_key_to_dict_without_user(self, api_key_repo):
        """Test converting API key without user to dict"""
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test Key"
        mock_api_key.permissions = {}
        mock_api_key.is_active = True
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None
        mock_api_key.expires_at = None

        # Simulate user not loaded
        delattr(mock_api_key, "user")

        result = api_key_repo._api_key_to_dict(mock_api_key)

        assert result["id"] == "test-uuid"
        assert result["user"] is None


class TestApiKeyRepositoryEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.fixture()
    def api_key_repo(self):
        mock_session = AsyncMock()
        return PostgreSQLApiKeyRepository(mock_session)

    @pytest.mark.asyncio()
    async def test_create_api_key_empty_permissions(self, api_key_repo):
        """Test creating API key with empty permissions"""
        api_key_repo.session.scalar = AsyncMock(return_value=123)
        api_key_repo.session.flush = AsyncMock()
        api_key_repo.session.refresh = AsyncMock()

        with patch("secrets.token_urlsafe", return_value="token"):
            with patch("uuid.uuid4", return_value="uuid"):
                result = await api_key_repo.create_api_key("123", "Test", None)

        # Should default to empty dict
        api_key_obj = api_key_repo.session.add.call_args[0][0]
        assert api_key_obj.permissions == {}

    @pytest.mark.asyncio()
    async def test_verify_api_key_with_future_expiration(self, api_key_repo):
        """Test verifying API key with future expiration date"""
        mock_user = Mock(spec=User)
        mock_user.is_active = True

        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test-uuid"
        mock_api_key.expires_at = datetime.now(UTC) + timedelta(days=365)  # Future
        mock_api_key.user = mock_user
        mock_api_key.user_id = 123
        mock_api_key.name = "Test"
        mock_api_key.permissions = {}
        mock_api_key.is_active = True
        mock_api_key.created_at = datetime.now(UTC)
        mock_api_key.last_used_at = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_api_key
        api_key_repo.session.execute = AsyncMock(return_value=mock_result)

        with patch.object(api_key_repo, "update_last_used", new_callable=AsyncMock):
            result = await api_key_repo.verify_api_key("test_key")

        assert result is not None  # Should be valid

    def test_datetime_to_str_edge_cases(self, api_key_repo):
        """Test datetime conversion edge cases in _api_key_to_dict"""
        mock_api_key = Mock(spec=ApiKey)
        mock_api_key.id = "test"
        mock_api_key.user_id = 123
        mock_api_key.name = "Test"
        mock_api_key.permissions = {}
        mock_api_key.is_active = True

        # Test with string datetime
        mock_api_key.created_at = "2024-01-01T00:00:00"
        mock_api_key.last_used_at = None
        mock_api_key.expires_at = datetime.now(UTC)

        result = api_key_repo._api_key_to_dict(mock_api_key)

        assert result["created_at"] == "2024-01-01T00:00:00"
        assert result["last_used_at"] is None
        assert result["expires_at"] is not None
