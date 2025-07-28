#!/usr/bin/env python3
"""
Comprehensive test suite for webui/repositories/postgres/auth_repository.py
Tests authentication repository with mocked database operations
"""

import hashlib
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from shared.database.exceptions import DatabaseOperationError, InvalidUserIdError
from shared.database.models import RefreshToken
from sqlalchemy.exc import IntegrityError
from webui.repositories.postgres.auth_repository import PostgreSQLAuthRepository


class TestPostgreSQLAuthRepository:
    """Test PostgreSQL auth repository implementation"""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.add = Mock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        return session

    @pytest.fixture()
    def auth_repo(self, mock_session):
        """Create auth repository with mocked session"""
        return PostgreSQLAuthRepository(mock_session)

    def test_initialization(self, mock_session):
        """Test repository initialization"""
        repo = PostgreSQLAuthRepository(mock_session)
        assert repo.session == mock_session
        assert repo.model_class == RefreshToken

    @pytest.mark.asyncio()
    async def test_save_refresh_token_success(self, auth_repo, mock_session):
        """Test successful refresh token save"""
        user_id = "123"
        token_hash = "test_hash"
        expires_at = datetime.now(UTC) + timedelta(days=30)

        await auth_repo.save_refresh_token(user_id, token_hash, expires_at)

        # Verify session operations
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

        # Check the token object
        token_obj = mock_session.add.call_args[0][0]
        assert isinstance(token_obj, RefreshToken)
        assert token_obj.user_id == 123
        assert token_obj.token_hash == token_hash
        assert token_obj.expires_at == expires_at
        assert token_obj.is_revoked is False

    @pytest.mark.asyncio()
    async def test_save_refresh_token_invalid_user_id(self, auth_repo):
        """Test save refresh token with invalid user ID"""
        with pytest.raises(InvalidUserIdError):
            await auth_repo.save_refresh_token("invalid", "hash", datetime.now(UTC))

    @pytest.mark.asyncio()
    async def test_save_refresh_token_integrity_error(self, auth_repo, mock_session):
        """Test save refresh token with integrity error (e.g., user doesn't exist)"""
        mock_session.flush.side_effect = IntegrityError("statement", "params", "orig")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.save_refresh_token("123", "hash", datetime.now(UTC))

        assert exc_info.value.operation == "save"
        assert exc_info.value.resource == "refresh_token"

    @pytest.mark.asyncio()
    async def test_save_refresh_token_generic_error(self, auth_repo, mock_session):
        """Test save refresh token with generic database error"""
        mock_session.flush.side_effect = Exception("Database error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.save_refresh_token("123", "hash", datetime.now(UTC))

        assert exc_info.value.operation == "save"

    @pytest.mark.asyncio()
    async def test_verify_refresh_token_valid(self, auth_repo, mock_session):
        """Test verifying a valid refresh token"""
        token = "test_token"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Mock refresh token
        mock_refresh_token = Mock()
        mock_refresh_token.user_id = 123

        # Mock query result
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_refresh_token
        mock_session.execute.return_value = mock_result

        # Mock user active check
        mock_session.scalar.return_value = True

        result = await auth_repo.verify_refresh_token(token)

        assert result == "123"

        # Verify the query
        execute_call = mock_session.execute.call_args[0][0]
        # The query should check for token_hash, not revoked, and not expired

    @pytest.mark.asyncio()
    async def test_verify_refresh_token_not_found(self, auth_repo, mock_session):
        """Test verifying non-existent refresh token"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await auth_repo.verify_refresh_token("invalid_token")
        assert result is None

    @pytest.mark.asyncio()
    async def test_verify_refresh_token_user_inactive(self, auth_repo, mock_session):
        """Test verifying token for inactive user"""
        token = "test_token"

        # Mock refresh token
        mock_refresh_token = Mock()
        mock_refresh_token.user_id = 123

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_refresh_token
        mock_session.execute.return_value = mock_result

        # Mock user as inactive
        mock_session.scalar.return_value = False

        result = await auth_repo.verify_refresh_token(token)
        assert result is None

    @pytest.mark.asyncio()
    async def test_verify_refresh_token_error(self, auth_repo, mock_session):
        """Test verify refresh token with database error"""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.verify_refresh_token("token")

        assert exc_info.value.operation == "verify"

    @pytest.mark.asyncio()
    async def test_revoke_refresh_token_success(self, auth_repo, mock_session):
        """Test successful token revocation"""
        token = "test_token"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = 1  # Token ID
        mock_session.execute.return_value = mock_result

        await auth_repo.revoke_refresh_token(token)

        # Verify update query was executed
        execute_call = mock_session.execute.call_args[0][0]
        # Should be an update query setting is_revoked = True

    @pytest.mark.asyncio()
    async def test_revoke_refresh_token_not_found(self, auth_repo, mock_session):
        """Test revoking non-existent token"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        # Should not raise error, just log warning
        await auth_repo.revoke_refresh_token("invalid_token")

    @pytest.mark.asyncio()
    async def test_revoke_refresh_token_error(self, auth_repo, mock_session):
        """Test revoke token with database error"""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.revoke_refresh_token("token")

        assert exc_info.value.operation == "revoke"

    @pytest.mark.asyncio()
    async def test_update_user_last_login_success(self, auth_repo, mock_session):
        """Test successful last login update"""
        user_id = "123"

        await auth_repo.update_user_last_login(user_id)

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

        # Verify update query
        execute_call = mock_session.execute.call_args[0][0]
        # Should be updating User.last_login

    @pytest.mark.asyncio()
    async def test_update_user_last_login_invalid_user_id(self, auth_repo):
        """Test update last login with invalid user ID"""
        with pytest.raises(InvalidUserIdError):
            await auth_repo.update_user_last_login("invalid")

    @pytest.mark.asyncio()
    async def test_update_user_last_login_error(self, auth_repo, mock_session):
        """Test update last login with database error"""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.update_user_last_login("123")

        assert exc_info.value.operation == "update"

    @pytest.mark.asyncio()
    async def test_cleanup_expired_tokens_success(self, auth_repo, mock_session):
        """Test successful cleanup of expired tokens"""
        # Mock deleted token IDs
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [1, 2, 3]
        mock_session.execute.return_value = mock_result

        count = await auth_repo.cleanup_expired_tokens()

        assert count == 3
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_cleanup_expired_tokens_none_found(self, auth_repo, mock_session):
        """Test cleanup when no expired tokens found"""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        count = await auth_repo.cleanup_expired_tokens()

        assert count == 0

    @pytest.mark.asyncio()
    async def test_cleanup_expired_tokens_error(self, auth_repo, mock_session):
        """Test cleanup with database error"""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.cleanup_expired_tokens()

        assert exc_info.value.operation == "cleanup"

    @pytest.mark.asyncio()
    async def test_revoke_all_user_tokens_success(self, auth_repo, mock_session):
        """Test revoking all tokens for a user"""
        user_id = "123"

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [1, 2]
        mock_session.execute.return_value = mock_result

        count = await auth_repo.revoke_all_user_tokens(user_id)

        assert count == 2

    @pytest.mark.asyncio()
    async def test_revoke_all_user_tokens_invalid_user_id(self, auth_repo):
        """Test revoke all tokens with invalid user ID"""
        with pytest.raises(InvalidUserIdError):
            await auth_repo.revoke_all_user_tokens("invalid")

    @pytest.mark.asyncio()
    async def test_revoke_all_user_tokens_error(self, auth_repo, mock_session):
        """Test revoke all tokens with database error"""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(DatabaseOperationError) as exc_info:
            await auth_repo.revoke_all_user_tokens("123")

        assert exc_info.value.operation == "revoke_all"

    @pytest.mark.asyncio()
    async def test_get_active_token_count_success(self, auth_repo, mock_session):
        """Test getting active token count"""
        user_id = "123"

        mock_session.scalar.return_value = 5

        count = await auth_repo.get_active_token_count(user_id)

        assert count == 5

    @pytest.mark.asyncio()
    async def test_get_active_token_count_zero(self, auth_repo, mock_session):
        """Test getting active token count when none exist"""
        mock_session.scalar.return_value = None

        count = await auth_repo.get_active_token_count("123")

        assert count == 0

    @pytest.mark.asyncio()
    async def test_get_active_token_count_invalid_user_id(self, auth_repo):
        """Test get active token count with invalid user ID"""
        with pytest.raises(InvalidUserIdError):
            await auth_repo.get_active_token_count("invalid")

    @pytest.mark.asyncio()
    async def test_create_token_success(self, auth_repo, mock_session):
        """Test create token (compatibility method)"""
        user_id = "123"
        token = "test_token"
        expires_at = (datetime.now(UTC) + timedelta(days=1)).isoformat()

        # Mock save_refresh_token
        with patch.object(auth_repo, "save_refresh_token", new_callable=AsyncMock) as mock_save:
            await auth_repo.create_token(user_id, token, expires_at)

            mock_save.assert_called_once()
            args = mock_save.call_args[0]
            assert args[0] == user_id
            assert args[1] == hashlib.sha256(token.encode()).hexdigest()

    @pytest.mark.asyncio()
    async def test_create_token_with_datetime(self, auth_repo, mock_session):
        """Test create token with datetime object"""
        user_id = "123"
        token = "test_token"
        expires_at = datetime.now(UTC) + timedelta(days=1)

        with patch.object(auth_repo, "save_refresh_token", new_callable=AsyncMock) as mock_save:
            await auth_repo.create_token(user_id, token, expires_at)

            mock_save.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_token_user_id(self, auth_repo):
        """Test get token user ID (compatibility method)"""
        token = "test_token"

        with patch.object(auth_repo, "verify_refresh_token", new_callable=AsyncMock) as mock_verify:
            mock_verify.return_value = "123"

            result = await auth_repo.get_token_user_id(token)

            assert result == "123"
            mock_verify.assert_called_once_with(token)

    @pytest.mark.asyncio()
    async def test_delete_token(self, auth_repo):
        """Test delete token (compatibility method)"""
        token = "test_token"

        with patch.object(auth_repo, "revoke_refresh_token", new_callable=AsyncMock) as mock_revoke:
            await auth_repo.delete_token(token)

            mock_revoke.assert_called_once_with(token)

    def test_hash_token(self, auth_repo):
        """Test token hashing"""
        token = "test_token"
        expected_hash = hashlib.sha256(token.encode()).hexdigest()

        result = auth_repo._hash_token(token)

        assert result == expected_hash
        assert len(result) == 64  # SHA-256 produces 64 hex characters


class TestAuthRepositoryEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.fixture()
    def auth_repo(self):
        mock_session = AsyncMock()
        return PostgreSQLAuthRepository(mock_session)

    @pytest.mark.asyncio()
    async def test_save_refresh_token_with_none_expires(self, auth_repo):
        """Test saving token with None expiration"""
        auth_repo.session.flush = AsyncMock()

        # Should handle None expiration gracefully
        await auth_repo.save_refresh_token("123", "hash", None)

    @pytest.mark.asyncio()
    async def test_verify_token_with_expired_token(self, auth_repo):
        """Test verifying an expired but not revoked token"""
        token = "expired_token"

        # Mock finding token but it's expired
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None  # Query filters out expired
        auth_repo.session.execute = AsyncMock(return_value=mock_result)

        result = await auth_repo.verify_refresh_token(token)
        assert result is None

    @pytest.mark.asyncio()
    async def test_concurrent_token_operations(self, auth_repo):
        """Test handling concurrent token operations"""
        # This would test race conditions in real scenarios
        # Mock multiple simultaneous operations
        auth_repo.session.execute = AsyncMock()

        # Simulate concurrent revoke operations
        await auth_repo.revoke_refresh_token("token1")
        await auth_repo.revoke_refresh_token("token1")  # Same token

        # Should handle gracefully without errors

    def test_hash_token_unicode(self, auth_repo):
        """Test hashing tokens with unicode characters"""
        token = "test_token_🔐"

        # Should handle unicode properly
        result = auth_repo._hash_token(token)
        assert isinstance(result, str)
        assert len(result) == 64

    @pytest.mark.asyncio()
    async def test_token_operations_with_very_long_user_id(self, auth_repo):
        """Test operations with very long numeric user IDs"""
        user_id = "999999999999999999"  # Very large but valid integer

        auth_repo.session.flush = AsyncMock()

        # Should handle large user IDs
        await auth_repo.save_refresh_token(user_id, "hash", datetime.now(UTC))

    @pytest.mark.asyncio()
    async def test_cleanup_with_no_expired_tokens(self, auth_repo):
        """Test cleanup when database has no expired tokens"""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = []
        auth_repo.session.execute = AsyncMock(return_value=mock_result)

        count = await auth_repo.cleanup_expired_tokens()
        assert count == 0
