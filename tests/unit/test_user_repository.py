#!/usr/bin/env python3
"""
Comprehensive test suite for webui/repositories/postgres/user_repository.py
Tests user repository with mocked database operations
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from passlib.context import CryptContext
from shared.database.exceptions import DatabaseOperationError, EntityAlreadyExistsError, InvalidUserIdError
from shared.database.models import User
from sqlalchemy.exc import IntegrityError
from webui.repositories.postgres.user_repository import PostgreSQLUserRepository


class TestPostgreSQLUserRepository:
    """Test PostgreSQL user repository implementation"""

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
    def user_repo(self, mock_session):
        """Create user repository with mocked session"""
        return PostgreSQLUserRepository(mock_session)

    def test_initialization(self, mock_session):
        """Test repository initialization"""
        repo = PostgreSQLUserRepository(mock_session)
        assert repo.session == mock_session
        assert repo.model == User
        assert isinstance(repo.pwd_context, CryptContext)

    @pytest.mark.asyncio()
    async def test_create_user_success(self, user_repo, mock_session):
        """Test successful user creation"""
        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed_password_123",
            "full_name": "Test User",
            "is_active": True,
            "is_superuser": False,
        }

        # Mock no existing user
        mock_session.scalar.return_value = None

        # Mock the created user
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = user_data["username"]
        mock_user.email = user_data["email"]
        mock_user.hashed_password = user_data["hashed_password"]
        mock_user.full_name = user_data["full_name"]
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        result = await user_repo.create_user(user_data)

        # Verify operations
        mock_session.scalar.assert_called_once()  # Check for existing user
        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()
        mock_session.refresh.assert_called_once()

        # Check the user object
        user_obj = mock_session.add.call_args[0][0]
        assert isinstance(user_obj, User)
        assert user_obj.username == "testuser"
        assert user_obj.email == "test@example.com"
        assert user_obj.is_active is True

    @pytest.mark.asyncio()
    async def test_create_user_missing_required_fields(self, user_repo):
        """Test creating user with missing required fields"""
        incomplete_data = {
            "username": "testuser",
            # Missing email and hashed_password
        }

        with pytest.raises(DatabaseOperationError) as exc_info:
            await user_repo.create_user(incomplete_data)
        
        assert "Username, email, and hashed_password are required" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_user_duplicate_username(self, user_repo, mock_session):
        """Test creating user with duplicate username"""
        existing_user = Mock(spec=User)
        existing_user.username = "testuser"
        existing_user.email = "other@example.com"

        mock_session.scalar.return_value = existing_user

        user_data = {"username": "testuser", "email": "new@example.com", "hashed_password": "hashed_password"}

        with pytest.raises(EntityAlreadyExistsError) as exc_info:
            await user_repo.create_user(user_data)

        assert "username: testuser" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_user_duplicate_email(self, user_repo, mock_session):
        """Test creating user with duplicate email"""
        existing_user = Mock(spec=User)
        existing_user.username = "otheruser"
        existing_user.email = "test@example.com"

        mock_session.scalar.return_value = existing_user

        user_data = {"username": "newuser", "email": "test@example.com", "hashed_password": "hashed_password"}

        with pytest.raises(EntityAlreadyExistsError) as exc_info:
            await user_repo.create_user(user_data)

        assert "email: test@example.com" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_get_user_by_id_success(self, user_repo, mock_session):
        """Test getting user by ID"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.hashed_password = "hashed"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        result = await user_repo.get_user_by_id("123")

        assert result is not None
        assert result["id"] == 123
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"

    @pytest.mark.asyncio()
    async def test_get_user_by_id_not_found(self, user_repo, mock_session):
        """Test getting non-existent user by ID"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await user_repo.get_user_by_id("999")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_user_by_id_invalid_id(self, user_repo):
        """Test getting user with invalid ID"""
        with pytest.raises(InvalidUserIdError):
            await user_repo.get_user_by_id("invalid")

    @pytest.mark.asyncio()
    async def test_get_user_by_username_success(self, user_repo, mock_session):
        """Test getting user by username"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = "hashed"
        mock_user.full_name = None
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        result = await user_repo.get_user_by_username("testuser")

        assert result is not None
        assert result["username"] == "testuser"

    @pytest.mark.asyncio()
    async def test_get_user_by_email_success(self, user_repo, mock_session):
        """Test getting user by email"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = "hashed"
        mock_user.full_name = None
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_user
        mock_session.execute.return_value = mock_result

        result = await user_repo.get_user_by_email("test@example.com")

        assert result is not None
        assert result["email"] == "test@example.com"

    @pytest.mark.asyncio()
    async def test_update_user_success(self, user_repo, mock_session):
        """Test successful user update"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "olduser"
        mock_user.email = "old@example.com"
        mock_user.full_name = "Old Name"
        mock_user.hashed_password = "hashed"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        mock_session.get.return_value = mock_user
        mock_session.scalar.return_value = None  # No conflicts

        updates = {"username": "newuser", "email": "new@example.com", "full_name": "New Name"}

        result = await user_repo.update_user("123", updates)

        # Verify updates were applied
        assert mock_user.username == "newuser"
        assert mock_user.email == "new@example.com"
        assert mock_user.full_name == "New Name"
        assert mock_user.updated_at > mock_user.created_at
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_user_not_found(self, user_repo, mock_session):
        """Test updating non-existent user"""
        mock_session.get.return_value = None

        result = await user_repo.update_user("999", {"username": "new"})
        assert result is None

    @pytest.mark.asyncio()
    async def test_update_user_invalid_id(self, user_repo):
        """Test updating user with invalid ID"""
        with pytest.raises(InvalidUserIdError):
            await user_repo.update_user("invalid", {"username": "new"})

    @pytest.mark.asyncio()
    async def test_update_user_duplicate_username(self, user_repo, mock_session):
        """Test updating user with duplicate username"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "currentuser"
        mock_user.email = "current@example.com"

        mock_session.get.return_value = mock_user

        # Mock existing user with target username
        existing_user = Mock(spec=User)
        existing_user.username = "takenuser"
        mock_session.scalar.return_value = existing_user

        with pytest.raises(EntityAlreadyExistsError) as exc_info:
            await user_repo.update_user("123", {"username": "takenuser"})

        assert "username: takenuser" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_update_user_ignores_invalid_fields(self, user_repo, mock_session):
        """Test that update ignores non-allowed fields"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.created_at = datetime.now(UTC)  # Should not be changeable
        mock_user.hashed_password = "original_hash"
        mock_user.full_name = "Test"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        original_created_at = mock_user.created_at

        mock_session.get.return_value = mock_user
        mock_session.scalar.return_value = None  # No conflicts

        updates = {
            "full_name": "New Name",  # Allowed
            "id": 999,  # Not allowed
            "created_at": datetime.now(UTC),  # Not allowed
        }

        await user_repo.update_user("123", updates)

        # Only full_name should be updated
        assert mock_user.full_name == "New Name"
        assert mock_user.id == 123  # Unchanged
        assert mock_user.created_at == original_created_at  # Unchanged

    @pytest.mark.asyncio()
    async def test_delete_user_success(self, user_repo, mock_session):
        """Test successful user deletion"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = 123
        mock_session.execute.return_value = mock_result

        result = await user_repo.delete_user("123")

        assert result is True
        mock_session.execute.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_user_not_found(self, user_repo, mock_session):
        """Test deleting non-existent user"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await user_repo.delete_user("999")

        assert result is False

    @pytest.mark.asyncio()
    async def test_delete_user_invalid_id(self, user_repo):
        """Test deleting user with invalid ID"""
        with pytest.raises(InvalidUserIdError):
            await user_repo.delete_user("invalid")

    @pytest.mark.asyncio()
    async def test_list_users_no_filters(self, user_repo, mock_session):
        """Test listing all users without filters"""
        mock_users = []
        for i in range(3):
            user = Mock(spec=User)
            user.id = i + 1
            user.username = f"user{i}"
            user.email = f"user{i}@example.com"
            user.hashed_password = "hashed"
            user.full_name = None
            user.is_active = True
            user.is_superuser = False
            user.created_at = datetime.now(UTC) - timedelta(days=i)
            user.updated_at = datetime.now(UTC)
            user.last_login = None
            mock_users.append(user)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_users
        mock_session.execute.return_value = mock_result

        result = await user_repo.list_users()

        assert len(result) == 3
        assert result[0]["username"] == "user0"
        assert result[1]["username"] == "user1"
        assert result[2]["username"] == "user2"

    @pytest.mark.asyncio()
    async def test_list_users_with_filters(self, user_repo, mock_session):
        """Test listing users with filters"""
        active_users = []
        for i in range(2):
            user = Mock(spec=User)
            user.id = i + 1
            user.username = f"activeuser{i}"
            user.email = f"active{i}@example.com"
            user.hashed_password = "hashed"
            user.full_name = None
            user.is_active = True
            user.is_superuser = False
            user.created_at = datetime.now(UTC)
            user.updated_at = datetime.now(UTC)
            user.last_login = None
            active_users.append(user)

        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = active_users
        mock_session.execute.return_value = mock_result

        result = await user_repo.list_users(is_active=True)

        assert len(result) == 2
        assert all(user["is_active"] for user in result)

    @pytest.mark.asyncio()
    async def test_verify_password_correct(self, user_repo, mock_session):
        """Test verifying correct password"""
        # Create a real password hash
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        password = "correct_password"
        hashed = pwd_context.hash(password)

        # Mock get_user_by_username
        with patch.object(user_repo, "get_user_by_username", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": 123, "username": "testuser", "hashed_password": hashed, "is_active": True}

            result = await user_repo.verify_password("testuser", password)

            assert result is not None
            assert result["username"] == "testuser"

    @pytest.mark.asyncio()
    async def test_verify_password_incorrect(self, user_repo):
        """Test verifying incorrect password"""
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        hashed = pwd_context.hash("correct_password")

        with patch.object(user_repo, "get_user_by_username", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": 123, "username": "testuser", "hashed_password": hashed, "is_active": True}

            result = await user_repo.verify_password("testuser", "wrong_password")

            assert result is None

    @pytest.mark.asyncio()
    async def test_verify_password_user_not_found(self, user_repo):
        """Test verifying password for non-existent user"""
        with patch.object(user_repo, "get_user_by_username", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            result = await user_repo.verify_password("nonexistent", "password")

            assert result is None

    @pytest.mark.asyncio()
    async def test_update_last_login_success(self, user_repo, mock_session):
        """Test updating last login timestamp"""
        await user_repo.update_last_login("123")

        mock_session.execute.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_last_login_invalid_id(self, user_repo):
        """Test updating last login with invalid ID"""
        with pytest.raises(InvalidUserIdError):
            await user_repo.update_last_login("invalid")

    @pytest.mark.asyncio()
    async def test_count_users_no_filter(self, user_repo, mock_session):
        """Test counting all users"""
        mock_session.scalar.return_value = 42

        count = await user_repo.count_users()

        assert count == 42

    @pytest.mark.asyncio()
    async def test_count_users_with_filter(self, user_repo, mock_session):
        """Test counting users with active filter"""
        mock_session.scalar.return_value = 35

        count = await user_repo.count_users(is_active=True)

        assert count == 35

    @pytest.mark.asyncio()
    async def test_count_users_none_result(self, user_repo, mock_session):
        """Test counting users when result is None"""
        mock_session.scalar.return_value = None

        count = await user_repo.count_users()

        assert count == 0

    def test_user_to_dict_none(self, user_repo):
        """Test converting None to dict"""
        result = user_repo._user_to_dict(None)
        assert result is None

    def test_user_to_dict_with_all_fields(self, user_repo):
        """Test converting user with all fields to dict"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.hashed_password = "hashed"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = datetime.now(UTC) - timedelta(hours=1)

        result = user_repo._user_to_dict(mock_user)

        assert result["id"] == 123
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["full_name"] == "Test User"
        assert result["hashed_password"] == "hashed"
        assert result["is_active"] is True
        assert result["is_superuser"] is False
        assert result["created_at"] is not None
        assert result["updated_at"] is not None
        assert result["last_login"] is not None

    def test_user_to_dict_with_null_fields(self, user_repo):
        """Test converting user with null optional fields"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.full_name = None
        mock_user.hashed_password = "hashed"
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        result = user_repo._user_to_dict(mock_user)

        assert result["full_name"] is None
        assert result["last_login"] is None


class TestUserRepositoryEdgeCases:
    """Test edge cases and error scenarios"""

    @pytest.fixture()
    def user_repo(self):
        mock_session = AsyncMock()
        return PostgreSQLUserRepository(mock_session)

    @pytest.mark.asyncio()
    async def test_create_user_with_defaults(self, user_repo):
        """Test creating user with default values"""
        user_repo.session.scalar = AsyncMock(return_value=None)
        user_repo.session.flush = AsyncMock()
        user_repo.session.refresh = AsyncMock()

        user_data = {
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed",
            # is_active and is_superuser should default
        }

        await user_repo.create_user(user_data)

        user_obj = user_repo.session.add.call_args[0][0]
        assert user_obj.is_active is True  # Default
        assert user_obj.is_superuser is False  # Default

    @pytest.mark.asyncio()
    async def test_integrity_error_handling(self, user_repo):
        """Test handling of database integrity errors"""
        user_repo.session.scalar = AsyncMock(return_value=None)
        user_repo.session.flush = AsyncMock(side_effect=IntegrityError("statement", "params", "orig"))

        with pytest.raises(DatabaseOperationError):
            await user_repo.create_user({"username": "test", "email": "test@example.com", "hashed_password": "hashed"})

    def test_datetime_to_str_edge_cases(self, user_repo):
        """Test datetime conversion edge cases"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "test"
        mock_user.email = "test@example.com"
        mock_user.full_name = None
        mock_user.hashed_password = "hashed"
        mock_user.is_active = True
        mock_user.is_superuser = False

        # Test with string datetime
        mock_user.created_at = "2024-01-01T00:00:00"
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        result = user_repo._user_to_dict(mock_user)

        assert result["created_at"] == "2024-01-01T00:00:00"
        assert result["updated_at"] is not None
        assert result["last_login"] is None

    @pytest.mark.asyncio()
    async def test_update_user_password(self, user_repo):
        """Test updating user password"""
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.hashed_password = "old_hash"
        mock_user.full_name = None
        mock_user.is_active = True
        mock_user.is_superuser = False
        mock_user.created_at = datetime.now(UTC)
        mock_user.updated_at = datetime.now(UTC)
        mock_user.last_login = None

        user_repo.session.get = AsyncMock(return_value=mock_user)
        user_repo.session.scalar = AsyncMock(return_value=None)
        user_repo.session.flush = AsyncMock()

        await user_repo.update_user("123", {"hashed_password": "new_hash"})

        assert mock_user.hashed_password == "new_hash"

    @pytest.mark.asyncio()
    async def test_get_user_shortcut_method(self, user_repo):
        """Test get_user method (shortcut for get_user_by_id)"""
        with patch.object(user_repo, "get_user_by_id", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"id": 123, "username": "test"}

            result = await user_repo.get_user("123")

            assert result["id"] == 123
            mock_get.assert_called_once_with("123")
