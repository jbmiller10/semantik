"""Unit tests for SQLite repository implementations.

These tests verify that the repository classes correctly wrap the underlying
database implementation functions, handling data transformations and error cases.
"""

from unittest.mock import patch

import pytest

from packages.shared.database.sqlite_repository import SQLiteJobRepository, SQLiteUserRepository


class TestSQLiteJobRepository:
    """Test suite for SQLiteJobRepository."""

    @pytest.fixture()
    def mock_db(self):
        """Create a mock database implementation."""
        with patch("packages.shared.database.sqlite_repository.db_impl") as mock:
            yield mock

    @pytest.fixture()
    def repository(self, mock_db):
        """Create a repository instance with mocked database."""
        repo = SQLiteJobRepository()
        repo.db = mock_db
        return repo

    @pytest.mark.asyncio()
    async def test_create_job_success(self, repository, mock_db):
        """Test successful job creation."""
        # Arrange
        job_data = {"name": "test_job", "user_id": 1}
        expected_job_id = "123"
        expected_job = {"id": expected_job_id, "name": "test_job", "user_id": 1}

        mock_db.create_job.return_value = expected_job_id
        mock_db.get_job.return_value = expected_job

        # Act
        result = await repository.create_job(job_data)

        # Assert
        assert result == expected_job
        mock_db.create_job.assert_called_once_with(job_data)
        mock_db.get_job.assert_called_once_with(expected_job_id)

    @pytest.mark.asyncio()
    async def test_create_job_retrieval_failure(self, repository, mock_db):
        """Test job creation when retrieval fails."""
        # Arrange
        job_data = {"name": "test_job"}
        mock_db.create_job.return_value = "123"
        mock_db.get_job.return_value = None

        # Act & Assert
        with pytest.raises(ValueError, match="Failed to retrieve created job"):
            await repository.create_job(job_data)

    @pytest.mark.asyncio()
    async def test_get_job_success(self, repository, mock_db):
        """Test successful job retrieval."""
        # Arrange
        job_id = "123"
        expected_job = {"id": job_id, "name": "test_job"}
        mock_db.get_job.return_value = expected_job

        # Act
        result = await repository.get_job(job_id)

        # Assert
        assert result == expected_job
        mock_db.get_job.assert_called_once_with(job_id)

    @pytest.mark.asyncio()
    async def test_get_job_not_found(self, repository, mock_db):
        """Test job retrieval when job doesn't exist."""
        # Arrange
        mock_db.get_job.return_value = None

        # Act
        result = await repository.get_job("nonexistent")

        # Assert
        assert result is None
        mock_db.get_job.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio()
    async def test_update_job_success(self, repository, mock_db):
        """Test successful job update."""
        # Arrange
        job_id = "123"
        updates = {"status": "processing"}
        existing_job = {"id": job_id, "name": "test_job", "status": "pending"}
        updated_job = {"id": job_id, "name": "test_job", "status": "processing"}

        mock_db.get_job.side_effect = [existing_job, updated_job]
        mock_db.update_job.return_value = None

        # Act
        result = await repository.update_job(job_id, updates)

        # Assert
        assert result == updated_job
        mock_db.update_job.assert_called_once_with(job_id, updates)
        assert mock_db.get_job.call_count == 2

    @pytest.mark.asyncio()
    async def test_update_nonexistent_job(self, repository, mock_db):
        """Test updating a job that doesn't exist."""
        # Arrange
        mock_db.get_job.return_value = None

        # Act
        result = await repository.update_job("nonexistent", {"status": "done"})

        # Assert
        assert result is None
        mock_db.get_job.assert_called_once()
        mock_db.update_job.assert_not_called()

    @pytest.mark.asyncio()
    async def test_delete_job_success(self, repository, mock_db):
        """Test successful job deletion."""
        # Arrange
        job_id = "123"
        mock_db.get_job.side_effect = [{"id": job_id}, None]  # Exists, then doesn't
        mock_db.delete_job.return_value = None

        # Act
        result = await repository.delete_job(job_id)

        # Assert
        assert result is True
        mock_db.delete_job.assert_called_once_with(job_id)
        assert mock_db.get_job.call_count == 2

    @pytest.mark.asyncio()
    async def test_delete_nonexistent_job(self, repository, mock_db):
        """Test deleting a job that doesn't exist."""
        # Arrange
        mock_db.get_job.return_value = None

        # Act
        result = await repository.delete_job("nonexistent")

        # Assert
        assert result is False
        mock_db.get_job.assert_called_once()
        mock_db.delete_job.assert_not_called()

    @pytest.mark.asyncio()
    async def test_list_jobs_with_user_id(self, repository, mock_db):
        """Test listing jobs with user ID filter."""
        # Arrange
        user_id = "42"
        expected_jobs = [{"id": "1", "user_id": 42}, {"id": "2", "user_id": 42}]
        mock_db.list_jobs.return_value = expected_jobs

        # Act
        result = await repository.list_jobs(user_id=user_id)

        # Assert
        assert result == expected_jobs
        mock_db.list_jobs.assert_called_once_with(user_id=42)

    @pytest.mark.asyncio()
    async def test_list_jobs_invalid_user_id(self, repository, mock_db):
        """Test listing jobs with invalid user ID."""
        # Act & Assert
        with pytest.raises(ValueError, match="user_id must be a valid integer"):
            await repository.list_jobs(user_id="not_an_int")

    @pytest.mark.asyncio()
    async def test_get_all_job_ids(self, repository, mock_db):
        """Test getting all job IDs."""
        # Arrange
        jobs = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        mock_db.list_jobs.return_value = jobs

        # Act
        result = await repository.get_all_job_ids()

        # Assert
        assert result == ["1", "2", "3"]
        mock_db.list_jobs.assert_called_once()


class TestSQLiteUserRepository:
    """Test suite for SQLiteUserRepository."""

    @pytest.fixture()
    def mock_db(self):
        """Create a mock database implementation."""
        with patch("packages.shared.database.sqlite_repository.db_impl") as mock:
            yield mock

    @pytest.fixture()
    def repository(self, mock_db):
        """Create a repository instance with mocked database."""
        repo = SQLiteUserRepository()
        repo.db = mock_db
        return repo

    @pytest.mark.asyncio()
    async def test_create_user_success(self, repository, mock_db):
        """Test successful user creation."""
        # Arrange
        user_data = {"username": "testuser", "email": "test@example.com"}
        expected_user = {"id": 1, "username": "testuser", "email": "test@example.com"}
        mock_db.create_user.return_value = expected_user

        # Act
        result = await repository.create_user(user_data)

        # Assert
        assert result == expected_user
        mock_db.create_user.assert_called_once_with(**user_data)

    @pytest.mark.asyncio()
    async def test_get_user_success(self, repository, mock_db):
        """Test successful user retrieval by ID."""
        # Arrange
        user_id = "1"
        expected_user = {"id": 1, "username": "testuser"}
        mock_db.get_user.return_value = expected_user

        # Act
        result = await repository.get_user(user_id)

        # Assert
        assert result == expected_user
        mock_db.get_user.assert_called_once_with(user_id)

    @pytest.mark.asyncio()
    async def test_get_user_not_found(self, repository, mock_db):
        """Test user retrieval when user doesn't exist."""
        # Arrange
        mock_db.get_user.return_value = None

        # Act
        result = await repository.get_user("nonexistent")

        # Assert
        assert result is None
        mock_db.get_user.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio()
    async def test_get_user_by_username_success(self, repository, mock_db):
        """Test successful user retrieval by username."""
        # Arrange
        username = "testuser"
        expected_user = {"id": 1, "username": username}
        mock_db.get_user.return_value = expected_user

        # Act
        result = await repository.get_user_by_username(username)

        # Assert
        assert result == expected_user
        # Note: SQLite implementation uses get_user for username lookup
        mock_db.get_user.assert_called_once_with(username)

    @pytest.mark.asyncio()
    async def test_update_user_placeholder(self, repository, mock_db):
        """Test user update placeholder implementation."""
        # Arrange
        user_id = "1"
        updates = {"email": "newemail@example.com"}
        existing_user = {"id": 1, "username": "testuser", "email": "old@example.com"}
        mock_db.get_user.return_value = existing_user

        # Act
        result = await repository.update_user(user_id, updates)

        # Assert
        # Currently returns unmodified user
        assert result == existing_user
        mock_db.get_user.assert_called_once_with(user_id)

    @pytest.mark.asyncio()
    async def test_update_nonexistent_user(self, repository, mock_db):
        """Test updating a user that doesn't exist."""
        # Arrange
        mock_db.get_user.return_value = None

        # Act
        result = await repository.update_user("nonexistent", {"email": "new@example.com"})

        # Assert
        assert result is None
        mock_db.get_user.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_user_placeholder(self, repository, mock_db):
        """Test user deletion placeholder implementation."""
        # Arrange
        user_id = "1"
        mock_db.get_user.return_value = {"id": 1, "username": "testuser"}

        # Act
        result = await repository.delete_user(user_id)

        # Assert
        # Currently always returns False (not implemented)
        assert result is False
        mock_db.get_user.assert_called_once_with(user_id)

    @pytest.mark.asyncio()
    async def test_delete_nonexistent_user(self, repository, mock_db):
        """Test deleting a user that doesn't exist."""
        # Arrange
        mock_db.get_user.return_value = None

        # Act
        result = await repository.delete_user("nonexistent")

        # Assert
        assert result is False
        mock_db.get_user.assert_called_once()

    @pytest.mark.asyncio()
    async def test_error_propagation(self, repository, mock_db):
        """Test that database errors are properly logged and re-raised."""
        # Arrange
        mock_db.create_user.side_effect = Exception("Database connection failed")

        # Act & Assert
        with pytest.raises(Exception, match="Database connection failed"):
            await repository.create_user({"username": "test"})
