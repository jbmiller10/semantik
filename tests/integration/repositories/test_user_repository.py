"""Integration tests for the PostgreSQLUserRepository using the real database session."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from shared.database.exceptions import EntityAlreadyExistsError, InvalidUserIdError
from sqlalchemy import select

from shared.database.models import User
from webui.repositories.postgres.user_repository import PostgreSQLUserRepository


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestPostgreSQLUserRepositoryIntegration:
    """Validate PostgreSQLUserRepository behaviour against the actual database."""

    @pytest.fixture()
    def repository(self, db_session):
        """Instantiate the repository with a real async session."""
        return PostgreSQLUserRepository(db_session)

    async def test_create_user_persists_expected_defaults(self, repository, db_session):
        """Creating a user should commit the row with hashed password and timestamps."""
        hashed_password = repository.pwd_context.hash("super-secret")
        user_data = {
            "username": f"user_{uuid4().hex[:8]}",
            "email": f"{uuid4().hex[:8]}@example.com",
            "hashed_password": hashed_password,
            "full_name": "Integration User",
            "is_active": True,
        }

        created_user = await repository.create_user(user_data)
        await db_session.commit()

        result = await db_session.execute(select(User).where(User.id == created_user["id"]))
        persisted = result.scalar_one()
        assert persisted.username == user_data["username"]
        assert persisted.email == user_data["email"]
        assert persisted.full_name == user_data["full_name"]
        assert persisted.is_active is True
        assert repository.pwd_context.verify("super-secret", persisted.hashed_password)
        assert isinstance(persisted.created_at, datetime)
        assert isinstance(persisted.updated_at, datetime)
        assert created_user["last_login"] is None

    async def test_create_user_duplicate_username_raises_error(self, repository, db_session):
        """Duplicate usernames should raise EntityAlreadyExistsError."""
        hashed_password = repository.pwd_context.hash("duplicate-secret")
        username = f"user_{uuid4().hex[:6]}"
        await repository.create_user(
            {
                "username": username,
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": hashed_password,
            }
        )
        await db_session.commit()

        with pytest.raises(EntityAlreadyExistsError):
            await repository.create_user(
                {
                    "username": username,
                    "email": f"{uuid4().hex[:6]}@example.com",
                    "hashed_password": repository.pwd_context.hash("other-secret"),
                }
            )

    async def test_update_user_changes_mutable_fields(self, repository, db_session):
        """Updating a user should persist the new values."""
        hashed_password = repository.pwd_context.hash("initial-pass")
        created = await repository.create_user(
            {
                "username": f"user_{uuid4().hex[:8]}",
                "email": f"{uuid4().hex[:8]}@example.com",
                "hashed_password": hashed_password,
                "full_name": "Original Name",
            }
        )
        await db_session.commit()

        updated = await repository.update_user(
            str(created["id"]),
            {
                "email": f"updated_{uuid4().hex[:8]}@example.com",
                "full_name": "Updated Name",
                "is_active": False,
            },
        )
        await db_session.commit()

        assert updated is not None
        assert updated["full_name"] == "Updated Name"
        assert updated["is_active"] is False

        result = await db_session.execute(select(User).where(User.id == created["id"]))
        persisted = result.scalar_one()
        assert persisted.full_name == "Updated Name"
        assert persisted.is_active is False

    async def test_delete_user_removes_row(self, repository, db_session):
        """Deleting a user should remove the row from the database."""
        created = await repository.create_user(
            {
                "username": f"user_{uuid4().hex[:8]}",
                "email": f"{uuid4().hex[:8]}@example.com",
                "hashed_password": repository.pwd_context.hash("delete-me"),
            }
        )
        await db_session.commit()

        deleted = await repository.delete_user(str(created["id"]))
        await db_session.commit()

        assert deleted is True

        result = await db_session.execute(select(User).where(User.id == created["id"]))
        assert result.scalar_one_or_none() is None

    async def test_delete_user_invalid_id_raises(self, repository):
        """Invalid identifiers should raise InvalidUserIdError instead of silently succeeding."""
        with pytest.raises(InvalidUserIdError):
            await repository.delete_user("invalid-id")
