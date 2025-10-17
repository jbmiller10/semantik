"""Integration coverage for PostgreSQLUserRepository using the real async session."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from shared.database.exceptions import DatabaseOperationError, EntityAlreadyExistsError, InvalidUserIdError
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import User
from packages.webui.repositories.postgres.user_repository import PostgreSQLUserRepository, pwd_context

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestUserRepositoryIntegration:
    """Exercise the user repository against the real database."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> PostgreSQLUserRepository:
        """Instantiate the repository with the shared async session."""
        return PostgreSQLUserRepository(db_session)

    async def _fetch_user(self, db_session: AsyncSession, user_id: int) -> User | None:
        """Helper to load a user directly from the database."""
        result = await db_session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def test_create_user_persists_defaults(
        self,
        repository: PostgreSQLUserRepository,
        db_session: AsyncSession,
    ) -> None:
        """Creating a user should persist it and return a populated dictionary."""
        hashed_password = pwd_context.hash("super-secret")
        user_data = {
            "username": f"int-user-{uuid4().hex[:8]}",
            "email": f"{uuid4().hex[:6]}@example.com",
            "hashed_password": hashed_password,
            "full_name": "Integration User",
        }

        created = await repository.create_user(user_data)

        assert created["username"] == user_data["username"]
        assert created["email"] == user_data["email"]
        assert created["is_active"] is True
        assert created["is_superuser"] is False
        assert created["hashed_password"] == hashed_password
        assert created["created_at"] is not None

        persisted = await self._fetch_user(db_session, int(created["id"]))
        assert persisted is not None
        assert persisted.username == user_data["username"]
        assert persisted.email == user_data["email"]
        assert persisted.is_active

    async def test_create_user_duplicate_username_raises(self, repository: PostgreSQLUserRepository) -> None:
        """Attempting to create a user with a duplicate username should raise."""
        username = f"dup-user-{uuid4().hex[:6]}"
        base_payload = {
            "username": username,
            "email": f"{uuid4().hex[:6]}@example.com",
            "hashed_password": pwd_context.hash("initial-pass"),
        }
        await repository.create_user(base_payload)

        with pytest.raises(EntityAlreadyExistsError):
            await repository.create_user(
                {
                    "username": username,
                    "email": f"{uuid4().hex[:6]}@example.com",
                    "hashed_password": pwd_context.hash("another-pass"),
                }
            )

    async def test_create_user_with_missing_fields_raises_database_error(
        self, repository: PostgreSQLUserRepository
    ) -> None:
        """Missing required fields should bubble up as DatabaseOperationError."""
        with pytest.raises(DatabaseOperationError):
            await repository.create_user({"username": "missing-email"})

    async def test_get_user_by_id_round_trips(self, repository: PostgreSQLUserRepository) -> None:
        """The repository should fetch users that exist and return None otherwise."""
        created = await repository.create_user(
            {
                "username": f"fetch-{uuid4().hex[:6]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash("pass"),
            }
        )

        fetched = await repository.get_user_by_id(str(created["id"]))
        assert fetched is not None
        assert fetched["username"] == created["username"]

        assert await repository.get_user_by_id("9999999") is None

        with pytest.raises(InvalidUserIdError):
            await repository.get_user_by_id("not-an-int")

    async def test_update_user_applies_mutations(
        self,
        repository: PostgreSQLUserRepository,
        db_session: AsyncSession,
    ) -> None:
        """Updating a user should persist allowable field changes."""
        created = await repository.create_user(
            {
                "username": f"update-{uuid4().hex[:6]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash("initial"),
            }
        )

        updated = await repository.update_user(
            str(created["id"]),
            {"full_name": "Updated Name", "is_active": False, "username": f"updated-{uuid4().hex[:6]}"},
        )

        assert updated is not None
        assert updated["full_name"] == "Updated Name"
        assert updated["is_active"] is False
        assert updated["username"].startswith("updated-")

        persisted = await self._fetch_user(db_session, int(created["id"]))
        assert persisted is not None
        assert persisted.full_name == "Updated Name"
        assert persisted.is_active is False

    async def test_update_user_duplicate_email_rejected(self, repository: PostgreSQLUserRepository) -> None:
        """Email uniqueness should be enforced during updates."""
        user_one = await repository.create_user(
            {
                "username": f"user-one-{uuid4().hex[:4]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash("first"),
            }
        )
        user_two = await repository.create_user(
            {
                "username": f"user-two-{uuid4().hex[:4]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash("second"),
            }
        )

        with pytest.raises(EntityAlreadyExistsError):
            await repository.update_user(str(user_two["id"]), {"email": user_one["email"]})

    async def test_delete_user_removes_record(
        self,
        repository: PostgreSQLUserRepository,
        db_session: AsyncSession,
    ) -> None:
        """Deleting a user should remove the row and return True."""
        created = await repository.create_user(
            {
                "username": f"delete-{uuid4().hex[:6]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash("delete"),
            }
        )

        deleted = await repository.delete_user(str(created["id"]))
        assert deleted is True

        persisted = await self._fetch_user(db_session, int(created["id"]))
        assert persisted is None

        assert await repository.delete_user(str(created["id"])) is False

    async def test_verify_password_returns_user_on_match(self, repository: PostgreSQLUserRepository) -> None:
        """Password verification should succeed for matching credentials."""
        password = "p@ssword!"
        created = await repository.create_user(
            {
                "username": f"verify-{uuid4().hex[:6]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash(password),
            }
        )

        verified = await repository.verify_password(created["username"], password)
        assert verified is not None
        assert verified["id"] == created["id"]

        assert await repository.verify_password(created["username"], "wrong") is None

    async def test_update_last_login_sets_timestamp(
        self,
        repository: PostgreSQLUserRepository,
        db_session: AsyncSession,
    ) -> None:
        """Updating last login should persist a UTC timestamp."""
        created = await repository.create_user(
            {
                "username": f"last-login-{uuid4().hex[:6]}",
                "email": f"{uuid4().hex[:6]}@example.com",
                "hashed_password": pwd_context.hash("last-login"),
            }
        )

        await repository.update_last_login(str(created["id"]))

        persisted = await self._fetch_user(db_session, int(created["id"]))
        assert persisted is not None
        assert persisted.last_login is not None
        assert isinstance(persisted.last_login, datetime)
        assert persisted.last_login.tzinfo is UTC
