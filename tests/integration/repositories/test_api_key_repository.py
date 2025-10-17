"""Integration tests for the PostgreSQLApiKeyRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from shared.database.exceptions import EntityNotFoundError, InvalidUserIdError
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import ApiKey, User
from packages.webui.repositories.postgres.api_key_repository import PostgreSQLApiKeyRepository

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestApiKeyRepositoryIntegration:
    """Exercise API key repository behavior against the real database."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> PostgreSQLApiKeyRepository:
        return PostgreSQLApiKeyRepository(db_session)

    async def _get_api_keys_for_user(self, db_session: AsyncSession, user_id: int) -> list[ApiKey]:
        result = await db_session.execute(select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at))
        return list(result.scalars())

    async def test_create_api_key_persists_hash_and_returns_plain_key(
        self,
        repository: PostgreSQLApiKeyRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        payload = await repository.create_api_key(str(test_user_db.id), "primary", {"search": True})

        assert payload["user_id"] == test_user_db.id
        assert payload["api_key"]
        assert payload["permissions"] == {"search": True}

        stored = await self._get_api_keys_for_user(db_session, test_user_db.id)
        assert len(stored) == 1
        assert stored[0].key_hash != payload["api_key"]
        assert stored[0].is_active is True

    async def test_create_api_key_missing_user_fails(
        self,
        repository: PostgreSQLApiKeyRepository,
    ) -> None:
        with pytest.raises(EntityNotFoundError):
            await repository.create_api_key("999999", "missing")

        with pytest.raises(InvalidUserIdError):
            await repository.create_api_key("not-int", "missing")

    async def test_get_api_key_and_hash_lookup(
        self,
        repository: PostgreSQLApiKeyRepository,
        test_user_db: User,
    ) -> None:
        created = await repository.create_api_key(str(test_user_db.id), "lookup")

        fetched = await repository.get_api_key(created["id"])
        assert fetched is not None
        assert fetched["id"] == created["id"]
        assert fetched["user"]["username"] == test_user_db.username

        key_hash = repository._hash_api_key(created["api_key"])  # noqa: SLF001 - test verifying behavior
        by_hash = await repository.get_api_key_by_hash(key_hash)
        assert by_hash is not None
        assert by_hash["id"] == created["id"]

    async def test_list_user_api_keys_orders_newest_first(
        self,
        repository: PostgreSQLApiKeyRepository,
        test_user_db: User,
    ) -> None:
        first = await repository.create_api_key(str(test_user_db.id), f"first-{uuid4().hex[:4]}")
        second = await repository.create_api_key(str(test_user_db.id), f"second-{uuid4().hex[:4]}")

        keys = await repository.list_user_api_keys(str(test_user_db.id))
        assert [key["id"] for key in keys] == [second["id"], first["id"]]

    async def test_update_api_key_modifies_allowed_fields(
        self,
        repository: PostgreSQLApiKeyRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        created = await repository.create_api_key(str(test_user_db.id), "updatable")
        expires_at = datetime.now(UTC) + timedelta(days=10)

        updated = await repository.update_api_key(
            created["id"], {"name": "updated", "is_active": False, "expires_at": expires_at}
        )

        assert updated is not None
        assert updated["name"] == "updated"
        assert updated["is_active"] is False
        assert updated["expires_at"].startswith(expires_at.isoformat()[:19])

        stored = await repository.get_api_key(created["id"])
        assert stored is not None
        assert stored["is_active"] is False

    async def test_delete_api_key_removes_row(
        self,
        repository: PostgreSQLApiKeyRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        created = await repository.create_api_key(str(test_user_db.id), "delete-me")

        assert await repository.delete_api_key(created["id"]) is True
        assert await repository.delete_api_key(created["id"]) is False

        stored = await self._get_api_keys_for_user(db_session, test_user_db.id)
        assert stored == []

    async def test_verify_api_key_checks_activity_and_expiration(
        self,
        repository: PostgreSQLApiKeyRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        created = await repository.create_api_key(str(test_user_db.id), "verify")
        api_key = created["api_key"]

        verified = await repository.verify_api_key(api_key)
        assert verified is not None
        assert verified["id"] == created["id"]

        # Expire the key and ensure verification fails
        await repository.update_api_key(created["id"], {"expires_at": datetime.now(UTC) - timedelta(days=1)})
        assert await repository.verify_api_key(api_key) is None

        # Reactivate with future expiration and ensure last_used timestamp is updated
        await repository.update_api_key(
            created["id"], {"expires_at": datetime.now(UTC) + timedelta(days=1), "is_active": True}
        )
        await repository.verify_api_key(api_key)

        stored_key = await repository.get_api_key(created["id"])
        assert stored_key is not None
        assert stored_key["last_used_at"] is not None

        # Deactivate the owning user and ensure verification fails
        test_user_db.is_active = False
        await db_session.flush()

        assert await repository.verify_api_key(api_key) is None

    async def test_cleanup_expired_keys_purges_records(
        self,
        repository: PostgreSQLApiKeyRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        await repository.create_api_key(str(test_user_db.id), "active")
        expired = await repository.create_api_key(str(test_user_db.id), "expired")
        await repository.update_api_key(expired["id"], {"expires_at": datetime.now(UTC) - timedelta(days=2)})

        purged = await repository.cleanup_expired_keys()
        assert purged == 1

        remaining = await self._get_api_keys_for_user(db_session, test_user_db.id)
        assert len(remaining) == 1
