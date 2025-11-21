"""Integration tests for PostgreSQLApiKeyRepository with the real database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from shared.database.exceptions import EntityNotFoundError, InvalidUserIdError
from sqlalchemy import select

from shared.database.models import ApiKey
from webui.repositories.postgres.api_key_repository import PostgreSQLApiKeyRepository


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestPostgreSQLApiKeyRepositoryIntegration:
    """Exercise API key repository behaviour end-to-end."""

    @pytest.fixture()
    def repository(self, db_session):
        """Repository under test backed by the integration session."""
        return PostgreSQLApiKeyRepository(db_session)

    async def test_create_api_key_persists_and_returns_plain_key(self, repository, db_session, test_user_db):
        """Creating an API key should persist the hashed record and return the plain token once."""
        created = await repository.create_api_key(str(test_user_db.id), "integration key", permissions={"read": True})
        await db_session.commit()

        assert created["user_id"] == test_user_db.id
        assert created["permissions"] == {"read": True}
        plain_key = created["api_key"]
        assert len(plain_key) > 30  # token_urlsafe(32) size check

        result = await db_session.execute(select(ApiKey).where(ApiKey.id == created["id"]))
        persisted = result.scalar_one()
        assert persisted.user_id == test_user_db.id
        assert persisted.permissions == {"read": True}
        assert persisted.is_active is True
        assert persisted.key_hash == repository._hash_api_key(plain_key)

    async def test_verify_api_key_returns_user_and_updates_last_used(self, repository, db_session, test_user_db):
        """Verification should succeed for active, non-expired keys and record last_used updates."""
        created = await repository.create_api_key(str(test_user_db.id), "verify key")
        plain_key = created["api_key"]
        await db_session.commit()

        verified = await repository.verify_api_key(plain_key)
        assert verified is not None
        assert verified["id"] == created["id"]
        assert verified["user_id"] == test_user_db.id

        # Trigger a second verification to exercise last_used update
        await repository.verify_api_key(plain_key)
        await db_session.commit()

        result = await db_session.execute(select(ApiKey).where(ApiKey.id == created["id"]))
        persisted = result.scalar_one()
        assert persisted.last_used_at is not None

    async def test_list_and_delete_user_keys(self, repository, db_session, test_user_db):
        """Listing should include created keys and deletion should remove them."""
        first = await repository.create_api_key(str(test_user_db.id), "list key 1")
        second = await repository.create_api_key(str(test_user_db.id), "list key 2")
        await db_session.commit()

        keys = await repository.list_user_api_keys(str(test_user_db.id))
        returned_ids = {item["id"] for item in keys}
        assert {first["id"], second["id"]} <= returned_ids

        deleted = await repository.delete_api_key(first["id"])
        await db_session.commit()
        assert deleted is True

        remaining = await repository.list_user_api_keys(str(test_user_db.id))
        assert first["id"] not in {item["id"] for item in remaining}

    async def test_create_api_key_invalid_user_id_raises(self, repository):
        """Non-numeric identifiers should raise InvalidUserIdError."""
        with pytest.raises(InvalidUserIdError):
            await repository.create_api_key("invalid", "bad key")

    async def test_create_api_key_missing_user_raises(self, repository, db_session):
        """Unknown users should raise EntityNotFoundError before insert."""
        fake_user_id = str(uuid4().int % 2_147_483_647)
        with pytest.raises(EntityNotFoundError):
            await repository.create_api_key(fake_user_id, "missing user")

    async def test_cleanup_expired_keys_removes_records(self, repository, db_session, test_user_db):
        """Expired keys should be deleted by cleanup."""
        created = await repository.create_api_key(str(test_user_db.id), "expiring key")
        await db_session.commit()

        # Manually expire the key
        await repository.update_api_key(created["id"], {"expires_at": datetime.now(UTC) - timedelta(days=1)})
        await db_session.commit()

        deleted_count = await repository.cleanup_expired_keys()
        await db_session.commit()
        assert deleted_count == 1

        remaining = await repository.get_api_key(created["id"])
        assert remaining is None
