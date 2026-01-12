"""Integration tests for PostgreSQLAuthRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from shared.database.exceptions import InvalidUserIdError
from shared.database.models import RefreshToken
from webui.repositories.postgres.auth_repository import PostgreSQLAuthRepository


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestPostgreSQLAuthRepositoryIntegration:
    """Verify token storage flows against the real database."""

    @pytest.fixture()
    def repository(self, db_session):
        return PostgreSQLAuthRepository(db_session)

    async def _create_token(self, repository, user_id: str, token_value: str, expires_at: datetime) -> None:
        """Helper to create a token using the interface methods."""
        token_hash = repository.hash_refresh_token(token_value)
        await repository.save_refresh_token(user_id, token_hash, expires_at)

    async def test_create_and_verify_token(self, repository, db_session, test_user_db):
        """Tokens created for a user should verify successfully."""
        token_value = f"token-value-{uuid4().hex}"
        expires_at = datetime.now(UTC) + timedelta(days=7)

        await self._create_token(repository, str(test_user_db.id), token_value, expires_at)
        await db_session.commit()

        verified_user_id = await repository.verify_refresh_token(token_value)
        assert verified_user_id == str(test_user_db.id)

        result = await db_session.execute(select(RefreshToken).where(RefreshToken.user_id == test_user_db.id))
        token = result.scalar_one()
        assert token.is_revoked is False
        assert token.expires_at == expires_at

    async def test_verify_token_inactive_user_returns_none(self, repository, db_session, test_user_db):
        """Inactive users should not be authenticated even if token exists."""
        test_user_db.is_active = False
        await db_session.commit()

        token_value = f"inactive-token-{uuid4().hex}"
        await self._create_token(
            repository, str(test_user_db.id), token_value, datetime.now(UTC) + timedelta(days=1)
        )
        await db_session.commit()

        assert await repository.verify_refresh_token(token_value) is None

    async def test_revoke_token_marks_record(self, repository, db_session, test_user_db):
        """Revoking a token should mark it as revoked in the database."""
        token_value = f"revoke-me-{uuid4().hex}"
        await self._create_token(
            repository, str(test_user_db.id), token_value, datetime.now(UTC) + timedelta(days=1)
        )
        await db_session.commit()

        await repository.revoke_refresh_token(token_value)
        await db_session.commit()

        result = await db_session.execute(select(RefreshToken).where(RefreshToken.user_id == test_user_db.id))
        token = result.scalar_one()
        assert token.is_revoked is True

    async def test_cleanup_expired_tokens(self, repository, db_session, test_user_db):
        """Expired tokens should be removed by cleanup."""
        await self._create_token(
            repository,
            str(test_user_db.id),
            f"expired-token-{uuid4().hex}",
            datetime.now(UTC) - timedelta(days=1),
        )
        await db_session.commit()

        deleted_count = await repository.cleanup_expired_tokens()
        await db_session.commit()
        assert deleted_count == 1

    async def test_invalid_user_id_raises(self, repository):
        """Non-numeric identifiers should raise InvalidUserIdError."""
        with pytest.raises(InvalidUserIdError):
            await repository.save_refresh_token("not-an-int", "hash", datetime.now(UTC))
