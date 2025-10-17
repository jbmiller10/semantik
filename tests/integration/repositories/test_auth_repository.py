"""Integration tests for PostgreSQLAuthRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from shared.database.exceptions import DatabaseOperationError, InvalidUserIdError
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import RefreshToken, User
from packages.webui.repositories.postgres.auth_repository import PostgreSQLAuthRepository

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestAuthRepositoryIntegration:
    """Exercise refresh token operations against the real database."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> PostgreSQLAuthRepository:
        return PostgreSQLAuthRepository(db_session)

    async def _fetch_tokens(self, db_session: AsyncSession, user_id: int) -> list[RefreshToken]:
        result = await db_session.execute(select(RefreshToken).where(RefreshToken.user_id == user_id))
        return list(result.scalars())

    async def test_save_refresh_token_persists_row(
        self,
        repository: PostgreSQLAuthRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        token_hash = repository._hash_token("refresh-token-1")  # noqa: SLF001 - verifying hashing path
        expires_at = datetime.now(UTC) + timedelta(days=30)

        await repository.save_refresh_token(str(test_user_db.id), token_hash, expires_at)

        tokens = await self._fetch_tokens(db_session, test_user_db.id)
        assert len(tokens) == 1
        assert tokens[0].token_hash == token_hash
        assert tokens[0].expires_at == expires_at
        assert tokens[0].is_revoked is False

    async def test_save_refresh_token_validates_user_id(self, repository: PostgreSQLAuthRepository) -> None:
        with pytest.raises(InvalidUserIdError):
            await repository.save_refresh_token("not-int", "hash", datetime.now(UTC))

    async def test_verify_refresh_token_success(
        self,
        repository: PostgreSQLAuthRepository,
        test_user_db: User,
    ) -> None:
        token_value = "refresh-token-2"
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token(token_value), datetime.now(UTC) + timedelta(days=7)
        )

        user_id = await repository.verify_refresh_token(token_value)
        assert user_id == str(test_user_db.id)

        # mark user inactive to ensure verification fails
        test_user_db.is_active = False
        await repository.session.flush()
        assert await repository.verify_refresh_token(token_value) is None

    async def test_verify_refresh_token_handles_invalid_cases(
        self,
        repository: PostgreSQLAuthRepository,
        test_user_db: User,
    ) -> None:
        token_value = "refresh-token-3"
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token(token_value), datetime.now(UTC) - timedelta(days=1)
        )
        # expired token should be ignored
        assert await repository.verify_refresh_token(token_value) is None

    async def test_revoke_refresh_token_marks_revoked(
        self,
        repository: PostgreSQLAuthRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        token_value = "refresh-token-4"
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token(token_value), datetime.now(UTC) + timedelta(days=1)
        )

        await repository.revoke_refresh_token(token_value)

        tokens = await self._fetch_tokens(db_session, test_user_db.id)
        assert tokens[0].is_revoked is True

    async def test_cleanup_expired_tokens_deletes_records(
        self,
        repository: PostgreSQLAuthRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token("active-token"), datetime.now(UTC) + timedelta(days=1)
        )
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token("expired-token"), datetime.now(UTC) - timedelta(days=1)
        )

        deleted = await repository.cleanup_expired_tokens()
        assert deleted == 1

        tokens = await self._fetch_tokens(db_session, test_user_db.id)
        assert len(tokens) == 1

    async def test_revoke_all_user_tokens_returns_count(
        self,
        repository: PostgreSQLAuthRepository,
        test_user_db: User,
    ) -> None:
        for suffix in range(3):
            await repository.save_refresh_token(
                str(test_user_db.id), repository._hash_token(f"bulk-{suffix}"), datetime.now(UTC) + timedelta(days=1)
            )

        revoked = await repository.revoke_all_user_tokens(str(test_user_db.id))
        assert revoked == 3
        assert await repository.get_active_token_count(str(test_user_db.id)) == 0

    async def test_get_active_token_count_ignores_revoked_and_expired(
        self,
        repository: PostgreSQLAuthRepository,
        db_session: AsyncSession,
        test_user_db: User,
    ) -> None:
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token("active"), datetime.now(UTC) + timedelta(days=1)
        )
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token("expired"), datetime.now(UTC) - timedelta(days=1)
        )
        await repository.save_refresh_token(
            str(test_user_db.id), repository._hash_token("revoked"), datetime.now(UTC) + timedelta(days=1)
        )
        await repository.revoke_refresh_token("revoked")

        count = await repository.get_active_token_count(str(test_user_db.id))
        assert count == 1

    async def test_create_token_round_trip(
        self,
        repository: PostgreSQLAuthRepository,
        test_user_db: User,
    ) -> None:
        expires_at = (datetime.now(UTC) + timedelta(hours=3)).isoformat()
        await repository.create_token(str(test_user_db.id), "api-token", expires_at)

        user_id = await repository.get_token_user_id("api-token")
        assert user_id == str(test_user_db.id)

        await repository.delete_token("api-token")
        assert await repository.get_token_user_id("api-token") is None

    async def test_verify_refresh_token_database_error_raises(
        self,
        repository: PostgreSQLAuthRepository,
    ) -> None:
        # Simulate database error by closing session; this should raise DatabaseOperationError
        await repository.session.close()
        with pytest.raises(DatabaseOperationError):
            await repository.verify_refresh_token("whatever")
