"""Repository for user-scoped chunking configuration profiles."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select, update

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import ChunkingConfigProfile

logger = logging.getLogger(__name__)


class ChunkingConfigProfileRepository:
    """CRUD helpers for ``ChunkingConfigProfile`` records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_profile(
        self,
        *,
        user_id: int,
        name: str,
        strategy: str,
        config: dict[str, Any],
        description: str | None = None,
        is_default: bool | None = False,
        tags: list[str] | None = None,
    ) -> ChunkingConfigProfile:
        """Create or update a profile by (user, name)."""

        normalized_name = name.strip()

        # Clear other defaults only when explicitly setting a new default
        if is_default:
            await self.session.execute(
                update(ChunkingConfigProfile)
                .where(ChunkingConfigProfile.created_by == user_id)
                .values(is_default=False)
            )

        existing_stmt = select(ChunkingConfigProfile).where(
            ChunkingConfigProfile.created_by == user_id,
            func.lower(ChunkingConfigProfile.name) == normalized_name.lower(),
        )
        existing = (await self.session.execute(existing_stmt)).scalar_one_or_none()

        tags_payload: list[str] | None = None
        if tags:
            tags_payload = [t.strip() for t in tags if isinstance(t, str) and t.strip()]

        if existing:
            existing.name = normalized_name
            existing.description = description
            existing.strategy = strategy
            existing.config = config
            if is_default is not None:
                existing.is_default = is_default
            existing.tags = tags_payload
            logger.debug("Updated chunking config profile %s for user %s", existing.id, user_id)
            await self.session.flush()
            return existing

        profile = ChunkingConfigProfile(
            name=normalized_name,
            description=description,
            strategy=strategy,
            config=config,
            created_by=user_id,
            is_default=is_default,
            tags=tags_payload,
        )
        self.session.add(profile)
        await self.session.flush()
        return profile

    async def list_profiles(
        self,
        *,
        user_id: int,
        strategy: str | None = None,
        is_default: bool | None = None,
    ) -> list[ChunkingConfigProfile]:
        """Return profiles filtered by user/strategy/default flag."""

        stmt = select(ChunkingConfigProfile).where(ChunkingConfigProfile.created_by == user_id)

        if strategy:
            stmt = stmt.where(func.lower(ChunkingConfigProfile.strategy) == strategy.lower())
        if is_default is not None:
            stmt = stmt.where(ChunkingConfigProfile.is_default.is_(is_default))

        stmt = stmt.order_by(ChunkingConfigProfile.updated_at.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def increment_usage(self, profile_id: int) -> None:
        """Increment usage counter for the profile."""

        await self.session.execute(
            update(ChunkingConfigProfile)
            .where(ChunkingConfigProfile.id == profile_id)
            .values(usage_count=ChunkingConfigProfile.usage_count + 1)
        )
        await self.session.flush()

    async def clear_defaults(self, user_id: int) -> None:
        """Set all profiles for user to non-default."""

        await self.session.execute(
            update(ChunkingConfigProfile).where(ChunkingConfigProfile.created_by == user_id).values(is_default=False)
        )
        await self.session.flush()
