"""MCP Profile Service for managing MCP search profile configurations."""

import logging
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, EntityNotFoundError
from shared.database.models import Collection, MCPProfile, MCPProfileCollection
from webui.api.v2.mcp_schemas import MCPClientConfig, MCPProfileCreate, MCPProfileUpdate

logger = logging.getLogger(__name__)


class MCPProfileService:
    """Service for managing MCP profile business logic."""

    def __init__(self, db_session: AsyncSession):
        """Initialize the MCP profile service.

        Args:
            db_session: AsyncSession instance for database operations.
        """
        self.db_session = db_session

    async def create(self, data: MCPProfileCreate, owner_id: int) -> MCPProfile:
        """Create a new MCP profile.

        Args:
            data: Profile creation data.
            owner_id: ID of the user creating the profile.

        Returns:
            Created MCPProfile instance.

        Raises:
            EntityAlreadyExistsError: If profile name already exists for user.
            EntityNotFoundError: If any collection_id doesn't exist.
            AccessDeniedError: If user doesn't own all specified collections.
        """
        # Check for duplicate name
        existing = await self._get_by_name(data.name, owner_id)
        if existing:
            raise EntityAlreadyExistsError("MCPProfile", data.name)

        # Validate collection access
        await self._validate_collection_access(data.collection_ids, owner_id)

        # Create profile
        profile = MCPProfile(
            id=str(uuid4()),
            name=data.name,
            description=data.description,
            owner_id=owner_id,
            enabled=data.enabled,
            search_type=data.search_type,
            result_count=data.result_count,
            use_reranker=data.use_reranker,
            score_threshold=data.score_threshold,
            hybrid_alpha=data.hybrid_alpha,
            search_mode=data.search_mode,
            rrf_k=data.rrf_k,
        )
        self.db_session.add(profile)

        # Create collection associations with ordering
        for order, collection_id in enumerate(data.collection_ids):
            assoc = MCPProfileCollection(
                profile_id=profile.id,
                collection_id=collection_id,
                order=order,
            )
            self.db_session.add(assoc)

        await self.db_session.flush()

        # Reload with collections
        return await self._get_with_collections(profile.id)

    async def list_for_user(
        self,
        user_id: int,
        *,
        enabled_only: bool = False,
    ) -> list[MCPProfile]:
        """List all MCP profiles for a user.

        Args:
            user_id: ID of the profile owner.
            enabled_only: If True, only return enabled profiles.

        Returns:
            List of MCPProfile instances with collections loaded.
        """
        stmt = (
            select(MCPProfile)
            .where(MCPProfile.owner_id == user_id)
            .options(selectinload(MCPProfile.collections))
            .order_by(MCPProfile.name)
        )

        if enabled_only:
            stmt = stmt.where(MCPProfile.enabled == True)  # noqa: E712

        result = await self.db_session.execute(stmt)
        return list(result.scalars().all())

    async def get(self, profile_id: str, owner_id: int) -> MCPProfile:
        """Get a profile by ID with owner check.

        Args:
            profile_id: UUID of the profile.
            owner_id: ID of the expected owner.

        Returns:
            MCPProfile instance with collections loaded.

        Raises:
            EntityNotFoundError: If profile doesn't exist.
            AccessDeniedError: If user doesn't own the profile.
        """
        profile = await self._get_with_collections(profile_id)
        if profile is None:
            raise EntityNotFoundError("MCPProfile", profile_id)

        if profile.owner_id != owner_id:
            raise AccessDeniedError(str(owner_id), "MCPProfile", profile_id)

        return profile

    async def update(
        self,
        profile_id: str,
        data: MCPProfileUpdate,
        owner_id: int,
    ) -> MCPProfile:
        """Update an MCP profile.

        Args:
            profile_id: UUID of the profile to update.
            data: Profile update data.
            owner_id: ID of the profile owner.

        Returns:
            Updated MCPProfile instance.

        Raises:
            EntityNotFoundError: If profile doesn't exist.
            AccessDeniedError: If user doesn't own the profile.
            EntityAlreadyExistsError: If new name conflicts with existing profile.
        """
        profile = await self.get(profile_id, owner_id)

        # Check for name conflict if name is changing
        if data.name is not None and data.name != profile.name:
            existing = await self._get_by_name(data.name, owner_id)
            if existing:
                raise EntityAlreadyExistsError("MCPProfile", data.name)

        # Validate new collection access if provided
        if data.collection_ids is not None:
            await self._validate_collection_access(data.collection_ids, owner_id)

            # Delete existing associations
            await self.db_session.execute(
                MCPProfileCollection.__table__.delete().where(MCPProfileCollection.profile_id == profile_id)
            )

            # Create new associations with ordering
            for order, collection_id in enumerate(data.collection_ids):
                assoc = MCPProfileCollection(
                    profile_id=profile_id,
                    collection_id=collection_id,
                    order=order,
                )
                self.db_session.add(assoc)

        # Update scalar fields
        update_fields = data.model_dump(exclude_unset=True, exclude={"collection_ids"})
        for field, value in update_fields.items():
            setattr(profile, field, value)

        await self.db_session.flush()

        # Reload with collections
        return await self._get_with_collections(profile_id)

    async def delete(self, profile_id: str, owner_id: int) -> None:
        """Delete an MCP profile.

        Args:
            profile_id: UUID of the profile to delete.
            owner_id: ID of the profile owner.

        Raises:
            EntityNotFoundError: If profile doesn't exist.
            AccessDeniedError: If user doesn't own the profile.
        """
        profile = await self.get(profile_id, owner_id)
        await self.db_session.delete(profile)
        await self.db_session.flush()

    async def get_config(self, profile_id: str, owner_id: int, webui_url: str) -> MCPClientConfig:
        """Get MCP client configuration for a profile.

        Args:
            profile_id: UUID of the profile.
            owner_id: ID of the profile owner.
            webui_url: Base URL of the Semantik WebUI.

        Returns:
            MCPClientConfig with server configuration.

        Raises:
            EntityNotFoundError: If profile doesn't exist.
            AccessDeniedError: If user doesn't own the profile.
        """
        profile = await self.get(profile_id, owner_id)

        return MCPClientConfig(
            server_name=f"semantik-{profile.name}",
            command="semantik-mcp",
            args=["serve", "--profile", profile.name],
            env={
                "SEMANTIK_WEBUI_URL": webui_url,
                "SEMANTIK_AUTH_TOKEN": "<your-access-token-or-api-key>",
            },
        )

    async def _get_by_name(self, name: str, owner_id: int) -> MCPProfile | None:
        """Get a profile by name for a user."""
        stmt = select(MCPProfile).where(
            MCPProfile.name == name,
            MCPProfile.owner_id == owner_id,
        )
        result = await self.db_session.execute(stmt)
        return result.scalar_one_or_none()

    async def _get_with_collections(self, profile_id: str) -> MCPProfile | None:
        """Get a profile with collections eagerly loaded and ordered."""
        # Load profile with collections via selectinload
        stmt = select(MCPProfile).where(MCPProfile.id == profile_id).options(selectinload(MCPProfile.collections))
        result = await self.db_session.execute(stmt)
        profile = result.scalar_one_or_none()

        if profile is None:
            return None

        # Get ordering from junction table
        order_stmt = select(MCPProfileCollection.collection_id, MCPProfileCollection.order).where(
            MCPProfileCollection.profile_id == profile_id
        )
        order_result = await self.db_session.execute(order_stmt)
        order_map = {row.collection_id: row.order for row in order_result}

        # Sort collections by order
        profile.collections.sort(key=lambda c: order_map.get(c.id, 0))

        return profile

    async def _validate_collection_access(
        self,
        collection_ids: list[str],
        owner_id: int,
    ) -> None:
        """Validate that user owns all specified collections.

        Args:
            collection_ids: List of collection UUIDs.
            owner_id: ID of the user.

        Raises:
            EntityNotFoundError: If any collection doesn't exist.
            AccessDeniedError: If user doesn't own any collection.
        """
        if not collection_ids:
            return

        stmt = select(Collection).where(Collection.id.in_(collection_ids))
        result = await self.db_session.execute(stmt)
        collections = {c.id: c for c in result.scalars().all()}

        # Check all collections exist
        missing = set(collection_ids) - set(collections.keys())
        if missing:
            raise EntityNotFoundError("Collection", ", ".join(missing))

        # Check ownership
        for collection_id in collection_ids:
            collection = collections[collection_id]
            if collection.owner_id != owner_id:
                raise AccessDeniedError(str(owner_id), "Collection", collection_id)
