"""Service for managing chunking strategies and configurations.

This service handles all business logic related to chunking strategies,
including creation, updates, and default strategy initialization.
"""

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import ChunkingStrategy

logger = logging.getLogger(__name__)


class ChunkingStrategyService:
    """Service for managing chunking strategies."""

    DEFAULT_STRATEGIES = [
        {
            "name": "character",
            "description": "Simple fixed-size character-based chunking using TokenTextSplitter",
            "is_active": True,
            "meta": {"supports_streaming": True},
        },
        {
            "name": "recursive",
            "description": "Smart sentence-aware splitting using SentenceSplitter",
            "is_active": True,
            "meta": {"supports_streaming": True, "recommended_default": True},
        },
        {
            "name": "markdown",
            "description": "Respects markdown structure using MarkdownNodeParser",
            "is_active": True,
            "meta": {"supports_streaming": True, "file_types": [".md", ".mdx"]},
        },
        {
            "name": "semantic",
            "description": "Uses AI embeddings to find natural boundaries using SemanticSplitterNodeParser",
            "is_active": False,
            "meta": {"supports_streaming": False, "requires_embeddings": True},
        },
        {
            "name": "hierarchical",
            "description": "Creates parent-child chunks using HierarchicalNodeParser",
            "is_active": False,
            "meta": {"supports_streaming": False},
        },
        {
            "name": "hybrid",
            "description": "Automatically selects strategy based on content",
            "is_active": False,
            "meta": {"supports_streaming": False},
        },
    ]

    def __init__(self, session: AsyncSession):
        """Initialize the service with a database session.

        Args:
            session: AsyncSession for database operations
        """
        self.session = session

    async def ensure_default_strategies(self) -> int:
        """Ensure default chunking strategies exist in the database.

        This method is idempotent and can be called multiple times safely.

        Returns:
            Number of strategies created
        """
        created_count = 0

        for strategy_data in self.DEFAULT_STRATEGIES:
            # Check if strategy already exists
            result = await self.session.execute(
                select(ChunkingStrategy).where(ChunkingStrategy.name == strategy_data["name"])
            )
            existing = result.scalar_one_or_none()

            if not existing:
                strategy = ChunkingStrategy(**strategy_data)
                self.session.add(strategy)
                created_count += 1
                logger.info(f"Created default chunking strategy: {strategy_data['name']}")
            else:
                logger.debug(f"Chunking strategy already exists: {strategy_data['name']}")

        if created_count > 0:
            await self.session.commit()
            logger.info(f"Created {created_count} default chunking strategies")

        return created_count

    async def get_all_strategies(self, active_only: bool = False) -> list[ChunkingStrategy]:
        """Get all chunking strategies.

        Args:
            active_only: If True, only return active strategies

        Returns:
            List of chunking strategies
        """
        query = select(ChunkingStrategy)
        if active_only:
            query = query.where(ChunkingStrategy.is_active.is_(True))

        result = await self.session.execute(query.order_by(ChunkingStrategy.name))
        return list(result.scalars().all())

    async def get_strategy_by_name(self, name: str) -> ChunkingStrategy | None:
        """Get a chunking strategy by name.

        Args:
            name: Strategy name

        Returns:
            ChunkingStrategy or None if not found
        """
        result = await self.session.execute(select(ChunkingStrategy).where(ChunkingStrategy.name == name))
        return result.scalar_one_or_none()

    async def get_default_strategy(self) -> ChunkingStrategy | None:
        """Get the recommended default chunking strategy.

        Returns:
            ChunkingStrategy or None if no default is set
        """
        result = await self.session.execute(
            select(ChunkingStrategy).where(
                ChunkingStrategy.is_active.is_(True),
                ChunkingStrategy.meta["recommended_default"].astext == "true",
            )
        )
        return result.scalar_one_or_none()

    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> ChunkingStrategy | None:
        """Update a chunking strategy.

        Args:
            strategy_id: Strategy ID
            updates: Dictionary of fields to update

        Returns:
            Updated strategy or None if not found
        """
        result = await self.session.execute(select(ChunkingStrategy).where(ChunkingStrategy.id == strategy_id))
        strategy = result.scalar_one_or_none()

        if not strategy:
            return None

        # Update allowed fields
        allowed_fields = {"description", "is_active", "meta"}
        for field, value in updates.items():
            if field in allowed_fields and hasattr(strategy, field):
                setattr(strategy, field, value)

        await self.session.commit()
        logger.info(f"Updated chunking strategy: {strategy.name}")
        return strategy
