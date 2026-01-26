"""Repository for agent conversation database operations.

This module provides async database access for agent conversations
following the repository pattern used throughout the codebase.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import selectinload

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import (
    DatabaseOperationError,
    EntityNotFoundError,
)
from webui.services.agent.models import (
    AgentConversation,
    ConversationStatus,
    ConversationUncertainty,
    UncertaintySeverity,
)

logger = logging.getLogger(__name__)


class AgentConversationRepository:
    """Repository for AgentConversation database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session.

        Args:
            session: AsyncSession instance for database operations
        """
        self.session = session

    async def create(
        self,
        user_id: int,
        source_id: int | None = None,
        inline_source_config: dict[str, Any] | None = None,
    ) -> AgentConversation:
        """Create a new agent conversation.

        Args:
            user_id: ID of the user creating the conversation
            source_id: Optional source to configure (existing source)
            inline_source_config: Optional inline source configuration for new sources.
                Stores: {"source_type": "...", "source_config": {...}, "_pending_secrets": {...}}
                The actual CollectionSource is created when the pipeline is applied.

        Returns:
            Created AgentConversation instance

        Raises:
            DatabaseOperationError: For database errors
        """
        try:
            conversation = AgentConversation(
                id=str(uuid4()),
                user_id=user_id,
                source_id=source_id,
                inline_source_config=inline_source_config,
                status=ConversationStatus.ACTIVE,
            )
            self.session.add(conversation)
            await self.session.flush()

            logger.info(f"Created conversation {conversation.id} for user {user_id}")
            return conversation

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}", exc_info=True)
            raise DatabaseOperationError("create", "agent_conversation", str(e)) from e

    async def get_by_id(
        self,
        conversation_id: str,
        include_uncertainties: bool = False,
    ) -> AgentConversation | None:
        """Get a conversation by ID.

        Args:
            conversation_id: UUID of the conversation
            include_uncertainties: Whether to eagerly load uncertainties

        Returns:
            AgentConversation instance or None if not found
        """
        try:
            query = select(AgentConversation).where(AgentConversation.id == conversation_id)

            if include_uncertainties:
                query = query.options(selectinload(AgentConversation.uncertainties))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(
                f"Failed to get conversation {conversation_id}: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("get", "agent_conversation", str(e)) from e

    async def get_by_id_for_user(
        self,
        conversation_id: str,
        user_id: int,
        include_uncertainties: bool = False,
    ) -> AgentConversation | None:
        """Get a conversation by ID with user ownership check.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            include_uncertainties: Whether to eagerly load uncertainties

        Returns:
            AgentConversation instance or None if not found or not owned
        """
        try:
            query = select(AgentConversation).where(
                AgentConversation.id == conversation_id,
                AgentConversation.user_id == user_id,
            )

            if include_uncertainties:
                query = query.options(selectinload(AgentConversation.uncertainties))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except Exception as e:
            logger.error(
                f"Failed to get conversation {conversation_id} for user {user_id}: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("get", "agent_conversation", str(e)) from e

    async def list_for_user(
        self,
        user_id: int,
        status: ConversationStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentConversation]:
        """List conversations for a user.

        Args:
            user_id: ID of the user
            status: Optional status filter
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of AgentConversation instances
        """
        try:
            query = select(AgentConversation).where(AgentConversation.user_id == user_id)

            if status is not None:
                query = query.where(AgentConversation.status == status)

            query = query.order_by(AgentConversation.updated_at.desc())
            query = query.limit(limit).offset(offset)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Failed to list conversations for user {user_id}: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("list", "agent_conversations", str(e)) from e

    async def update_status(
        self,
        conversation_id: str,
        user_id: int,
        status: ConversationStatus,
    ) -> AgentConversation:
        """Update conversation status.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            status: New status

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            conversation.status = status
            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Updated conversation {conversation_id} status to {status.value}")
            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to update conversation status: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    async def update_pipeline(
        self,
        conversation_id: str,
        user_id: int,
        pipeline_config: dict[str, Any],
    ) -> AgentConversation:
        """Update the current pipeline configuration.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            pipeline_config: Serialized PipelineDAG configuration

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            conversation.current_pipeline = pipeline_config
            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(f"Updated pipeline for conversation {conversation_id}")
            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to update conversation pipeline: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    async def update_source_analysis(
        self,
        conversation_id: str,
        user_id: int,
        source_analysis: dict[str, Any],
    ) -> AgentConversation:
        """Update the source analysis results.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            source_analysis: Results from SourceAnalyzer sub-agent

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            conversation.source_analysis = source_analysis
            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(f"Updated source analysis for conversation {conversation_id}")
            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to update source analysis: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    async def update_metadata(
        self,
        conversation_id: str,
        user_id: int,
        metadata: dict[str, Any],
    ) -> AgentConversation:
        """Update conversation extra_data (metadata).

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            metadata: Fields to merge into existing extra_data

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            # Merge with existing extra_data
            existing = conversation.extra_data or {}
            existing.update(metadata)
            conversation.extra_data = existing
            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.debug(f"Updated extra_data for conversation {conversation_id}")
            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to update conversation extra_data: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    async def update_summary(
        self,
        conversation_id: str,
        user_id: int,
        summary: str,
    ) -> AgentConversation:
        """Update the conversation summary for recovery.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            summary: Conversation summary text

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            conversation.summary = summary
            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to update conversation summary: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    async def set_collection(
        self,
        conversation_id: str,
        user_id: int,
        collection_id: str,
    ) -> AgentConversation:
        """Set the collection created from this conversation.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            collection_id: UUID of the created collection

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            conversation.collection_id = collection_id
            conversation.status = ConversationStatus.APPLIED
            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Linked conversation {conversation_id} to collection {collection_id}")
            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to set collection: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    async def set_source_id(
        self,
        conversation_id: str,
        user_id: int,
        source_id: int,
    ) -> AgentConversation:
        """Set the source_id on a conversation after creating from inline config.

        Also clears the _pending_secrets from inline_source_config.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            source_id: ID of the newly created CollectionSource

        Returns:
            Updated AgentConversation instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            conversation.source_id = source_id

            # Clear pending secrets from inline config (they're now stored encrypted)
            if conversation.inline_source_config and "_pending_secrets" in conversation.inline_source_config:
                # Create a copy without secrets
                cleaned_config = {k: v for k, v in conversation.inline_source_config.items() if k != "_pending_secrets"}
                conversation.inline_source_config = cleaned_config

            conversation.updated_at = datetime.now(UTC)
            await self.session.flush()

            logger.info(f"Set source_id {source_id} on conversation {conversation_id}")
            return conversation

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to set source_id: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "agent_conversation", str(e)) from e

    # Uncertainty methods

    async def add_uncertainty(
        self,
        conversation_id: str,
        user_id: int,
        severity: UncertaintySeverity,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> ConversationUncertainty:
        """Add an uncertainty to a conversation.

        Args:
            conversation_id: UUID of the conversation
            user_id: ID of the user (must own the conversation)
            severity: Uncertainty severity level
            message: Human-readable description
            context: Additional context data

        Returns:
            Created ConversationUncertainty instance

        Raises:
            EntityNotFoundError: If conversation not found or not owned by user
        """
        try:
            # Verify conversation exists and is owned by user
            conversation = await self.get_by_id_for_user(conversation_id, user_id)
            if not conversation:
                raise EntityNotFoundError("agent_conversation", conversation_id)

            uncertainty = ConversationUncertainty(
                id=str(uuid4()),
                conversation_id=conversation_id,
                severity=severity,
                message=message,
                context=context,
            )
            self.session.add(uncertainty)
            await self.session.flush()

            logger.info(f"Added {severity.value} uncertainty to conversation {conversation_id}")
            return uncertainty

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to add uncertainty: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("create", "conversation_uncertainty", str(e)) from e

    async def resolve_uncertainty(
        self,
        uncertainty_id: str,
        resolved_by: str,
    ) -> ConversationUncertainty:
        """Mark an uncertainty as resolved.

        Args:
            uncertainty_id: UUID of the uncertainty
            resolved_by: How it was resolved (user_confirmed, pipeline_adjusted, etc.)

        Returns:
            Updated ConversationUncertainty instance

        Raises:
            EntityNotFoundError: If uncertainty not found
        """
        try:
            result = await self.session.execute(
                select(ConversationUncertainty).where(ConversationUncertainty.id == uncertainty_id)
            )
            uncertainty = result.scalar_one_or_none()

            if not uncertainty:
                raise EntityNotFoundError("conversation_uncertainty", uncertainty_id)

            uncertainty.resolved = True
            uncertainty.resolved_by = resolved_by
            await self.session.flush()

            logger.info(f"Resolved uncertainty {uncertainty_id} by {resolved_by}")
            return uncertainty

        except EntityNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to resolve uncertainty: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("update", "conversation_uncertainty", str(e)) from e

    async def get_blocking_uncertainties(
        self,
        conversation_id: str,
    ) -> list[ConversationUncertainty]:
        """Get unresolved blocking uncertainties for a conversation.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            List of unresolved blocking uncertainties
        """
        try:
            result = await self.session.execute(
                select(ConversationUncertainty).where(
                    ConversationUncertainty.conversation_id == conversation_id,
                    ConversationUncertainty.severity == UncertaintySeverity.BLOCKING,
                    ConversationUncertainty.resolved == False,  # noqa: E712
                )
            )
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Failed to get blocking uncertainties: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("list", "conversation_uncertainties", str(e)) from e

    async def get_uncertainties(
        self,
        conversation_id: str,
        include_resolved: bool = False,
    ) -> list[ConversationUncertainty]:
        """Get all uncertainties for a conversation.

        Args:
            conversation_id: UUID of the conversation
            include_resolved: Whether to include resolved uncertainties

        Returns:
            List of uncertainties
        """
        try:
            query = select(ConversationUncertainty).where(ConversationUncertainty.conversation_id == conversation_id)

            if not include_resolved:
                query = query.where(ConversationUncertainty.resolved == False)  # noqa: E712

            query = query.order_by(ConversationUncertainty.created_at)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Failed to get uncertainties: {e}",
                exc_info=True,
            )
            raise DatabaseOperationError("list", "conversation_uncertainties", str(e)) from e
