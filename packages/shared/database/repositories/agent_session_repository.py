"""Repository implementation for AgentSession and AgentSessionMessage models."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import Select, and_, delete, func, select, update

from shared.database.agent_session import AgentSession, AgentSessionMessage
from shared.database.exceptions import DatabaseOperationError, EntityNotFoundError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.agents.types import AgentMessage

logger = logging.getLogger(__name__)


class AgentSessionRepository:
    """Repository providing CRUD operations for agent sessions."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        agent_plugin_id: str,
        *,
        user_id: int | None = None,
        collection_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        title: str | None = None,
        parent_session_id: str | None = None,
    ) -> AgentSession:
        """Create a new agent session."""
        session = AgentSession(
            id=str(uuid4()),
            external_id=str(uuid4())[:8],  # Short ID for URLs
            agent_plugin_id=agent_plugin_id,
            user_id=user_id,
            collection_id=collection_id,
            agent_config=agent_config or {},
            title=title,
            parent_session_id=parent_session_id,
        )

        try:
            self.session.add(session)
            await self.session.flush()
            logger.info("Created agent session %s (external: %s)", session.id, session.external_id)
            return session
        except Exception as exc:
            logger.error("Failed to create agent session: %s", exc)
            raise DatabaseOperationError("create", "agent_session", str(exc)) from exc

    async def get_by_id(self, session_id: str) -> AgentSession | None:
        """Get session by internal ID."""
        stmt: Select[tuple[AgentSession]] = select(AgentSession).where(AgentSession.id == session_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_external_id(self, external_id: str) -> AgentSession | None:
        """Get session by external (URL-safe) ID."""
        stmt: Select[tuple[AgentSession]] = select(AgentSession).where(AgentSession.external_id == external_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_sdk_session_id(self, sdk_session_id: str) -> AgentSession | None:
        """Get session by SDK session ID (for resume)."""
        stmt: Select[tuple[AgentSession]] = select(AgentSession).where(AgentSession.sdk_session_id == sdk_session_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: int,
        *,
        status: str | None = None,
        collection_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AgentSession], int]:
        """List sessions for a user with pagination."""
        stmt: Select[tuple[AgentSession]] = (
            select(AgentSession)
            .where(AgentSession.user_id == user_id)
            .order_by(AgentSession.last_activity_at.desc())
            .limit(limit)
            .offset(offset)
        )

        if status:
            stmt = stmt.where(AgentSession.status == status)

        if collection_id:
            stmt = stmt.where(AgentSession.collection_id == collection_id)

        sessions = list((await self.session.execute(stmt)).scalars().all())

        # Count total
        count_stmt = select(func.count(AgentSession.id)).where(AgentSession.user_id == user_id)
        if status:
            count_stmt = count_stmt.where(AgentSession.status == status)
        if collection_id:
            count_stmt = count_stmt.where(AgentSession.collection_id == collection_id)

        total = await self.session.scalar(count_stmt) or 0

        return sessions, total

    async def add_message(
        self,
        session_id: str,
        message: AgentMessage,
    ) -> AgentSessionMessage:
        """Add a message to a session."""
        db_session = await self.get_by_id(session_id)
        if not db_session:
            raise EntityNotFoundError("agent_session", session_id)

        # Update session's message list
        db_session.add_message(message.to_dict())

        # Update stats if present
        if message.usage:
            db_session.update_stats(
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                cost_usd=message.cost_usd or 0,
            )

        # Create denormalized message record
        msg_record = AgentSessionMessage(
            id=str(uuid4()),
            session_id=session_id,
            message_id=message.id,
            sequence_number=message.sequence_number,
            role=message.role.value,
            type=message.type.value,
            content=message.content,
            tool_name=message.tool_name,
            tool_call_id=message.tool_call_id,
            tool_input=message.tool_input,
            tool_output=message.tool_output,
            model=message.model,
            input_tokens=message.usage.input_tokens if message.usage else None,
            output_tokens=message.usage.output_tokens if message.usage else None,
            cost_usd=int(message.cost_usd * 10000) if message.cost_usd else None,
        )

        self.session.add(msg_record)
        await self.session.flush()

        return msg_record

    async def get_messages(
        self,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        after_sequence: int | None = None,
    ) -> list[AgentSessionMessage]:
        """Get messages for a session."""
        stmt: Select[tuple[AgentSessionMessage]] = (
            select(AgentSessionMessage)
            .where(AgentSessionMessage.session_id == session_id)
            .order_by(AgentSessionMessage.sequence_number)
            .limit(limit)
            .offset(offset)
        )

        if after_sequence is not None:
            stmt = stmt.where(AgentSessionMessage.sequence_number > after_sequence)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_sdk_session_id(
        self,
        session_id: str,
        sdk_session_id: str,
    ) -> None:
        """Update SDK session ID (after first execution)."""
        await self.session.execute(
            update(AgentSession).where(AgentSession.id == session_id).values(sdk_session_id=sdk_session_id)
        )
        await self.session.flush()

    async def update_title(self, session_id: str, title: str) -> None:
        """Update session title."""
        await self.session.execute(update(AgentSession).where(AgentSession.id == session_id).values(title=title))
        await self.session.flush()

    async def archive(self, session_id: str) -> None:
        """Archive a session."""
        await self.session.execute(
            update(AgentSession)
            .where(AgentSession.id == session_id)
            .values(status="archived", archived_at=datetime.now(UTC))
        )
        await self.session.flush()

    async def delete(self, session_id: str) -> None:
        """Soft-delete a session."""
        await self.session.execute(update(AgentSession).where(AgentSession.id == session_id).values(status="deleted"))
        await self.session.flush()

    async def hard_delete(self, session_id: str) -> None:
        """Permanently delete a session and its messages."""
        # Delete messages first (due to FK constraint)
        await self.session.execute(delete(AgentSessionMessage).where(AgentSessionMessage.session_id == session_id))
        # Delete session
        await self.session.execute(delete(AgentSession).where(AgentSession.id == session_id))
        await self.session.flush()

    async def cleanup_old_sessions(
        self,
        older_than_days: int = 90,
        status: str = "deleted",
    ) -> int:
        """Delete sessions older than specified days with given status."""
        cutoff = datetime.now(UTC) - timedelta(days=older_than_days)

        # Get session IDs to delete
        result = await self.session.execute(
            select(AgentSession.id).where(
                and_(
                    AgentSession.status == status,
                    AgentSession.updated_at < cutoff,
                )
            )
        )
        session_ids = [row[0] for row in result.fetchall()]

        if not session_ids:
            return 0

        # Delete messages
        await self.session.execute(delete(AgentSessionMessage).where(AgentSessionMessage.session_id.in_(session_ids)))

        # Delete sessions
        await self.session.execute(delete(AgentSession).where(AgentSession.id.in_(session_ids)))

        await self.session.flush()

        logger.info("Cleaned up %d old agent sessions", len(session_ids))
        return len(session_ids)

    async def increment_fork_count(self, session_id: str) -> None:
        """Increment the fork count for a session."""
        await self.session.execute(
            update(AgentSession).where(AgentSession.id == session_id).values(fork_count=AgentSession.fork_count + 1)
        )
        await self.session.flush()
