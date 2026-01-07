"""Service layer for agent operations.

This module orchestrates agent execution, session management, and tool
resolution following the three-layer architecture (Router -> Service -> Repository).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from shared.agents.exceptions import (
    AgentError,
    AgentExecutionError,
    AgentInterruptedError,
    SessionNotFoundError,
)
from shared.agents.tools.registry import get_tool_registry
from shared.agents.types import (
    AgentCapabilities,
    AgentContext,
    AgentMessage,
    AgentUseCase,
    MessageRole,
    MessageType,
)
from shared.database.repositories.agent_session_repository import AgentSessionRepository
from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.agents.tools.base import AgentTool
    from shared.database.agent_session import AgentSession
    from shared.plugins.types.agent import AgentPlugin

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing agent plugin operations.

    Responsibilities:
    - Load and cache agent plugin instances
    - Create/resume sessions via AgentSessionRepository
    - Resolve tools from ToolRegistry
    - Execute agents and stream responses
    - Persist messages to database
    - Support interruption, forking, archiving, deletion

    Usage:
        service = AgentService(db, session_repo)
        async for msg in service.execute("claude-agent", "Hello"):
            print(msg.content)
    """

    def __init__(
        self,
        db: AsyncSession,
        session_repo: AgentSessionRepository | None = None,
    ) -> None:
        """Initialize the agent service.

        Args:
            db: AsyncSession for database operations.
            session_repo: Optional pre-built repository (for testing).
        """
        self._db = db
        self._session_repo = session_repo or AgentSessionRepository(db)

        # Plugin instance cache: keyed by "{plugin_id}:{config_hash}"
        self._instances: dict[str, AgentPlugin] = {}

        # Active executions for interruption support: keyed by session_id
        self._active_executions: dict[str, AgentPlugin] = {}

        # Lock for thread-safe instance cache access
        self._instance_lock = asyncio.Lock()

    # =========================================================================
    # Plugin Management
    # =========================================================================

    async def _get_instance(
        self,
        plugin_id: str,
        config: dict[str, Any] | None = None,
    ) -> AgentPlugin:
        """Get or create a cached plugin instance.

        Plugin instances are cached by plugin_id and config hash to avoid
        repeated initialization overhead.

        Args:
            plugin_id: The registered plugin identifier.
            config: Optional plugin configuration.

        Returns:
            Initialized AgentPlugin instance.

        Raises:
            ValueError: If plugin not found in registry.
            AgentInitializationError: If plugin fails to initialize.
        """
        # Ensure agent plugins are loaded
        load_plugins(plugin_types={"agent"})

        # Build cache key from plugin_id and config hash
        config_hash = self._compute_config_hash(config)
        cache_key = f"{plugin_id}:{config_hash}"

        async with self._instance_lock:
            if cache_key in self._instances:
                return self._instances[cache_key]

            # Look up in registry
            record = plugin_registry.get("agent", plugin_id)
            if not record:
                raise ValueError(f"Agent plugin not found: {plugin_id}")

            # Instantiate and initialize
            instance: AgentPlugin = record.plugin_class(config)
            await instance.initialize(config)

            self._instances[cache_key] = instance
            logger.info(
                "Initialized agent plugin: %s (config_hash=%s)",
                plugin_id,
                config_hash,
            )

            return instance

    @staticmethod
    def _compute_config_hash(config: dict[str, Any] | None) -> str:
        """Compute stable hash for config dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            8-character hex hash or "default" if no config.
        """
        if not config:
            return "default"

        # Sort keys for stable serialization
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:8]

    # =========================================================================
    # Agent Discovery
    # =========================================================================

    async def list_agents(self) -> list[dict[str, Any]]:
        """List all available agent plugins.

        Returns:
            List of agent metadata dictionaries.
        """
        load_plugins(plugin_types={"agent"})
        records = plugin_registry.list_records(plugin_type="agent")

        return [
            {
                "id": r.plugin_id,
                "version": r.plugin_version,
                "manifest": r.manifest.to_dict() if r.manifest else None,
                "capabilities": r.plugin_class.get_capabilities().to_dict(),
                "use_cases": [uc.value for uc in r.plugin_class.supported_use_cases()],
            }
            for r in records
        ]

    async def get_agent(self, plugin_id: str) -> dict[str, Any] | None:
        """Get detailed agent information.

        Args:
            plugin_id: The agent plugin identifier.

        Returns:
            Agent details or None if not found.
        """
        load_plugins(plugin_types={"agent"})
        record = plugin_registry.get("agent", plugin_id)

        if not record:
            return None

        return {
            "id": record.plugin_id,
            "version": record.plugin_version,
            "manifest": record.manifest.to_dict() if record.manifest else None,
            "capabilities": record.plugin_class.get_capabilities().to_dict(),
            "use_cases": [uc.value for uc in record.plugin_class.supported_use_cases()],
            "config_schema": record.plugin_class.get_config_schema(),
        }

    async def find_agent_for_use_case(self, use_case: AgentUseCase) -> str | None:
        """Find the best agent for a specific use case.

        Args:
            use_case: The target use case.

        Returns:
            Plugin ID of suitable agent, or None if none found.
        """
        load_plugins(plugin_types={"agent"})
        records = plugin_registry.list_records(plugin_type="agent")

        for record in records:
            supported = record.plugin_class.supported_use_cases()
            if use_case in supported:
                return str(record.plugin_id)
        return None

    async def get_capabilities(self, plugin_id: str) -> AgentCapabilities | None:
        """Get capabilities for a specific agent.

        Args:
            plugin_id: The agent plugin identifier.

        Returns:
            AgentCapabilities or None if not found.
        """
        load_plugins(plugin_types={"agent"})
        record = plugin_registry.get("agent", plugin_id)

        if not record:
            return None

        return record.plugin_class.get_capabilities()

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute(
        self,
        plugin_id: str,
        prompt: str,
        *,
        context: AgentContext | None = None,
        session_id: str | None = None,
        config: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = True,
    ) -> AsyncIterator[AgentMessage]:
        """Execute an agent and stream responses.

        This is the primary execution method. It:
        1. Resolves or creates a session
        2. Gets or creates a plugin instance
        3. Resolves requested tools from the registry
        4. Executes the agent, streaming messages
        5. Persists complete messages to the database
        6. Handles errors with proper error messages

        Transaction Boundaries:
        - Session creation is committed BEFORE execution starts
        - Each complete message is committed after persistence
        - Errors are persisted before re-raising

        Args:
            plugin_id: Agent plugin identifier.
            prompt: User message or task description.
            context: Optional runtime context.
            session_id: External ID to resume existing session.
            config: Plugin configuration overrides.
            tools: Tool names to make available.
            system_prompt: Override default system prompt.
            model: Override default model.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum output tokens.
            stream: Whether to stream partial responses.

        Yields:
            AgentMessage objects (partials and finals).

        Raises:
            ValueError: If session not found or plugin invalid.
            SessionNotFoundError: If session_id doesn't exist.
            AgentExecutionError: If execution fails.
            AgentInterruptedError: If interrupted.
        """
        # Phase 1: Resolve session
        db_session = await self._resolve_session(
            plugin_id=plugin_id,
            session_id=session_id,
            config=config,
            context=context,
        )

        # Phase 2: Get plugin instance
        instance = await self._get_instance(plugin_id, config)

        # Phase 3: Resolve tools
        tool_instances = self._resolve_tools(tools)

        # Phase 4: Build execution context
        exec_context = self._build_execution_context(
            context=context,
            db_session=db_session,
        )

        # Phase 5: Register for interruption
        self._active_executions[str(db_session.id)] = instance

        try:
            # Phase 6: Execute and stream
            async for message in self._execute_with_persistence(
                instance=instance,
                prompt=prompt,
                db_session=db_session,
                context=exec_context,
                tools=tool_instances,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            ):
                yield message

        except AgentInterruptedError:
            # User-initiated interruption - record and re-raise
            await self._persist_interruption(db_session)
            raise

        except AgentError:
            # Agent-specific errors - already structured, re-raise
            raise

        except Exception as e:
            # Unexpected errors - wrap and persist
            await self._persist_error(db_session, e)
            raise AgentExecutionError(
                f"Execution failed: {e}",
                adapter=plugin_id,
                cause=str(e),
            ) from e

        finally:
            # Phase 7: Cleanup
            self._active_executions.pop(str(db_session.id), None)

    async def _resolve_session(
        self,
        plugin_id: str,
        session_id: str | None,
        config: dict[str, Any] | None,
        context: AgentContext | None,
    ) -> AgentSession:
        """Resolve existing session or create new one.

        Args:
            plugin_id: Agent plugin identifier.
            session_id: External session ID for resumption.
            config: Plugin configuration.
            context: Runtime context with user/collection info.

        Returns:
            AgentSession (existing or newly created).

        Raises:
            SessionNotFoundError: If session_id provided but not found.
        """
        if session_id:
            # Resume existing session
            db_session = await self._session_repo.get_by_external_id(session_id)
            if not db_session:
                raise SessionNotFoundError(
                    f"Session not found: {session_id}",
                    session_id=session_id,
                )
            return db_session

        # Create new session
        db_session = await self._session_repo.create(
            agent_plugin_id=plugin_id,
            user_id=context.user_id if context else None,
            collection_id=context.collection_id if context else None,
            agent_config=config,
        )

        # CRITICAL: Commit BEFORE execution to avoid race conditions
        await self._db.commit()

        logger.info(
            "Created agent session: %s (external: %s)",
            db_session.id,
            db_session.external_id,
        )

        return db_session

    def _resolve_tools(self, tool_names: list[str] | None) -> list[AgentTool] | None:
        """Resolve tool names to tool instances.

        Args:
            tool_names: List of tool names to resolve.

        Returns:
            List of AgentTool instances, or None if no tools requested.
        """
        if not tool_names:
            return None

        registry = get_tool_registry()
        tools = registry.get_by_names(tool_names)

        if len(tools) != len(tool_names):
            resolved = {t.name for t in tools}
            missing = set(tool_names) - resolved
            logger.warning("Some tools not found: %s", missing)

        return tools if tools else None

    def _build_execution_context(
        self,
        context: AgentContext | None,
        db_session: AgentSession,
    ) -> AgentContext:
        """Build complete execution context.

        Args:
            context: Optional incoming context.
            db_session: Database session.

        Returns:
            Complete AgentContext for execution.
        """
        if context is None:
            context = AgentContext(request_id=str(db_session.id))

        # Set session ID in context
        context.session_id = str(db_session.id)

        return context

    async def _execute_with_persistence(
        self,
        instance: AgentPlugin,
        prompt: str,
        db_session: AgentSession,
        context: AgentContext,
        tools: list[AgentTool] | None,
        system_prompt: str | None,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> AsyncIterator[AgentMessage]:
        """Execute agent with message persistence.

        Streams messages from the agent, persisting complete messages
        to the database.

        Args:
            instance: The agent plugin instance.
            prompt: User prompt.
            db_session: Database session record.
            context: Execution context.
            tools: Resolved tool instances.
            system_prompt: Optional system prompt override.
            model: Optional model override.
            temperature: Optional temperature.
            max_tokens: Optional max tokens.
            stream: Whether to stream.

        Yields:
            AgentMessage objects.
        """
        # Add user message to session
        user_message = AgentMessage(
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content=prompt,
            sequence_number=db_session.message_count,
        )
        await self._session_repo.add_message(str(db_session.id), user_message)
        await self._db.commit()

        yield user_message

        # Track sequence for response messages
        sequence = db_session.message_count

        # Execute agent
        async for message in instance.execute(
            prompt,
            context=context,
            tools=tools,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=db_session.sdk_session_id,
            stream=stream,
        ):
            # Update sequence number for non-partial messages
            if not message.is_partial:
                # Create new message with updated sequence number
                message = AgentMessage(
                    id=message.id,
                    role=message.role,
                    type=message.type,
                    content=message.content,
                    tool_name=message.tool_name,
                    tool_call_id=message.tool_call_id,
                    tool_input=message.tool_input,
                    tool_output=message.tool_output,
                    timestamp=message.timestamp,
                    model=message.model,
                    usage=message.usage,
                    cost_usd=message.cost_usd,
                    is_partial=message.is_partial,
                    sequence_number=sequence,
                    error_code=message.error_code,
                    error_details=message.error_details,
                )

                # Persist complete messages
                await self._session_repo.add_message(str(db_session.id), message)
                await self._db.commit()
                sequence += 1

                # Capture SDK session ID if present (for resume functionality)
                if (
                    hasattr(instance, "_adapter")
                    and instance._adapter is not None
                    and hasattr(instance._adapter, "_current_session_id")
                ):
                    sdk_session_id = instance._adapter._current_session_id
                    if sdk_session_id and sdk_session_id != db_session.sdk_session_id:
                        await self._session_repo.update_sdk_session_id(
                            str(db_session.id),
                            sdk_session_id,
                        )
                        await self._db.commit()

            yield message

    async def _persist_error(
        self,
        db_session: AgentSession,
        error: Exception,
    ) -> None:
        """Persist an error message to the session.

        Args:
            db_session: Database session record.
            error: The exception that occurred.
        """
        error_message = AgentMessage(
            role=MessageRole.ERROR,
            type=MessageType.ERROR,
            content=str(error),
            error_code=type(error).__name__,
            error_details={"original_error": str(error)},
            sequence_number=db_session.message_count,
        )
        await self._session_repo.add_message(str(db_session.id), error_message)
        await self._db.commit()

    async def _persist_interruption(
        self,
        db_session: AgentSession,
    ) -> None:
        """Persist an interruption message to the session.

        Args:
            db_session: Database session record.
        """
        interrupt_message = AgentMessage(
            role=MessageRole.SYSTEM,
            type=MessageType.METADATA,
            content="Execution interrupted by user",
            sequence_number=db_session.message_count,
        )
        await self._session_repo.add_message(str(db_session.id), interrupt_message)
        await self._db.commit()

    # =========================================================================
    # Interruption
    # =========================================================================

    async def interrupt(self, session_id: str) -> None:
        """Interrupt an active execution.

        Args:
            session_id: External session ID.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        instance = self._active_executions.get(str(db_session.id))
        if instance:
            await instance.interrupt()
            logger.info("Interrupted execution for session: %s", session_id)
        else:
            logger.debug("No active execution found for session: %s", session_id)

    # =========================================================================
    # Session Management
    # =========================================================================

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session details by external ID.

        Args:
            session_id: External session ID.

        Returns:
            Session dictionary or None if not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        return db_session.to_dict() if db_session else None

    async def get_messages(
        self,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        after_sequence: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages for a session.

        Args:
            session_id: External session ID.
            limit: Maximum messages to return.
            offset: Pagination offset.
            after_sequence: Only return messages after this sequence.

        Returns:
            List of message dictionaries.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        messages = await self._session_repo.get_messages(
            str(db_session.id),
            limit=limit,
            offset=offset,
            after_sequence=after_sequence,
        )

        return [
            {
                "id": m.message_id,
                "sequence": m.sequence_number,
                "role": m.role,
                "type": m.type,
                "content": m.content,
                "tool_name": m.tool_name,
                "tool_call_id": m.tool_call_id,
                "tool_input": m.tool_input,
                "tool_output": m.tool_output,
                "model": m.model,
                "input_tokens": m.input_tokens,
                "output_tokens": m.output_tokens,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]

    async def list_sessions(
        self,
        user_id: int,
        *,
        status: str | None = None,
        collection_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List sessions for a user.

        Args:
            user_id: User ID.
            status: Filter by status.
            collection_id: Filter by collection.
            limit: Maximum sessions to return.
            offset: Pagination offset.

        Returns:
            Tuple of (session list, total count).
        """
        sessions, total = await self._session_repo.list_by_user(
            user_id,
            status=status,
            collection_id=collection_id,
            limit=limit,
            offset=offset,
        )
        return [s.to_dict() for s in sessions], total

    async def update_session_title(
        self,
        session_id: str,
        title: str,
    ) -> dict[str, Any]:
        """Update session title.

        Args:
            session_id: External session ID.
            title: New title.

        Returns:
            Updated session dictionary.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        await self._session_repo.update_title(str(db_session.id), title)
        await self._db.commit()

        # Refresh to get updated data
        db_session = await self._session_repo.get_by_id(str(db_session.id))
        return db_session.to_dict() if db_session else {}

    # =========================================================================
    # Session Forking
    # =========================================================================

    async def fork_session(self, session_id: str) -> str:
        """Fork a session (create branch in conversation).

        Creates a new session with the same messages as the parent,
        enabling "what-if" exploration.

        Args:
            session_id: External ID of session to fork.

        Returns:
            External ID of the new forked session.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        # Create forked session
        new_session = await self._session_repo.create(
            agent_plugin_id=db_session.agent_plugin_id,
            user_id=db_session.user_id,
            collection_id=db_session.collection_id,
            agent_config=db_session.agent_config,
            title=f"Fork of {db_session.title or db_session.external_id}",
            parent_session_id=str(db_session.id),
        )

        # Copy messages (via JSONB assignment)
        new_session.messages = list(db_session.messages) if db_session.messages else []
        new_session.message_count = db_session.message_count

        # Copy SDK session ID for potential resume
        new_session.sdk_session_id = db_session.sdk_session_id

        # Increment parent's fork count
        await self._session_repo.increment_fork_count(str(db_session.id))

        await self._db.commit()

        logger.info(
            "Forked session %s -> %s",
            session_id,
            new_session.external_id,
        )

        return str(new_session.external_id)

    # =========================================================================
    # Session Lifecycle
    # =========================================================================

    async def archive_session(self, session_id: str) -> None:
        """Archive a session (soft delete, preserves data).

        Args:
            session_id: External session ID.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        await self._session_repo.archive(str(db_session.id))
        await self._db.commit()

        logger.info("Archived session: %s", session_id)

    async def delete_session(self, session_id: str) -> None:
        """Soft-delete a session.

        Args:
            session_id: External session ID.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        await self._session_repo.delete(str(db_session.id))
        await self._db.commit()

        logger.info("Deleted session: %s", session_id)

    async def hard_delete_session(self, session_id: str) -> None:
        """Permanently delete a session and all messages.

        Args:
            session_id: External session ID.

        Raises:
            SessionNotFoundError: If session not found.
        """
        db_session = await self._session_repo.get_by_external_id(session_id)
        if not db_session:
            raise SessionNotFoundError(
                f"Session not found: {session_id}",
                session_id=session_id,
            )

        await self._session_repo.hard_delete(str(db_session.id))
        await self._db.commit()

        logger.info("Hard deleted session: %s", session_id)

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def cleanup_old_sessions(
        self,
        older_than_days: int = 90,
    ) -> int:
        """Clean up old deleted sessions.

        Args:
            older_than_days: Delete sessions older than this.

        Returns:
            Number of sessions deleted.
        """
        count = await self._session_repo.cleanup_old_sessions(
            older_than_days=older_than_days,
            status="deleted",
        )
        await self._db.commit()

        logger.info("Cleaned up %d old sessions", count)
        return int(count)
