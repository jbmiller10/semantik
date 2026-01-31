"""
Agent API v2 endpoints for pipeline builder conversations.

This module provides RESTful API endpoints for managing agent conversations
that help users configure document processing pipelines.

Exception handling is centralized via global exception handlers registered
in webui.middleware.exception_handlers.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from fastapi.responses import StreamingResponse

from shared.database import get_db
from shared.database.exceptions import EntityNotFoundError
from shared.database.repositories.collection_source_repository import CollectionSourceRepository
from shared.llm.exceptions import LLMNotConfiguredError
from shared.llm.factory import LLMServiceFactory
from shared.llm.types import LLMQualityTier
from webui.api.v2.agent_schemas import (
    AgentMessageResponse,
    AnswerQuestionRequest,
    AnswerQuestionResponse,
    ApplyPipelineRequest,
    ApplyPipelineResponse,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    CreateConversationRequest,
    MessageResponse,
    PauseResumeResponse,
    SendMessageRequest,
    UncertaintyResponse,
)
from webui.auth import get_current_user
from webui.services.agent.exceptions import AgentBusyError, AgentError, ConversationNotActiveError
from webui.services.agent.message_store import MessageStore
from webui.services.agent.models import ConversationStatus
from webui.services.agent.orchestrator import AgentOrchestrator
from webui.services.agent.repository import AgentConversationRepository
from webui.services.chunking.container import get_async_redis_client

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/agent", tags=["agent-v2"])


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_message_store() -> MessageStore:
    """Get a MessageStore instance with Redis client."""
    redis_client = await get_async_redis_client()
    if redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Message store unavailable - Redis not connected",
        )
    return MessageStore(redis_client)


async def _verify_llm_configured(session: AsyncSession, user_id: int) -> None:
    """Verify the user has LLM configured for HIGH tier.

    Raises:
        HTTPException: If LLM is not configured
    """
    factory = LLMServiceFactory(session)
    has_llm = await factory.has_provider_configured(user_id, LLMQualityTier.HIGH)
    if not has_llm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM not configured. Please configure an LLM provider in settings before using the agent.",
        )


async def _get_conversation_for_user(
    session: AsyncSession,
    conversation_id: str,
    user_id: int,
    include_uncertainties: bool = False,
) -> Any:
    """Get a conversation with ownership verification.

    Raises:
        HTTPException: If conversation not found or not owned by user
    """
    repo = AgentConversationRepository(session)
    conversation = await repo.get_by_id_for_user(
        conversation_id,
        user_id,
        include_uncertainties=include_uncertainties,
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return conversation


# =============================================================================
# Endpoints
# =============================================================================


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "LLM not configured, invalid source, or invalid connector type"},
        404: {"description": "Source not found"},
    },
)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Start a new agent conversation for pipeline building.

    Creates a new conversation associated with either:
    - An existing collection source (via source_id)
    - An inline source configuration (via inline_source)

    When using inline_source, the actual CollectionSource record is created
    when the pipeline is applied. Secrets are stored temporarily in the
    conversation's inline_source_config field.

    Requires the user to have LLM configured (HIGH tier).
    """
    user_id = int(current_user["id"])

    # Verify LLM is configured
    await _verify_llm_configured(db, user_id)

    source_id: int | None = None
    inline_source_config: dict[str, Any] | None = None

    if request.source_id is not None:
        # Using existing source - verify ownership
        source_repo = CollectionSourceRepository(db)
        source = await source_repo.get_by_id(request.source_id)
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source {request.source_id} not found",
            )

        # Verify user owns the source's parent collection (IDOR protection)
        from shared.database.repositories.collection_repository import CollectionRepository

        collection_repo = CollectionRepository(db)
        collection = await collection_repo.get_by_uuid(str(source.collection_id))
        if not collection or collection.owner_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source {request.source_id} not found",
            )

        source_id = request.source_id

    elif request.inline_source is not None:
        # Using inline source config - validate connector type exists
        from webui.services.connector_registry import get_connector_definition

        try:
            # Verify the connector type is valid by attempting to get its definition
            get_connector_definition(request.inline_source.source_type)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

        # Store inline source config with encrypted pending secrets
        inline_source_config = {
            "source_type": request.inline_source.source_type,
            "source_config": request.inline_source.source_config,
        }

        # Encrypt secrets immediately for defense in depth
        # They will be decrypted and moved to ConnectorSecret on apply
        if request.secrets:
            from shared.config import settings
            from shared.utils.encryption import EncryptionNotConfiguredError, encrypt_secret

            try:
                encrypted_secrets = {key: encrypt_secret(value) for key, value in request.secrets.items()}
                inline_source_config["_pending_secrets"] = encrypted_secrets
            except EncryptionNotConfiguredError as e:
                # In production, fail rather than store plaintext secrets
                if settings.ENVIRONMENT == "production":
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Server encryption not configured. Contact administrator.",
                    ) from e
                # Development only: store plaintext with warning
                logger.warning("CONNECTOR_SECRETS_KEY not configured - secrets stored unencrypted (dev only)")
                inline_source_config["_pending_secrets"] = request.secrets

    # Create conversation
    repo = AgentConversationRepository(db)
    conversation = await repo.create(
        user_id=user_id,
        source_id=source_id,
        inline_source_config=inline_source_config,
    )
    await db.commit()

    logger.info(f"Created conversation {conversation.id} for user {user_id}")

    return ConversationResponse(
        id=conversation.id,
        status=conversation.status.value,
        source_id=conversation.source_id,
        created_at=conversation.created_at,
    )


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=AgentMessageResponse,
    responses={
        400: {"description": "Conversation not active"},
        404: {"description": "Conversation not found"},
    },
)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AgentMessageResponse:
    """Send a message and get the agent's response.

    Processes the user's message through the agent orchestrator,
    which may execute tools and update the pipeline configuration.
    """
    user_id = int(current_user["id"])

    # Get conversation with ownership check
    conversation = await _get_conversation_for_user(
        db,
        conversation_id,
        user_id,
        include_uncertainties=False,
    )

    # Create orchestrator dependencies
    message_store = await _get_message_store()
    llm_factory = LLMServiceFactory(db)

    # Create and run orchestrator
    orchestrator = AgentOrchestrator(
        conversation=conversation,
        session=db,
        llm_factory=llm_factory,
        message_store=message_store,
    )

    try:
        response = await orchestrator.handle_message(request.message)
    except AgentBusyError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ConversationNotActiveError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except LLMNotConfiguredError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LLM not configured. Please configure an LLM provider in settings.",
        ) from e
    except AgentError as e:
        logger.exception(f"Agent error in conversation {conversation_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent error: {e}",
        ) from e

    return AgentMessageResponse(
        response=response.content,
        pipeline_updated=response.pipeline_updated,
        uncertainties_added=[
            UncertaintyResponse(
                id=u["id"],
                severity=u["severity"],
                message=u["message"],
                resolved=u["resolved"],
                context=u.get("context"),
            )
            for u in response.uncertainties_added
        ],
        tool_calls=[{"name": tc["name"], "arguments": tc["arguments"]} for tc in response.tool_calls_made],
    )


@router.post(
    "/conversations/{conversation_id}/messages/stream",
    responses={
        200: {"content": {"text/event-stream": {}}},
        400: {"description": "Conversation not active"},
        404: {"description": "Conversation not found"},
    },
)
async def send_message_stream(
    conversation_id: str,
    request: SendMessageRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Send a message and stream the agent's response via SSE.

    Returns Server-Sent Events as the agent processes the request.
    Each event has the format:
        event: {type}
        data: {json}

    Event types:
    - tool_call_start: Tool execution starting
    - tool_call_end: Tool execution completed
    - subagent_start: Sub-agent spawned
    - subagent_end: Sub-agent completed
    - uncertainty: Uncertainty flagged
    - pipeline_update: Pipeline modified
    - content: Final response text
    - done: Stream complete with full metadata
    - error: Error occurred
    """
    user_id = int(current_user["id"])

    # Verify conversation exists and user has access (quick ownership check)
    # We only use this for permission validation - conversation will be re-loaded
    # inside the generator with an independent session
    await _get_conversation_for_user(
        db,
        conversation_id,
        user_id,
        include_uncertainties=False,
    )

    # Get message from request before entering generator
    message = request.message

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from orchestrator streaming.

        Uses an independent database session for all operations to avoid
        session cancellation when the SSE connection is closed. The request-
        bound session (from Depends(get_db)) gets cancelled when the HTTP
        connection closes, but we need database operations to complete.

        The conversation is re-loaded inside this generator with the independent
        session to prevent detached object errors when the request session closes.
        """
        import json

        from asyncpg.exceptions import InterfaceError as AsyncpgInterfaceError

        from shared.database.postgres_database import pg_connection_manager

        try:
            # Create independent session for all database operations
            # This survives SSE connection closure
            async with pg_connection_manager.get_session() as llm_session:
                # Re-load conversation with the independent session to avoid
                # detached object errors when the request-bound session closes
                conversation = await _get_conversation_for_user(
                    llm_session,
                    conversation_id,
                    user_id,
                    include_uncertainties=False,
                )

                # Create orchestrator with independent session for all DB ops
                message_store = await _get_message_store()
                llm_factory = LLMServiceFactory(llm_session)

                orchestrator = AgentOrchestrator(
                    conversation=conversation,
                    session=llm_session,  # Use independent session for all DB ops
                    llm_factory=llm_factory,
                    message_store=message_store,
                )

                try:
                    async for event in orchestrator.handle_message_streaming(message):
                        # Format as SSE: event: type\ndata: json\n\n
                        # Note: Only serialize event.data, not the whole event object
                        # The event type is already in the SSE 'event:' line
                        event_line = f"event: {event.event.value}\n"
                        data_line = f"data: {json.dumps(event.data)}\n\n"
                        yield event_line + data_line
                except AsyncpgInterfaceError as e:
                    # DB connection closed during long-running LLM operation
                    # Rollback before context manager exit to prevent PendingRollbackError
                    logger.warning(f"DB connection closed during SSE stream for conversation {conversation_id}: {e}")
                    with contextlib.suppress(Exception):
                        await llm_session.rollback()
                    error_data = json.dumps(
                        {
                            "message": "Connection timeout during processing. Results may be incomplete.",
                            "recoverable": True,
                        }
                    )
                    yield f"event: error\ndata: {error_data}\n\n"
                    return
                except Exception as e:
                    # Rollback before context manager exit to prevent PendingRollbackError
                    logger.exception(f"SSE streaming error for conversation {conversation_id}: {e}")
                    with contextlib.suppress(Exception):
                        await llm_session.rollback()
                    error_data = json.dumps({"message": str(e)})
                    yield f"event: error\ndata: {error_data}\n\n"
                    return

        except Exception as e:
            # Failed to initialize session or load conversation
            logger.exception(f"Failed to initialize SSE stream for conversation {conversation_id}: {e}")
            error_data = json.dumps({"message": "Service temporarily unavailable. Please try again."})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post(
    "/conversations/{conversation_id}/answer",
    response_model=AnswerQuestionResponse,
    responses={
        400: {"description": "Invalid answer or conversation not active"},
        404: {"description": "Conversation not found"},
    },
)
async def answer_question(
    conversation_id: str,
    request: AnswerQuestionRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AnswerQuestionResponse:
    """Answer a question from the agent.

    Submits an answer to a pending question, either by selecting
    a pre-defined option or providing a custom response.
    """
    user_id = int(current_user["id"])

    # Get conversation with ownership check
    conversation = await _get_conversation_for_user(
        db,
        conversation_id,
        user_id,
        include_uncertainties=False,
    )

    # Verify conversation is active
    if conversation.status != ConversationStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conversation is not active (status: {conversation.status.value})",
        )

    # Store the answer in the message store for the orchestrator to pick up
    message_store = await _get_message_store()

    # Build answer message
    answer_content = f"[Selected option: {request.option_id}]" if request.option_id else request.custom_response or ""

    # Store as user message with question metadata
    from webui.services.agent.message_store import ConversationMessage

    answer_msg = ConversationMessage.create(
        "user",
        answer_content,
        metadata={
            "question_id": request.question_id,
            "option_id": request.option_id,
            "is_question_answer": True,
        },
    )
    await message_store.append_message(conversation_id, answer_msg)

    logger.info(f"Stored answer to question {request.question_id} for conversation {conversation_id}")

    return AnswerQuestionResponse(
        success=True,
        message="Answer recorded",
    )


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
    responses={
        404: {"description": "Conversation not found"},
    },
)
async def get_conversation(
    conversation_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationDetailResponse:
    """Get full conversation state including messages and pipeline.

    Returns the complete conversation details including message history,
    pipeline configuration, source analysis, and uncertainties.
    """
    user_id = int(current_user["id"])

    # Get conversation with uncertainties
    conversation = await _get_conversation_for_user(
        db,
        conversation_id,
        user_id,
        include_uncertainties=True,
    )

    # Get messages from Redis
    messages: list[MessageResponse] = []
    message_load_error: str | None = None
    try:
        message_store = await _get_message_store()
        raw_messages = await message_store.get_messages(conversation_id)
        messages = [
            MessageResponse(
                role=m.role,
                content=m.content,
                timestamp=datetime.fromisoformat(m.timestamp),
                metadata=m.metadata,
            )
            for m in raw_messages
        ]
    except Exception as e:
        logger.warning(f"Failed to load messages for conversation {conversation_id}: {e}")
        message_load_error = f"Failed to load messages: {e}"

    # Build inline_source_config response, filtering out secrets
    from webui.api.v2.agent_schemas import InlineSourceConfigResponse

    inline_source_config_response: InlineSourceConfigResponse | None = None
    if conversation.inline_source_config:
        inline_source_config_response = InlineSourceConfigResponse(
            source_type=conversation.inline_source_config.get("source_type", ""),
            source_config=conversation.inline_source_config.get("source_config", {}),
            # Note: _pending_secrets is intentionally NOT included
        )

    return ConversationDetailResponse(
        id=conversation.id,
        status=conversation.status.value,
        source_id=conversation.source_id,
        inline_source_config=inline_source_config_response,
        collection_id=conversation.collection_id,
        current_pipeline=conversation.current_pipeline,
        source_analysis=conversation.source_analysis,
        uncertainties=[
            UncertaintyResponse(
                id=u.id,
                severity=u.severity.value,
                message=u.message,
                resolved=u.resolved,
                context=u.context,
            )
            for u in conversation.uncertainties
        ],
        messages=messages,
        message_load_error=message_load_error,
        summary=conversation.summary,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.post(
    "/conversations/{conversation_id}/apply",
    response_model=ApplyPipelineResponse,
    responses={
        400: {"description": "No pipeline configured or blocking uncertainties"},
        404: {"description": "Conversation not found"},
    },
)
async def apply_pipeline(
    conversation_id: str,
    request: ApplyPipelineRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ApplyPipelineResponse:
    """Apply the configured pipeline to create a collection.

    Creates a new collection using the pipeline configuration from the
    conversation and optionally starts the indexing operation.
    """
    user_id = int(current_user["id"])

    # Get conversation with uncertainties
    conversation = await _get_conversation_for_user(
        db,
        conversation_id,
        user_id,
        include_uncertainties=True,
    )

    # Verify conversation is active
    if conversation.status != ConversationStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conversation is not active (status: {conversation.status.value})",
        )

    # Verify pipeline is configured
    if not conversation.current_pipeline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pipeline configured. Use the agent to build a pipeline first.",
        )

    # Check for blocking uncertainties (unless force=True)
    if not request.force:
        repo = AgentConversationRepository(db)
        blocking = await repo.get_blocking_uncertainties(conversation_id)
        if blocking:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot apply: {len(blocking)} blocking uncertainties. Resolve them or use force=true.",
            )

    # Create collection using the apply_pipeline tool
    from webui.services.agent.tools import ApplyPipelineTool

    context = {
        "session": db,
        "user_id": user_id,
        "conversation": conversation,
    }
    apply_tool = ApplyPipelineTool(context)

    result = await apply_tool.execute(
        collection_name=request.collection_name,
        force=request.force,
        start_indexing=True,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("error", "Failed to apply pipeline"),
        )

    await db.commit()

    return ApplyPipelineResponse(
        collection_id=result["collection_id"],
        collection_name=result["collection_name"],
        operation_id=result.get("operation_id"),
        status=result.get("status", "created"),
    )


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    responses={},
)
async def list_conversations(
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationListResponse:
    """List user's conversations with optional status filter.

    Returns a paginated list of conversations belonging to the current user.
    """
    user_id = int(current_user["id"])

    # Parse status filter
    status_enum: ConversationStatus | None = None
    if status_filter:
        try:
            status_enum = ConversationStatus(status_filter)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}. Valid values: active, applied, abandoned",
            ) from e

    repo = AgentConversationRepository(db)
    conversations = await repo.list_for_user(
        user_id=user_id,
        status=status_enum,
        limit=limit,
        offset=skip,
    )

    return ConversationListResponse(
        conversations=[
            ConversationResponse(
                id=c.id,
                status=c.status.value,
                source_id=c.source_id,
                created_at=c.created_at,
            )
            for c in conversations
        ],
        total=len(conversations),  # TODO: Add count query for accurate total
    )


@router.patch(
    "/conversations/{conversation_id}/status",
    response_model=ConversationResponse,
    responses={
        400: {"description": "Invalid status transition"},
        404: {"description": "Conversation not found"},
    },
)
async def update_conversation_status(
    conversation_id: str,
    new_status: str = Query(..., alias="status", description="New status (abandoned)"),
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationResponse:
    """Update conversation status (e.g., mark as abandoned).

    Only allows transition to 'abandoned' status.
    """
    user_id = int(current_user["id"])

    # Parse and validate status
    if new_status != "abandoned":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only set status to 'abandoned'",
        )

    repo = AgentConversationRepository(db)

    try:
        conversation = await repo.update_status(
            conversation_id=conversation_id,
            user_id=user_id,
            status=ConversationStatus.ABANDONED,
        )
        await db.commit()
    except EntityNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        ) from e

    return ConversationResponse(
        id=conversation.id,
        status=conversation.status.value,
        source_id=conversation.source_id,
        created_at=conversation.created_at,
    )


@router.post(
    "/conversations/{conversation_id}/pause",
    response_model=PauseResumeResponse,
    responses={
        400: {"description": "Conversation not active"},
        404: {"description": "Conversation not found"},
    },
)
async def pause_agent(
    conversation_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PauseResumeResponse:
    """Pause agent processing (switch to manual mode).

    Signals the agent to stop processing after the current operation.
    User can continue editing the pipeline manually.
    """
    user_id = int(current_user["id"])

    conversation = await _get_conversation_for_user(db, conversation_id, user_id, include_uncertainties=False)

    if conversation.status != ConversationStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conversation is not active (status: {conversation.status.value})",
        )

    # Store pause state in conversation metadata
    repo = AgentConversationRepository(db)
    await repo.update_metadata(
        conversation_id=conversation_id,
        user_id=user_id,
        metadata={"is_paused": True},
    )
    await db.commit()

    logger.info(f"Agent paused for conversation {conversation_id}")

    return PauseResumeResponse(
        success=True,
        is_paused=True,
        message="Agent paused. You can now edit the pipeline manually.",
    )


@router.post(
    "/conversations/{conversation_id}/resume",
    response_model=PauseResumeResponse,
    responses={
        400: {"description": "Conversation not active or not paused"},
        404: {"description": "Conversation not found"},
    },
)
async def resume_agent(
    conversation_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PauseResumeResponse:
    """Resume agent processing (switch to assisted mode).

    Re-enables agent processing. The agent will continue from
    where it left off, incorporating any manual changes.
    """
    user_id = int(current_user["id"])

    conversation = await _get_conversation_for_user(db, conversation_id, user_id, include_uncertainties=False)

    if conversation.status != ConversationStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conversation is not active (status: {conversation.status.value})",
        )

    # Clear pause state
    repo = AgentConversationRepository(db)
    await repo.update_metadata(
        conversation_id=conversation_id,
        user_id=user_id,
        metadata={"is_paused": False},
    )
    await db.commit()

    logger.info(f"Agent resumed for conversation {conversation_id}")

    return PauseResumeResponse(
        success=True,
        is_paused=False,
        message="Agent resumed. Assisted mode active.",
    )
