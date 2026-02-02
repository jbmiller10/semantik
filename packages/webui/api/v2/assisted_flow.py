"""Assisted flow API endpoints.

Provides endpoints for the Claude Agent SDK-powered pipeline
configuration assistant.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, status

from shared.database import get_db
from webui.api.v2.assisted_flow_schemas import (
    StartFlowRequest,
    StartFlowResponse,
)
from webui.auth import get_current_user
from webui.services.assisted_flow.source_stats import get_source_stats

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/assisted-flow", tags=["assisted-flow"])


@router.post("/start", response_model=StartFlowResponse)
async def start_assisted_flow(
    request: StartFlowRequest,
    db: AsyncSession = Depends(get_db),
    user: dict[str, Any] = Depends(get_current_user),
) -> StartFlowResponse:
    """Start a new assisted flow session.

    Creates a new SDK session with the pipeline configuration tools
    and returns a session ID for subsequent message requests.

    Args:
        request: Contains source_id to configure
        db: Database session
        user: Authenticated user

    Returns:
        Session ID and source info
    """
    try:
        # Get source stats for initial context
        stats = await get_source_stats(db, request.source_id)

        # TODO: Initialize SDK session with tools and prompts
        # For now, return a placeholder
        session_id = f"session_{request.source_id}"

        return StartFlowResponse(
            session_id=session_id,
            source_name=stats["source_name"],
        )

    except Exception as e:
        logger.error(f"Failed to start assisted flow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
