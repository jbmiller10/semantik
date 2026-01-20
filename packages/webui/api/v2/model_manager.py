"""Model manager API endpoints (superuser-only)."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from webui.api.schemas import ErrorResponse
from webui.auth import get_current_user

router = APIRouter(prefix="/api/v2/models", tags=["models-v2"])


def _require_superuser(current_user: dict[str, Any]) -> None:
    """Raise 403 if user is not a superuser."""
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser access required for model management",
        )


@router.get(
    "",
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Superuser access required"},
    },
)
async def list_models(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """List all curated models with installation status.

    Requires superuser access.
    """
    _require_superuser(current_user)
    # Placeholder - implementation in Phase 1A
    return {"models": [], "message": "Not yet implemented"}
