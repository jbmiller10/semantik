"""Centralized authentication for VecPipe internal API."""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from shared.config import settings


def require_internal_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    """Verify the internal API key for protected endpoints."""
    expected_key = settings.INTERNAL_API_KEY
    if not expected_key:
        raise HTTPException(status_code=500, detail="Internal API key is not configured")
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")
