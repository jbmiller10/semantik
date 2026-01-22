"""Shared error helpers for VecPipe search."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


async def response_json(response: Any) -> Any:
    """Return response.json(), awaiting if the underlying object is awaitable.

    Some tests use AsyncMock responses where .json() is async.
    """
    data = response.json()
    if inspect.isawaitable(data):
        data = await data
    return data


async def maybe_raise_for_status(response: Any) -> None:
    """Call response.raise_for_status(), awaiting if it returns an awaitable.

    httpx returns None, but tests may use AsyncMock.
    """
    raise_for_status = getattr(response, "raise_for_status", None)
    if not callable(raise_for_status):
        return
    maybe_coro = raise_for_status()
    if inspect.isawaitable(maybe_coro):
        await maybe_coro


def extract_qdrant_error(exc: httpx.HTTPStatusError) -> str:
    """Best-effort extraction of a human-readable Qdrant error message."""
    default_detail = "Vector database error"

    try:
        resp = getattr(exc, "response", None)
        if resp is None:
            return default_detail

        payload = resp.json()
        if inspect.isawaitable(payload):
            return default_detail

        if isinstance(payload, dict):
            status = payload.get("status", {})
            if isinstance(status, dict) and status.get("error"):
                return str(status["error"])

            if payload.get("error"):
                return str(payload.get("error"))
    except Exception as parse_exc:  # pragma: no cover - best effort only
        logger.debug("Failed parsing Qdrant error payload: %s", parse_exc, exc_info=True)

    return default_detail
