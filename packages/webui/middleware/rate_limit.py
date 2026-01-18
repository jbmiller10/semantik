"""
Rate limiting middleware for the webui API.

This middleware sets the user in request.state for rate limiting purposes.
"""

import hashlib
import logging
from collections.abc import Callable
from typing import Any

import jwt
from fastapi import Request
from jwt.exceptions import InvalidTokenError
from slowapi.middleware import _find_route_handler, _should_exempt, sync_check_limits
from starlette.middleware.base import BaseHTTPMiddleware

from shared.config import settings
from shared.database import get_db_session
from webui.rate_limiter import ensure_limiter_runtime_state
from webui.repositories.postgres.api_key_repository import PostgreSQLApiKeyRepository

logger = logging.getLogger(__name__)


def _stable_fallback_user_id(username: str) -> int:
    digest = hashlib.sha256(username.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _looks_like_jwt(token: str) -> bool:
    """Return True if the token looks like a JWT (header.payload.signature)."""
    return token.count(".") == 2


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to set user in request.state for rate limiting."""

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """
        Process the request and set user in request.state.

        Args:
            request: FastAPI request object
            call_next: Next middleware in the chain

        Returns:
            Response from the next middleware
        """
        # Try to get user from authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
            try:
                # Try to decode the token and get user
                user = await self.get_user_from_token(request, token)
                if user:
                    request.state.user = user
            except Exception as e:
                # If token is invalid, don't set user
                # Rate limiting will fall back to IP address
                logger.debug(f"Could not extract user from token for rate limiting: {e}")

        limiter = getattr(request.app.state, "limiter", None)
        inject_headers = False

        if limiter:
            ensure_limiter_runtime_state(limiter)
            if not limiter.enabled:
                return await call_next(request)

            handler = _find_route_handler(request.app.routes, request.scope)
            if _should_exempt(limiter, handler):
                logger.debug("Skipping middleware rate limit check due to exemption")
                return await call_next(request)

            error_response, inject_headers = sync_check_limits(
                limiter,
                request,
                handler,
                request.app,
            )
            if getattr(request.state, "view_rate_limit", None):
                limit, args = request.state.view_rate_limit
                logger.debug(
                    "Rate limit check: limit=%s args=%s remaining=%s",
                    limit,
                    args,
                    limiter.limiter.get_window_stats(limit, *args)[1],
                )
            if error_response is not None:
                return error_response

        response = await call_next(request)

        if limiter and inject_headers and getattr(request.state, "view_rate_limit", None):
            response = limiter._inject_headers(response, request.state.view_rate_limit)

        return response

    async def get_user_from_token(self, request: Request, token: str) -> dict[str, Any] | None:
        """
        Get user from bearer token for rate limiting.

        Supports:
        - JWT tokens (decoded locally, no DB lookup)
        - API keys (verified via DB to resolve owning user)

        Args:
            request: FastAPI request object (used to cache API key verification results)
            token: Bearer token

        Returns:
            User dictionary or None if invalid
        """
        if _looks_like_jwt(token):
            try:
                # Decode the JWT token
                payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.ALGORITHM])

                username = payload.get("sub")
                user_id = payload.get("user_id")

                if username:
                    # Return user info for rate limiting
                    # The actual user ID should come from the token payload
                    resolved_user_id: int | None = None
                    if user_id is not None:
                        try:
                            resolved_user_id = int(user_id)
                        except (TypeError, ValueError):
                            resolved_user_id = None
                    if resolved_user_id is None:
                        resolved_user_id = _stable_fallback_user_id(str(username))
                    return {
                        "id": resolved_user_id,
                        "username": str(username),
                    }
            except InvalidTokenError as e:
                logger.debug(f"Invalid JWT token for rate limiting: {e}")
            except Exception as e:
                logger.debug(f"Error decoding JWT token for rate limiting: {e}")
            return None

        # Avoid DB lookups for obviously invalid/short tokens.
        if len(token) < 20:
            return None

        # API key path (or other non-JWT bearer tokens). Verify once and cache
        # the result so auth dependencies can reuse it without a second DB lookup.
        if getattr(request.state, "api_key_auth_checked", False):
            api_key_data = getattr(request.state, "api_key_auth", None)
        else:
            api_key_data = None
            try:
                async for session in get_db_session():
                    repo = PostgreSQLApiKeyRepository(session)
                    api_key_data = await repo.verify_api_key(token, update_last_used=True)
            except Exception as e:
                logger.debug(f"Error verifying API key for rate limiting: {e}")
                return None
            request.state.api_key_auth = api_key_data
            request.state.api_key_auth_checked = True

        if not isinstance(api_key_data, dict):
            return None

        user_info = api_key_data.get("user") or {}
        user_id_raw = user_info.get("id") or api_key_data.get("user_id")
        if user_id_raw is None:
            return None
        try:
            user_id_int = int(user_id_raw)
        except (TypeError, ValueError):
            return None

        username = user_info.get("username")
        return {
            "id": user_id_int,
            "username": str(username) if username else "api_key",
        }

        return None
