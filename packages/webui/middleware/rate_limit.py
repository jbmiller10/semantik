"""
Rate limiting middleware for the webui API.

This middleware sets the user in request.state for rate limiting purposes.
"""

import logging
import hashlib
from collections.abc import Callable
from typing import Any

import jwt
from fastapi import Request
from jwt.exceptions import InvalidTokenError
from slowapi.middleware import _find_route_handler, _should_exempt, sync_check_limits
from starlette.middleware.base import BaseHTTPMiddleware

from shared.config import settings
from webui.rate_limiter import ensure_limiter_runtime_state

logger = logging.getLogger(__name__)


def _stable_fallback_user_id(username: str) -> int:
    digest = hashlib.sha256(username.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


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
            token = auth_header.split(" ", 1)[1]
            try:
                # Try to decode the token and get user
                user = await self.get_user_from_token(token)
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

    async def get_user_from_token(self, token: str) -> dict[str, Any] | None:
        """
        Get user from JWT token.

        Args:
            token: JWT token

        Returns:
            User dictionary or None if invalid
        """
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
            logger.debug(f"Invalid token for rate limiting: {e}")
        except Exception as e:
            logger.debug(f"Error decoding token for rate limiting: {e}")

        return None
