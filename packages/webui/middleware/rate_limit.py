"""
Rate limiting middleware for the webui API.

This middleware sets the user in request.state for rate limiting purposes.
"""

import logging
from collections.abc import Callable
from typing import Any

import jwt
from fastapi import Request
from jwt.exceptions import InvalidTokenError
from shared.config import settings
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


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

        # Continue with the request
        return await call_next(request)

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
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])

            username = payload.get("sub")
            user_id = payload.get("user_id")

            if username:
                # Return user info for rate limiting
                # The actual user ID should come from the token payload
                return {
                    "id": user_id if user_id else hash(username) % 1000000,
                    "username": username,
                }
        except InvalidTokenError as e:
            logger.debug(f"Invalid token for rate limiting: {e}")
        except Exception as e:
            logger.debug(f"Error decoding token for rate limiting: {e}")

        return None
