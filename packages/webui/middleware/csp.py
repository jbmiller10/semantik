"""
Content Security Policy (CSP) middleware for XSS prevention.

This middleware adds CSP headers to all responses, particularly
important for chunking endpoints that handle user-provided metadata.
"""

from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class CSPMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add Content Security Policy headers.

    This helps prevent XSS attacks by restricting what resources
    the browser is allowed to load and execute.
    """

    # Default CSP policy - very restrictive
    DEFAULT_CSP = (
        "default-src 'self'; "
        "worker-src 'self' blob:; "
        "child-src 'self' blob:; "
        "script-src 'self' blob: 'wasm-unsafe-eval' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "  # Allow inline styles for UI frameworks
        "img-src 'self' data: https:; "  # Allow data URIs and HTTPS images
        "font-src 'self' data:; "
        "connect-src 'self' blob:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "upgrade-insecure-requests"
    )

    # Specific CSP for chunking endpoints (even more restrictive)
    CHUNKING_CSP = (
        "default-src 'none'; "
        "script-src 'none'; "
        "style-src 'none'; "
        "img-src 'none'; "
        "font-src 'none'; "
        "connect-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "form-action 'none'"
    )

    def __init__(self, app: Any, strict_mode: bool = False) -> None:
        """
        Initialize CSP middleware.

        Args:
            app: The ASGI application
            strict_mode: If True, use the most restrictive CSP for all endpoints
        """
        super().__init__(app)
        self.strict_mode = strict_mode

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """
        Process the request and add CSP headers to the response.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            Response with CSP headers added
        """
        # Process the request
        response = await call_next(request)

        # Determine which CSP to use based on the path
        path = request.url.path

        if self.strict_mode or "/chunking" in path or "/chunk" in path:
            # Use strict CSP for chunking endpoints
            csp_policy = self.CHUNKING_CSP
        else:
            # Use default CSP for other endpoints
            csp_policy = self.DEFAULT_CSP

        # Add CSP header
        response.headers["Content-Security-Policy"] = csp_policy

        # Add other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        # Note: X-XSS-Protection is deprecated and removed - CSP provides better protection
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response
