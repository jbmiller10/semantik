"""Semantik MCP server exposing read-only search capabilities."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from webui.mcp.client import SemantikAPIClient, SemantikAPIError
from webui.mcp.tools import (
    build_diagnostics_tool,
    build_get_chunk_tool,
    build_get_document_content_tool,
    build_get_document_tool,
    build_list_documents_tool,
    build_search_tool,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from mcp import Tool
    from starlette.requests import Request


# --------------------------------------------------------------------------
# Context variable for authenticated user in service mode
# --------------------------------------------------------------------------


@dataclass
class MCPAuthContext:
    """Authenticated user context for MCP requests in service mode."""

    user_id: int
    username: str


# Context variable to pass auth info from middleware to handlers
_mcp_auth_context: contextvars.ContextVar[MCPAuthContext | None] = contextvars.ContextVar(
    "mcp_auth_context", default=None
)


# --------------------------------------------------------------------------
# Authentication Middleware for HTTP Transport (Service Mode)
# --------------------------------------------------------------------------

# Cache TTL for API key validation (seconds)
_VALIDATION_CACHE_TTL = 60.0


class MCPAuthMiddleware(BaseHTTPMiddleware):
    """Validate client API keys on incoming MCP requests.

    This middleware:
    - Skips the /health endpoint
    - Extracts Bearer token from Authorization header
    - Validates via WebUI (with caching)
    - Stores user_id in request.state for profile filtering
    - Returns 401 if invalid
    """

    def __init__(self, app: Any, api_client: SemantikAPIClient) -> None:
        super().__init__(app)
        self.api_client = api_client
        # Cache: api_key_hash -> (validation_result, expiry_time)
        self._validation_cache: dict[str, tuple[dict[str, Any], float]] = {}

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process each request, validating auth for MCP endpoints."""
        path = request.url.path

        # Skip /health endpoint
        if path == "/health":
            return await call_next(request)  # type: ignore[no-any-return]

        # Extract Bearer token from Authorization header
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header. Use: Bearer <your-api-key>"},
                status_code=401,
            )

        api_key = auth_header[7:].strip()  # Remove "Bearer " prefix
        if not api_key:
            return JSONResponse(
                {"error": "Empty API key in Authorization header"},
                status_code=401,
            )

        # Validate the API key (with caching)
        validation_result = await self._validate_with_cache(api_key)

        if not validation_result.get("valid"):
            error_msg = validation_result.get("error", "Invalid API key")
            return JSONResponse({"error": error_msg}, status_code=401)

        # Store user info in request.state for downstream use
        request.state.mcp_user_id = validation_result.get("user_id")
        request.state.mcp_username = validation_result.get("username")

        # Set context variable for access in MCP handlers
        user_id = validation_result.get("user_id")
        username = validation_result.get("username", "")
        if user_id is not None:
            _mcp_auth_context.set(MCPAuthContext(user_id=user_id, username=username))

        try:
            return await call_next(request)  # type: ignore[no-any-return]
        finally:
            # Clear context after request
            _mcp_auth_context.set(None)

    async def _validate_with_cache(self, api_key: str) -> dict[str, Any]:
        """Validate API key, using cache to reduce WebUI calls."""
        import hashlib

        # Use hash of API key as cache key (don't store raw keys)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        now = time.monotonic()

        # Check cache
        if key_hash in self._validation_cache:
            result, expiry = self._validation_cache[key_hash]
            if now < expiry:
                logger.debug("API key validation cache hit for %s...", key_hash[:8])
                return result

        # Cache miss - validate via WebUI
        try:
            fresh_result = await self.api_client.validate_api_key(api_key)
            # Cache the result
            self._validation_cache[key_hash] = (fresh_result, now + _VALIDATION_CACHE_TTL)

            # Clean up expired entries periodically
            if len(self._validation_cache) > 100:
                self._cleanup_cache(now)

            return fresh_result  # type: ignore[no-any-return]
        except SemantikAPIError as e:
            logger.error("Failed to validate API key: %s", e)
            return {"valid": False, "error": "Authentication service unavailable"}

    def _cleanup_cache(self, now: float) -> None:
        """Remove expired cache entries."""
        expired = [k for k, (_, exp) in self._validation_cache.items() if exp <= now]
        for k in expired:
            del self._validation_cache[k]


class SemantikMCPServer:
    """MCP server exposing Semantik search tools.

    Supports two modes:
    - User mode (stdio): Uses auth_token for user-specific profile access
    - Service mode (http): Uses internal_api_key, validates client API keys,
      and filters profiles per-user per-session
    """

    def __init__(
        self,
        *,
        webui_url: str,
        auth_token: str | None = None,
        internal_api_key: str | None = None,
        profile_filter: list[str] | None = None,
    ) -> None:
        """Initialize the MCP server.

        Args:
            webui_url: Base URL of the Semantik WebUI.
            auth_token: User auth token (for stdio/user mode).
            internal_api_key: Internal API key (for http/service mode).
            profile_filter: Optional list of profile names to expose (stdio only).
        """
        # Validate that at least one auth method is provided
        if not auth_token and not internal_api_key:
            raise ValueError("Either auth_token or internal_api_key must be provided")

        self._service_mode = bool(internal_api_key)
        self.webui_url = webui_url

        if internal_api_key:
            self.api_client = SemantikAPIClient(webui_url, internal_api_key=internal_api_key)
        else:
            self.api_client = SemantikAPIClient(webui_url, auth_token)

        self.profile_filter = set(profile_filter) if profile_filter else None

        self.server = Server("semantik")
        # All-profiles cache (for service mode)
        self._all_profiles_cache: list[dict[str, Any]] | None = None
        self._all_profiles_cache_expires_at: float = 0.0
        self._all_profiles_cache_lock = asyncio.Lock()
        # User-mode profiles cache (for stdio)
        self._profiles_cache: list[dict[str, Any]] | None = None
        self._profiles_cache_expires_at: float = 0.0
        self._profiles_cache_lock = asyncio.Lock()

        self._setup_handlers()

    @property
    def service_mode(self) -> bool:
        """True if running in service mode (HTTP with internal API key)."""
        return self._service_mode

    async def _get_all_profiles_cached(self) -> list[dict[str, Any]]:
        """Get all enabled profiles from WebUI (service mode).

        This fetches all profiles across all users and caches them.
        Used in HTTP/service mode where we filter by authenticated user.
        """
        now = time.monotonic()
        if self._all_profiles_cache is not None and now < self._all_profiles_cache_expires_at:
            logger.debug("All-profiles cache hit (TTL: %.1fs remaining)", self._all_profiles_cache_expires_at - now)
            return self._all_profiles_cache

        async with self._all_profiles_cache_lock:
            now = time.monotonic()
            if self._all_profiles_cache is not None and now < self._all_profiles_cache_expires_at:
                logger.debug("All-profiles cache hit (TTL: %.1fs remaining)", self._all_profiles_cache_expires_at - now)
                return self._all_profiles_cache

            logger.debug("All-profiles cache miss, fetching from API")
            try:
                profiles: list[dict[str, Any]] = await self.api_client.get_all_profiles()
            except SemantikAPIError as e:
                logger.error("Failed to fetch all profiles from Semantik API: %s", e)
                if self._all_profiles_cache is not None:
                    logger.warning("Using stale all-profiles cache due to API error")
                    return self._all_profiles_cache
                raise

            self._all_profiles_cache = profiles
            # 10-second TTL balances responsiveness (new profiles appear quickly)
            # against reducing API calls during rapid tool invocations
            self._all_profiles_cache_expires_at = now + 10.0
            logger.debug("All-profiles cache updated with %d profiles (TTL: 10s)", len(profiles))
            return profiles

    def _filter_profiles_for_user(self, profiles: list[dict[str, Any]], user_id: int) -> list[dict[str, Any]]:
        """Filter profiles to those owned by the specified user."""
        return [p for p in profiles if p.get("owner_id") == user_id]

    async def _get_profiles_cached(self, user_id: int | None = None) -> list[dict[str, Any]]:
        """Get profiles for tool operations.

        In stdio mode (user mode): Returns user's profiles directly.
        In HTTP mode (service mode): Returns all profiles filtered by user_id.

        Args:
            user_id: User ID to filter by (only used in service mode).
        """
        if self._service_mode:
            # Service mode: fetch all profiles, filter by user
            all_profiles = await self._get_all_profiles_cached()

            if user_id is not None:
                profiles = self._filter_profiles_for_user(all_profiles, user_id)
                logger.debug("Filtered %d profiles for user %d", len(profiles), user_id)
            else:
                # No user context - shouldn't happen in properly authenticated requests
                logger.warning("Service mode request without user context")
                profiles = []

            return profiles

        # User mode: original behavior
        now = time.monotonic()
        if self._profiles_cache is not None and now < self._profiles_cache_expires_at:
            logger.debug("Profile cache hit (TTL: %.1fs remaining)", self._profiles_cache_expires_at - now)
            return self._profiles_cache

        async with self._profiles_cache_lock:
            now = time.monotonic()
            if self._profiles_cache is not None and now < self._profiles_cache_expires_at:
                logger.debug("Profile cache hit (TTL: %.1fs remaining)", self._profiles_cache_expires_at - now)
                return self._profiles_cache

            logger.debug("Profile cache miss, fetching from API")
            try:
                profiles = await self.api_client.get_profiles(enabled_only=True)
            except SemantikAPIError as e:
                logger.error("Failed to fetch profiles from Semantik API: %s", e)
                if self._profiles_cache is not None:
                    logger.warning("Using stale profile cache due to API error")
                    return self._profiles_cache
                raise

            if self.profile_filter is not None:
                profiles = [p for p in profiles if p.get("name") in self.profile_filter]
                logger.debug("Profile filter applied: %s -> %d profiles", list(self.profile_filter), len(profiles))

            self._profiles_cache = profiles
            # 10-second TTL balances responsiveness (new profiles appear quickly)
            # against reducing API calls during rapid tool invocations
            self._profiles_cache_expires_at = now + 10.0
            logger.debug("Profile cache updated with %d profiles (TTL: 10s)", len(profiles))
            return profiles

    def _get_current_user_id(self) -> int | None:
        """Get the authenticated user ID from context (service mode only)."""
        auth_context = _mcp_auth_context.get()
        return auth_context.user_id if auth_context else None

    @staticmethod
    def _allowed_collection_ids(profiles: list[dict[str, Any]]) -> set[str]:
        allowed: set[str] = set()
        for profile in profiles:
            for collection in profile.get("collections", []) or []:
                cid = collection.get("id")
                if isinstance(cid, str) and cid:
                    allowed.add(cid)
        return allowed

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def _list_tools() -> list[Tool]:
            return await self.list_tools()

        @self.server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            return await self.call_tool(name, arguments)

    async def list_tools(self) -> list[Tool]:
        # Get user ID from auth context (service mode) or None (user mode)
        user_id = self._get_current_user_id()
        profiles = await self._get_profiles_cached(user_id)

        tools: list[Tool] = []
        for profile in profiles:
            name = str(profile.get("name") or "")
            if not name:
                continue
            tools.append(
                build_search_tool(
                    name=f"search_{name}",
                    description=str(profile.get("description") or ""),
                    profile=profile,
                )
            )

        tools.append(build_get_document_tool())
        tools.append(build_get_document_content_tool())
        tools.append(build_get_chunk_tool())
        tools.append(build_list_documents_tool())
        tools.append(build_diagnostics_tool())
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        start_time = time.monotonic()
        logger.info("Tool call: %s", name)
        logger.debug("Tool arguments: %s", arguments)

        # Get user ID from auth context (service mode) or None (user mode)
        user_id = self._get_current_user_id()

        try:
            # Handle diagnostics first - it has its own error handling
            if name == "diagnostics":
                logger.debug("Executing diagnostics tool")
                payload = await self._execute_diagnostics()
                return self._as_text_result(payload)

            profiles = await self._get_profiles_cached(user_id)
            profiles_by_name = {p.get("name"): p for p in profiles if p.get("name")}

            if name.startswith("search_"):
                profile_name = name.removeprefix("search_")
                profile = profiles_by_name.get(profile_name)
                if profile is None:
                    raise ValueError(f"Profile not found or not enabled: {profile_name}")
                logger.info("Executing search (profile=%s, query=%s)", profile_name, arguments.get("query", "")[:50])
                payload = await self._execute_search(profile, arguments)
                duration = time.monotonic() - start_time
                logger.info("Search completed in %.2fs (results=%d)", duration, len(payload.get("results", [])))
                return self._as_text_result(payload)

            if name in {"get_document", "get_document_content", "get_chunk", "list_documents"}:
                allowed = self._allowed_collection_ids(profiles)
                collection_id = arguments.get("collection_id")
                if collection_id not in allowed:
                    raise ValueError("Collection is not accessible via the exposed profile(s)")

            if name == "get_document":
                payload = await self.api_client.get_document(
                    collection_id=arguments["collection_id"],
                    document_id=arguments["document_id"],
                )
                return self._as_text_result(payload)

            if name == "get_document_content":
                payload = await self._execute_get_document_content(
                    collection_id=arguments["collection_id"],
                    document_id=arguments["document_id"],
                )
                return self._as_text_result(payload)

            if name == "get_chunk":
                payload = await self.api_client.get_chunk(
                    collection_id=arguments["collection_id"],
                    chunk_id=arguments["chunk_id"],
                )
                return self._as_text_result(payload)

            if name == "list_documents":
                payload = await self.api_client.list_documents(
                    collection_id=arguments["collection_id"],
                    page=int(arguments.get("page") or 1),
                    per_page=int(arguments.get("per_page") or 50),
                    status=arguments.get("status"),
                )
                return self._as_text_result(payload)

            raise ValueError(f"Unknown tool: {name}")

        except SemantikAPIError as exc:
            return self._as_text_result({"error": str(exc)}, is_error=True)
        except ValueError as exc:
            # Expected user errors (invalid arguments, unknown tool, etc.)
            return self._as_text_result({"error": str(exc)}, is_error=True)
        except (KeyError, TypeError, AttributeError) as exc:
            # Programming errors - log with full context
            logger.exception("Unexpected error in tool call (likely a bug): %s", exc)
            return self._as_text_result({"error": "Internal error processing request"}, is_error=True)
        except Exception as exc:
            logger.exception("Unexpected exception in tool call: %s", exc)
            return self._as_text_result({"error": "Unexpected error"}, is_error=True)

    async def _execute_search(self, profile: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("Missing required argument: query")

        collections = profile.get("collections") or []
        if not isinstance(collections, list) or not collections:
            raise ValueError("Profile has no collections configured")

        collection_uuids: list[str] = []
        for c in collections:
            cid = c.get("id") if isinstance(c, dict) else None
            if isinstance(cid, str) and cid:
                collection_uuids.append(cid)

        collection_uuids = collection_uuids[:10]
        if not collection_uuids:
            raise ValueError("Profile has no valid collection IDs configured")

        default_k = int(profile.get("result_count") or 10)
        k = int(arguments.get("k") or default_k)
        k = max(1, min(100, k))

        search_type = str(arguments.get("search_type") or profile.get("search_type") or "semantic")
        use_reranker = bool(
            arguments.get("use_reranker") if "use_reranker" in arguments else profile.get("use_reranker")
        )

        score_threshold = arguments.get("score_threshold")
        if score_threshold is None:
            score_threshold = profile.get("score_threshold")
        score_threshold_value = float(score_threshold or 0.0)

        # New search_mode parameter for sparse/hybrid search
        # Use profile defaults if not specified in arguments
        search_mode = str(arguments.get("search_mode") or profile.get("search_mode") or "dense")
        rrf_k = int(arguments.get("rrf_k") or profile.get("rrf_k") or 60)

        # HyDE query expansion
        use_hyde = bool(arguments.get("use_hyde") if "use_hyde" in arguments else profile.get("use_hyde", False))

        data = await self.api_client.search(
            collection_uuids=collection_uuids,
            query=query,
            k=k,
            search_type=search_type,
            search_mode=search_mode,
            rrf_k=rrf_k,
            use_reranker=use_reranker,
            use_hyde=use_hyde,
            score_threshold=score_threshold_value,
            include_content=True,
        )

        return self._format_search_results(data, max_snippet_chars=400)

    async def _execute_get_document_content(self, *, collection_id: str, document_id: str) -> dict[str, Any]:
        content, content_type = await self.api_client.get_document_content(collection_id, document_id)
        content_type_value = content_type or "application/octet-stream"

        if b"\x00" in content:
            return {
                "content_type": content_type_value,
                "error": "Document content appears to be binary. Prefer get_chunk for extracted text.",
            }

        max_bytes = 200_000
        truncated = False
        if len(content) > max_bytes:
            content = content[:max_bytes]
            truncated = True

        text = content.decode("utf-8", errors="replace")
        return {
            "content_type": content_type_value,
            "truncated": truncated,
            "text": text,
        }

    async def _execute_diagnostics(self) -> dict[str, Any]:
        """Execute the diagnostics tool to gather server status."""
        now = time.monotonic()
        diagnostics: dict[str, Any] = {
            "server_name": "semantik",
            "webui_url": self.api_client.base_url,
        }

        # Check WebUI connection and token validity
        connection_status: dict[str, Any] = {"connected": False}
        try:
            # Try to fetch profiles to verify connection and auth
            profiles = await self.api_client.get_profiles(enabled_only=True)
            connection_status["connected"] = True
            connection_status["authenticated"] = True
            connection_status["profile_count"] = len(profiles)
        except SemantikAPIError as e:
            error_str = str(e)
            connection_status["error"] = error_str
            if "401" in error_str or "unauthorized" in error_str.lower():
                connection_status["authenticated"] = False
            elif "connection" in error_str.lower() or "refused" in error_str.lower():
                connection_status["connected"] = False
            profiles = []
        diagnostics["connection"] = connection_status

        # Profile filter info (if active)
        if self.profile_filter:
            diagnostics["profile_filter"] = list(self.profile_filter)
            # Filter profiles as the server would
            profiles = [p for p in profiles if p.get("name") in self.profile_filter]

        # Profile summary
        profile_summaries: list[dict[str, Any]] = []
        for profile in profiles:
            collections = profile.get("collections") or []
            profile_summaries.append(
                {
                    "name": profile.get("name"),
                    "enabled": profile.get("enabled", False),
                    "collection_count": len(collections),
                    "search_type": profile.get("search_type", "semantic"),
                    "use_reranker": profile.get("use_reranker", True),
                }
            )
        diagnostics["profiles"] = profile_summaries
        diagnostics["available_tools"] = (
            len(profile_summaries)  # search_* tools
            + 5  # get_document, get_document_content, get_chunk, list_documents, diagnostics
        )

        # Cache status
        cache_status: dict[str, Any] = {}
        if self._profiles_cache is not None:
            cache_status["cached"] = True
            cache_status["cached_profile_count"] = len(self._profiles_cache)
            ttl_remaining = max(0.0, self._profiles_cache_expires_at - now)
            cache_status["ttl_remaining_seconds"] = round(ttl_remaining, 2)
        else:
            cache_status["cached"] = False
        diagnostics["cache"] = cache_status

        return diagnostics

    @staticmethod
    def _format_search_results(raw: dict[str, Any], *, max_snippet_chars: int) -> dict[str, Any]:
        results_in = raw.get("results") or []
        if not isinstance(results_in, list):
            results_in = []

        results_out: list[dict[str, Any]] = []
        for item in results_in:
            if not isinstance(item, dict):
                continue
            snippet = item.get("text") or ""
            if isinstance(snippet, str):
                snippet = snippet.strip()
                if len(snippet) > max_snippet_chars:
                    snippet = snippet[:max_snippet_chars].rstrip() + "…"
            else:
                snippet = ""

            results_out.append(
                {
                    "collection_id": item.get("collection_id"),
                    "collection_name": item.get("collection_name"),
                    "document_id": item.get("document_id"),
                    "chunk_id": item.get("chunk_id"),
                    "score": item.get("score"),
                    "file_path": item.get("file_path"),
                    "file_name": item.get("file_name"),
                    "text": snippet,
                    "metadata": item.get("metadata") or {},
                }
            )

        return {
            "query": raw.get("query"),
            "total_results": raw.get("total_results", len(results_out)),
            "results": results_out,
            "partial_failure": raw.get("partial_failure", False),
            "failed_collections": raw.get("failed_collections"),
        }

    @staticmethod
    def _as_text_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
        return {
            "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
            "isError": is_error,
        }

    async def run(self) -> None:
        logger.info("Starting Semantik MCP server (stdio transport)")
        try:
            async with stdio_server() as (read_stream, write_stream):
                logger.info("MCP server ready, handling requests...")
                await self.server.run(read_stream, write_stream, self.server.create_initialization_options())
        except Exception as e:
            logger.error("MCP server failed: %s", e, exc_info=True)
            raise
        finally:
            logger.info("Shutting down MCP server")
            await self.api_client.close()

    async def run_http(self, host: str = "0.0.0.0", port: int = 9090) -> None:
        """Run the MCP server with HTTP transport.

        In service mode (with internal_api_key), adds authentication middleware
        that validates client API keys on incoming requests.
        """
        import uvicorn

        logger.info("Starting Semantik MCP HTTP server on %s:%d", host, port)
        if self._service_mode:
            logger.info("Service mode enabled - client API key authentication required")

        session_manager = StreamableHTTPSessionManager(
            app=self.server,
            json_response=False,
            stateless=False,
        )

        @asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
            async with session_manager.run():
                try:
                    yield
                finally:
                    await self.api_client.close()

        async def health_endpoint(_request: Request) -> JSONResponse:
            return JSONResponse({"status": "healthy", "server": "semantik-mcp"})

        # Build middleware list
        middleware_list = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ]

        # Add auth middleware in service mode
        if self._service_mode:
            middleware_list.append(
                Middleware(MCPAuthMiddleware, api_client=self.api_client),
            )

        app = Starlette(
            routes=[
                Route("/health", endpoint=health_endpoint, methods=["GET"]),
                Route(
                    "/mcp/{path:path}",
                    endpoint=session_manager.handle_request,
                    methods=["GET", "POST", "DELETE"],
                ),
                Route(
                    "/mcp",
                    endpoint=session_manager.handle_request,
                    methods=["GET", "POST", "DELETE"],
                ),
            ],
            lifespan=lifespan,
            middleware=middleware_list,
        )

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
