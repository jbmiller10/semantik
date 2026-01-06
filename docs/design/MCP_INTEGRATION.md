# MCP Integration Design Document

Model Context Protocol (MCP) support for Semantik, enabling LLM agents to search collections via standardized tool interfaces.

## Overview

### Goals

1. Allow LLM clients (Claude Desktop, Cursor, custom agents) to search Semantik collections
2. Support multiple "search profiles" with scoped collection access
3. Provide configurable default search parameters per profile
4. Enable profile filtering at runtime to prevent tool confusion in specialized agents
5. Manage profiles via WebUI with easy client configuration export

### Non-Goals (v1)

- Write operations (creating collections, adding documents)
- SSE transport (stdio only for v1)
- Multi-user/multi-tenant complexity (self-hosted focus)
- Plugin architecture (core feature)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        MCP Clients                               │
│  (Claude Desktop, Cursor, Custom Agents)                         │
└──────────────────┬───────────────────────────────────────────────┘
                   │ stdio (JSON-RPC)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Semantik MCP Server                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Profile Filter (--profile flag)                           │  │
│  │  Only exposes tools for specified profile(s)               │  │
│  └────────────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Tools                                                     │  │
│  │  • search_{name}  - Query collections (scoped by profile)  │  │
│  │  • get_document   - Retrieve document metadata             │  │
│  │  • get_chunk      - Retrieve chunk content                 │  │
│  │  • list_documents - List documents in a collection          │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────┬───────────────────────────────────────────────┘
                   │ HTTP (internal)
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Semantik API (WebUI)                           │
│  • /api/v2/search                                                │
│  • /api/v2/collections                                           │
│  • /api/v2/collections/{collection}/documents                    │
│  • /api/v2/collections/{collection}/documents/{document}          │
│  • /api/v2/collections/{collection}/documents/{document}/content  │
│  • /api/v2/chunking/collections/{collection}/chunks               │
│  • /api/v2/chunking/collections/{collection}/chunks/{chunk}       │
│  • /api/v2/mcp/profiles                                          │
└──────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Single MCP Server, Multiple Profiles**: One server implementation that dynamically exposes tools based on configured profiles. Profile filtering happens at invocation time via CLI flags.

2. **Profile-Scoped Search**: Each profile defines which collections are searchable and default search parameters. When `--profile coding` is used, only the `coding` profile's collections are accessible.

3. **Core Feature, Not Plugin**: MCP is tightly coupled to collections and search. The plugin system is for swappable implementations (embeddings, chunking), not for features that span the core domain.

4. **Authenticate via WebUI**: The MCP server authenticates to the WebUI API using Bearer auth. For persistent MCP clients, prefer long-lived API keys; JWT access tokens also work but expire (default 24 hours) and require refresh.

## Data Model

### MCPProfile

New database model for MCP search profiles:

```python
# packages/shared/database/models.py

from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import relationship

class MCPProfile(Base):
    """MCP search profile configuration."""

    __tablename__ = "mcp_profiles"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))  # UUID as string

    # Identity
    name = Column(String(64), nullable=False)  # e.g., "coding", "personal", "taxes"

    description = Column(Text, nullable=False)  # Shown to LLM as tool description

    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    owner = relationship("User", back_populates="mcp_profiles")

    enabled = Column(Boolean, nullable=False, default=True)

    # Search defaults
    search_type = Column(String(32), nullable=False, default="semantic")  # semantic, hybrid, keyword, question, code

    result_count = Column(Integer, nullable=False, default=10)

    use_reranker = Column(Boolean, nullable=False, default=True)

    score_threshold = Column(Float, nullable=True)

    hybrid_alpha = Column(Float, nullable=True)  # Only used when search_type=hybrid

    # Metadata
    created_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), onupdate=func.now())

    # Relationships
    collections = relationship("Collection", secondary="mcp_profile_collections", back_populates="mcp_profiles")

    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_mcp_profiles_owner_name"),)


class MCPProfileCollection(Base):
    """Junction table for MCP profile to collection mapping."""

    __tablename__ = "mcp_profile_collections"

    profile_id = Column(
        String,
        ForeignKey("mcp_profiles.id", ondelete="CASCADE"),
        primary_key=True,
    )

    collection_id = Column(
        String,
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Ordering: lower values are searched first (affects result ranking)
    order = Column(Integer, nullable=False, default=0)


# Add to User model:
# mcp_profiles = relationship("MCPProfile", back_populates="owner", cascade="all, delete-orphan")

# Add to Collection model:
# mcp_profiles = relationship("MCPProfile", secondary="mcp_profile_collections", back_populates="collections")
```

Model relationships:
- `User.mcp_profiles`: one-to-many (add `cascade="all, delete-orphan"`)
- `Collection.mcp_profiles`: many-to-many via `mcp_profile_collections`

### Migration

```python
# alembic/versions/xxxx_add_mcp_profiles.py

def upgrade():
    op.create_table(
        "mcp_profiles",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("search_type", sa.String(32), nullable=False, server_default="semantic"),
        sa.Column("result_count", sa.Integer, nullable=False, server_default=sa.text("10")),
        sa.Column("use_reranker", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("score_threshold", sa.Float, nullable=True),
        sa.Column("hybrid_alpha", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("owner_id", "name", name="uq_mcp_profiles_owner_name"),
    )
    op.create_index("ix_mcp_profiles_owner_id", "mcp_profiles", ["owner_id"])

    op.create_table(
        "mcp_profile_collections",
        sa.Column("profile_id", sa.String(),
                  sa.ForeignKey("mcp_profiles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("collection_id", sa.String(),
                  sa.ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("order", sa.Integer, nullable=False, server_default=sa.text("0")),
    )
```

## API Endpoints

### Profile Management

```
POST   /api/v2/mcp/profiles              Create profile
GET    /api/v2/mcp/profiles              List profiles
GET    /api/v2/mcp/profiles/{id}         Get profile
PUT    /api/v2/mcp/profiles/{id}         Update profile
DELETE /api/v2/mcp/profiles/{id}         Delete profile
GET    /api/v2/mcp/profiles/{id}/config  Get client config snippet
```

List profiles query parameters:
- `enabled`: filter by enabled state
- `name`: filter by exact profile name (optional)
- `page`, `per_page`: pagination

### Read Endpoints Used by MCP Tools

```
POST   /api/v2/search
GET    /api/v2/collections/{collection_uuid}/documents
GET    /api/v2/collections/{collection_uuid}/documents/{document_uuid}
GET    /api/v2/collections/{collection_uuid}/documents/{document_uuid}/content
GET    /api/v2/chunking/collections/{collection_uuid}/chunks/{chunk_id}
```

**Note**: All profile endpoints (`/api/v2/mcp/profiles*`) implicitly filter by the authenticated user. A token can only see and manage profiles owned by its associated user.

### Request/Response Models

```python
# packages/webui/api/v2/schemas/mcp.py

class MCPProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = Field(..., min_length=1, max_length=1000)
    collection_ids: list[str] = Field(..., min_length=1, description="Collection UUIDs (as strings), ordered by search priority")
    enabled: bool = True
    search_type: Literal["semantic", "hybrid", "keyword", "question", "code"] = "semantic"
    result_count: int = Field(default=10, ge=1, le=100)
    use_reranker: bool = True
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid_alpha: float | None = Field(default=None, ge=0.0, le=1.0)


class MCPProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    description: str | None = Field(default=None, min_length=1, max_length=1000)
    collection_ids: list[str] | None = Field(default=None, min_length=1, description="Collection UUIDs (as strings), ordered by search priority")
    enabled: bool | None = None
    search_type: Literal["semantic", "hybrid", "keyword", "question", "code"] | None = None
    result_count: int | None = Field(default=None, ge=1, le=100)
    use_reranker: bool | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    hybrid_alpha: float | None = Field(default=None, ge=0.0, le=1.0)


class MCPProfileResponse(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool
    search_type: str
    result_count: int
    use_reranker: bool
    score_threshold: float | None
    hybrid_alpha: float | None
    collections: list[CollectionSummary]
    created_at: datetime
    updated_at: datetime


class MCPClientConfig(BaseModel):
    """Claude Desktop / MCP client configuration snippet."""
    server_name: str
    command: str
    args: list[str]
    env: dict[str, str]
```

### Endpoint Implementation

```python
# packages/webui/api/v2/mcp_profiles.py

router = APIRouter(prefix="/api/v2/mcp/profiles", tags=["mcp"])


@router.post("", response_model=MCPProfileResponse, status_code=201)
async def create_profile(
    profile: MCPProfileCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new MCP search profile."""
    service = MCPProfileService(db)
    return await service.create(profile, owner=current_user)


@router.get("", response_model=list[MCPProfileResponse])
async def list_profiles(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all MCP profiles for the current user."""
    service = MCPProfileService(db)
    return await service.list_for_user(current_user.id)


@router.get("/{profile_id}", response_model=MCPProfileResponse)
async def get_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific MCP profile."""
    service = MCPProfileService(db)
    return await service.get(profile_id, owner_id=current_user.id)


@router.put("/{profile_id}", response_model=MCPProfileResponse)
async def update_profile(
    profile_id: str,
    profile: MCPProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an MCP profile."""
    service = MCPProfileService(db)
    return await service.update(profile_id, profile, owner_id=current_user.id)


@router.delete("/{profile_id}", status_code=204)
async def delete_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an MCP profile."""
    service = MCPProfileService(db)
    await service.delete(profile_id, owner_id=current_user.id)


@router.get("/{profile_id}/config", response_model=MCPClientConfig)
async def get_client_config(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get MCP client configuration snippet for this profile.
    Returns JSON suitable for Claude Desktop's mcp_servers config.
    """
    service = MCPProfileService(db)
    profile = await service.get(profile_id, owner_id=current_user.id)

    return MCPClientConfig(
        server_name=f"semantik-{profile.name}",
        command="semantik-mcp",
        args=["serve", "--profile", profile.name],
        env={
            "SEMANTIK_WEBUI_URL": settings.WEBUI_URL,
            "SEMANTIK_AUTH_TOKEN": "<your-access-token-or-api-key>",
        },
    )
```

## MCP Server Implementation

### CLI Entry Point

```python
# packages/webui/mcp/cli.py

import asyncio
import click
from .server import SemantikMCPServer


@click.group()
def mcp():
    """MCP server commands."""
    pass


@mcp.command()
@click.option(
    "--profile", "-p",
    multiple=True,
    help="Profile(s) to expose. Can be specified multiple times. "
         "If not specified, all enabled profiles are exposed."
)
@click.option(
    "--webui-url",
    envvar="SEMANTIK_WEBUI_URL",
    default="http://localhost:8080",
    help="Semantik WebUI base URL"
)
@click.option(
    "--auth-token",
    envvar="SEMANTIK_AUTH_TOKEN",
    required=True,
    help="Semantik auth token (JWT access token or API key)"
)
def serve(profile: tuple[str, ...], webui_url: str, auth_token: str):
    """
    Start the MCP server (stdio transport).

    Examples:

        # Expose only the 'coding' profile
        semantik-mcp serve --profile coding

        # Expose all enabled profiles
        semantik-mcp serve
    """
    server = SemantikMCPServer(
        webui_url=webui_url,
        auth_token=auth_token,
        profile_filter=list(profile) if profile else None,
    )
    asyncio.run(server.run())
```

### Server Implementation

```python
# packages/webui/mcp/server.py

import asyncio
import json
import sys
import time
from typing import Any
from mcp import Server, Tool
from mcp.server.stdio import stdio_server

from .client import SemantikAPIClient
from .tools import (
    build_search_tool,
    build_get_document_tool,
    build_get_document_content_tool,
    build_get_chunk_tool,
    build_list_documents_tool,
)


class SemantikMCPServer:
    """MCP server exposing Semantik search capabilities."""

    def __init__(
        self,
        webui_url: str,
        auth_token: str,
        profile_filter: list[str] | None = None,
    ):
        self.api_client = SemantikAPIClient(webui_url, auth_token)
        self.profile_filter = profile_filter
        self.server = Server("semantik")
        self._profiles_cache: list[dict[str, Any]] | None = None
        self._profiles_cache_expires_at: float = 0.0
        self._profiles_cache_lock = asyncio.Lock()
        self._setup_handlers()

    async def _get_profiles_cached(self) -> list[dict[str, Any]]:
        """Fetch profiles with a short TTL cache to reduce per-call latency."""
        now = time.monotonic()
        if self._profiles_cache is not None and now < self._profiles_cache_expires_at:
            return self._profiles_cache

        async with self._profiles_cache_lock:
            now = time.monotonic()
            if self._profiles_cache is not None and now < self._profiles_cache_expires_at:
                return self._profiles_cache

            profiles = await self.api_client.get_profiles(enabled_only=True)
            if self.profile_filter:
                profiles = [p for p in profiles if p["name"] in self.profile_filter]

            self._profiles_cache = profiles
            self._profiles_cache_expires_at = now + 10.0
            return profiles

    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return available tools based on profile filter."""
            profiles = await self._get_profiles_cached()

            tools = []

            # Always expose stable tool names: search_{profile_name}
            for profile in profiles:
                tools.append(
                    build_search_tool(
                        name=f"search_{profile['name']}",
                        description=profile["description"],
                        profile=profile,
                    )
                )

            # Always include utility tools
            tools.append(build_get_document_tool())
            tools.append(build_get_document_content_tool())
            tools.append(build_get_chunk_tool())
            tools.append(build_list_documents_tool())

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
            """Execute a tool call."""

            profiles = await self._get_profiles_cached()
            profiles_by_name = {p["name"]: p for p in profiles}

            if name.startswith("search_"):
                profile_name = name.removeprefix("search_")
                profile = profiles_by_name.get(profile_name)
                if profile is None:
                    raise ValueError(f"Profile not found or not enabled: {profile_name}")
                return await self._execute_search(profile, arguments)

            # Utility tools must also be scope-checked to prevent bypassing profile filters.
            if name in {"get_document", "get_document_content", "get_chunk", "list_documents"}:
                allowed_collection_ids = {str(c["id"]) for p in profiles for c in p.get("collections", [])}
                if arguments.get("collection_id") not in allowed_collection_ids:
                    raise ValueError("Collection is not accessible via the exposed profile(s)")

            if name == "get_document":
                return await self.api_client.get_document(
                    collection_id=arguments["collection_id"],
                    document_id=arguments["document_id"],
                )

            if name == "get_document_content":
                return await self.api_client.get_document_content(
                    collection_id=arguments["collection_id"],
                    document_id=arguments["document_id"],
                )

            if name == "get_chunk":
                return await self.api_client.get_chunk(
                    collection_id=arguments["collection_id"],
                    chunk_id=arguments["chunk_id"],
                )

            if name == "list_documents":
                return await self.api_client.list_documents(
                    collection_id=arguments["collection_id"],
                    page=arguments.get("page", 1),
                    per_page=arguments.get("per_page", 50),
                    status=arguments.get("status"),
                )

            raise ValueError(f"Unknown tool: {name}")

    async def _execute_search(self, profile, arguments: dict[str, Any]) -> dict:
        """Execute a search using profile defaults and provided arguments."""
        query = arguments["query"]

        # Use profile defaults, allow override
        search_params = {
            "query": query,
            "collection_uuids": [str(c["id"]) for c in profile["collections"]],
            "k": arguments.get("k", profile["result_count"]),
            "search_type": arguments.get("search_type", profile["search_type"]),
            "use_reranker": arguments.get("use_reranker", profile["use_reranker"]),
        }

        score_threshold = arguments.get("score_threshold", profile.get("score_threshold"))
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold

        hybrid_alpha = arguments.get("hybrid_alpha", profile.get("hybrid_alpha"))
        if hybrid_alpha is not None and search_params["search_type"] == "hybrid":
            search_params["hybrid_alpha"] = hybrid_alpha

        if "hybrid_mode" in arguments:
            search_params["hybrid_mode"] = arguments["hybrid_mode"]

        if "keyword_mode" in arguments:
            search_params["keyword_mode"] = arguments["keyword_mode"]

        results = await self.api_client.search(**search_params)

        return self._format_search_results(results)

    def _format_search_results(self, results: dict) -> dict:
        """Format search results for MCP response."""
        formatted = {
            "query": results["query"],
            "total_results": results["total_results"],
            "results": [
                {
                    "collection_id": r["collection_id"],
                    "collection_name": r.get("collection_name"),
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "score": r["score"],
                    "text": r["text"],
                    "file_path": r.get("file_path"),
                    "metadata": r.get("metadata", {}),
                }
                for r in results["results"]
            ],
        }
        return formatted

    async def run(self):
        """Run the MCP server with stdio transport."""
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        finally:
            await self.api_client.close()
```

### Tool Definitions

**Search Pagination Strategy**: Search results are limited to `k=100` maximum per call. This is intentional:
- Semantic search quality degrades beyond the top 100 results
- Large result sets bloat LLM context windows unnecessarily
- For comprehensive coverage, LLMs should refine queries rather than paginate

When an LLM needs more than 100 results, the recommended approach is:
1. Use multiple focused queries with different keywords/angles
2. Use `list_documents` to browse a collection's full contents
3. Use `get_document_content` to retrieve specific documents in full

```python
# packages/webui/mcp/tools.py

from mcp import Tool


def build_search_tool(name: str, description: str, profile) -> Tool:
    """Build a search tool for a profile."""
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (natural language)",
                },
                "k": {
                    "type": "integer",
                    "description": f"Number of results to return (default: {profile['result_count']})",
                    "minimum": 1,
                    "maximum": 100,
                },
                "search_type": {
                    "type": "string",
                    "enum": ["semantic", "hybrid", "keyword", "question", "code"],
                    "description": f"Search type (default: {profile['search_type']})",
                },
                "use_reranker": {
                    "type": "boolean",
                    "description": f"Enable reranking (default: {profile['use_reranker']})",
                },
                "score_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum score threshold (overrides profile default if set)",
                },
                "hybrid_alpha": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Hybrid alpha (only when search_type=hybrid)",
                },
                "hybrid_mode": {
                    "type": "string",
                    "enum": ["weighted", "filter"],
                    "description": "Hybrid mode (weighted or filter)",
                },
                "keyword_mode": {
                    "type": "string",
                    "enum": ["any", "all"],
                    "description": "Keyword matching mode when using hybrid search",
                },
            },
            "required": ["query"],
        },
    )


def build_get_document_tool() -> Tool:
    """Build the get_document tool."""
    return Tool(
        name="get_document",
        description="Retrieve document metadata by ID. Use this to get more information about a document from search results.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {
                    "type": "string",
                    "description": "The collection ID (UUID)",
                },
                "document_id": {
                    "type": "string",
                    "description": "The document ID (UUID)",
                },
            },
            "required": ["collection_id", "document_id"],
        },
    )


def build_get_document_content_tool() -> Tool:
    """Build the get_document_content tool."""
    return Tool(
        name="get_document_content",
        description="Retrieve the full content of a document. Use this when you need the complete document text, not just chunks. Warning: may be large for big documents.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {
                    "type": "string",
                    "description": "The collection ID (UUID)",
                },
                "document_id": {
                    "type": "string",
                    "description": "The document ID (UUID)",
                },
            },
            "required": ["collection_id", "document_id"],
        },
    )


def build_get_chunk_tool() -> Tool:
    """Build the get_chunk tool."""
    return Tool(
        name="get_chunk",
        description="Retrieve the full content of a chunk by ID. Use this to get complete text when search results are truncated.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {
                    "type": "string",
                    "description": "The collection ID (UUID). Required because chunks are stored in a partitioned table.",
                },
                "chunk_id": {
                    "type": "string",
                    "description": "The chunk ID",
                },
            },
            "required": ["collection_id", "chunk_id"],
        },
    )


def build_list_documents_tool() -> Tool:
    """Build the list_documents tool."""
    return Tool(
        name="list_documents",
        description="List documents in a collection (paginated). Use this when you need document IDs and metadata.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {
                    "type": "string",
                    "description": "The collection ID (UUID)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                },
                "per_page": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                },
                "status": {
                    "type": "string",
                    "description": "Optional document status filter (e.g., completed, failed)",
                },
            },
            "required": ["collection_id"],
        },
    )
```

### API Client

```python
# packages/webui/mcp/client.py

import httpx
from typing import Any


class SemantikAPIClient:
    """HTTP client for Semantik WebUI API."""

    def __init__(self, webui_url: str, auth_token: str):
        self.base_url = webui_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {auth_token}"}
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=30.0,
        )

    async def get_profiles(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Fetch MCP profiles visible to the token."""
        params = {"enabled": "true"} if enabled_only else None
        response = await self._client.get("/api/v2/mcp/profiles", params=params)
        response.raise_for_status()
        return response.json()

    async def search(self, **params) -> dict:
        """Execute a search."""
        response = await self._client.post("/api/v2/search", json=params)
        response.raise_for_status()
        return response.json()

    async def list_documents(
        self,
        collection_id: str,
        page: int = 1,
        per_page: int = 50,
        status: str | None = None,
    ) -> dict:
        """List documents in a collection (paginated)."""
        params = {"page": page, "per_page": per_page}
        if status is not None:
            params["status"] = status
        response = await self._client.get(f"/api/v2/collections/{collection_id}/documents", params=params)
        response.raise_for_status()
        return response.json()

    async def get_document(self, collection_id: str, document_id: str) -> dict:
        """Get document metadata."""
        response = await self._client.get(f"/api/v2/collections/{collection_id}/documents/{document_id}")
        response.raise_for_status()
        return response.json()

    async def get_document_content(self, collection_id: str, document_id: str) -> dict:
        """Get full document content."""
        response = await self._client.get(f"/api/v2/collections/{collection_id}/documents/{document_id}/content")
        response.raise_for_status()
        return response.json()

    async def get_chunk(self, collection_id: str, chunk_id: str) -> dict:
        """Get chunk content."""
        response = await self._client.get(f"/api/v2/chunking/collections/{collection_id}/chunks/{chunk_id}")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self._client.aclose()
```

## WebUI Components

### Routes

```
/settings/mcp          MCP Profiles list page
/settings/mcp/new      Create profile page
/settings/mcp/:id      Edit profile page
```

### TypeScript Types

```typescript
// apps/webui-react/src/components/mcp/types.ts

export type MCPProfile = {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  search_type: 'semantic' | 'hybrid' | 'keyword' | 'question' | 'code';
  result_count: number;
  use_reranker: boolean;
  score_threshold: number | null;
  hybrid_alpha: number | null;
  collections: Array<{ id: string; name: string }>;
  created_at: string;
  updated_at: string;
};

export type MCPProfileFormData = {
  name: string;
  description: string;
  collection_ids: string[];
  enabled: boolean;
  search_type: MCPProfile['search_type'];
  result_count: number;
  use_reranker: boolean;
  score_threshold: number | null;
  hybrid_alpha: number | null;
};

export type ConfigModalProps = {
  open: boolean;
  onClose: (open: boolean) => void;
  profile: MCPProfile;
};
```

### Profile List Page

```typescript
// apps/webui-react/src/pages/settings/MCPProfilesPage.tsx

import { useMCPProfiles, useDeleteMCPProfile } from '@/hooks/useMCPProfiles';

export function MCPProfilesPage() {
  const { data: profiles, isLoading } = useMCPProfiles();
  const deleteProfile = useDeleteMCPProfile();

  return (
    <SettingsLayout>
      <PageHeader
        title="MCP Profiles"
        description="Configure search profiles for MCP clients like Claude Desktop"
        action={
          <Button as={Link} to="/settings/mcp/new">
            Create Profile
          </Button>
        }
      />

      {isLoading ? (
        <LoadingSpinner />
      ) : profiles?.length === 0 ? (
        <EmptyState
          title="No MCP profiles"
          description="Create a profile to expose your collections to AI assistants"
          action={<Button as={Link} to="/settings/mcp/new">Create Profile</Button>}
        />
      ) : (
        <ProfileList>
          {profiles?.map((profile) => (
            <ProfileCard
              key={profile.id}
              profile={profile}
              onDelete={() => deleteProfile.mutate(profile.id)}
            />
          ))}
        </ProfileList>
      )}
    </SettingsLayout>
  );
}
```

### Profile Card Component

```typescript
// apps/webui-react/src/components/mcp/ProfileCard.tsx

interface ProfileCardProps {
  profile: MCPProfile;
  onDelete: () => void;
}

export function ProfileCard({ profile, onDelete }: ProfileCardProps) {
  const [showConfig, setShowConfig] = useState(false);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>{profile.name}</CardTitle>
            <CardDescription>{profile.description}</CardDescription>
          </div>
          <Toggle
            checked={profile.enabled}
            onChange={() => {/* toggle enabled */}}
          />
        </div>
      </CardHeader>

      <CardContent>
        <div className="space-y-2 text-sm">
          <div>
            <span className="text-muted-foreground">Collections: </span>
            {profile.collections.map(c => c.name).join(', ')}
          </div>
          <div>
            <span className="text-muted-foreground">Search type: </span>
            {profile.search_type}
          </div>
          <div>
            <span className="text-muted-foreground">Results: </span>
            {profile.result_count}
            {profile.use_reranker && ' (with reranking)'}
          </div>
        </div>
      </CardContent>

      <CardFooter className="flex justify-between">
        <div className="flex gap-2">
          <Button variant="outline" as={Link} to={`/settings/mcp/${profile.id}`}>
            Edit
          </Button>
          <Button variant="outline" onClick={() => setShowConfig(true)}>
            Connection Info
          </Button>
        </div>
        <Button variant="ghost" onClick={onDelete}>
          Delete
        </Button>
      </CardFooter>

      <ConfigModal
        open={showConfig}
        onClose={() => setShowConfig(false)}
        profile={profile}
      />
    </Card>
  );
}
```

### Config Modal

```typescript
// apps/webui-react/src/components/mcp/ConfigModal.tsx

export function ConfigModal({ open, onClose, profile }: ConfigModalProps) {
  const { data: config } = useMCPProfileConfig(profile.id);

  const toolName = `search_${profile.name}`;

  const configJson = JSON.stringify({
    [config?.server_name ?? '']: {
      command: config?.command,
      args: config?.args,
      env: config?.env,
    }
  }, null, 2);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Claude Desktop Configuration</DialogTitle>
          <DialogDescription>
            Add this to your Claude Desktop config file
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Tool name - helps users understand how to invoke this profile */}
          <div>
            <Label>Tool name</Label>
            <div className="flex items-center gap-2">
              <code className="text-sm bg-muted px-3 py-2 rounded font-mono">
                {toolName}
              </code>
              <CopyButton value={toolName} />
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              Claude will use this tool name to search this profile's collections.
            </p>
          </div>

          <div>
            <Label>Config file location</Label>
            <div className="space-y-2">
              <code className="block text-sm bg-muted p-2 rounded">
                macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
              </code>
              <code className="block text-sm bg-muted p-2 rounded">
                Linux: ~/.config/Claude/claude_desktop_config.json
              </code>
              <code className="block text-sm bg-muted p-2 rounded">
                Windows: %APPDATA%\\Claude\\claude_desktop_config.json
              </code>
            </div>
          </div>

          <div>
            <Label>Add to mcpServers</Label>
            <div className="relative">
              <pre className="text-sm bg-muted p-4 rounded overflow-x-auto">
                {configJson}
              </pre>
              <CopyButton value={configJson} className="absolute top-2 right-2" />
            </div>
          </div>

          <Alert>
            <AlertDescription>
              Replace <code>&lt;your-access-token-or-api-key&gt;</code> with a token
              that has access to the collections in this profile (API keys are
              recommended for persistent MCP clients).
            </AlertDescription>
          </Alert>
        </div>
      </DialogContent>
    </Dialog>
  );
}
```

### Create/Edit Form

```typescript
// apps/webui-react/src/components/mcp/ProfileForm.tsx

export function ProfileForm({ profile, onSubmit }: ProfileFormProps) {
  const { data: collections } = useCollections();

  const form = useForm<MCPProfileFormData>({
    defaultValues: profile ?? {
      name: '',
      description: '',
      collection_ids: [],
      enabled: true,
      search_type: 'semantic',
      result_count: 10,
      use_reranker: true,
      score_threshold: null,
      hybrid_alpha: null,
    },
  });

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        {/* Basic Info */}
        <FormField
          name="name"
          label="Profile Name"
          description="Used as the tool name (lowercase, no spaces)"
        >
          <Input placeholder="coding" pattern="^[a-z][a-z0-9_-]*$" />
        </FormField>

        <FormField
          name="description"
          label="Description"
          description="Shown to the AI to help it understand what this profile searches"
        >
          <Textarea
            placeholder="Search coding documentation, API references, and technical guides"
          />
        </FormField>

        {/* Collection Selection */}
        <FormField
          name="collection_ids"
          label="Collections"
          description="Which collections this profile can search"
        >
          <CollectionMultiSelect collections={collections} />
        </FormField>

        {/* Search Defaults */}
        <Collapsible>
          <CollapsibleTrigger>Search Defaults</CollapsibleTrigger>
          <CollapsibleContent className="space-y-4 pt-4">
            <FormField name="search_type" label="Search Type">
              <Select>
                <SelectItem value="semantic">Semantic</SelectItem>
                <SelectItem value="hybrid">Hybrid</SelectItem>
                <SelectItem value="keyword">Keyword</SelectItem>
                <SelectItem value="question">Question</SelectItem>
                <SelectItem value="code">Code</SelectItem>
              </Select>
            </FormField>

            <FormField name="result_count" label="Default Results">
              <Input type="number" min={1} max={100} />
            </FormField>

            <FormField name="use_reranker" label="Use Reranker">
              <Switch />
            </FormField>

            <FormField name="score_threshold" label="Score Threshold">
              <Input type="number" min={0} max={1} step={0.1} />
            </FormField>

            {form.watch('search_type') === 'hybrid' && (
              <FormField name="hybrid_alpha" label="Hybrid Alpha">
                <Slider min={0} max={1} step={0.1} />
              </FormField>
            )}
          </CollapsibleContent>
        </Collapsible>

        <div className="flex justify-end gap-2">
          <Button type="button" variant="outline" onClick={() => navigate(-1)}>
            Cancel
          </Button>
          <Button type="submit">
            {profile ? 'Update Profile' : 'Create Profile'}
          </Button>
        </div>
      </form>
    </Form>
  );
}
```

### React Query Hooks

```typescript
// apps/webui-react/src/hooks/useMCPProfiles.ts

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { mcpApi } from '@/services/api/v2/mcp';

export function useMCPProfiles() {
  return useQuery({
    queryKey: ['mcp-profiles'],
    queryFn: () => mcpApi.listProfiles(),
  });
}

export function useMCPProfile(id: string) {
  return useQuery({
    queryKey: ['mcp-profiles', id],
    queryFn: () => mcpApi.getProfile(id),
  });
}

export function useMCPProfileConfig(id: string) {
  return useQuery({
    queryKey: ['mcp-profiles', id, 'config'],
    queryFn: () => mcpApi.getProfileConfig(id),
  });
}

export function useCreateMCPProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: mcpApi.createProfile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-profiles'] });
    },
  });
}

export function useUpdateMCPProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: MCPProfileFormData }) =>
      mcpApi.updateProfile(id, data),
    onSuccess: (_, { id }) => {
      queryClient.invalidateQueries({ queryKey: ['mcp-profiles'] });
      queryClient.invalidateQueries({ queryKey: ['mcp-profiles', id] });
    },
  });
}

export function useDeleteMCPProfile() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: mcpApi.deleteProfile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-profiles'] });
    },
  });
}
```

## File Structure

```
packages/
  shared/
    database/
      models.py                    # Add MCPProfile, MCPProfileCollection

  webui/
    api/v2/
      mcp_profiles.py              # API endpoints
      documents.py                 # Add document metadata endpoint for MCP tools
      chunking.py                  # Add chunk retrieval endpoint for MCP tools
      schemas/
        mcp.py                     # Pydantic models
    services/
      mcp_profile_service.py       # Business logic
    mcp/
      __init__.py
      cli.py                       # CLI entry point
      server.py                    # MCP server implementation
      client.py                    # Semantik WebUI API client
      tools.py                     # Tool definitions

apps/
  webui-react/
    src/
      pages/settings/
        MCPProfilesPage.tsx        # List page
        MCPProfileCreatePage.tsx   # Create page
        MCPProfileEditPage.tsx     # Edit page
      components/mcp/
        ProfileCard.tsx
        ProfileForm.tsx
        ConfigModal.tsx
        CollectionMultiSelect.tsx
      hooks/
        useMCPProfiles.ts
      services/api/v2/
        mcp.ts                     # API client

alembic/
  versions/
    xxxx_add_mcp_profiles.py       # Migration
```

## Configuration

### Environment Variables

```bash
# MCP Server (used by CLI)
SEMANTIK_WEBUI_URL=http://localhost:8080  # Semantik WebUI base URL
SEMANTIK_AUTH_TOKEN=...                   # JWT access token or API key

# Optional
SEMANTIK_MCP_LOG_LEVEL=INFO               # Logging level for MCP server
```

### pyproject.toml Entry Point

```toml
[project.scripts]
semantik-mcp = "webui.mcp.cli:mcp"
```

### Dependencies

- Python: add the MCP SDK dependency (module imported as `mcp`) alongside existing `httpx`
- CLI: `click` (already used for other CLI entry points)

## Authentication & Authorization

### Flow

1. User obtains an auth token for the MCP server (JWT access token or API key)
2. User creates an MCP profile with scoped collection access
3. MCP server authenticates to the WebUI API using `Authorization: Bearer <token>`
4. Tool calls enforce both (a) token permissions and (b) profile collection scope

Profile validation rules:
- On create/update, validate the authenticated user can access every `collection_id` in the profile.
- When exporting client config, do not embed secrets; always return placeholders.
- At runtime, searches may still fail with 403 if the token is more restricted than the profile; surface the error clearly.

### Security Considerations

- Treat MCP tokens as secrets (Claude/Cursor configs are plaintext on disk)
- The MCP server runs locally, so stdio transport is acceptable
- Profile scoping must apply to all tools (including `get_document`, `get_chunk`, and `list_documents`)
- Do not use the internal `X-Internal-Api-Key` header (reserved for WebUI ↔ search service)

### Error Scenarios

- `401 Not authenticated`: missing/invalid token
- `403 Access denied`: token cannot access collection or collection not in profile scope
- `404 Not found`: profile/document/chunk does not exist (or is not visible to the token)

## Testing Strategy

### Unit Tests

```python
# tests/webui/mcp/test_server.py

async def test_single_profile_exposes_search_tool():
    """Single profile mode exposes a stable profile tool plus a convenience alias."""
    server = SemantikMCPServer(
        webui_url="http://test",
        auth_token="test",
        profile_filter=["coding"],
    )
    # Mock API client
    server.api_client.get_profiles = AsyncMock(return_value=[
        {
            "name": "coding",
            "description": "Code docs",
            "enabled": True,
            "search_type": "semantic",
            "result_count": 10,
            "use_reranker": True,
            "score_threshold": None,
            "hybrid_alpha": None,
            "collections": [],
        }
    ])

    tools = await server.server.list_tools()

    tool_names = [t.name for t in tools]
    assert "search_coding" in tool_names
    assert "search" in tool_names


async def test_multiple_profiles_expose_prefixed_tools():
    """Multiple profiles expose 'search_{name}' tools."""
    server = SemantikMCPServer(...)
    server.api_client.get_profiles = AsyncMock(return_value=[
        {"name": "coding", "description": "Code docs", "enabled": True, "search_type": "semantic", "result_count": 10, "use_reranker": True, "collections": []},
        {"name": "personal", "description": "Personal notes", "enabled": True, "search_type": "semantic", "result_count": 10, "use_reranker": True, "collections": []},
    ])

    tools = await server.server.list_tools()

    tool_names = [t.name for t in tools]
    assert "search_coding" in tool_names
    assert "search_personal" in tool_names
```

### Integration Tests

```python
# tests/webui/api/v2/test_mcp_profiles.py

async def test_create_profile(client, auth_headers, test_collection):
    response = await client.post(
        "/api/v2/mcp/profiles",
        headers=auth_headers,
        json={
            "name": "test-profile",
            "description": "Test description",
            "collection_ids": [str(test_collection.id)],
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-profile"
    assert len(data["collections"]) == 1


async def test_get_client_config(client, auth_headers, test_profile):
    response = await client.get(
        f"/api/v2/mcp/profiles/{test_profile.id}/config",
        headers=auth_headers,
    )

    assert response.status_code == 200
    config = response.json()
    assert config["command"] == "semantik-mcp"
    assert "--profile" in config["args"]
```

### E2E Tests

```python
# tests/e2e/test_mcp_flow.py

async def test_mcp_search_flow():
    """Test full MCP search flow."""
    # 1. Create collection with documents
    # 2. Create MCP profile for collection
    # 3. Start MCP server in subprocess
    # 4. Send MCP tool call via stdio
    # 5. Verify search results
```

## Implementation Phases

### Phase 1: Foundation (Backend Core)

**Goal**: Establish the data layer and API for MCP profiles.

#### 1.1 Database Models
- [ ] Add `MCPProfile` model to `packages/shared/database/models.py`
  - All fields as specified (id, name, description, owner_id, enabled, search defaults)
  - `__table_args__` with unique constraint on (owner_id, name)
- [ ] Add `MCPProfileCollection` junction table with `order` column
- [ ] Add `mcp_profiles` relationship to `User` model (one-to-many, cascade delete)
- [ ] Add `mcp_profiles` relationship to `Collection` model (many-to-many)
- [ ] Create Alembic migration with proper `nullable=False` and `server_default` values
- [ ] Run migration locally, verify schema

#### 1.2 Pydantic Schemas
- [ ] Create `packages/webui/api/v2/schemas/mcp.py`
- [ ] `MCPProfileCreate` with validation (name pattern, description 1000 chars, collection_ids)
- [ ] `MCPProfileUpdate` (all fields optional)
- [ ] `MCPProfileResponse` (includes collections as summaries)
- [ ] `MCPClientConfig` (server_name, command, args, env)
- [ ] `CollectionSummary` if not already defined (id, name)

#### 1.3 Service Layer
- [ ] Create `packages/webui/services/mcp_profile_service.py`
- [ ] `MCPProfileService` class with async methods:
  - `create(data, owner)` - validate collection access, set order from list position
  - `list_for_user(user_id, enabled_only=False)` - return profiles with collections
  - `get(profile_id, owner_id)` - 404 if not found or wrong owner
  - `update(profile_id, data, owner_id)` - validate collection access if changed
  - `delete(profile_id, owner_id)` - cascade handled by FK
  - `get_config(profile_id, owner_id)` - return MCPClientConfig
- [ ] Collection access validation helper (user can only add collections they own)

#### 1.4 API Endpoints
- [ ] Create `packages/webui/api/v2/mcp_profiles.py`
- [ ] Register router in main app
- [ ] `POST /api/v2/mcp/profiles` - create profile (201)
- [ ] `GET /api/v2/mcp/profiles` - list profiles (filter by enabled, paginate)
- [ ] `GET /api/v2/mcp/profiles/{id}` - get single profile
- [ ] `PUT /api/v2/mcp/profiles/{id}` - update profile
- [ ] `DELETE /api/v2/mcp/profiles/{id}` - delete profile (204)
- [ ] `GET /api/v2/mcp/profiles/{id}/config` - get client config snippet

#### 1.5 Tests
- [ ] Unit tests for `MCPProfileService`
  - Test create with valid/invalid collection_ids
  - Test owner scoping (can't see other users' profiles)
  - Test collection ordering preserved
- [ ] Integration tests for API endpoints
  - Test CRUD operations
  - Test auth required (401)
  - Test owner isolation (403/404)
  - Test config endpoint returns valid structure

**Exit Criteria**: All profile CRUD operations work via API, tests pass.

---

### Phase 2: MCP Server

**Goal**: Implement the MCP server that connects to Semantik's API.

#### 2.1 Dependencies
- [ ] Add `mcp` SDK to dependencies in `pyproject.toml`
- [ ] Add `click` if not already present (for CLI)
- [ ] Add `httpx` if not already present (for async HTTP client)

#### 2.2 API Client
- [ ] Create `packages/webui/mcp/client.py`
- [ ] `SemantikAPIClient` class:
  - Constructor with base_url, auth_token, httpx.AsyncClient
  - `get_profiles(enabled_only=True)` - fetch user's profiles
  - `search(**params)` - POST to /api/v2/search
  - `list_documents(collection_id, page, per_page, status)`
  - `get_document(collection_id, document_id)`
  - `get_document_content(collection_id, document_id)`
  - `get_chunk(collection_id, chunk_id)`
  - `close()` - cleanup
- [ ] Error handling (raise on non-2xx, preserve error messages)

#### 2.3 Tool Definitions
- [ ] Create `packages/webui/mcp/tools.py`
- [ ] `build_search_tool(name, description, profile)` - dynamic schema with profile defaults
- [ ] `build_get_document_tool()` - static schema
- [ ] `build_get_document_content_tool()` - static schema with size warning
- [ ] `build_get_chunk_tool()` - static schema
- [ ] `build_list_documents_tool()` - static schema with pagination

#### 2.4 Server Implementation
- [ ] Create `packages/webui/mcp/server.py`
- [ ] `SemantikMCPServer` class:
  - Constructor with webui_url, auth_token, profile_filter
  - Profile caching with 10s TTL and async lock
  - `_get_profiles_cached()` - fetch and filter profiles
  - `_setup_handlers()` - register list_tools and call_tool
  - `list_tools()` handler:
    - Expose `search_{name}` for each profile
    - Always include utility tools
  - `call_tool()` handler:
    - Route search calls to correct profile
    - Scope-check utility tools against allowed collections
    - Clear error messages for unknown tools / access denied
  - `_execute_search(profile, arguments)` - merge profile defaults with overrides
  - `_format_search_results(results)` - clean output for LLM consumption
  - `run()` - stdio_server context manager

#### 2.5 CLI Entry Point
- [ ] Create `packages/webui/mcp/cli.py`
- [ ] `@click.group() mcp` - command group
- [ ] `@mcp.command() serve` with options:
  - `--profile / -p` (multiple)
  - `--webui-url` (envvar SEMANTIK_WEBUI_URL)
  - `--auth-token` (envvar SEMANTIK_AUTH_TOKEN, required)
- [ ] Add entry point to `pyproject.toml`: `semantik-mcp = "webui.mcp.cli:mcp"`
- [ ] Test CLI locally: `poetry run semantik-mcp serve --help`

#### 2.6 Tests
- [ ] Unit tests for `SemantikMCPServer`
  - Test single profile exposes search_{name} tool
  - Test multiple profiles expose prefixed tools
  - Confirm no generic `search` tool is exposed (always use `search_{profile_name}`)
  - Test utility tools scope-checked against profile collections
- [ ] Unit tests for `SemantikAPIClient` (mock httpx)
- [ ] Integration test: start server, send tool calls via stdio mock

**Exit Criteria**: `semantik-mcp serve --profile <name>` works against running Semantik instance.

---

### Phase 3: WebUI (Frontend)

**Goal**: Build the profile management UI.

#### 3.1 TypeScript Types
- [ ] Create `apps/webui-react/src/types/mcp.ts` (or add to existing types)
- [ ] `MCPProfile` type with all fields
- [ ] `MCPProfileFormData` type for form state
- [ ] `MCPClientConfig` type
- [ ] Add per-profile search result formatting fields (e.g., output mode + snippet limits) so the frontend can configure how `search_{profile}` results are shaped
- [ ] `ConfigModalProps` type

#### 3.2 API Service
- [ ] Create `apps/webui-react/src/services/api/v2/mcp.ts`
- [ ] `mcpApi` object with methods:
  - `listProfiles()` - GET /api/v2/mcp/profiles
  - `getProfile(id)` - GET /api/v2/mcp/profiles/{id}
  - `createProfile(data)` - POST /api/v2/mcp/profiles
  - `updateProfile(id, data)` - PUT /api/v2/mcp/profiles/{id}
  - `deleteProfile(id)` - DELETE /api/v2/mcp/profiles/{id}
  - `getProfileConfig(id)` - GET /api/v2/mcp/profiles/{id}/config

#### 3.3 React Query Hooks
- [ ] Create `apps/webui-react/src/hooks/useMCPProfiles.ts`
- [ ] `useMCPProfiles()` - list query
- [ ] `useMCPProfile(id)` - single profile query
- [ ] `useMCPProfileConfig(id)` - config query
- [ ] `useCreateMCPProfile()` - mutation with cache invalidation
- [ ] `useUpdateMCPProfile()` - mutation with typed params
- [ ] `useDeleteMCPProfile()` - mutation with cache invalidation

#### 3.4 Routes
- [ ] Add routes to router configuration:
  - `/settings/mcp` - list page
  - `/settings/mcp/new` - create page
  - `/settings/mcp/:id` - edit page
- [ ] Add "MCP Profiles" to settings navigation menu

#### 3.5 Components
- [ ] `MCPProfilesPage` - list page with loading/empty states
- [ ] `MCPProfileCreatePage` - wraps ProfileForm for create
- [ ] `MCPProfileEditPage` - fetches profile, wraps ProfileForm for edit
- [ ] `ProfileCard` - displays profile summary, enable toggle, actions
- [ ] `ProfileForm` - form with all fields:
  - Name input with pattern validation
  - Description textarea (1000 char limit shown)
  - CollectionMultiSelect with drag-to-reorder for priority
  - Search type select
  - Result count input
  - Reranker toggle
  - Collapsible advanced section (score_threshold, hybrid_alpha)
  - Search result formatting section:
    - Output mode (e.g., compact vs rich) to control `_format_search_results` output for MCP tools
    - Snippet length / truncation limits to reduce token bloat
    - Optional toggles for including extra fields (paths, scores, metadata) as needed per profile
- [ ] `ConfigModal` - displays:
  - Tool name (`search_{name}`) with copy button
  - Config file locations (macOS/Linux/Windows)
  - JSON config snippet with copy button
  - Token placeholder warning
  - Note on result formatting: search tool returns intentionally compact results; use `get_document`, `get_document_content`, and `get_chunk` for deeper retrieval
- [ ] `CollectionMultiSelect` - searchable multi-select with ordering

#### 3.6 Tests
- [ ] Component tests for ProfileForm validation
- [ ] Component tests for ConfigModal rendering
- [ ] Integration tests for create/edit flows (mock API)

**Exit Criteria**: Can create, edit, delete profiles via WebUI; config modal shows correct info.

---

### Phase 4: Integration & Polish

**Goal**: End-to-end testing, error handling, documentation.

#### 4.1 E2E Tests
- [ ] Create `tests/e2e/test_mcp_flow.py`
- [ ] Test full flow:
  1. Create collection with documents via API
  2. Wait for indexing to complete
  3. Create MCP profile via API
  4. Start MCP server in subprocess
  5. Send list_tools via stdio, verify tools present
  6. Send search tool call, verify results returned
  7. Send get_document tool call, verify metadata
  8. Send get_document_content tool call, verify content
  9. Send get_chunk tool call, verify chunk content
  10. Cleanup (delete profile, collection)
- [ ] Test profile filtering (--profile flag)
- [ ] Test scope enforcement (can't access collections outside profile)

#### 4.2 Error Handling
- [ ] Review all API endpoints for consistent error responses
- [ ] Add specific error codes/messages:
  - `PROFILE_NOT_FOUND` - 404
  - `PROFILE_NAME_EXISTS` - 409 conflict
  - `COLLECTION_ACCESS_DENIED` - 403
  - `COLLECTION_NOT_FOUND` - 404 (when adding to profile)
- [ ] MCP server error handling:
  - Connection refused to WebUI → clear error message
  - Auth token expired/invalid → clear error message
  - Profile not found → tool error response
  - Collection scope violation → tool error response
- [ ] Add retry logic to API client for transient failures

#### 4.3 Logging & Diagnostics
- [ ] Add structured logging to MCP server
  - Log level configurable via SEMANTIK_MCP_LOG_LEVEL
  - Log tool calls (name, profile, duration)
  - Log errors with context
- [ ] Add `--verbose` flag to CLI for debug output
- [ ] Consider adding a `diagnostics` tool for debugging:
  - Lists available profiles
  - Shows connection status
  - Shows token permissions

#### 4.4 Documentation
- [ ] Update main README with MCP feature overview
- [ ] Create `docs/mcp.md` user guide:
  - What is MCP
  - Creating your first profile
  - Configuring Claude Desktop
  - Configuring Cursor
  - Troubleshooting common issues
- [ ] Add inline help text in WebUI forms
- [ ] Document environment variables in deployment docs

#### 4.5 Manual Testing Checklist
- [ ] Fresh install: create user, collection, profile, connect Claude Desktop
- [ ] Search returns relevant results
- [ ] Score threshold filters low-quality results
- [ ] Reranker improves result quality
- [ ] Profile disable stops tool from appearing
- [ ] Delete profile removes from list_tools
- [ ] Token expiry shows clear error
- [ ] Wrong token shows 401 error
- [ ] Collection removed from profile can't be searched

**Exit Criteria**: All tests pass, documentation complete, feature ready for release.

---

### Phase Summary

| Phase | Scope | Key Deliverables |
|-------|-------|------------------|
| 1 | Backend Core | DB models, migration, API, service layer |
| 2 | MCP Server | CLI, server, tools, API client |
| 3 | WebUI | Profile management pages, config modal |
| 4 | Polish | E2E tests, error handling, docs |

**Recommended Order**: Phases 1-2 can be developed in parallel by different developers. Phase 3 depends on Phase 1 (API). Phase 4 depends on all previous phases.

## Future Considerations

### SSE Transport (v2)
For web-based MCP clients, add SSE transport option:
```bash
semantik-mcp serve --transport sse --port 3001
```

### Write Operations (v2)
Extend profiles with write permissions:
- `create_collection`
- `add_document`
- `delete_document`

### Profile Sharing (v2)
Allow sharing profiles between users (for team scenarios).

### Metrics & Logging
Track MCP usage for analytics:
- Queries per profile
- Popular search terms
- Response times
