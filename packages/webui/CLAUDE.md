<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding/removing API endpoints or routers
     - Changing service methods or patterns
     - Modifying authentication/authorization
     - Altering WebSocket channels or limits
     - Adding new Celery tasks or background jobs
     - Adding new middleware or changing order
     - Modifying MCP server tools or profiles
     - Adding/modifying plugin system features
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>WebUI Service</name>
  <purpose>FastAPI backend service handling API routing, authentication, WebSocket management, task orchestration, MCP server, and plugin system</purpose>
  <location>packages/webui/</location>
</component>

<architecture>
  <pattern>Three-layer: API Routers -> Service Layer -> Repository Layer</pattern>
  <key-principle>NEVER put business logic in routers - delegate to services</key-principle>
  <anti-pattern>
    <!-- BAD: Direct DB calls in router -->
    @router.post("/")
    async def create(db: AsyncSession = Depends(get_db)):
        db.add(model)  # WRONG

    <!-- GOOD: Delegate to service -->
    @router.post("/")
    async def create(service: Service = Depends(get_service)):
        return await service.create()  # CORRECT
  </anti-pattern>
</architecture>

<key-files>
  <file path="main.py">Application entry point, FastAPI app factory, lifespan management</file>
  <file path="celery_app.py">Celery worker configuration with beat schedule</file>
  <file path="websocket_manager.py">WebSocket connection management entry point</file>
  <file path="auth.py">JWT authentication, user models, password hashing</file>
  <file path="dependencies.py">FastAPI dependency injection utilities</file>
  <file path="background_tasks.py">Redis cleanup background tasks with circuit breaker</file>
  <file path="startup_tasks.py">Application startup initialization (default data)</file>
</key-files>

<api-structure>
  <base-routers>
    <router path="api/auth.py">Authentication endpoints (login, register, refresh)</router>
    <router path="api/health.py">Health check endpoints</router>
    <router path="api/metrics.py">Prometheus metrics</router>
    <router path="api/internal.py">Internal API endpoints (worker use)</router>
    <router path="api/settings.py">Application settings API</router>
    <router path="api/models.py">Embedding model information</router>
    <router path="api/root.py">Static file serving, catch-all routes</router>
  </base-routers>

  <v2-routers prefix="/api/v2">
    <router path="api/v2/collections.py">Collection CRUD, source management, sync control</router>
    <router path="api/v2/operations.py">Async operation management, WebSocket progress</router>
    <router path="api/v2/search.py">Semantic/hybrid search endpoints</router>
    <router path="api/v2/documents.py">Document management within collections</router>
    <router path="api/v2/chunking.py">Chunking strategies, preview, comparison</router>
    <router path="api/v2/sources.py">Collection source CRUD (nested under collections)</router>
    <router path="api/v2/connectors.py">Connector catalog and preview/validation</router>
    <router path="api/v2/plugins.py">Plugin management, install/uninstall, health</router>
    <router path="api/v2/mcp_profiles.py">MCP profile CRUD, client config generation</router>
    <router path="api/v2/projections.py">UMAP/t-SNE embedding visualizations</router>
    <router path="api/v2/extractors.py">Text extractor plugin listing</router>
    <router path="api/v2/rerankers.py">Reranker plugin listing</router>
    <router path="api/v2/embedding.py">Embedding model endpoints</router>
    <router path="api/v2/directory_scan.py">Directory scanning with WebSocket progress</router>
    <router path="api/v2/partition_monitoring.py">Qdrant partition health monitoring</router>
    <router path="api/v2/system.py">System info, GPU status, resource usage</router>
  </v2-routers>
</api-structure>

<services>
  <service name="CollectionService">
    <responsibility>Collection lifecycle, source management, sync scheduling</responsibility>
    <location>services/collection_service.py</location>
    <critical-methods>
      - create_collection: Creates collection and optionally dispatches INDEX
      - add_source: Adds source with secrets, dispatches APPEND
      - trigger_sync: Manual sync run for continuous collections
      - pause_sync / resume_sync: Sync scheduling control
      - reindex: Blue-green reindexing with validation
      - delete_collection: Cascading deletion with cleanup
    </critical-methods>
  </service>

  <service name="SourceService">
    <responsibility>Collection source CRUD, secret management</responsibility>
    <location>services/source_service.py</location>
    <key-feature>Encrypted secret storage via ConnectorSecretRepository</key-feature>
  </service>

  <service name="ChunkingOrchestrator">
    <responsibility>Coordinates chunking operations across specialized services</responsibility>
    <location>services/chunking/orchestrator.py</location>
    <sub-services>
      - ChunkingProcessor: Core chunking algorithms
      - ChunkingCache: Redis-based caching
      - ChunkingMetrics: Performance tracking
      - ChunkingValidator: Input validation
      - ChunkingConfigManager: Strategy configuration
    </sub-services>
    <usage>
      from webui.services.factory import get_chunking_orchestrator
      orchestrator = await get_chunking_orchestrator(db)
      result = await orchestrator.preview_chunks(content, strategy, config)
    </usage>
  </service>

  <service name="SearchService">
    <responsibility>Multi-collection search, reranking, result aggregation</responsibility>
    <location>services/search_service.py</location>
  </service>

  <service name="PluginService">
    <responsibility>External plugin management, config validation, health checks</responsibility>
    <location>services/plugin_service.py</location>
    <features>
      - JSON Schema config validation with suggestions
      - Health check with metrics recording
      - Plugin state file sync for VecPipe
      - Enable/disable with connector cache invalidation
    </features>
  </service>

  <service name="MCPProfileService">
    <responsibility>MCP search profile management</responsibility>
    <location>services/mcp_profile_service.py</location>
    <key-feature>Generates MCP client configuration for Claude Desktop</key-feature>
  </service>

  <service name="ProjectionService">
    <responsibility>UMAP/t-SNE embedding projections</responsibility>
    <location>services/projection_service.py</location>
  </service>

  <service name="factory.py">
    <responsibility>Dependency injection factory functions for all services</responsibility>
    <usage>
      from webui.services.factory import get_collection_service
      service = Depends(get_collection_service)
    </usage>
  </service>
</services>

<mcp-integration>
  <purpose>Expose Semantik search to MCP-capable LLM clients (Claude Desktop, Cursor)</purpose>
  <location>mcp/</location>
  <components>
    <component path="mcp/server.py">SemantikMCPServer with stdio transport</component>
    <component path="mcp/client.py">SemantikAPIClient for WebUI API calls</component>
    <component path="mcp/tools.py">Tool builders for search, documents, chunks</component>
    <component path="mcp/cli.py">CLI entry point (semantik-mcp serve)</component>
  </components>
  <tools>
    - search_{profile_name}: Semantic search scoped to profile collections
    - get_document: Retrieve document metadata
    - get_document_content: Retrieve full document content
    - get_chunk: Retrieve specific chunk by ID
    - list_documents: Paginated document listing
    - diagnostics: Server and profile status
  </tools>
  <profile-api>
    POST /api/v2/mcp/profiles: Create profile
    GET /api/v2/mcp/profiles: List user profiles
    GET /api/v2/mcp/profiles/{id}/config: Get Claude Desktop config snippet
  </profile-api>
</mcp-integration>

<plugin-system>
  <purpose>Extensible plugin architecture for connectors, extractors, rerankers</purpose>
  <api-prefix>/api/v2/plugins</api-prefix>
  <features>
    - Registry-based plugin discovery (remote + local)
    - Install/uninstall via pip with audit logging
    - JSON Schema config validation with helpful suggestions
    - Health checks with timeout and metrics
    - Enable/disable with state file sync to VecPipe
  </features>
  <plugin-types>embedding, chunking, connector, reranker, extractor</plugin-types>
  <security>
    - Admin-only install/uninstall
    - Package name validation
    - Audit logging for all operations
    - Rate limiting on install/uninstall/health
  </security>
</plugin-system>

<middleware-stack order="execution">
  <!-- Order matters! Listed in execution order -->
  <middleware order="1">CorrelationMiddleware - Request ID/tracing propagation</middleware>
  <middleware order="2">CSPMiddleware - Content Security Policy headers</middleware>
  <middleware order="3">CORSMiddleware - Cross-origin request handling</middleware>
  <middleware order="4">RateLimitMiddleware - SlowAPI rate limiting</middleware>
  <middleware order="5">Exception handlers - Global + chunking-specific</middleware>
</middleware-stack>

<websocket>
  <manager>websocket/scalable_manager.py (ScalableWebSocketManager)</manager>
  <architecture>Redis Pub/Sub for horizontal scaling</architecture>
  <limits>10 connections per user, 10,000 total</limits>
  <authentication>
    <methods>
      - JWT token: Via subprotocol, query parameter (?token=), or first message
      - API key: Via subprotocol, query parameter (?token=), or first message
    </methods>
    <token-routing>Same as HTTP - format-based detection (JWT has dots, API keys have smtk_ prefix)</token-routing>
    <api-key-constraints>API keys never grant superuser access for WebSocket connections</api-key-constraints>
  </authentication>
  <endpoints>
    - /ws/operations: Global operation updates
    - /ws/operations/{operation_id}: Specific operation progress
    - /ws/directory-scan/{scan_id}: Directory scan progress
  </endpoints>
  <channels>
    - operation-progress:{operation_id}
    - collection-updates:{collection_id}
  </channels>
  <features>
    - Lua scripts for atomic connection registration
    - Heartbeat-based connection cleanup
    - Graceful failover with stale detection
  </features>
</websocket>

<celery-tasks>
  <location>tasks/</location>
  <modules>
    <module path="tasks/ingestion.py">INDEX, APPEND, REMOVE_SOURCE operations</module>
    <module path="tasks/reindex.py">Blue-green REINDEX with validation</module>
    <module path="tasks/projection.py">UMAP/t-SNE projection computation</module>
    <module path="tasks/cleanup.py">Periodic cleanup (results, collections, Qdrant)</module>
    <module path="tasks/sync_dispatcher.py">Continuous sync scheduling</module>
    <module path="tasks/utils.py">Shared utilities, constants, base task class</module>
    <module path="chunking_tasks.py">Chunking-specific Celery tasks</module>
  </modules>

  <pattern>Transaction BEFORE task dispatch</pattern>
  <critical-flow>
    1. Create operation record in DB
    2. Commit transaction
    3. Dispatch Celery task with operation_id
    4. Return operation ID to client
  </critical-flow>

  <operation-types>
    - INDEX: Initial collection indexing
    - APPEND: Add documents to collection
    - REINDEX: Blue-green reindexing with validation
    - DELETE: Collection deletion with Qdrant cleanup
    - REMOVE_SOURCE: Remove documents by source
    - PROJECTION: Embedding visualization computation
  </operation-types>

  <beat-schedule>
    - cleanup-old-results: Daily (30 days retention)
    - refresh-collection-chunking-stats: Hourly
    - monitor-partition-health: Every 6 hours
    - dispatch-sync-sources: Every 60 seconds (continuous sync)
  </beat-schedule>
</celery-tasks>

<background-tasks>
  <location>background_tasks.py</location>
  <purpose>Redis memory management and cleanup</purpose>
  <features>
    - Circuit breaker pattern for Redis failures
    - Exponential backoff on consecutive failures
    - Stream length limits (1000 events max)
    - TTL configuration by data type
  </features>
  <ttl-config>
    - operation_active: 24 hours
    - operation_completed: 5 minutes
    - operation_failed: 1 minute
    - websocket_state: 15 minutes
    - cache_default: 5 minutes
  </ttl-config>
</background-tasks>

<security>
  <authentication>
    <jwt>JWT with 24h access tokens, 30d refresh tokens</jwt>
    <api-keys>
      <description>API keys for headless/programmatic access (MCP servers, automation)</description>
      <format>smtk_&lt;uuid_prefix&gt;_&lt;secret&gt; (e.g., smtk_550e8400_Wq3xY5pZ...)</format>
      <usage>Authorization: Bearer &lt;api_key&gt;</usage>
      <token-routing>
        - JWT tokens detected by format: header.payload.signature (2 dots)
        - API keys detected by prefix: smtk_
        - Token format checked first, no fallback between types
      </token-routing>
      <constraints>
        - API keys NEVER grant is_superuser=true (security constraint)
        - Keys include audit metadata: _auth_method, _api_key_id, _api_key_name
        - Expired keys return 401
        - Revoked keys (is_active=false) return 401
        - Inactive user's keys return 401
      </constraints>
      <management>
        - Create: POST /api/v2/api-keys
        - List: GET /api/v2/api-keys
        - Revoke: PATCH /api/v2/api-keys/{id} with {"is_active": false}
      </management>
    </api-keys>
  </authentication>
  <authorization>Owner-based collection access (owner_id check)</authorization>
  <validation>Pydantic models for all API inputs</validation>
  <secrets>
    - JWT_SECRET required in production
    - CONNECTOR_SECRETS_KEY for encrypted source credentials
    - INTERNAL_API_KEY auto-generated for worker communication
  </secrets>
  <rate-limiting>
    - SlowAPI with per-endpoint limits
    - Circuit breaker for rate limit exceeded
    - Special limits for plugin operations
  </rate-limiting>
</security>

<testing>
  <location>tests/webui/</location>
  <patterns>
    - Use TestClient with skip_lifespan=True
    - Mock Redis and Celery for unit tests
    - Use pytest-asyncio for async tests
  </patterns>
  <environment>
    - TESTING=true enables in-memory Celery transports
    - Test fixtures in conftest.py
  </environment>
</testing>

<common-pitfalls>
  <pitfall>
    <issue>Forgetting to commit before Celery dispatch</issue>
    <consequence>Race condition - task runs before data exists</consequence>
    <solution>Always await db.commit() before celery_app.send_task()</solution>
  </pitfall>
  <pitfall>
    <issue>Business logic in routers</issue>
    <consequence>Untestable, violates separation of concerns</consequence>
    <solution>Move logic to service layer, routers only handle HTTP</solution>
  </pitfall>
  <pitfall>
    <issue>Not checking collection status before operations</issue>
    <consequence>Concurrent operations conflict</consequence>
    <solution>Check active operations via OperationRepository</solution>
  </pitfall>
  <pitfall>
    <issue>Using inherited DB connections in Celery workers</issue>
    <consequence>Connection errors after fork</consequence>
    <solution>pg_connection_manager.reset() in worker_process_init signal</solution>
  </pitfall>
  <pitfall>
    <issue>Hardcoding plugin health check timeouts</issue>
    <consequence>Slow plugins cause worker starvation</consequence>
    <solution>Use HEALTH_CHECK_TIMEOUT_SECONDS constant (10s)</solution>
  </pitfall>
</common-pitfalls>

<development>
  <commands>
    - Run server: `make docker-dev-up` or `python -m webui.main`
    - Run worker: `celery -A webui.celery_app worker --loglevel=info`
    - Run beat: `celery -A webui.celery_app beat --loglevel=info`
    - Run tests: `pytest tests/webui/ -v`
    - Lint: `ruff check packages/webui/`
    - Type check: `mypy packages/webui/`
  </commands>
</development>
