<component>
  <name>WebUI Service</name>
  <purpose>FastAPI backend service handling API routing, authentication, WebSocket management, and task orchestration</purpose>
  <location>packages/webui/</location>
</component>

<architecture>
  <pattern>Three-layer: API Routers → Service Layer → Repository Layer</pattern>
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
  <file path="main.py">Application entry point, FastAPI app initialization</file>
  <file path="app.py">FastAPI app factory with middleware setup</file>
  <file path="celery_app.py">Celery worker configuration</file>
  <file path="tasks.py">Async task definitions for collections</file>
  <file path="chunking_tasks.py">Chunking-specific async tasks</file>
  <file path="websocket_manager.py">WebSocket connection management</file>
  <file path="auth.py">JWT authentication implementation</file>
  <file path="dependencies.py">FastAPI dependency injection</file>
</key-files>

<api-structure>
  <router path="api/v2/collections.py">Collection CRUD operations</router>
  <router path="api/v2/operations.py">Async operation management</router>
  <router path="api/v2/search.py">Search endpoints</router>
  <router path="api/v2/documents.py">Document management</router>
  <router path="api/v2/chunking.py">Chunking strategies and preview</router>
</api-structure>

<services>
  <service name="CollectionService">
    <responsibility>Collection lifecycle management</responsibility>
    <location>services/collection_service.py</location>
    <critical-methods>
      - create_collection: Creates collection and dispatches INDEX operation
      - add_source: Adds source and dispatches APPEND operation
      - reindex: Dispatches REINDEX operation
      - delete_collection: Handles cascading deletion
    </critical-methods>
  </service>
  
  <service name="ChunkingService">
    <responsibility>Document chunking orchestration</responsibility>
    <location>services/chunking_service.py</location>
    <critical-methods>
      - preview_chunks: Preview with caching
      - execute_ingestion_chunking: Production chunking with fallback
      - compare_strategies: Multi-strategy comparison
    </critical-methods>
  </service>
  
  <service name="SearchService">
    <responsibility>Search orchestration and reranking</responsibility>
    <location>services/search_service.py</location>
  </service>
</services>

<middleware-stack>
  <!-- Order matters! -->
  <middleware order="1">CORS - Cross-origin requests</middleware>
  <middleware order="2">RequestID - Unique request tracking</middleware>
  <middleware order="3">Correlation - Distributed tracing</middleware>
  <middleware order="4">RateLimit - Request throttling</middleware>
  <middleware order="5">Exception - Global error handling</middleware>
</middleware-stack>

<websocket>
  <architecture>Redis Pub/Sub for horizontal scaling</architecture>
  <limits>10 connections per user, 10,000 total</limits>
  <authentication>JWT token via first message after connection</authentication>
  <channels>
    - operation-progress:{operation_id}
    - collection-updates:{collection_id}
  </channels>
</websocket>

<celery-tasks>
  <pattern>Transaction BEFORE task dispatch</pattern>
  <critical-flow>
    1. Create operation in DB
    2. Commit transaction
    3. Dispatch Celery task
    4. Return operation ID to client
  </critical-flow>
  <task-types>
    - INDEX: Initial collection indexing
    - APPEND: Add documents to collection
    - REINDEX: Blue-green reindexing
    - DELETE: Collection deletion
    - REMOVE_SOURCE: Remove documents by source
  </task-types>
</celery-tasks>

<security>
  <authentication>JWT with 24h access, 30d refresh tokens</authentication>
  <authorization>Owner-based collection access</authorization>
  <validation>Pydantic models for all inputs</validation>
  <secrets>JWT_SECRET required in production</secrets>
</security>

<testing>
  <requirement>All new endpoints need integration tests</requirement>
  <location>tests/webui/</location>
  <patterns>Use TestClient, mock Redis/Celery</patterns>
</testing>

<common-pitfalls>
  <pitfall>
    <issue>Forgetting to commit before Celery dispatch</issue>
    <consequence>Race condition - task runs before data exists</consequence>
  </pitfall>
  <pitfall>
    <issue>Business logic in routers</issue>
    <consequence>Untestable, violates separation of concerns</consequence>
  </pitfall>
  <pitfall>
    <issue>Not checking collection status before operations</issue>
    <consequence>Concurrent operations conflict</consequence>
  </pitfall>
</common-pitfalls>