<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding new task types or operations
     - Changing progress tracking patterns
     - Modifying timeout or retry configuration
     - Altering async/sync handling patterns
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>Celery Tasks</name>
  <purpose>Async operation execution for INDEX, APPEND, REINDEX, DELETE, PROJECTION, BENCHMARK</purpose>
  <location>packages/webui/tasks/</location>
</component>

<architecture>
  <flow>Service creates Operation → DB commit → Celery dispatch → Task executes → Progress via Redis</flow>
  <key-principle>ALWAYS commit database transaction BEFORE dispatching task</key-principle>
</architecture>

<task-modules>
  <module path="ingestion.py">INDEX, APPEND, REMOVE_SOURCE operations with pipeline executor</module>
  <module path="reindex.py">Blue-green REINDEX with validation (vector count, search quality)</module>
  <module path="projection.py">UMAP/t-SNE embedding projection computation</module>
  <module path="benchmark.py">Search quality benchmark execution via BenchmarkExecutor</module>
  <module path="cleanup.py">Periodic cleanup (old results, orphaned collections, stale benchmarks)</module>
  <module path="sync_dispatcher.py">Continuous sync scheduling for collection sources</module>
  <module path="parallel_ingestion.py">Multi-process document ingestion with billiard.Pool</module>
  <module path="utils.py">Constants, CeleryTaskWithOperationUpdates, helper functions</module>
  <module path="error_classifier.py">Error classification for retry decisions</module>
  <module path="qdrant_utils.py">Qdrant point building and batch operations</module>
</task-modules>

<critical-pattern>
  TRANSACTION BEFORE DISPATCH - Never dispatch a task for uncommitted data:

  # CORRECT
  operation = await operation_repo.create(...)
  await session.commit()  # Data exists in DB!
  celery_app.send_task("task_name", args=[str(operation.uuid)])

  # WRONG - Race condition!
  celery_app.send_task("task_name", args=[str(operation.uuid)])
  await session.commit()  # Task may run before data exists
</critical-pattern>

<progress-tracking>
  CeleryTaskWithOperationUpdates provides Redis Stream + Pub/Sub:

  async with CeleryTaskWithOperationUpdates(operation_id) as updater:
      updater.set_user_id(user_id)
      updater.set_collection_id(collection_id)
      await updater.send_update("scanning_documents", {"status": "scanning"})

  Stream key: operation-progress:{operation_id}
  Pub/Sub channels: operation:{operation_id}, user:{user_id}

  WebSocket clients subscribe to these channels for real-time updates.
</progress-tracking>

<constants>
  Timeouts:
    OPERATION_SOFT_TIME_LIMIT = 3600   # 1 hour soft limit
    OPERATION_HARD_TIME_LIMIT = 7200   # 2 hour hard limit

  Batching:
    EMBEDDING_BATCH_SIZE = 100
    VECTOR_UPLOAD_BATCH_SIZE = 100
    DOCUMENT_REMOVAL_BATCH_SIZE = 100

  Retries:
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 60  # seconds

  Cleanup:
    DEFAULT_DAYS_TO_KEEP = 30
    CLEANUP_DELAY_PER_10K_VECTORS = 60  # Extra 1 min per 10k vectors
</constants>

<async-in-celery>
  Celery workers run synchronously, but tasks need async DB/Redis access.
  Use resolve_awaitable_sync() which maintains per-thread event loops:

  from webui.tasks.utils import resolve_awaitable_sync

  def sync_celery_task():
      result = resolve_awaitable_sync(async_db_operation())

  CRITICAL: pg_connection_manager.reset() is called automatically when
  event loop changes - asyncpg connections are bound to specific loops.
</async-in-celery>

<patching-for-tests>
  Tasks import via webui.tasks namespace. To patch dependencies:

  @patch("webui.tasks.qdrant_manager")
  @patch("webui.tasks.ChunkingOrchestrator")
  def test_something(mock_chunking, mock_qdrant):
      ...

  Use resolve_qdrant_manager() / resolve_qdrant_manager_class()
  to get the current (potentially patched) instances at runtime.
</patching-for-tests>

<beat-schedule>
  Defined in celery_app.py (NOT in task modules):
  - cleanup-old-results: Daily (30 days retention)
  - refresh-collection-chunking-stats: Hourly
  - monitor-partition-health: Every 6 hours
  - dispatch-sync-sources: Every 60 seconds
</beat-schedule>

<internal-api-calls>
  Tasks calling VecPipe endpoints need authentication:

  from webui.tasks.utils import _build_internal_api_headers

  headers = _build_internal_api_headers()  # X-Internal-Api-Key header
  async with httpx.AsyncClient() as client:
      response = await client.post(url, headers=headers, json=payload)
</internal-api-calls>

<gotchas>
  <gotcha>ALWAYS commit before dispatch - race condition prevention</gotcha>
  <gotcha>Reset DB connections after fork: pg_connection_manager.reset()</gotcha>
  <gotcha>Use _sanitize_error_message() in logs to remove PII</gotcha>
  <gotcha>Cleanup delay scales with vector count (calculate_cleanup_delay())</gotcha>
  <gotcha>Beat schedule is in celery_app.py, not in task modules</gotcha>
  <gotcha>Internal API calls require X-Internal-Api-Key header</gotcha>
  <gotcha>EMBEDDING_CONCURRENCY_PER_WORKER env var controls embed parallelism</gotcha>
</gotchas>

<testing>
  <command>uv run pytest tests/unit/tasks/ -v</command>
  <command>uv run pytest tests/webui/tasks/ -v</command>
  <note>stub_celery_send_task fixture (autouse) stubs all Celery dispatches</note>
</testing>
