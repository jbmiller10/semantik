# Benchmarks API (v2)

Semantik’s benchmark system lets you upload a ground-truth dataset, map it to a collection, resolve document references, then run a configuration matrix and retrieve aggregated metrics.

Base URL: `http://localhost:8080`  
Auth: JWT (`Authorization: Bearer <token>`) unless `DISABLE_AUTH=true`.

## REST Endpoints

### Datasets

- `POST /api/v2/benchmark-datasets`  
  Upload a dataset JSON file (multipart form: `name`, optional `description`, and `file`).
- `GET /api/v2/benchmark-datasets?page=1&per_page=50`  
  List datasets owned by the current user.
- `GET /api/v2/benchmark-datasets/{dataset_id}`  
  Get one dataset.
- `DELETE /api/v2/benchmark-datasets/{dataset_id}`  
  Delete a dataset (and all its mappings).

### Dataset ↔ Collection Mappings

- `POST /api/v2/benchmark-datasets/{dataset_id}/mappings`  
  Create a mapping to a collection (`{"collection_id": "<uuid>"}`).
- `GET /api/v2/benchmark-datasets/{dataset_id}/mappings`  
  List mappings for a dataset.
- `GET /api/v2/benchmark-datasets/{dataset_id}/mappings/{mapping_id}`  
  Get a mapping.
- `POST /api/v2/benchmark-datasets/{dataset_id}/mappings/{mapping_id}/resolve`  
  Resolve doc references to real document IDs.
  - Returns `200` when completed synchronously.
  - Returns `202` with an `operation_uuid` when queued asynchronously.

### Benchmarks

- `POST /api/v2/benchmarks`  
  Create a benchmark definition (pre-creates all run configurations).
- `GET /api/v2/benchmarks?page=1&per_page=50&status_filter=running`  
  List benchmarks (optional status filter).
- `GET /api/v2/benchmarks/{benchmark_id}`  
  Get a benchmark.
- `POST /api/v2/benchmarks/{benchmark_id}/start`  
  Start execution (creates an Operation and dispatches Celery).
- `POST /api/v2/benchmarks/{benchmark_id}/cancel`  
  Cancel execution (cooperative; workers poll cancellation state).
- `GET /api/v2/benchmarks/{benchmark_id}/results`  
  Aggregated run metrics and timing data.
- `GET /api/v2/benchmarks/{benchmark_id}/runs/{run_id}/queries?page=1&per_page=50`  
  Per-query results for a run (paginated).
- `DELETE /api/v2/benchmarks/{benchmark_id}`  
  Delete benchmark + runs + results.

## Dataset Upload Format

Datasets are JSON. Minimal example:

```json
{
  "schema_version": "1.0",
  "metadata": { "source": "manual" },
  "queries": [
    {
      "query_key": "q1",
      "query_text": "how do I configure auth?",
      "metadata": { "topic": "security" },
      "relevant_docs": [
        { "doc_ref": { "uri": "file:///abs/path/doc.md" }, "relevance_grade": 3 }
      ]
    }
  ]
}
```

Notes:
- `schema_version` defaults to `"1.0"` if omitted.
- Each query requires a stable identifier and text:
  - `query_key` (also accepts `query_id` for compatibility)
  - `query_text` (also accepts `query` for compatibility)
- Relevance judgments:
  - `relevant_docs` is the preferred field.
  - `relevant_doc_refs` is accepted for backward compatibility.
- `relevance_grade` must be an int in `[0..3]` (defaults to `2`).
- `doc_ref` can be:
  - an object (recommended), e.g. `{ "uri": "file:///..." }`
  - a string (shorthand), treated as `{"uri": "<string>"}`.

## Mapping Resolution Semantics

Resolution is deterministic and uses this priority order:
1. `document_id` (must exist in the mapped collection)
2. `uri` (exact match to `Document.uri`)
3. `content_hash` (only if unique within the collection)
4. `path` (treated as uri-like; exact match to `Document.uri`)
5. `file_name` (only if unique within the collection)

Large mappings/collections are routed to async resolution based on the `BENCHMARK_MAPPING_RESOLVE_*` thresholds.

## WebSocket Progress Events

Progress is delivered over the Operations WebSocket streams. See [WEBSOCKET_API.md](../WEBSOCKET_API.md) for connection/auth details.

Message envelope (published into Redis and emitted to WebSocket clients):

```json
{
  "timestamp": "2026-01-19T12:00:00Z",
  "type": "<event_type>",
  "data": { "operation_id": "...", "collection_id": "...", "...": "..." }
}
```

### `benchmark_progress`

Emitted during benchmark execution. Core fields:
- `benchmark_id`: benchmark UUID
- `status`: `pending|running|completed|failed|cancelled`
- `total_runs`, `completed_runs`, `failed_runs`
- `primary_k`, `k_values_for_metrics`
- `stage`: `starting|indexing|evaluating|completed`
- `current_run` (optional): `{ run_id, run_order, config, total_queries, completed_queries, stage, ... }`
- `last_completed_run` (optional): includes `status`, `metrics`, `metrics_flat`, `timing`, `error_message` if failed

### `benchmark_mapping_resolution_progress`

Emitted during async mapping resolution. Core fields:
- `mapping_id`, `dataset_id`, `collection_id`
- `stage`: `starting|loading_documents|resolving|finalizing|completed|failed`
- `total_refs`, `processed_refs`, `resolved_refs`, `ambiguous_refs`, `unresolved_refs`

## Related Configuration

Environment variables controlling quotas/routing live in `packages/shared/config/webui.py` and are documented in [CONFIGURATION.md](../CONFIGURATION.md) and `.env.docker.example`.
