# WebSocket API

## Overview

Semantik exposes WebSocket endpoints for real‑time progress of long‑running operations (indexing, reindexing, projections, source removal) and directory scans. WebSockets are **read‑only**: start work through the REST API, then subscribe here to stream updates without polling.

All WebSocket URLs below assume default ports from `packages/shared/config/webui.py`.

## Endpoints

### 1. Global Operations Channel

**Endpoint**: `ws://localhost:8080/ws/operations?token=<jwt_token>`

Streams updates for **all** operations owned by the authenticated user. Useful for dashboards or “active operations” views.

### 2. Operation Progress Channel

**Endpoint**: `ws://localhost:8080/ws/operations/{operation_id}?token=<jwt_token>`

Streams updates for a single operation. `operation_id` is the operation UUID returned by endpoints like:
- `POST /api/v2/collections/{id}/add-source`
- `POST /api/v2/collections/{id}/reindex`
- `POST /api/v2/collections/{id}/projections`
- `DELETE /api/v2/collections/{id}/sources/{source_id}`

### 3. Directory Scan Progress Channel

**Endpoint**: `ws://localhost:8080/ws/directory-scan/{scan_id}?token=<jwt_token>`

`scan_id` comes from starting a scan via:

```http
POST /api/v2/directory-scan/preview
Authorization: Bearer <jwt_token>
Content-Type: application/json

{
  "path": "/absolute/path/to/documents",
  "recursive": true
}
```

The REST response includes `scan_id`; connect to this WebSocket to follow progress until completion.

## Authentication

WebSockets authenticate via the same JWT access token used for REST, passed as a query parameter:

`?token=<jwt_token>`

If `DISABLE_AUTH=true` in development, the token is optional.

## Message Formats

### Operation Messages

Operations are published by Celery tasks using `webui.tasks.utils.CeleryTaskWithOperationUpdates` and forwarded to WebSocket subscribers.

**Common envelope:**

```json
{
  "timestamp": "2025-12-12T01:23:45.123Z",
  "type": "<event_type>",
  "data": { "...": "..." }
}
```

On connect, the server first sends a **current state** snapshot:

```json
{
  "timestamp": "2025-12-12T01:23:45.123Z",
  "type": "current_state",
  "data": {
    "status": "processing",
    "operation_type": "index",
    "created_at": "2025-12-12T01:20:00Z",
    "started_at": "2025-12-12T01:21:00Z",
    "completed_at": null,
    "error_message": null
  }
}
```

After that, updates arrive as event messages. `type` is a string; current task emitters use (non‑exhaustive):

- `operation_started`
- `scanning_documents`, `scanning_completed`
- `processing_embeddings`
- `document_processed`
- `index_completed`, `append_completed`
- `removing_documents`, `remove_source_completed`
- `reindex_preflight`, `staging_created`, `reprocessing_progress`, `validation_complete`, `reindex_completed`
- `projection_enqueued`, `projection_started`, `projection_fetch_progress`, `projection_completed`, `projection_failed`
- `operation_completed`

**Payload (`data`) is task‑specific**, but commonly includes:
- `status`: e.g., `processing`, `completed`, `failed`, `cancelled`
- `progress`: 0–100 (float)
- `message`: human‑readable status text
- `metadata`: optional dict (ETA, current file, etc.)
- `warnings` or `error`/`error_message`
- identifiers like `operation_id`, `collection_id`, depending on the task

**Example progress update:**

```json
{
  "timestamp": "2025-12-12T01:24:30.000Z",
  "type": "reprocessing_progress",
  "data": {
    "status": "processing",
    "progress": 40.0,
    "message": "Embedding 200/500 documents"
  }
}
```

Clients should treat unknown `type` values as forward‑compatible and rely on `data.status` for final state. If a task fails before emitting a final event, the WebSocket may close after the operation enters an error state; clients can confirm via `GET /api/v2/operations/{operation_id}`.

### Directory Scan Messages

Directory scan updates are sent directly as `DirectoryScanProgress` objects (no outer envelope).

**Schema:**

```json
{
  "type": "started|counting|progress|warning|error|completed",
  "scan_id": "<scan_id>",
  "data": { "...": "..." }
}
```

**Example progress update:**

```json
{
  "type": "progress",
  "scan_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "data": {
    "files_scanned": 520,
    "total_files": 1250,
    "current_path": "/documents/specs/spec-v2.pdf",
    "percentage": 41.6
  }
}
```

`completed` messages include the final scan results in `data`, matching `DirectoryScanResponse` (files list, totals, warnings).

## Ping / Keepalive

- Operations channels accept either a JSON ping `{"type":"ping"}` or a legacy plain `"ping"` string; server responds with `{"type":"pong"}`.
- Directory scan channel accepts JSON ping `{"type":"ping"}` and responds with `{"type":"pong"}`.

## Minimal Client Example (JavaScript)

```js
const token = localStorage.getItem("authToken");

// Operation progress
const opWs = new WebSocket(`ws://localhost:8080/ws/operations/${operationId}?token=${token}`);
opWs.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.data);
};

// Directory scan progress
const scanWs = new WebSocket(`ws://localhost:8080/ws/directory-scan/${scanId}?token=${token}`);
scanWs.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(msg.type, msg.data);
};
```

---

If you add new Celery task update types, keep this document aligned with the strings emitted via `CeleryTaskWithOperationUpdates.send_update()` and the `DirectoryScanProgress` schema in `webui.api.schemas`.
