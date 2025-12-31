# WebSocket API

Real-time progress updates for operations and directory scans. Start work via REST API, then subscribe via WebSocket to stream updates (no polling).

## Endpoints

**Global operations:** `ws://localhost:8080/ws/operations`
All operations for authenticated user.

**Single operation:** `ws://localhost:8080/ws/operations/{operation_id}`
One operation's progress.

**Directory scan:** `ws://localhost:8080/ws/directory-scan/{scan_id}`
Get `scan_id` from `POST /api/v2/directory-scan/preview`.

## Auth

**Preferred (v7.1+):** Pass JWT via WebSocket subprotocol header:
```javascript
const ws = new WebSocket(url, [`access_token.${token}`]);
```

The server echoes back the accepted subprotocol on successful authentication. This approach is more secure as tokens don't appear in server logs, browser history, or referrer headers.

**Deprecated:** Query parameter `?token=<jwt>` still works for backward compatibility but will be removed in a future version.

Authentication is optional if `DISABLE_AUTH=true`.

## Message Formats

### Operation Messages

Format: `{"timestamp": "...", "type": "<event>", "data": {...}}`

On connect, get a `current_state` snapshot. Then stream events like:
- `operation_started`, `scanning_documents`, `processing_embeddings`
- `index_completed`, `append_completed`, `reindex_completed`
- `projection_started`, `projection_completed`

Common `data` fields: `status`, `progress` (0-100), `message`, `metadata`

Example: `{"timestamp": "...", "type": "reprocessing_progress", "data": {"status": "processing", "progress": 40.0, "message": "Embedding 200/500"}}`

Unknown `type` values are forward-compatible. Check `GET /api/v2/operations/{id}` if WebSocket closes unexpectedly.

### Directory Scan Messages

Format: `{"type": "started|progress|completed", "scan_id": "...", "data": {...}}`

Example: `{"type": "progress", "scan_id": "...", "data": {"files_scanned": 520, "total_files": 1250, "percentage": 41.6}}`

## Keepalive

Send `{"type":"ping"}`, get `{"type":"pong"}`.

## Client Example

```js
// Preferred: subprotocol authentication (v7.1+)
const ws = new WebSocket(
  `ws://localhost:8080/ws/operations/${id}`,
  [`access_token.${token}`]
);
ws.onmessage = (e) => console.log(JSON.parse(e.data));

// Deprecated: query parameter authentication (still supported)
// const ws = new WebSocket(`ws://localhost:8080/ws/operations/${id}?token=${token}`);
```
