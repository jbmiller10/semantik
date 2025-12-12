# Chunking API Examples

Quick examples for curl, Python, JavaScript, WebSocket.

## Auth

```bash
curl -X POST /api/auth/login -d '{"username":"...","password":"..."}'
# Returns access_token
```

## Strategies

| Strategy | ID | Best For |
|----------|-----|----------|
| Fixed | `fixed_size` | Simple splitting |
| Semantic | `semantic` | Topic boundaries |
| Recursive | `recursive` | General purpose (default) |
| Markdown | `markdown` | MD structure |
| Hierarchical | `hierarchical` | Parent-child |
| Hybrid | `hybrid` | Auto-switch |

## Status Values

`pending` | `in_progress` | `completed` | `failed` | `cancelled` | `partial`

## curl Examples

```bash
# List strategies
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v2/chunking/strategies

# Preview
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v2/chunking/preview \
  -d '{"content":"...","strategy":"semantic","max_chunks":10}'

# Start collection chunking
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v2/chunking/collections/$COLL_ID/chunk \
  -d '{"strategy":"semantic","config":{"chunk_size":512}}'

# Get progress
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/api/v2/chunking/operations/$OP_ID/progress
```

## Python

```python
import requests

client = requests.Session()
client.headers.update({"Authorization": f"Bearer {token}"})
base = "http://localhost:8080/api/v2/chunking"

# List strategies
strategies = client.get(f"{base}/strategies").json()

# Preview
preview = client.post(f"{base}/preview", json={
    "content": "...",
    "strategy": "semantic",
    "config": {"chunk_size": 512, "chunk_overlap": 50}
}).json()

# Start chunking
op = client.post(f"{base}/collections/{coll_id}/chunk", json={
    "strategy": "semantic",
    "config": {"chunk_size": 512}
}).json()

# Poll progress
progress = client.get(f"{base}/operations/{op['operation_id']}/progress").json()
```

## JavaScript/TypeScript

```typescript
const api = "http://localhost:8080/api/v2/chunking";
const headers = { "Authorization": `Bearer ${token}` };

// List strategies
const strategies = await fetch(`${api}/strategies`, { headers }).then(r => r.json());

// Preview
const preview = await fetch(`${api}/preview`, {
  method: "POST",
  headers,
  body: JSON.stringify({ content: "...", strategy: "semantic" })
}).then(r => r.json());

// Start chunking
const op = await fetch(`${api}/collections/${collId}/chunk`, {
  method: "POST",
  headers,
  body: JSON.stringify({ strategy: "semantic", config: { chunk_size: 512 } })
}).then(r => r.json());

// WebSocket progress
const ws = new WebSocket(`ws://localhost:8080/ws/operations/${op.operation_id}?token=${token}`);
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  console.log(msg.data.progress, msg.data.status);
};
```

## WebSocket

```javascript
// IMPORTANT: Use operation_id from API response, NOT websocket_channel
const ws = new WebSocket(`ws://localhost:8080/ws/operations/${operationId}?token=${token}`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  // Message: { type: string, data: { status, progress, message, ... } }

  const { status, progress } = msg.data;
  console.log(`${progress}% - ${status}`);

  if (msg.type === "operation_completed" || status === "completed") {
    console.log("Done!");
    ws.close();
  }
  if (msg.type === "operation_failed" || status === "failed") {
    console.error(msg.data.error_message);
    ws.close();
  }
};
```

## Error Handling

```python
# Python
try:
    result = requests.post(url, json=payload)
    result.raise_for_status()
except requests.HTTPError as e:
    if e.response.status_code == 400:
        print("Validation error:", e.response.json())
    elif e.response.status_code == 429:
        retry_after = e.response.headers.get("X-RateLimit-Reset-After", 60)
        print(f"Rate limited, retry in {retry_after}s")
```

```javascript
// JavaScript
try {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json();
    if (response.status === 429) {
      console.log("Rate limited");
    }
    throw new Error(error.error || response.statusText);
  }
  return await response.json();
} catch (error) {
  console.error(error);
}
```

## Common Patterns

**Find best strategy**:
```python
# Get recommendation
rec = client.post(f"{base}/strategies/recommend",
                  params={"file_types": ["pdf", "md"]}).json()
print(rec["recommended_strategy"])

# Test with preview
preview = client.post(f"{base}/preview", json={
    "content": sample,
    "strategy": rec["recommended_strategy"],
    "config": rec["suggested_config"]
}).json()
```

**Compare strategies**:
```python
comparison = client.post(f"{base}/compare", json={
    "content": sample,
    "strategies": ["semantic", "recursive", "fixed_size"]
}).json()

for comp in comparison["comparisons"]:
    print(f"{comp['strategy']}: {comp['total_chunks']} chunks, quality={comp['quality_score']}")
```

**Process collection with progress**:
```python
import asyncio
import websockets

async def process_with_progress(coll_id):
    op = client.post(f"{base}/collections/{coll_id}/chunk", json={
        "strategy": "semantic"
    }).json()

    ws_url = f"ws://localhost:8080/ws/operations/{op['operation_id']}?token={token}"
    async with websockets.connect(ws_url) as ws:
        while True:
            msg = json.loads(await ws.recv())
            progress = msg.get("data", {}).get("progress")
            status = msg.get("data", {}).get("status")

            if progress:
                print(f"{progress:.1f}%")
            if status in ("completed", "failed"):
                break

asyncio.run(process_with_progress("coll_123"))
```

## Tips

- Always use POST for reranking (GET doesn't support it)
- WebSocket URL uses `operation_id` directly, not `websocket_channel` field
- Preview cached 15min - reuse `preview_id` when testing
- Rate limits: preview 10/min, compare 5/min, chunking 5/5min
- Chunk sizes: 100-300 (Q&A), 300-600 (general), 600-1000 (summarization)
