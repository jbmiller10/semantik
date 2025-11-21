# WebSocket API Documentation

## Overview

Semantik provides WebSocket endpoints for real-time updates during operation processing and directory scanning. This enables responsive user interfaces that show live progress without polling.

## WebSocket Endpoints

### 1. Operation Progress WebSocket

**Endpoint**: `ws://localhost:8080/ws/operations/{operation_id}`

Provides real-time updates during operation processing (indexing, reindexing, etc.).

**Authentication**: Pass JWT token as query parameter: `?token=<jwt_token>`

**Connection Example**:
```javascript
const token = localStorage.getItem('authToken');
const ws = new WebSocket(`ws://localhost:8080/ws/operations/abc123-operation-id?token=${token}`);

ws.onopen = () => {
    console.log('Connected to operation progress');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Progress update:', data);
};
```

### 2. Directory Scan WebSocket

**Endpoint**: `ws://localhost:8080/ws/scan/{scan_id}`

Provides real-time updates during directory scanning operations.

**Connection Example**:
```javascript
const scanId = crypto.randomUUID();
const ws = new WebSocket(`ws://localhost:8080/ws/scan/${scanId}`);

ws.onopen = () => {
    // Start scan
    ws.send(JSON.stringify({
        action: 'scan',
        path: '/documents',
        recursive: true
    }));
};
```

## Message Formats

### Operation Progress Messages

#### Server → Client Messages

**Operation Started**
```json
{
    "type": "operation_started",
    "operation_id": "abc123-operation-id",
    "operation_type": "index",
    "total_files": 42
}
```

**File Processing**
```json
{
    "type": "file_processing",
    "current_file": "/documents/report.pdf",
    "processed_files": 10,
    "total_files": 42,
    "status": "Processing"
}
```

**File Completed**
```json
{
    "type": "file_completed",
    "processed_files": 11,
    "total_files": 42
}
```

**Operation Completed**
```json
{
    "type": "operation_completed",
    "operation_id": "abc123-operation-id",
    "status": "completed",
    "message": "Operation completed successfully"
}
```

**Error**
```json
{
    "type": "error",
    "message": "Failed to process file: Permission denied"
}
```

### Directory Scan Messages

#### Client → Server Messages

**Start Scan**
```json
{
    "action": "scan",
    "path": "/path/to/directory",
    "recursive": true
}
```

**Cancel Scan**
```json
{
    "action": "cancel"
}
```

#### Server → Client Messages

**Scan Progress**
```json
{
    "type": "progress",
    "scanned": 100,
    "total": 500,
    "current_path": "/documents/subfolder/file.txt"
}
```

**Scan Completed**
```json
{
    "type": "completed",
    "files": [
        {
            "path": "/documents/file1.pdf",
            "size": 1024000,
            "modified": "2024-01-15T10:00:00Z",
            "extension": ".pdf"
        }
    ],
    "count": 42,
    "total_size": 52428800,
    "warnings": ["Permission denied: /documents/private"]
}
```

**Scan Error**
```json
{
    "type": "error",
    "message": "Failed to scan directory: Directory not found"
}
```

## Client Implementation

### JavaScript/TypeScript Example

```typescript
class OperationProgressClient {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 3000;
    private token: string;

    constructor(token: string) {
        this.token = token;
    }

    connect(operationId: string) {
        const wsUrl = `ws://localhost:8080/ws/operations/${operationId}?token=${this.token}`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            this.setupEventHandlers();
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
        }
    }

    private setupEventHandlers() {
        if (!this.ws) return;

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            
            // Reconnect on abnormal closure
            if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnect();
            }
        };
    }

    private handleMessage(data: any) {
        switch (data.type) {
            case 'operation_started':
                console.log(`Operation started: ${data.total_files} files to process`);
                break;
            
            case 'file_processing':
                console.log(`Processing: ${data.current_file} (${data.processed_files}/${data.total_files})`);
                break;
            
            case 'file_completed':
                console.log(`Progress: ${data.processed_files}/${data.total_files}`);
                break;
            
            case 'operation_completed':
                console.log('Operation completed!');
                this.disconnect();
                break;
            
            case 'error':
                console.error('Operation error:', data.message);
                break;
        }
    }

    private reconnect() {
        this.reconnectAttempts++;
        console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            this.connect(operationId);
        }, this.reconnectDelay);
    }

    disconnect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.close(1000, 'Client disconnect');
        }
    }
}
```

### React Hook Example

```tsx
import { useEffect, useRef, useState } from 'react';

interface OperationProgress {
    currentFile?: string;
    processedFiles: number;
    totalFiles: number;
    status: 'connecting' | 'processing' | 'completed' | 'error';
    error?: string;
}

export function useOperationProgress(operationId: string, token: string): OperationProgress {
    const [progress, setProgress] = useState<OperationProgress>({
        processedFiles: 0,
        totalFiles: 0,
        status: 'connecting'
    });
    
    const wsRef = useRef<WebSocket | null>(null);
    
    useEffect(() => {
        if (!operationId || !token) return;
        
        const ws = new WebSocket(`ws://localhost:8080/ws/operations/${operationId}?token=${token}`);
        wsRef.current = ws;
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'operation_started':
                    setProgress(prev => ({
                        ...prev,
                        totalFiles: data.total_files,
                        status: 'processing'
                    }));
                    break;
                
                case 'file_processing':
                    setProgress({
                        currentFile: data.current_file,
                        processedFiles: data.processed_files,
                        totalFiles: data.total_files,
                        status: 'processing'
                    });
                    break;
                
                case 'operation_completed':
                    setProgress(prev => ({
                        ...prev,
                        status: 'completed'
                    }));
                    break;
                
                case 'error':
                    setProgress(prev => ({
                        ...prev,
                        status: 'error',
                        error: data.message
                    }));
                    break;
            }
        };
        
        ws.onerror = () => {
            setProgress(prev => ({
                ...prev,
                status: 'error',
                error: 'WebSocket connection failed'
            }));
        };
        
        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [operationId, token]);
    
    return progress;
}
```

## Server Implementation

### Connection Manager

The server uses a `ConnectionManager` class to handle multiple WebSocket connections:

```python
class WebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, channel_id: str, user_id: str):
        await websocket.accept()
        connection_key = f"{user_id}:{channel_id}"
        if connection_key not in self.active_connections:
            self.active_connections[connection_key] = []
        self.active_connections[connection_key].append(websocket)
    
    def disconnect(self, websocket: WebSocket, channel_id: str, user_id: str):
        connection_key = f"{user_id}:{channel_id}"
        if connection_key in self.active_connections:
            self.active_connections[connection_key].remove(websocket)
            if not self.active_connections[connection_key]:
                del self.active_connections[connection_key]
    
    async def send_update(self, channel_id: str, message_type: str, data: dict[str, Any]):
        # Send to all connections subscribed to this channel
        for key in self.active_connections:
            if channel_id in key:
                disconnected = []
                for connection in self.active_connections[key]:
                    try:
                        await connection.send_json({
                            "type": message_type,
                            **data
                        })
                    except Exception:
                        disconnected.append(connection)
                
                # Clean up disconnected clients
                for conn in disconnected:
                    user_id = key.split(":")[0]
                    self.disconnect(conn, channel_id, user_id)
```

### Sending Updates During Processing

The Celery tasks send updates via the CeleryTaskWithOperationUpdates context manager:

```python
# From tasks.py - example of sending updates during indexing
async with CeleryTaskWithOperationUpdates(operation_id) as updater:
    # Operation started
    await updater.send_update(
        "operation_started", 
        {"status": "processing", "type": operation_type}
    )
    
    # Scanning documents
    await updater.send_update(
        "scanning_documents", 
        {"status": "scanning", "source_path": source_path}
    )
    
    # Scan completed
    await updater.send_update(
        "scanning_completed",
        {
            "status": "scanning_completed",
            "total_files_found": scan_stats["total_documents_found"],
            "new_documents_registered": scan_stats["new_documents_registered"],
            "scan_duration": scan_duration
        }
    )
    
    # Processing documents
    await updater.send_update(
        "document_processed",
        {
            "processed": processed_count,
            "failed": failed_count,
            "total": len(documents),
            "current_file": document.file_path,
            "progress_percent": (processed_count / len(documents)) * 100
        }
    )
    
    # Operation completed
    await updater.send_update(
        "operation_completed", 
        {"status": "completed", "result": result}
    )
```

**Update Flow Architecture**:

1. **Celery Task** → Sends update via `updater.send_update()`
2. **Redis Stream** → Message stored in `operation-progress:{operation_id}` stream
3. **WebSocket Manager** → Consumes from Redis stream
4. **WebSocket Clients** → Receive real-time updates

This architecture ensures:
- Updates persist even if WebSocket disconnects temporarily
- Multiple WebSocket connections can receive the same updates
- Updates are delivered in order
- System can scale horizontally

### WebSocket Endpoint Registration

The WebSocket endpoint is registered at the FastAPI application level, not in the router:

```python
# In main.py or app initialization
from webui.api.v2.operations import operation_websocket

# Register WebSocket endpoint directly on the app
app.add_api_websocket_route(
    "/ws/operations/{operation_id}",
    operation_websocket,
    name="operation_progress_ws"
)
```

This is important because:
- WebSocket routes must be registered at the app level, not in APIRouter
- The endpoint handles its own authentication via query parameters
- The route is outside the `/api/v2` prefix used by REST endpoints

## Connection Management

### Auto-Reconnection

The client implementations include automatic reconnection logic:

- Reconnects on abnormal closures (not code 1000)
- Maximum of 5 reconnection attempts by default
- Fixed interval between attempts (3 seconds)
- Resets attempt counter on successful connection

### Connection States

```javascript
// WebSocket.readyState values
0 - CONNECTING: Connection not yet established
1 - OPEN: Connection established, ready to communicate
2 - CLOSING: Connection closing handshake in progress
3 - CLOSED: Connection closed or couldn't be established
```

### Error Handling

Both client and server handle various error scenarios:

1. **Connection Failures**: Client falls back to polling or shows error state
2. **Message Send Failures**: Server removes failed connections automatically
3. **Invalid Messages**: Should be validated (currently minimal validation)
4. **Network Interruptions**: Auto-reconnection handles temporary issues

## Security Considerations

✅ **Security Update**: The operation WebSocket endpoint now includes authentication via JWT tokens passed as query parameters.

### Recommended Security Enhancements

1. **Authentication (Already Implemented)**:
```python
# Extract token from query parameters
token = websocket.query_params.get("token")

try:
    # Authenticate the user
    user = await get_current_user_websocket(token)
    user_id = str(user["id"])
except ValueError as e:
    # Authentication failed
    await websocket.close(code=1008, reason=str(e))
    return

# Verify user has permission to access this operation
try:
    operation = await repo.get_by_uuid_with_permission_check(
        operation_uuid=operation_id,
        user_id=int(user["id"]),
    )
except EntityNotFoundError:
    await websocket.close(code=1008, reason=f"Operation '{operation_id}' not found")
    return
except AccessDeniedError:
    await websocket.close(code=1008, reason="You don't have access to this operation")
    return
```

2. **Connection Limits (Available in WebSocketManager)**:
```python
# The WebSocketManager includes built-in connection limits:
MAX_USER_CONNECTIONS = 10  # Per user limit
MAX_OPERATION_CONNECTIONS = 100  # Per operation limit

# Automatically enforced when connecting:
await ws_manager.connect(websocket, channel_id, user_id)
# Raises ConnectionLimitExceeded if limits are exceeded
```

3. **Message Validation**:
```python
from pydantic import BaseModel, validator

class ScanMessage(BaseModel):
    action: Literal["scan", "cancel"]
    path: Optional[str] = None
    recursive: Optional[bool] = True
    
    @validator('path')
    def validate_path(cls, v, values):
        if values.get('action') == 'scan' and not v:
            raise ValueError("Path required for scan action")
        # Validate path doesn't escape allowed directories
        return validate_safe_path(v)
```

## Performance Considerations

### Message Frequency

Avoid overwhelming clients with too frequent updates:

```python
# Throttle updates to max 1 per second per file
last_update_time = 0
UPDATE_THROTTLE = 1.0  # seconds

async def send_progress_update(job_id: str, data: dict):
    current_time = time.time()
    if current_time - last_update_time >= UPDATE_THROTTLE:
        await manager.send_update(job_id, data)
        last_update_time = current_time
```

### Connection Scaling

For high-scale deployments:

1. **Use Redis Pub/Sub** for multi-instance coordination
2. **Implement connection pooling** for database queries
3. **Add WebSocket compression** for large messages
4. **Monitor connection metrics** (connections/job, message rates)

## Operation Lifecycle and Progress Tracking

### Operation Types and Their Progress Stages

**INDEX Operation**:
1. `scanning_documents` - Finding files in the source directory
2. `scanning_completed` - Document registration complete
3. `processing_embeddings` - Generating embeddings
4. `document_processed` - Individual document progress
5. `index_completed` - Collection ready

**APPEND Operation**:
1. `scanning_documents` - Finding new files
2. `scanning_completed` - New documents identified
3. `document_processed` - Processing new documents
4. `append_completed` - New documents added

**REINDEX Operation**:
1. `reindex_preflight` - Analyzing current state
2. `staging_created` - New collection created
3. `reprocessing_progress` - Reprocessing all documents
4. `validation_complete` - Quality checks passed
5. `reindex_completed` - Collection switched

**REMOVE_SOURCE Operation**:
1. `removing_documents` - Removing documents from source
2. `remove_source_completed` - Source removed

### Progress Calculation

Progress percentage is calculated differently for each operation type:

```javascript
// For document processing
const progressPercent = (processedFiles / totalFiles) * 100;

// For reindexing (includes validation)
const reindexProgress = {
    scanning: 10,
    staging: 20,
    reprocessing: 70,  // Main work
    validation: 90,
    switching: 100
};
```

## Testing WebSocket Connections

### Manual Testing with wscat

```bash
# Install wscat
npm install -g wscat

# Get auth token first
TOKEN=$(curl -s -X POST http://localhost:8080/api/v2/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test@example.com","password":"password"}' | jq -r .access_token)

# Connect to operation WebSocket
wscat -c "ws://localhost:8080/ws/operations/YOUR_OPERATION_ID?token=$TOKEN"
```

### Automated Testing

```python
import asyncio
import websockets
import json
from datetime import datetime

async def test_operation_progress():
    # First, get auth token
    import httpx
    async with httpx.AsyncClient() as client:
        auth_response = await client.post(
            "http://localhost:8080/api/v2/auth/login",
            json={"username": "test@example.com", "password": "password"}
        )
        token = auth_response.json()["access_token"]
        
        # Create a new operation (e.g., index a collection)
        collection_response = await client.post(
            "http://localhost:8080/api/v2/collections",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "Test Collection",
                "description": "WebSocket test",
                "source_path": "/test/documents"
            }
        )
        operation_id = collection_response.json()["operation_id"]
    
    # Connect to WebSocket
    uri = f"ws://localhost:8080/ws/operations/{operation_id}?token={token}"
    
    messages_received = []
    
    async with websockets.connect(uri) as websocket:
        # Collect messages for 30 seconds or until completion
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < 30:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                data = json.loads(message)
                messages_received.append(data)
                
                print(f"[{data['timestamp']}] {data['type']}: {data.get('data', {})}")
                
                # Check for completion
                if data["type"] == "status_update" and data["data"]["status"] in ["completed", "failed"]:
                    break
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send(json.dumps({"type": "ping"}))
        
    # Verify message sequence
    message_types = [msg["type"] for msg in messages_received]
    assert "current_state" in message_types, "Should receive current state on connect"
    assert "operation_started" in message_types or "scanning_documents" in message_types
    
    print(f"\nReceived {len(messages_received)} messages")
    print(f"Message types: {message_types}")

# Run test
asyncio.run(test_operation_progress())
```

### Browser-based Testing

```html
<!DOCTYPE html>
<html>
<head>
    <title>WebSocket Operation Progress Test</title>
</head>
<body>
    <h1>Operation Progress WebSocket Test</h1>
    <div id="status">Disconnected</div>
    <div id="messages"></div>
    
    <script>
        // Get auth token from localStorage or prompt
        const token = localStorage.getItem('authToken') || prompt('Enter auth token:');
        const operationId = prompt('Enter operation ID:');
        
        if (token && operationId) {
            const ws = new WebSocket(`ws://localhost:8080/ws/operations/${operationId}?token=${token}`);
            const statusEl = document.getElementById('status');
            const messagesEl = document.getElementById('messages');
            
            ws.onopen = () => {
                statusEl.textContent = 'Connected';
                statusEl.style.color = 'green';
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                const messageEl = document.createElement('div');
                messageEl.innerHTML = `
                    <strong>${message.type}</strong> (${new Date(message.timestamp).toLocaleTimeString()})<br>
                    <pre>${JSON.stringify(message.data, null, 2)}</pre>
                    <hr>
                `;
                messagesEl.appendChild(messageEl);
                
                // Auto-scroll to bottom
                messagesEl.scrollTop = messagesEl.scrollHeight;
            };
            
            ws.onerror = (error) => {
                statusEl.textContent = 'Error';
                statusEl.style.color = 'red';
            };
            
            ws.onclose = (event) => {
                statusEl.textContent = `Disconnected (${event.code}: ${event.reason})`;
                statusEl.style.color = 'orange';
            };
            
            // Keep-alive ping every 30 seconds
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, 30000);
        }
    </script>
</body>
</html>
```

### Load Testing

```python
async def load_test_websockets(num_connections: int, token: str):
    tasks = []
    for i in range(num_connections):
        operation_id = f"load-test-{i}"
        task = asyncio.create_task(maintain_connection(operation_id, token))
        tasks.append(task)
    
    # Run all connections for 60 seconds
    await asyncio.sleep(60)
    
    # Cancel all tasks
    for task in tasks:
        task.cancel()
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
```

### Common Issues

1. **Connection Immediately Closes**
   - Check WebSocket endpoint URL
   - Verify job_id exists
   - Check for CORS issues

2. **No Messages Received**
   - Ensure operation is actually processing
   - Check WebSocketManager has connection registered
   - Verify message sending code is reached
   - Check Redis pub/sub is working correctly

3. **Messages Delayed or Batched**
   - Check for blocking operations in message handler
   - Verify async/await used correctly
   - Look for throttling logic

## Future Enhancements

Planned improvements for the WebSocket API:

1. **Authentication & Authorization**: Add JWT token validation
2. **Heartbeat/Ping-Pong**: Implement keep-alive mechanism
3. **Message Compression**: Enable permessage-deflate
4. **Rate Limiting**: Prevent message flooding
5. **Metrics & Monitoring**: Track connection health
6. **Binary Protocol**: Support for binary messages
7. **Presence System**: Track online users per job
8. **Message History**: Allow replay of recent messages

## Best Practices

### Client Implementation

1. **Always implement reconnection logic** - Network interruptions are common
2. **Handle all message types gracefully** - Unknown message types should be logged, not cause errors
3. **Implement exponential backoff** - Don't hammer the server with rapid reconnection attempts
4. **Send periodic pings** - Keep connections alive through proxies and load balancers
5. **Clean up on unmount** - Always close connections when components unmount

### Server Implementation

1. **Use structured message format** - Consistent timestamp, type, and data fields
2. **Send initial state on connection** - Clients should know the current state immediately
3. **Implement connection limits** - Prevent DOS attacks and resource exhaustion
4. **Use Redis Streams for persistence** - Ensures message delivery even after disconnections
5. **Clean up completed operations** - Remove Redis streams after operations complete

### Error Handling

```javascript
// Comprehensive error handling example
class RobustOperationClient {
    handleError(error: any, context: string) {
        console.error(`Error in ${context}:`, error);
        
        // Notify user appropriately
        if (error.code === 1008) {
            this.notifyUser('Authentication failed. Please log in again.');
        } else if (error.code === 1011) {
            this.notifyUser('Server error. Please try again later.');
        } else if (error.message?.includes('network')) {
            this.notifyUser('Network connection lost. Reconnecting...');
        } else {
            this.notifyUser('An unexpected error occurred.');
        }
        
        // Log to monitoring service
        this.logToMonitoring({
            error: error.message || 'Unknown error',
            context,
            operationId: this.operationId,
            timestamp: new Date().toISOString()
        });
    }
}
```

## Monitoring and Observability

### Key Metrics to Track

1. **Connection Metrics**:
   - Active connections per operation
   - Connection duration
   - Reconnection frequency
   - Authentication failures

2. **Message Metrics**:
   - Messages sent per second
   - Message size distribution
   - Message type frequency
   - Failed message deliveries

3. **Operation Metrics**:
   - Operation duration by type
   - Success/failure rates
   - Progress update frequency
   - Resource usage per operation

### Example Monitoring Implementation

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
ws_connections_total = Counter(
    'websocket_connections_total',
    'Total WebSocket connections',
    ['operation_type', 'status']
)

ws_active_connections = Gauge(
    'websocket_active_connections',
    'Currently active WebSocket connections',
    ['operation_type']
)

ws_message_duration = Histogram(
    'websocket_message_duration_seconds',
    'Time to process WebSocket messages',
    ['message_type']
)

# Track in your WebSocket handler
async def track_connection(operation_type: str):
    ws_connections_total.labels(
        operation_type=operation_type,
        status='connected'
    ).inc()
    
    ws_active_connections.labels(
        operation_type=operation_type
    ).inc()
    
    try:
        yield
    finally:
        ws_active_connections.labels(
            operation_type=operation_type
        ).dec()
```

## Conclusion

The WebSocket API provides efficient real-time updates for collection operations in Semantik's collection-centric architecture. Key features include:

- **Secure authentication** via JWT tokens with permission verification
- **Persistent message delivery** through Redis Streams
- **Comprehensive progress tracking** for all operation types
- **Automatic reconnection support** with exponential backoff
- **Horizontal scalability** through Redis-based message distribution
- **Resource limits** to prevent abuse and ensure stability

The API is designed to provide a smooth user experience with real-time feedback during long-running operations while maintaining security and reliability at scale.
