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

Updates are sent at key points during operation processing:

```python
# Operation started
await ws_manager.send_update(f"operation:{operation_id}", "operation_started", {
    "operation_id": operation_id,
    "operation_type": "index",
    "total_files": total_files
})

# Processing each file
await ws_manager.send_update(f"operation:{operation_id}", "file_processing", {
    "current_file": file_path,
    "processed_files": processed_count,
    "total_files": total_files,
    "status": "Processing"
})

# File completed
await ws_manager.send_update(f"operation:{operation_id}", "file_completed", {
    "processed_files": processed_count,
    "total_files": total_files
})

# Operation completed
await ws_manager.send_update(f"operation:{operation_id}", "operation_completed", {
    "operation_id": operation_id,
    "status": "completed",
    "message": "Operation completed successfully"
})
```

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

## Testing WebSocket Connections

### Manual Testing

Use the built-in test page:
```bash
# Open in browser
http://localhost:8080/tests/websocket-tests.html
```

### Automated Testing

```python
import asyncio
import websockets
import json

async def test_operation_progress():
    token = "your-jwt-token"  # Get from auth endpoint
    operation_id = "test-operation-id"
    uri = f"ws://localhost:8080/ws/operations/{operation_id}?token={token}"
    
    async with websockets.connect(uri) as websocket:
        # Wait for connection
        await asyncio.sleep(0.1)
        
        # Should receive updates
        message = await websocket.recv()
        data = json.loads(message)
        
        assert data["type"] in ["operation_started", "file_processing", "error"]
        print(f"Received: {data}")

# Run test
asyncio.run(test_operation_progress())
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

## Conclusion

The WebSocket API provides efficient real-time updates for operation processing and directory scanning. The operation WebSocket endpoint includes authentication via JWT tokens and proper permission checks. The WebSocketManager handles connection limits, graceful disconnections, and Redis-based message distribution for scalability.