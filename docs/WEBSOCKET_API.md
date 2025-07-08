# WebSocket API Documentation

## Overview

Semantik provides WebSocket endpoints for real-time updates during job processing and directory scanning. This enables responsive user interfaces that show live progress without polling.

## WebSocket Endpoints

### 1. Job Progress WebSocket

**Endpoint**: `ws://localhost:8080/ws/{job_id}`

Provides real-time updates during embedding job processing.

**Connection Example**:
```javascript
const ws = new WebSocket('ws://localhost:8080/ws/abc123-job-id');

ws.onopen = () => {
    console.log('Connected to job progress');
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

### Job Progress Messages

#### Server → Client Messages

**Job Started**
```json
{
    "type": "job_started",
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

**Job Completed**
```json
{
    "type": "job_completed",
    "message": "Job completed successfully"
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
class JobProgressClient {
    private ws: WebSocket | null = null;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 3000;

    connect(jobId: string) {
        const wsUrl = `ws://localhost:8080/ws/${jobId}`;
        
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
            case 'job_started':
                console.log(`Job started: ${data.total_files} files to process`);
                break;
            
            case 'file_processing':
                console.log(`Processing: ${data.current_file} (${data.processed_files}/${data.total_files})`);
                break;
            
            case 'file_completed':
                console.log(`Progress: ${data.processed_files}/${data.total_files}`);
                break;
            
            case 'job_completed':
                console.log('Job completed!');
                this.disconnect();
                break;
            
            case 'error':
                console.error('Job error:', data.message);
                break;
        }
    }

    private reconnect() {
        this.reconnectAttempts++;
        console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
        
        setTimeout(() => {
            this.connect(jobId);
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

interface JobProgress {
    currentFile?: string;
    processedFiles: number;
    totalFiles: number;
    status: 'connecting' | 'processing' | 'completed' | 'error';
    error?: string;
}

export function useJobProgress(jobId: string): JobProgress {
    const [progress, setProgress] = useState<JobProgress>({
        processedFiles: 0,
        totalFiles: 0,
        status: 'connecting'
    });
    
    const wsRef = useRef<WebSocket | null>(null);
    
    useEffect(() => {
        if (!jobId) return;
        
        const ws = new WebSocket(`ws://localhost:8080/ws/${jobId}`);
        wsRef.current = ws;
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            switch (data.type) {
                case 'job_started':
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
                
                case 'job_completed':
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
    }, [jobId]);
    
    return progress;
}
```

## Server Implementation

### Connection Manager

The server uses a `ConnectionManager` class to handle multiple WebSocket connections:

```python
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
    
    async def send_update(self, job_id: str, message: dict[str, Any]):
        if job_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            # Clean up disconnected clients
            for conn in disconnected:
                self.disconnect(conn, job_id)
```

### Sending Updates During Processing

Updates are sent at key points during job processing:

```python
# Job started
await manager.send_update(job_id, {
    "type": "job_started",
    "total_files": total_files
})

# Processing each file
await manager.send_update(job_id, {
    "type": "file_processing",
    "current_file": file_path,
    "processed_files": processed_count,
    "total_files": total_files,
    "status": "Processing"
})

# File completed
await manager.send_update(job_id, {
    "type": "file_completed",
    "processed_files": processed_count,
    "total_files": total_files
})

# Job completed
await manager.send_update(job_id, {
    "type": "job_completed",
    "message": "Job completed successfully"
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

⚠️ **Important**: The current WebSocket implementation does not include authentication checks. Consider these security improvements:

### Recommended Security Enhancements

1. **Add Authentication**:
```python
@app.websocket("/ws/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str, token: str = Query(...)):
    # Verify JWT token before accepting connection
    user = verify_token(token)
    if not user:
        await websocket.close(code=1008, reason="Unauthorized")
        return
    
    # Check if user has access to this job
    if not user_can_access_job(user, job_id):
        await websocket.close(code=1008, reason="Forbidden")
        return
    
    await websocket_endpoint(websocket, job_id)
```

2. **Connection Limits**:
```python
MAX_CONNECTIONS_PER_USER = 10
MAX_CONNECTIONS_PER_JOB = 50

async def connect(self, websocket: WebSocket, job_id: str, user_id: str):
    # Check connection limits
    if self.count_user_connections(user_id) >= MAX_CONNECTIONS_PER_USER:
        raise ConnectionLimitExceeded("User connection limit reached")
    
    if len(self.active_connections.get(job_id, [])) >= MAX_CONNECTIONS_PER_JOB:
        raise ConnectionLimitExceeded("Job connection limit reached")
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

async def test_job_progress():
    uri = "ws://localhost:8080/ws/test-job-id"
    
    async with websockets.connect(uri) as websocket:
        # Wait for connection
        await asyncio.sleep(0.1)
        
        # Should receive updates
        message = await websocket.recv()
        data = json.loads(message)
        
        assert data["type"] in ["job_started", "file_processing", "error"]
        print(f"Received: {data}")

# Run test
asyncio.run(test_job_progress())
```

### Load Testing

```python
async def load_test_websockets(num_connections: int):
    tasks = []
    for i in range(num_connections):
        job_id = f"load-test-{i}"
        task = asyncio.create_task(maintain_connection(job_id))
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
   - Ensure job is actually processing
   - Check ConnectionManager has connection registered
   - Verify message sending code is reached

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

The WebSocket API provides efficient real-time updates for job processing and directory scanning. While the current implementation is functional, adding authentication and enhanced connection management would improve security and reliability for production deployments.