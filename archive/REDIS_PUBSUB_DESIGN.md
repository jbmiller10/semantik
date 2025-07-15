# Redis Pub/Sub Design for Real-time Job Updates

## Overview

This document outlines the design for implementing Redis-based real-time updates in Semantik, replacing the current WebSocket polling approach. The design uses Redis Streams for guaranteed delivery and resilience.

## Current State

- WebSocket connections are established but no real-time updates are sent
- Job status must be polled via REST API
- No mechanism for Celery workers to push updates to connected clients

## Proposed Architecture

### 1. Redis Streams vs Simple Pub/Sub

We'll use **Redis Streams** instead of simple pub/sub for several reasons:

- **Guaranteed delivery**: Messages persist until acknowledged
- **Consumer groups**: Multiple API servers can share the load
- **Message history**: Clients can catch up on missed messages
- **Built-in backpressure**: Automatic flow control

### 2. Message Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Celery    │────▶│   Redis     │────▶│  WebSocket  │────▶│   React     │
│   Worker    │     │   Stream    │     │   Manager   │     │   Client    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                         │
      └─────────── Job Updates ────────────────┘
```

### 3. Implementation Details

#### 3.1 Stream Structure

Each job gets its own stream with a TTL:
```
Stream Key: job:updates:{job_id}
TTL: 24 hours (configurable)
```

Message format:
```json
{
  "timestamp": "2024-12-19T10:30:45Z",
  "type": "status_update|progress|error|completion",
  "data": {
    "status": "processing",
    "progress": 45,
    "current_file": "/path/to/file.pdf",
    "processed_files": 23,
    "total_files": 50,
    "message": "Processing document..."
  }
}
```

#### 3.2 Celery Worker Updates

```python
# packages/webui/tasks.py
import aioredis
from datetime import datetime

class CeleryTaskWithUpdates:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.redis = redis.Redis.from_url(settings.REDIS_URL)
        self.stream_key = f"job:updates:{job_id}"
    
    async def send_update(self, update_type: str, data: dict):
        """Send update to Redis Stream"""
        message = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": update_type,
            "data": data
        }
        
        # Add to stream with automatic ID
        self.redis.xadd(
            self.stream_key,
            {"message": json.dumps(message)},
            maxlen=1000  # Keep last 1000 messages
        )
        
        # Set TTL on first message
        self.redis.expire(self.stream_key, 86400)  # 24 hours

@celery_app.task(bind=True)
def process_embedding_job_task(self, job_id: str):
    updater = CeleryTaskWithUpdates(job_id)
    
    # Send updates during processing
    for i, file in enumerate(files):
        asyncio.run(updater.send_update("progress", {
            "status": "processing",
            "current_file": file.path,
            "processed_files": i,
            "total_files": len(files)
        }))
```

#### 3.3 WebSocket Manager with Redis Streams

```python
# packages/webui/websocket_manager.py
import asyncio
import aioredis
import json
from typing import Dict, Set
from fastapi import WebSocket

class RedisStreamWebSocketManager:
    def __init__(self):
        self.redis = None
        self.connections: Dict[str, Set[WebSocket]] = {}
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
        self.consumer_group = f"webui-{uuid.uuid4().hex[:8]}"
        
    async def startup(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True
        )
        
    async def shutdown(self):
        """Clean up resources"""
        # Cancel all consumer tasks
        for task in self.consumer_tasks.values():
            task.cancel()
        
        # Close Redis connection
        if self.redis:
            await self.redis.close()
    
    async def connect(self, websocket: WebSocket, job_id: str, user_id: str):
        """Handle new WebSocket connection"""
        await websocket.accept()
        
        # Store connection
        key = f"{user_id}:{job_id}"
        if key not in self.connections:
            self.connections[key] = set()
        self.connections[key].add(websocket)
        
        # Start consumer task if not exists
        if job_id not in self.consumer_tasks:
            task = asyncio.create_task(
                self._consume_updates(job_id)
            )
            self.consumer_tasks[job_id] = task
        
        # Send catch-up messages
        await self._send_history(websocket, job_id)
    
    async def disconnect(self, websocket: WebSocket, job_id: str, user_id: str):
        """Handle WebSocket disconnection"""
        key = f"{user_id}:{job_id}"
        if key in self.connections:
            self.connections[key].discard(websocket)
            if not self.connections[key]:
                del self.connections[key]
        
        # Stop consumer if no connections
        if not any(job_id in k for k in self.connections):
            if job_id in self.consumer_tasks:
                self.consumer_tasks[job_id].cancel()
                del self.consumer_tasks[job_id]
    
    async def _consume_updates(self, job_id: str):
        """Consume updates from Redis Stream"""
        stream_key = f"job:updates:{job_id}"
        
        try:
            # Create consumer group
            try:
                await self.redis.xgroup_create(
                    stream_key,
                    self.consumer_group,
                    id="0"
                )
            except Exception:
                pass  # Group already exists
            
            while True:
                # Read from stream with blocking
                messages = await self.redis.xreadgroup(
                    self.consumer_group,
                    f"consumer-{job_id}",
                    {stream_key: ">"},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                if messages:
                    for stream, stream_messages in messages:
                        for msg_id, data in stream_messages:
                            # Parse message
                            message = json.loads(data["message"])
                            
                            # Send to all connected clients for this job
                            await self._broadcast_to_job(job_id, message)
                            
                            # Acknowledge message
                            await self.redis.xack(stream_key, self.consumer_group, msg_id)
                
                await asyncio.sleep(0.1)  # Small delay between reads
                
        except asyncio.CancelledError:
            # Clean up consumer group
            try:
                await self.redis.xgroup_delconsumer(
                    stream_key,
                    self.consumer_group,
                    f"consumer-{job_id}"
                )
            except Exception:
                pass
            raise
    
    async def _send_history(self, websocket: WebSocket, job_id: str):
        """Send historical messages to newly connected client"""
        stream_key = f"job:updates:{job_id}"
        
        try:
            # Read last 100 messages
            messages = await self.redis.xrange(
                stream_key,
                min="-",
                max="+",
                count=100
            )
            
            for msg_id, data in messages:
                message = json.loads(data["message"])
                await websocket.send_json(message)
                
        except Exception as e:
            logger.warning(f"Failed to send history: {e}")
    
    async def _broadcast_to_job(self, job_id: str, message: dict):
        """Broadcast message to all connections for a job"""
        disconnected = []
        
        for key, websockets in self.connections.items():
            if job_id in key:
                for websocket in websockets.copy():
                    try:
                        await websocket.send_json(message)
                    except Exception:
                        disconnected.append((key, websocket))
        
        # Clean up disconnected clients
        for key, websocket in disconnected:
            self.connections[key].discard(websocket)

# Global instance
ws_manager = RedisStreamWebSocketManager()
```

#### 3.4 FastAPI Integration

```python
# packages/webui/main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await ws_manager.startup()
    yield
    # Shutdown
    await ws_manager.shutdown()

app = FastAPI(lifespan=lifespan)

# Update WebSocket endpoint
@app.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    job_id: str,
    current_user: dict = Depends(get_current_user_ws)
):
    await ws_manager.connect(websocket, job_id, str(current_user["id"]))
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, job_id, str(current_user["id"]))
```

### 4. Connection Resilience

#### 4.1 Automatic Reconnection (Client Side)

```typescript
// apps/webui-react/src/hooks/useJobWebSocket.ts
export function useJobWebSocket(jobId: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState<JobUpdate[]>([]);
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();
  const reconnectAttempts = useRef(0);

  const connect = useCallback(() => {
    const token = getAuthToken();
    const wsUrl = `${WS_BASE_URL}/ws/jobs/${jobId}?token=${token}`;
    
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => {
      setIsConnected(true);
      reconnectAttempts.current = 0;
    };
    
    ws.current.onmessage = (event) => {
      const update = JSON.parse(event.data);
      setMessages(prev => [...prev, update]);
    };
    
    ws.current.onclose = () => {
      setIsConnected(false);
      
      // Exponential backoff reconnection
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
      reconnectAttempts.current++;
      
      reconnectTimeout.current = setTimeout(() => {
        connect();
      }, delay);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, [jobId]);

  useEffect(() => {
    connect();
    
    return () => {
      clearTimeout(reconnectTimeout.current);
      ws.current?.close();
    };
  }, [connect]);

  return { isConnected, messages };
}
```

#### 4.2 Server-Side Resilience

- Redis connection pooling with automatic reconnection
- Consumer group ensures messages aren't lost if a server restarts
- Health checks to verify Redis connectivity
- Graceful shutdown to clean up consumers

### 5. Monitoring and Observability

#### 5.1 Metrics

```python
# Prometheus metrics
redis_stream_messages_sent = Counter(
    'redis_stream_messages_sent_total',
    'Total messages sent to Redis streams',
    ['job_id', 'message_type']
)

websocket_connections_active = Gauge(
    'websocket_connections_active',
    'Number of active WebSocket connections'
)

redis_stream_lag = Histogram(
    'redis_stream_consumer_lag_seconds',
    'Lag between message creation and consumption'
)
```

#### 5.2 Logging

- Log all connection/disconnection events
- Log Redis stream errors
- Log message processing times
- Include correlation IDs for tracing

### 6. Security Considerations

1. **Authentication**: WebSocket connections require valid JWT token
2. **Authorization**: Users can only subscribe to their own jobs
3. **Rate Limiting**: Limit update frequency per job
4. **Message Validation**: Validate all messages before broadcasting
5. **TTL**: Automatic cleanup of old streams

### 7. Migration Plan

1. **Phase 1**: Deploy Redis Stream infrastructure
2. **Phase 2**: Update Celery workers to send updates
3. **Phase 3**: Deploy new WebSocket manager
4. **Phase 4**: Update frontend to use new WebSocket endpoint
5. **Phase 5**: Remove old polling code

### 8. Testing Strategy

#### Unit Tests
- Test Redis Stream operations
- Test WebSocket manager methods
- Test message serialization/deserialization

#### Integration Tests
- Test end-to-end message flow
- Test reconnection scenarios
- Test consumer group behavior

#### Load Tests
- Test with 1000+ concurrent WebSocket connections
- Test with high message throughput
- Test Redis memory usage

### 9. Configuration

```python
# Environment variables
REDIS_URL = "redis://localhost:6379/0"
REDIS_STREAM_TTL = 86400  # 24 hours
REDIS_STREAM_MAX_LEN = 1000  # Max messages per stream
WS_HEARTBEAT_INTERVAL = 30  # seconds
WS_MAX_CONNECTIONS_PER_USER = 10
```

### 10. Future Enhancements

1. **Stream Compression**: Compress messages for bandwidth efficiency
2. **Batch Updates**: Aggregate multiple updates before sending
3. **Priority Queues**: High-priority updates bypass normal queue
4. **WebSocket Compression**: Enable per-message deflate
5. **Multi-region Support**: Cross-region message replication

## Conclusion

This Redis Streams-based design provides a robust, scalable solution for real-time updates with guaranteed delivery, automatic failover, and excellent observability. The implementation can be done incrementally without disrupting existing functionality.