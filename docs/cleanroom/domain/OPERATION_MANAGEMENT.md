# OPERATION_MANAGEMENT - Cleanroom Documentation

## 1. Component Overview

The OPERATION_MANAGEMENT component provides comprehensive asynchronous operation management for the Semantik application. It handles all long-running collection operations through a unified architecture that combines Celery task processing, real-time WebSocket notifications, and robust state management.

### Core Purpose
- Manage lifecycle of long-running collection operations (indexing, appending, reindexing, deletion)
- Provide real-time progress updates to frontend clients via WebSocket
- Ensure reliable task execution with proper error handling and recovery
- Track operation history and metrics for audit and performance monitoring

### Key Capabilities
- Asynchronous task execution using Celery workers
- Real-time progress streaming via Redis Streams and WebSocket
- Operation cancellation with proper resource cleanup
- Comprehensive state management with atomic transitions
- Resource tracking and performance metrics collection

## 2. Architecture & Design Patterns

### 2.1 Operation Lifecycle State Machine

```
PENDING → PROCESSING → COMPLETED
   ↓         ↓           
   └─────→ CANCELLED    
             ↓          
          FAILED       
```

**State Definitions:**
- `PENDING`: Operation created but not yet started by worker
- `PROCESSING`: Operation actively being executed by Celery worker
- `COMPLETED`: Operation finished successfully
- `FAILED`: Operation encountered unrecoverable error
- `CANCELLED`: Operation cancelled by user or system

### 2.2 Layered Architecture

```
┌─────────────────────────────────────────────┐
│           Frontend (React)                  │
│     useOperationProgress Hook               │
└─────────────────────────────────────────────┘
                    ↓ WebSocket
┌─────────────────────────────────────────────┐
│        API Layer (FastAPI)                  │
│    packages/webui/api/v2/operations.py      │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        Service Layer                        │
│  packages/webui/services/operation_service  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│        Repository Layer                     │
│  packages/shared/database/repositories/     │
│         operation_repository.py             │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Celery Workers + Redis Streams          │
│      packages/webui/tasks.py                │
└─────────────────────────────────────────────┘
```

### 2.3 Communication Patterns

**Async Task Submission:**
1. API creates operation record with PENDING status
2. Celery task submitted with operation UUID
3. Task ID stored in operation record
4. Worker picks up task and transitions to PROCESSING

**Real-time Updates:**
1. Worker sends updates to Redis Stream: `operation-progress:{operation_id}`
2. Worker publishes to Redis Pub/Sub: `operation:{operation_id}`
3. ScalableWebSocketManager subscribes to channels
4. Updates forwarded to connected WebSocket clients

## 3. Key Interfaces & Contracts

### 3.1 Operation Model

**Location:** `packages/shared/database/models.py`

```python
class Operation(Base):
    __tablename__ = "operations"
    
    # Identity
    id = Column(Integer, primary_key=True)
    uuid = Column(String, unique=True, nullable=False)  # External reference
    
    # Relationships
    collection_id = Column(String, ForeignKey("collections.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Operation Details
    type = Column(Enum(OperationType))  # INDEX, APPEND, REINDEX, etc.
    status = Column(Enum(OperationStatus))  # PENDING, PROCESSING, etc.
    task_id = Column(String)  # Celery task ID
    
    # Configuration & Results
    config = Column(JSON, nullable=False)  # Operation-specific config
    error_message = Column(Text)
    meta = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True))
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
```

### 3.2 Operation Types

**Location:** `packages/shared/database/models.py`

```python
class OperationType(str, enum.Enum):
    INDEX = "index"           # Initial collection indexing
    APPEND = "append"         # Add documents to existing collection
    REINDEX = "reindex"       # Blue-green reindexing
    REMOVE_SOURCE = "remove_source"  # Remove documents by source
    DELETE = "delete"         # Delete entire collection
```

### 3.3 API Endpoints

**Location:** `packages/webui/api/v2/operations.py`

```python
# Get operation details
GET /api/v2/operations/{operation_uuid}
Response: OperationResponse

# Cancel operation
DELETE /api/v2/operations/{operation_uuid}
Response: OperationResponse

# List operations with filtering
GET /api/v2/operations?status=processing&type=index&page=1&per_page=50
Response: List[OperationResponse]

# WebSocket for real-time updates
WS /ws/operations/{operation_id}?token={jwt_token}
```

### 3.4 Service Interface

**Location:** `packages/webui/services/operation_service.py`

```python
class OperationService:
    async def get_operation(operation_uuid: str, user_id: int) -> Operation
    async def cancel_operation(operation_uuid: str, user_id: int) -> Operation
    async def list_operations(
        user_id: int,
        status_list: Optional[List[OperationStatus]],
        operation_type: Optional[OperationType],
        offset: int,
        limit: int
    ) -> Tuple[List[Operation], int]
    async def verify_websocket_access(operation_uuid: str, user_id: int) -> Operation
```

## 4. Data Flow & Dependencies

### 4.1 Operation Creation Flow

```
User Request → API Endpoint → Collection Service
                                    ↓
                            Create Operation Record
                                    ↓
                            Submit Celery Task
                                    ↓
                            Store Task ID
                                    ↓
                            Return Operation UUID
```

### 4.2 Task Processing Flow

```
Celery Worker Picks Up Task
            ↓
    Update Status to PROCESSING
            ↓
    Send "operation_started" to Redis
            ↓
    Execute Operation Logic
            ↓
    Send Progress Updates to Redis
            ↓
    Update Status to COMPLETED/FAILED
            ↓
    Send Final Status to Redis
```

### 4.3 Real-time Update Flow

```
Worker → Redis Stream → ScalableWebSocketManager → WebSocket Client
   ↓                           ↓
Redis Pub/Sub         Connection Registry
                              ↓
                      Multi-instance Support
```

### 4.4 Dependencies

**Internal Dependencies:**
- `shared.database.models`: Operation, OperationStatus, OperationType
- `shared.database.repositories`: OperationRepository
- `webui.celery_app`: Celery application instance
- `webui.websocket.scalable_manager`: WebSocket management

**External Dependencies:**
- PostgreSQL: Operation persistence
- Redis: Task queue, streams, pub/sub
- Celery: Distributed task execution
- Qdrant: Vector operations

## 5. Critical Implementation Details

### 5.1 Task Configuration

**Location:** `packages/webui/tasks.py`

```python
# Timeout Configuration
OPERATION_SOFT_TIME_LIMIT = 3600  # 1 hour soft limit
OPERATION_HARD_TIME_LIMIT = 7200  # 2 hour hard limit

# Retry Configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 60  # seconds

# Batch Processing
EMBEDDING_BATCH_SIZE = 100
VECTOR_UPLOAD_BATCH_SIZE = 100
DOCUMENT_REMOVAL_BATCH_SIZE = 100

# Redis Stream Configuration
REDIS_STREAM_MAX_LEN = 1000  # Keep last 1000 messages
REDIS_STREAM_TTL = 86400  # 24 hours
```

### 5.2 Progress Reporting Mechanism

**Location:** `packages/webui/tasks.py`

```python
class CeleryTaskWithOperationUpdates:
    """Context manager for sending operation updates"""
    
    async def send_update(self, update_type: str, data: dict):
        # Write to Redis Stream
        await redis_client.xadd(
            f"operation-progress:{operation_id}",
            {"message": json.dumps(message)},
            maxlen=REDIS_STREAM_MAX_LEN
        )
        
        # Publish to Pub/Sub for WebSocket distribution
        await redis_client.publish(
            f"operation:{operation_id}",
            json.dumps(pub_message)
        )
```

### 5.3 Cancellation Implementation

**Location:** `packages/webui/services/operation_service.py`

```python
async def cancel_operation(self, operation_uuid: str, user_id: int):
    # 1. Update database status to CANCELLED
    operation = await self.operation_repo.cancel(
        operation_uuid=operation_uuid,
        user_id=user_id
    )
    
    # 2. Revoke Celery task if running
    if operation.task_id:
        celery_app.control.revoke(
            operation.task_id, 
            terminate=True  # Force termination
        )
    
    # 3. Commit transaction
    await self.db_session.commit()
```

### 5.4 WebSocket Connection Management

**Location:** `packages/webui/websocket/scalable_manager.py`

```python
class ScalableWebSocketManager:
    # Connection limits
    max_connections_per_user = 10
    max_total_connections = 10000
    
    # Horizontal scaling via Redis Pub/Sub
    async def connect(websocket, user_id, operation_id):
        # Register connection in Redis
        await redis_client.hset(
            "websocket:connections",
            connection_id,
            json.dumps(metadata)
        )
        
        # Subscribe to channels
        await pubsub.subscribe(f"operation:{operation_id}")
        
        # Send initial state
        await _send_operation_state(websocket, operation_id)
```

### 5.5 State Transition Validation

**Location:** `packages/shared/database/repositories/operation_repository.py`

```python
async def cancel(self, operation_uuid: str, user_id: int):
    # Validate current status allows cancellation
    if operation.status not in (
        OperationStatus.PENDING, 
        OperationStatus.PROCESSING
    ):
        raise ValidationError(
            f"Cannot cancel operation in {operation.status} status"
        )
    
    # Atomic status update
    operation.status = OperationStatus.CANCELLED
    operation.completed_at = datetime.now(UTC)
```

## 6. Security Considerations

### 6.1 Operation Ownership

**Access Control:**
- Operations are owned by the creating user
- Access granted if: user owns operation OR owns collection OR collection is public
- Permission checks enforced at service layer

```python
# Repository permission check
if (operation.user_id != user_id and
    operation.collection.owner_id != user_id and
    not operation.collection.is_public):
    raise AccessDeniedError(user_id, "operation", operation_uuid)
```

### 6.2 WebSocket Authentication

**JWT Token Validation:**
```python
# WebSocket authentication via query parameter
token = websocket.query_params.get("token")
user = await get_current_user_websocket(token)

# Verify operation access
await service.verify_websocket_access(
    operation_uuid=operation_id,
    user_id=user["id"]
)
```

### 6.3 Input Validation

**Configuration Validation:**
- Operation config cannot be empty
- Collection must exist and be accessible
- Operation type must be valid enum value
- File paths sanitized and validated

### 6.4 Resource Limits

**Protection Against Abuse:**
- Max 10 WebSocket connections per user
- Max 10,000 total connections per instance
- Operation timeouts (1 hour soft, 2 hour hard)
- Redis stream length limited to 1000 messages

## 7. Testing Requirements

### 7.1 Unit Tests

**Location:** `tests/unit/test_operation_service.py`

```python
class TestOperationService:
    # Permission Tests
    - test_get_operation_success
    - test_get_operation_not_found
    - test_get_operation_access_denied
    
    # Cancellation Tests
    - test_cancel_operation_success
    - test_cancel_operation_invalid_status
    - test_cancel_operation_with_celery_task
    
    # Listing Tests
    - test_list_operations_filtering
    - test_list_operations_pagination
```

**Location:** `tests/unit/test_operation_repository.py`

```python
class TestOperationRepository:
    # CRUD Operations
    - test_create_operation
    - test_get_by_uuid
    - test_update_status
    
    # State Transitions
    - test_pending_to_processing
    - test_processing_to_completed
    - test_cancel_pending_operation
    
    # Permission Checks
    - test_permission_check_owner
    - test_permission_check_collection_owner
```

### 7.2 Integration Tests

**Location:** `tests/webui/api/v2/test_operations.py`

```python
class TestOperationsAPI:
    # API Endpoints
    - test_get_operation_endpoint
    - test_cancel_operation_endpoint
    - test_list_operations_with_filters
    
    # WebSocket Tests
    - test_websocket_authentication
    - test_websocket_progress_updates
    - test_websocket_disconnection
```

### 7.3 WebSocket Tests

**Test Coverage Required:**
```python
# Connection Management
- test_websocket_connect_with_valid_token
- test_websocket_connect_with_invalid_token
- test_websocket_permission_denied

# Message Flow
- test_receive_operation_updates
- test_handle_ping_pong
- test_connection_cleanup

# Scalability
- test_multiple_connections_same_operation
- test_cross_instance_messaging
- test_connection_limit_enforcement
```

### 7.4 Task Tests

**Celery Task Testing:**
```python
# Task Execution
- test_process_collection_operation_success
- test_task_failure_handling
- test_task_retry_logic
- test_task_timeout

# Progress Updates
- test_redis_stream_updates
- test_progress_throttling
- test_final_status_update
```

## 8. Common Pitfalls & Best Practices

### 8.1 Common Pitfalls

**State Consistency Issues:**
```python
# WRONG: Non-atomic status update
operation.status = OperationStatus.PROCESSING
# ... other operations that might fail ...
await db.commit()  # Status inconsistent if failure above

# RIGHT: Use transactions
async with db.begin():
    operation.status = OperationStatus.PROCESSING
    # All operations within transaction
```

**WebSocket Memory Leaks:**
```python
# WRONG: Not cleaning up connections
connections[connection_id] = websocket

# RIGHT: Always cleanup in finally block
try:
    connections[connection_id] = websocket
    # ... handle connection ...
finally:
    await ws_manager.disconnect(connection_id)
```

**Task ID Race Condition:**
```python
# WRONG: Setting task_id after other operations
operation = create_operation()
# ... other operations ...
operation.task_id = celery_task.id

# RIGHT: Set task_id immediately
task_id = celery_task.request.id
await operation_repo.set_task_id(operation_id, task_id)
```

### 8.2 Best Practices

**1. Always Use Context Managers for Resources:**
```python
async with CeleryTaskWithOperationUpdates(operation_id) as updater:
    # Automatic cleanup on exit
    await updater.send_update("processing", data)
```

**2. Implement Proper Error Recovery:**
```python
try:
    result = await process_operation()
except Exception as e:
    await operation_repo.update_status(
        operation_id, 
        OperationStatus.FAILED,
        error_message=str(e)
    )
    raise  # Let Celery handle retry
finally:
    # Always update final status
    await ensure_status_updated()
```

**3. Use Optimistic Locking for Concurrent Updates:**
```python
# Check operation hasn't changed before cancellation
operation = await get_with_lock(operation_id)
if operation.status in CANCELLABLE_STATES:
    operation.status = OperationStatus.CANCELLED
```

**4. Implement Progress Throttling:**
```python
# Avoid flooding with updates
last_update = self._progress_throttle.get(operation_id)
if last_update and (now - last_update) < THROTTLE_THRESHOLD:
    return  # Skip this update
```

**5. Handle WebSocket Reconnection:**
```python
// Frontend: Automatic reconnection
useEffect(() => {
    if (!isConnected && operationId) {
        reconnectWebSocket();
    }
}, [isConnected, operationId]);
```

## 9. Configuration & Environment

### 9.1 Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_STREAM_DB=0
REDIS_PUBSUB_DB=2

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/1
CELERY_TASK_TIME_LIMIT=7200
CELERY_TASK_SOFT_TIME_LIMIT=3600

# WebSocket Configuration
WS_MAX_CONNECTIONS_PER_USER=10
WS_MAX_TOTAL_CONNECTIONS=10000
WS_HEARTBEAT_INTERVAL=30
```

### 9.2 Celery Task Configuration

**Location:** `packages/webui/celery_app.py`

```python
celery_app.conf.update(
    task_acks_late=True,  # Late acknowledgment for reliability
    task_reject_on_worker_lost=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours hard limit
    task_soft_time_limit=3600,  # 1 hour soft limit
    worker_prefetch_multiplier=1,  # One task at a time
    worker_max_tasks_per_child=100,  # Restart after 100 tasks
)
```

### 9.3 Redis Stream Configuration

```python
# Stream key pattern
STREAM_KEY_PATTERN = "operation-progress:{operation_id}"

# Pub/Sub channels
OPERATION_CHANNEL = "operation:{operation_id}"
USER_CHANNEL = "user:{user_id}"
COLLECTION_CHANNEL = "collection:{collection_id}"

# TTL and retention
STREAM_MAX_LENGTH = 1000  # messages
STREAM_TTL = 86400  # 24 hours
```

### 9.4 Retry Policies

```python
# Operation-specific retry configuration
RETRY_CONFIG = {
    OperationType.INDEX: {
        "max_retries": 3,
        "retry_delay": 60,
        "retry_backoff": True
    },
    OperationType.REINDEX: {
        "max_retries": 2,
        "retry_delay": 120,
        "retry_backoff": True
    }
}
```

## 10. Integration Points

### 10.1 Celery Worker Integration

**Worker Initialization:**
```python
# Worker starts and connects to Redis
# Subscribes to task queue
# Processes operations sequentially
# Sends updates via Redis streams
```

**Task Lifecycle:**
1. Task received from queue
2. Operation record loaded from database
3. Task ID stored immediately
4. Processing begins with status update
5. Progress streamed to Redis
6. Final status recorded
7. Task acknowledged to Celery

### 10.2 WebSocket Integration

**Connection Flow:**
```python
# 1. Client connects with JWT token
ws = new WebSocket(`/ws/operations/${operationId}?token=${token}`)

# 2. Server validates token and permissions
user = await get_current_user_websocket(token)
await verify_operation_access(operation_id, user_id)

# 3. Subscribe to Redis channels
await ws_manager.connect(websocket, user_id, operation_id)

# 4. Stream updates to client
while connected:
    message = await receive_from_redis()
    await websocket.send_json(message)
```

### 10.3 Frontend Integration

**React Hook Usage:**
```typescript
// apps/webui-react/src/hooks/useOperationProgress.ts
const { isConnected, readyState } = useOperationProgress(
    operationId,
    {
        onComplete: () => refreshCollection(),
        onError: (error) => showError(error),
        showToasts: true
    }
);
```

**State Updates:**
```typescript
// Automatic cache updates via React Query
updateOperationInCache(operationId, {
    status: status as OperationStatus,
    progress: progress,
});
```

### 10.4 Database Integration

**Transaction Boundaries:**
```python
async with AsyncSessionLocal() as db:
    # All database operations within session
    operation_repo = OperationRepository(db)
    collection_repo = CollectionRepository(db)
    
    # Atomic operations
    async with db.begin():
        await operation_repo.update_status(...)
        await collection_repo.update_stats(...)
```

### 10.5 Monitoring Integration

**Prometheus Metrics:**
```python
# Operation metrics
operation_duration_seconds = Histogram(
    'operation_duration_seconds',
    'Time spent processing operations',
    ['operation_type']
)

operation_total = Counter(
    'operations_total',
    'Total number of operations',
    ['operation_type', 'status']
)

# Resource metrics
collection_cpu_seconds_total = Counter(...)
collection_memory_usage_bytes = Gauge(...)
```

### 10.6 Audit Log Integration

```python
# Every operation creates audit trail
await create_audit_log(
    collection_id=operation.collection_id,
    operation_id=operation.id,
    user_id=operation.user_id,
    action=f"operation_{operation.type}_{status}",
    details={
        "duration": duration,
        "documents_processed": doc_count
    }
)
```

## Key Files Reference

- **Operation Service:** `/home/john/semantik/packages/webui/services/operation_service.py`
- **Operation API:** `/home/john/semantik/packages/webui/api/v2/operations.py`
- **Operation Repository:** `/home/john/semantik/packages/shared/database/repositories/operation_repository.py`
- **Operation Models:** `/home/john/semantik/packages/shared/database/models.py` (lines 86-104, 323-358)
- **Celery Tasks:** `/home/john/semantik/packages/webui/tasks.py`
- **WebSocket Manager:** `/home/john/semantik/packages/webui/websocket/scalable_manager.py`
- **Frontend Hook:** `/home/john/semantik/apps/webui-react/src/hooks/useOperationProgress.ts`
- **Operation Tests:** `/home/john/semantik/tests/unit/test_operation_service.py`

## Critical Notes for LLM Agents

1. **Never bypass permission checks** - Always use service layer methods that include authorization
2. **Always set task_id immediately** - This is critical for cancellation support
3. **Use proper status transitions** - Only certain transitions are valid (see state machine)
4. **Handle WebSocket disconnections gracefully** - Cleanup resources in finally blocks
5. **Respect resource limits** - Don't exceed connection limits or timeout configurations
6. **Use transactions for atomic updates** - Especially when updating multiple related entities
7. **Send progress updates judiciously** - Implement throttling to avoid overwhelming clients
8. **Test cancellation thoroughly** - Ensure resources are cleaned up properly
9. **Monitor Redis memory usage** - Streams can grow large if not properly managed
10. **Implement proper error recovery** - Always update operation status even on failure