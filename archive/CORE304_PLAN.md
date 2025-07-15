# Redis Pub/Sub WebSocket Implementation Plan

## Overview
Implement distributed WebSocket state synchronization using Redis Streams to enable real-time job progress updates across multiple API server instances.

## Implementation Steps

### Phase 1: Create Redis WebSocket Manager
1. Create `packages/webui/websocket_manager.py`:
   - Implement `RedisStreamWebSocketManager` class based on the design in REDIS_PUBSUB_DESIGN.md
   - Use Redis Streams for message persistence and guaranteed delivery
   - Support consumer groups for multi-server deployments
   - Include methods for connection management, message consumption, and broadcasting

### Phase 2: Update Celery Tasks
1. Modify `packages/webui/tasks.py`:
   - Add Redis client initialization in tasks
   - Create helper class `CeleryTaskWithUpdates` for sending updates
   - Replace `celery_task.update_state()` calls with Redis Stream publishing
   - Send structured messages: job_started, file_processing, file_completed, error, job_completed

### Phase 3: Integrate with FastAPI
1. Update `packages/webui/main.py`:
   - Add application lifespan manager for Redis connection lifecycle
   - Initialize global `ws_manager` instance on startup
   - Clean up resources on shutdown

2. Update `packages/webui/api/jobs.py`:
   - Replace local `ConnectionManager` with imported `ws_manager`
   - Update WebSocket endpoint to use new manager
   - Add user authentication to WebSocket connections

### Phase 4: Configuration and Dependencies
1. Add Redis configuration:
   - Add `REDIS_URL` to environment variables
   - Configure Redis Stream TTL and max message length
   - Add connection pool settings

2. Update dependencies:
   - Add `redis` or `aioredis` to pyproject.toml
   - Ensure compatible versions with existing packages

### Phase 5: Testing and Monitoring
1. Create unit tests for WebSocket manager
2. Add integration tests for end-to-end message flow
3. Implement Prometheus metrics for monitoring
4. Add logging for debugging and troubleshooting

## Key Benefits
- **Scalability**: Support multiple API servers
- **Reliability**: Guaranteed message delivery with Redis Streams
- **Resilience**: Automatic reconnection and message history
- **Performance**: Efficient pub/sub pattern instead of polling
- **Observability**: Built-in metrics and logging

## Migration Strategy
- Deploy changes incrementally without breaking existing functionality
- Test thoroughly in development environment
- Monitor Redis memory usage and performance
- Provide rollback capability if issues arise