# Celery Task Implementation for Embedding Jobs

This document describes the implementation of CORE-302: Refactoring the embedding job processing to use Celery tasks.

## Changes Made

### 1. Created `celery_app.py`
- Located at: `packages/webui/celery_app.py`
- Configures Celery to use Redis as broker and result backend
- Sets appropriate task limits and worker configuration
- Auto-discovers tasks from the webui module

### 2. Refactored `tasks.py`
- Located at: `packages/webui/tasks.py`
- Contains the main `process_embedding_job_task` Celery task
- Implements async processing using `asyncio` within the Celery task
- Maintains all original processing logic from the previous implementation
- Updates Celery task state for real-time progress tracking

### 3. Updated `jobs.py`
- Removed the local `process_embedding_job` function
- Replaced `asyncio.create_task` with Celery task execution
- Updated task cancellation to use Celery's revoke mechanism
- Added WebSocket polling for Celery task state updates

### 4. Docker Configuration
- Updated worker service command to use `webui.celery_app`
- Updated Flower service command for monitoring

## Architecture

### Task Execution Flow

1. **Job Creation**: When a user creates an embedding job, the API endpoint:
   - Creates job records in the database
   - Creates a Qdrant collection
   - Submits the job to Celery: `process_embedding_job_task.delay(job_id)`
   - Stores the Celery task ID for tracking

2. **Task Processing**: The Celery worker:
   - Picks up the task from Redis queue
   - Runs the async processing logic in an event loop
   - Updates task state for progress tracking
   - Updates database records throughout processing
   - Handles file extraction, chunking, embedding generation, and Qdrant uploads

3. **Progress Updates**: WebSocket connections:
   - Poll Celery task state every 2 seconds
   - Send updates to connected clients
   - Clean up when tasks complete or fail

### Key Features

- **Async Support**: Uses native Python async/await within Celery tasks
- **Progress Tracking**: Updates Celery task state for real-time monitoring
- **Cancellation**: Supports task cancellation via Celery's revoke mechanism
- **Error Handling**: Maintains original error handling and retry logic
- **Resource Management**: Preserves memory optimization and garbage collection

## Testing

### Manual Testing Steps

1. **Start all services**:
   ```bash
   docker compose up -d
   ```

2. **Verify worker is running**:
   ```bash
   docker compose logs worker
   ```
   Look for: "ready" and task registration messages

3. **Monitor with Flower**:
   - Open http://localhost:5555
   - Login with admin:admin
   - Check "Workers" tab for active workers
   - Check "Tasks" tab during job processing

4. **Create an embedding job**:
   - Use the web UI to create a new job
   - Monitor progress in the UI
   - Check Flower for task execution

### Automated Testing

The Celery worker will be tested as part of the integration test suite. Key test scenarios:

1. Task discovery and registration
2. Successful job processing
3. Error handling and retry logic
4. Task cancellation
5. Progress update accuracy

## Benefits

1. **Scalability**: Can scale workers horizontally for parallel processing
2. **Reliability**: Tasks survive worker crashes and can be retried
3. **Monitoring**: Flower provides real-time task monitoring
4. **Decoupling**: API remains responsive during long-running tasks
5. **Resource Isolation**: Each worker has its own resource allocation

## Configuration

### Environment Variables

- `CELERY_BROKER_URL`: Redis broker URL (default: redis://redis:6379/0)
- `CELERY_RESULT_BACKEND`: Result backend URL (default: redis://redis:6379/0)

### Celery Settings

Key settings in `celery_app.py`:
- `task_soft_time_limit`: 1 hour soft limit
- `task_time_limit`: 2 hour hard limit
- `worker_prefetch_multiplier`: 1 (disable prefetching for long tasks)
- `worker_max_tasks_per_child`: 100 (restart after 100 tasks)

## Migration Notes

- Existing jobs will continue to work as before
- No database schema changes required
- WebSocket API remains unchanged
- Frontend requires no modifications

## Future Enhancements

1. **Task Scheduling**: Add periodic tasks for maintenance
2. **Priority Queues**: Implement task priorities
3. **Rate Limiting**: Add per-user rate limits
4. **Task Chaining**: Chain multiple processing steps
5. **Batch Processing**: Process multiple files in parallel within a task