# Celery/Redis Task Queue Setup

This document describes the Celery/Redis task queue infrastructure added to Semantik.

## Overview

The following services have been added to support distributed task processing:

1. **Redis** - Message broker and result backend for Celery
2. **Celery Worker** - Task execution service with GPU support
3. **Flower** - Web-based monitoring tool for Celery

## Services Configuration

### Redis
- **Port**: 6379
- **Container**: semantik-redis
- **Data persistence**: Uses named volume `redis_data`
- **Memory**: 512MB limit, 256MB reserved

### Celery Worker
- **Container**: semantik-worker
- **GPU Support**: Yes (configurable via CUDA_VISIBLE_DEVICES)
- **Concurrency**: 1 worker process (configurable)
- **Memory**: 4GB limit, 2GB reserved
- **Scalable**: Can be scaled horizontally

### Flower Monitoring
- **Port**: 5555
- **Container**: semantik-flower
- **Default Auth**: admin:admin (⚠️ **CHANGE IN PRODUCTION** - configurable via FLOWER_BASIC_AUTH)
- **Memory**: 512MB limit, 256MB reserved

> ⚠️ **PRODUCTION SECURITY WARNING**: The default Flower authentication (admin:admin) is for development only. For production deployments:
> - Change the default credentials by setting `FLOWER_BASIC_AUTH=username:strongpassword` in your environment
> - Consider using a reverse proxy with proper SSL/TLS termination
> - Restrict access to Flower UI using firewall rules or VPN
> - Use OAuth2 authentication if available in your infrastructure

## Quick Start

1. **Start all services**:
   ```bash
   docker compose up -d
   ```

2. **Check service status**:
   ```bash
   docker compose ps
   ```

3. **View logs**:
   ```bash
   # All services
   docker compose logs -f
   
   # Specific service
   docker compose logs -f worker
   ```

4. **Access Flower monitoring**:
   - Open http://localhost:5555 in your browser
   - Login with admin:admin (or configured credentials)

## Scaling Workers

To run multiple workers for parallel processing:

```bash
# Scale to 3 workers
docker compose up -d --scale worker=3

# Scale back to 1 worker
docker compose up -d --scale worker=1
```

## Environment Variables

### For Celery/Redis Configuration

- `CELERY_BROKER_URL`: Redis broker URL (default: redis://redis:6379/0)
- `CELERY_RESULT_BACKEND`: Result backend URL (default: redis://redis:6379/0)
- `CELERY_CONCURRENCY`: Number of worker processes (default: 1)
- `FLOWER_BASIC_AUTH`: Flower authentication (default: admin:admin) ⚠️ **CHANGE IN PRODUCTION**

### For GPU Configuration

- `CUDA_VISIBLE_DEVICES`: GPU device IDs to use (default: 0)
- `MODEL_MAX_MEMORY_GB`: Max GPU memory per model (default: 8)

## Testing the Setup

Run the provided test script to verify the configuration:

```bash
python test_celery_setup.py
```

## Implemented Celery Tasks

The following Celery tasks have been implemented:

### process_embedding_job_task
- **Module**: `packages/webui/tasks.py`
- **Purpose**: Processes document embedding jobs asynchronously
- **Features**:
  - Async execution within Celery
  - Real-time progress updates via task state
  - Full error handling and retry logic
  - Memory optimization and garbage collection

### test_task
- **Module**: `packages/webui/tasks.py`
- **Purpose**: Simple test task to verify Celery is working

To create additional Celery tasks, add them to `packages/webui/tasks.py`:

```python
from webui.celery_app import celery_app

@celery_app.task(bind=True)
def your_new_task(self, param1, param2):
    """Your task description."""
    # Task implementation
    return {"status": "completed", "result": "data"}
```

## Monitoring and Debugging

1. **View worker logs**:
   ```bash
   docker compose logs -f worker
   ```

2. **Check Redis connectivity**:
   ```bash
   docker compose exec redis redis-cli ping
   ```

3. **Monitor tasks in Flower**:
   - Tasks tab: View running/pending tasks
   - Workers tab: Monitor worker health
   - Broker tab: Check message queue status

## Troubleshooting

1. **Worker not starting**: Check logs with `docker compose logs worker`
2. **GPU not available**: Verify CUDA_VISIBLE_DEVICES and driver installation
3. **Redis connection errors**: Ensure Redis service is running
4. **Flower not accessible**: Check if port 5555 is not already in use

## GPU Resource Management

The application includes GPU scheduling to prevent resource contention when multiple workers need GPU access:

### GPU Scheduler Features
- **Distributed Locking**: Uses Redis to coordinate GPU allocation across workers
- **Automatic Fallback**: Falls back to CPU if no GPU is available
- **Memory Management**: Clears GPU memory after each task
- **Timeout Handling**: Prevents GPU hogging with configurable timeouts

### Configuration
- `CUDA_VISIBLE_DEVICES`: Set which GPUs workers can use (e.g., "0,1,2")
- GPU allocation timeout: 2 hours per task (configurable in `gpu_scheduler.py`)
- Wait timeout: 5 minutes to acquire a GPU (configurable)

### Scaling Considerations
- Each worker can use only one GPU at a time
- Workers will wait for GPU availability or fall back to CPU
- Monitor GPU memory usage through Flower task details
- For maximum throughput: `num_workers ≤ num_gpus`

### Monitoring GPU Usage
```bash
# Check GPU allocation status
docker compose exec worker python -c "
from shared.gpu_scheduler import get_gpu_scheduler
scheduler = get_gpu_scheduler()
import json
print(json.dumps(scheduler.get_gpu_status(), indent=2))
"
```

## Database Connection Management

When scaling workers, it's important to manage database connections properly to avoid exhaustion:

### Connection Pooling
The application includes a connection pool module (`shared.database.connection_pool`) that:
- Limits connections per worker to prevent exhaustion
- Uses SQLite WAL mode for better concurrency
- Automatically handles broken connections
- Provides timeout handling

### Worker Scaling Considerations
- Each worker maintains its own connection pool (max 3 connections by default)
- When scaling workers, consider total connections: `workers * connections_per_worker`
- For SQLite, excessive connections can lead to lock contention
- For production with many workers, consider migrating to PostgreSQL

### Configuration
To adjust connection limits, modify `connection_pool.py`:
```python
max_connections_per_worker = 3  # Adjust based on your needs
```

## Periodic Tasks

The application includes periodic cleanup tasks that run automatically:

### Cleanup Task
- **Task**: `cleanup_old_results`
- **Schedule**: Runs daily at midnight
- **Purpose**: Cleans up old Celery results and archives old job records
- **Configuration**: Keeps results for 7 days by default

### Running Celery Beat

To enable periodic tasks, you need to run Celery Beat scheduler:

```bash
# Option 1: Add beat service to docker-compose.yml
beat:
  build:
    context: .
    dockerfile: Dockerfile
    target: runtime
  container_name: semantik-beat
  command: ["celery", "-A", "webui.celery_app", "beat", "--loglevel=info"]
  environment:
    - PYTHONPATH=/app/packages
    - CELERY_BROKER_URL=redis://redis:6379/0
  depends_on:
    - redis
  restart: unless-stopped

# Option 2: Run beat in an existing worker (not recommended for production)
docker compose exec worker celery -A webui.celery_app beat --loglevel=info
```

### Custom Periodic Tasks

To add custom periodic tasks, update the beat_schedule in `celery_app.py`:

```python
beat_schedule={
    'cleanup-old-results': {
        'task': 'webui.tasks.cleanup_old_results',
        'schedule': 86400.0,  # Daily
        'args': (7,),
    },
    'your-custom-task': {
        'task': 'webui.tasks.your_task_name',
        'schedule': 3600.0,  # Hourly
        'kwargs': {'param': 'value'},
    },
}
```

## Security Considerations

### Production Deployment

When deploying to production, ensure the following security measures:

1. **Change Default Credentials**:
   - Set strong Flower authentication: `FLOWER_BASIC_AUTH=username:strongpassword`
   - Never use the default `admin:admin` credentials in production

2. **Network Security**:
   - Redis port (6379) is exposed by default for development
   - In production, consider:
     - Using Docker internal networking only (remove port mapping)
     - Implementing Redis authentication with `requirepass`
     - Using SSL/TLS for Redis connections

3. **Flower Security**:
   - Always run Flower behind a reverse proxy with SSL/TLS
   - Consider additional authentication layers (OAuth, LDAP, etc.)
   - Restrict access to Flower UI by IP address if possible

4. **Worker Security**:
   - Run workers with minimal privileges
   - Limit resource usage to prevent DoS
   - Monitor worker logs for suspicious activity

5. **Data Security**:
   - Ensure Redis persistence is configured securely
   - Encrypt sensitive task data before queuing
   - Set appropriate task result expiration times

## Next Steps

With this infrastructure in place, you can now:
1. Implement asynchronous document processing tasks
2. Move heavy operations from API endpoints to background tasks
3. Add task scheduling and periodic tasks
4. Implement progress tracking for long-running operations