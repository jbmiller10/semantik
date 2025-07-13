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
- **Default Auth**: admin:admin (configurable via FLOWER_BASIC_AUTH)
- **Memory**: 512MB limit, 256MB reserved

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
- `FLOWER_BASIC_AUTH`: Flower authentication (default: admin:admin)

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

## Next Steps

With this infrastructure in place, you can now:
1. Implement asynchronous document processing tasks
2. Move heavy operations from API endpoints to background tasks
3. Add task scheduling and periodic tasks
4. Implement progress tracking for long-running operations