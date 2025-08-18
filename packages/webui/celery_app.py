"""Celery application configuration."""

import logging
import os
from typing import Any

from celery import Celery
from celery.signals import worker_process_init

# Configure logging
logger = logging.getLogger(__name__)

# Create Celery instance
celery_app = Celery(
    "webui",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["webui.tasks"],  # Auto-discover tasks in the tasks module
)

# Configure Celery
celery_app.conf.update(
    # Task serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Task execution limits
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
    task_acks_late=True,  # Tasks will be acknowledged after they have been executed
    task_reject_on_worker_lost=True,  # Reject tasks when worker shuts down
    # Worker configuration
    worker_prefetch_multiplier=1,  # Disable prefetching for long-running tasks
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks
    worker_hijack_root_logger=False,  # Don't hijack the root logger
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour (reduced from 1 day)
    result_persistent=True,  # Store results persistently
    # Connection retry configuration
    broker_connection_retry_on_startup=True,  # Retry broker connection on startup
    broker_connection_max_retries=10,  # Max retries for broker connection
    broker_connection_retry=True,  # Retry broker connection on failure
    broker_connection_retry_delay=1.0,  # Initial retry delay
    broker_connection_retry_max_delay=30.0,  # Max retry delay
    broker_connection_retry_backoff_factor=2.0,  # Exponential backoff factor
    # Retry configuration
    task_default_retry_delay=60,  # 60 seconds
    task_max_retries=3,
    # Enable task events for monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-results": {
            "task": "webui.tasks.cleanup_old_results",
            "schedule": 86400.0,  # Run daily (24 hours)
            "args": (7,),  # Keep results for 7 days
        },
        "refresh-collection-chunking-stats": {
            "task": "webui.tasks.refresh_collection_chunking_stats",
            "schedule": 3600.0,  # Run hourly
            "options": {
                "queue": "default",
                "priority": 5,  # Medium priority
            },
        },
        "monitor-partition-health": {
            "task": "webui.tasks.monitor_partition_health",
            "schedule": 21600.0,  # Run every 6 hours
            "options": {
                "queue": "default",
                "priority": 3,  # Higher priority for monitoring
            },
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["webui"])


# Worker initialization
@worker_process_init.connect
def init_worker_process(**kwargs: Any) -> None:  # noqa: ARG001
    """Initialize worker process - prepare for database connections."""
    logger.info("Worker process initialized - database will be initialized per task")
