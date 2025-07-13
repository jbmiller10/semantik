"""Celery application configuration."""
import os
from celery import Celery

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
    result_expires=86400,  # Results expire after 1 day
    result_persistent=True,  # Store results persistently
    
    # Retry configuration
    task_default_retry_delay=60,  # 60 seconds
    task_max_retries=3,
    
    # Enable task events for monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["webui"])