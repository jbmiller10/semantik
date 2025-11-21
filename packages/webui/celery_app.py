"""Celery application configuration."""

import logging
import os
from typing import Any

from celery import Celery
from celery.signals import worker_process_init
from shared.config import settings as shared_settings
from shared.config.internal_api_key import ensure_internal_api_key
from shared.config.runtime import ensure_webui_directories, require_jwt_secret
from shared.database.postgres_database import pg_connection_manager

# Configure logging
logger = logging.getLogger(__name__)


def _is_truthy(value: str | None) -> bool:
    """Return True if the provided environment toggle is truthy."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_base_config() -> dict[str, Any]:
    """Return the baseline Celery configuration shared across environments."""
    return {
        # Task serialization
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,
        # Task execution limits
        "task_soft_time_limit": 3600,  # 1 hour soft limit
        "task_time_limit": 7200,  # 2 hour hard limit
        "task_acks_late": True,  # Tasks acknowledged after execution
        "task_reject_on_worker_lost": True,
        # Worker configuration
        "worker_prefetch_multiplier": 1,  # Disable prefetching for long tasks
        "worker_max_tasks_per_child": 100,  # Restart worker after 100 tasks
        "worker_hijack_root_logger": False,
        # Result backend settings
        "result_expires": 3600,  # Results expire after 1 hour
        "result_persistent": True,
        # Connection retry configuration
        "broker_connection_retry_on_startup": True,
        "broker_connection_max_retries": 10,
        "broker_connection_retry": True,
        "broker_connection_retry_delay": 1.0,
        "broker_connection_retry_max_delay": 30.0,
        "broker_connection_retry_backoff_factor": 2.0,
        # Retry configuration
        "task_default_retry_delay": 60,
        "task_max_retries": 3,
        # Enable task events for monitoring
        "worker_send_task_events": True,
        "task_send_sent_event": True,
        # Beat schedule for periodic tasks
        "beat_schedule": {
            "cleanup-old-results": {
                "task": "webui.tasks.cleanup_old_results",
                "schedule": 86400.0,  # Run daily
                "args": (7,),
            },
            "refresh-collection-chunking-stats": {
                "task": "webui.tasks.refresh_collection_chunking_stats",
                "schedule": 3600.0,  # Run hourly
                "options": {
                    "queue": "default",
                    "priority": 5,
                },
            },
            "monitor-partition-health": {
                "task": "webui.tasks.monitor_partition_health",
                "schedule": 21600.0,  # Every 6 hours
                "options": {
                    "queue": "default",
                    "priority": 3,
                },
            },
        },
    }


def _build_testing_overrides() -> dict[str, Any]:
    """Return configuration overrides that make Celery safe inside tests."""
    return {
        "worker_send_task_events": False,
        "task_send_sent_event": False,
        "broker_connection_retry_on_startup": False,
        "broker_connection_retry": False,
        "result_persistent": False,
        "beat_schedule": {},  # Disable periodic tasks during tests
    }


def _create_celery_app() -> Celery:
    """Instantiate and configure the Celery app."""
    try:
        ensure_webui_directories(shared_settings)
        require_jwt_secret(shared_settings)
        ensure_internal_api_key(shared_settings)
    except RuntimeError as exc:
        logger.error("Failed to initialise internal API key for Celery: %s", exc)
        raise

    testing_mode = _is_truthy(os.getenv("TESTING"))

    broker_url_env = "CELERY_BROKER_URL"
    backend_url_env = "CELERY_RESULT_BACKEND"

    if testing_mode:
        # Allow overrides while defaulting to in-memory transports during tests.
        broker_url = os.getenv("CELERY_TEST_BROKER_URL", "memory://")
        backend_url = os.getenv("CELERY_TEST_RESULT_BACKEND", "cache+memory://")
        logger.debug("Initializing Celery in testing mode with in-memory transports.")
    else:
        broker_url = os.getenv(broker_url_env, "redis://localhost:6379/0")
        backend_url = os.getenv(backend_url_env, "redis://localhost:6379/0")

    celery = Celery(
        "webui",
        broker=broker_url,
        backend=backend_url,
        include=["webui.tasks", "webui.chunking_tasks"],
    )

    config = _build_base_config()
    if testing_mode:
        config.update(_build_testing_overrides())

    celery.conf.update(config)
    celery.autodiscover_tasks(["webui"])

    return celery


# Create Celery instance with environment-aware configuration.
celery_app = _create_celery_app()


# Worker initialization
@worker_process_init.connect
def init_worker_process(**kwargs: Any) -> None:  # noqa: ARG001
    """Initialize worker process - prepare for database connections."""
    # Reset the connection manager state to ensure we don't use inherited
    # connections from the parent process (which are not fork-safe).
    # This forces a fresh engine creation when the first task runs.
    pg_connection_manager._engine = None
    pg_connection_manager._sessionmaker = None
    logger.info("Worker process initialized - database connection reset")
