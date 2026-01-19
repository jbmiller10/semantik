"""Domain-specific Celery task package.

This module glues together the newly modularized task implementations while
preserving the public surface previously exposed via ``webui.tasks``.
Importing this package registers all task modules so existing task names remain
available to Celery.
"""

from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any

import httpx

from webui.services.chunking.container import resolve_celery_chunking_orchestrator

from .benchmark import run_benchmark
from .cleanup import (
    cleanup_old_collections,
    cleanup_old_results,
    cleanup_qdrant_collections,
    cleanup_stale_benchmarks,
    monitor_partition_health,
    refresh_collection_chunking_stats,
)
from .ingestion import (
    _handle_task_failure,
    _handle_task_failure_async,
    _process_append_operation,
    _process_append_operation_impl,
    _process_collection_operation_async,
    _process_index_operation,
    _process_remove_source_operation,
    process_collection_operation,
    test_task,
)
from .projection import _process_projection_operation, compute_projection
from .reindex import (
    _cleanup_staging_resources,
    _process_reindex_operation,
    _process_reindex_operation_impl,
    _validate_reindex,
    reindex_handler,
)
from .utils import (
    CLEANUP_DELAY_MAX_SECONDS,
    CLEANUP_DELAY_MIN_SECONDS,
    CLEANUP_DELAY_PER_10K_VECTORS,
    CLEANUP_DELAY_SECONDS,
    DEFAULT_DAYS_TO_KEEP,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    DOCUMENT_REMOVAL_BATCH_SIZE,
    EMBEDDING_BATCH_SIZE,
    OPERATION_HARD_TIME_LIMIT,
    OPERATION_SOFT_TIME_LIMIT,
    REDIS_STREAM_MAX_LEN,
    REDIS_STREAM_TTL,
    REINDEX_SCORE_DIFF_THRESHOLD,
    REINDEX_SEARCH_MISMATCH_THRESHOLD,
    REINDEX_VECTOR_COUNT_VARIANCE,
    VECTOR_UPLOAD_BATCH_SIZE,
    CeleryTaskWithOperationUpdates,
    _audit_log_operation,
    _build_internal_api_headers,
    _get_internal_api_key,
    _record_operation_metrics,
    _sanitize_audit_details,
    _sanitize_error_message,
    _update_collection_metrics,
    calculate_cleanup_delay,
    celery_app,
    executor,
    logger,
    parse_file_thread_safe,
    settings,
)

__all__ = [
    # Ingestion tasks & helpers
    "process_collection_operation",
    "_process_collection_operation_async",
    "_process_index_operation",
    "_process_append_operation",
    "_process_append_operation_impl",
    "_process_remove_source_operation",
    "_process_projection_operation",
    "_handle_task_failure",
    "_handle_task_failure_async",
    "test_task",
    "compute_projection",
    # Reindex helpers
    "_process_reindex_operation",
    "_process_reindex_operation_impl",
    "_cleanup_staging_resources",
    "_validate_reindex",
    "reindex_handler",
    # Benchmark tasks
    "run_benchmark",
    # Cleanup tasks
    "cleanup_old_results",
    "cleanup_old_collections",
    "cleanup_qdrant_collections",
    "cleanup_stale_benchmarks",
    "refresh_collection_chunking_stats",
    "monitor_partition_health",
    # Utilities & shared constants
    "asyncio",
    "httpx",
    "CeleryTaskWithOperationUpdates",
    "resolve_celery_chunking_orchestrator",
    "celery_app",
    "executor",
    "parse_file_thread_safe",
    "calculate_cleanup_delay",
    "qdrant_manager",
    "settings",
    "logger",
    "_audit_log_operation",
    "_build_internal_api_headers",
    "_get_internal_api_key",
    "_record_operation_metrics",
    "_sanitize_error_message",
    "_sanitize_audit_details",
    "_update_collection_metrics",
    "CLEANUP_DELAY_SECONDS",
    "CLEANUP_DELAY_MIN_SECONDS",
    "CLEANUP_DELAY_MAX_SECONDS",
    "CLEANUP_DELAY_PER_10K_VECTORS",
    "DEFAULT_DAYS_TO_KEEP",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DOCUMENT_REMOVAL_BATCH_SIZE",
    "EMBEDDING_BATCH_SIZE",
    "OPERATION_SOFT_TIME_LIMIT",
    "OPERATION_HARD_TIME_LIMIT",
    "REINDEX_VECTOR_COUNT_VARIANCE",
    "REINDEX_SEARCH_MISMATCH_THRESHOLD",
    "REINDEX_SCORE_DIFF_THRESHOLD",
    "REDIS_STREAM_MAX_LEN",
    "REDIS_STREAM_TTL",
    "VECTOR_UPLOAD_BATCH_SIZE",
]


def _load_module(name: str) -> Any:
    return import_module(f"webui.tasks.{name}")


_PROXY_MODULES = tuple(
    _load_module(name) for name in ("ingestion", "projection", "reindex", "cleanup", "benchmark", "utils")
)


def __getattr__(name: str) -> Any:  # noqa: N807
    """Proxy attribute access to underlying task modules for test patches."""
    for module in _PROXY_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


def __setattr__(name: str, value: Any) -> None:  # noqa: N807
    """Propagate attribute assignments to the originating module when possible."""
    for module in _PROXY_MODULES:
        if hasattr(module, name):
            setattr(module, name, value)

    globals()[name] = value
