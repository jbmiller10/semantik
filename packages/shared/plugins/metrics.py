"""Prometheus metrics for plugin operations.

This module provides observability into plugin loading, health checks,
and overall plugin system state.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

from shared.metrics.prometheus import registry

if TYPE_CHECKING:
    from collections.abc import Generator

# Plugin Loading Metrics
PLUGIN_LOADS_TOTAL = Counter(
    "semantik_plugin_loads_total",
    "Total plugin load attempts",
    ["plugin_type", "plugin_id", "source", "status"],
    registry=registry,
)

PLUGIN_LOAD_DURATION = Histogram(
    "semantik_plugin_load_duration_seconds",
    "Plugin load duration in seconds",
    ["plugin_type"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=registry,
)

# Plugin Health Check Metrics
PLUGIN_HEALTH_CHECKS_TOTAL = Counter(
    "semantik_plugin_health_checks_total",
    "Total health check attempts",
    ["plugin_id", "result"],  # result: healthy, unhealthy, timeout, error
    registry=registry,
)

PLUGIN_HEALTH_CHECK_DURATION = Histogram(
    "semantik_plugin_health_check_duration_seconds",
    "Health check duration in seconds",
    ["plugin_id"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    registry=registry,
)

# Plugin State Gauges
PLUGINS_LOADED_GAUGE = Gauge(
    "semantik_plugins_loaded",
    "Number of loaded plugins by type and source",
    ["plugin_type", "source"],
    registry=registry,
)

PLUGINS_ENABLED_GAUGE = Gauge(
    "semantik_plugins_enabled",
    "Number of enabled plugins by type",
    ["plugin_type"],
    registry=registry,
)

PLUGINS_HEALTHY_GAUGE = Gauge(
    "semantik_plugins_healthy",
    "Number of healthy plugins by type",
    ["plugin_type"],
    registry=registry,
)

# Dependency Validation Metrics
PLUGIN_DEPENDENCY_WARNINGS_TOTAL = Counter(
    "semantik_plugin_dependency_warnings_total",
    "Total dependency validation warnings",
    ["plugin_id", "warning_type"],  # warning_type: missing, version, disabled
    registry=registry,
)


def record_plugin_load(
    plugin_type: str,
    plugin_id: str,
    source: str,
    success: bool,
    duration: float,
) -> None:
    """Record a plugin load attempt.

    Args:
        plugin_type: Type of plugin (embedding, chunking, etc.)
        plugin_id: Unique plugin identifier
        source: Plugin source (builtin, external)
        success: Whether the load succeeded
        duration: Time taken to load in seconds
    """
    status = "success" if success else "failure"
    PLUGIN_LOADS_TOTAL.labels(
        plugin_type=plugin_type,
        plugin_id=plugin_id,
        source=source,
        status=status,
    ).inc()
    PLUGIN_LOAD_DURATION.labels(plugin_type=plugin_type).observe(duration)


def record_health_check(
    plugin_id: str,
    result: str,
    duration: float,
) -> None:
    """Record a health check result.

    Args:
        plugin_id: Unique plugin identifier
        result: Health check result (healthy, unhealthy, timeout, error)
        duration: Time taken for health check in seconds
    """
    PLUGIN_HEALTH_CHECKS_TOTAL.labels(
        plugin_id=plugin_id,
        result=result,
    ).inc()
    PLUGIN_HEALTH_CHECK_DURATION.labels(plugin_id=plugin_id).observe(duration)


def record_dependency_warning(
    plugin_id: str,
    warning_type: str,
) -> None:
    """Record a dependency validation warning.

    Args:
        plugin_id: Plugin with the dependency issue
        warning_type: Type of warning (missing, version, disabled)
    """
    PLUGIN_DEPENDENCY_WARNINGS_TOTAL.labels(
        plugin_id=plugin_id,
        warning_type=warning_type,
    ).inc()


def update_plugin_gauges(
    loaded_by_type_source: dict[tuple[str, str], int],
    enabled_by_type: dict[str, int],
    healthy_by_type: dict[str, int],
) -> None:
    """Update all plugin state gauges.

    Args:
        loaded_by_type_source: Map of (plugin_type, source) -> count
        enabled_by_type: Map of plugin_type -> enabled count
        healthy_by_type: Map of plugin_type -> healthy count
    """
    for (plugin_type, source), count in loaded_by_type_source.items():
        PLUGINS_LOADED_GAUGE.labels(
            plugin_type=plugin_type,
            source=source,
        ).set(count)

    for plugin_type, count in enabled_by_type.items():
        PLUGINS_ENABLED_GAUGE.labels(plugin_type=plugin_type).set(count)

    for plugin_type, count in healthy_by_type.items():
        PLUGINS_HEALTHY_GAUGE.labels(plugin_type=plugin_type).set(count)


@contextmanager
def timed_operation() -> Generator[dict[str, float], None, None]:
    """Context manager for timing operations.

    Yields a dict that will contain 'duration' after the context exits.

    Example:
        with timed_operation() as timing:
            # do work
        print(timing['duration'])  # seconds elapsed
    """
    timing: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["duration"] = time.perf_counter() - start
