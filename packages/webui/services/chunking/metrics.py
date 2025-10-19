"""
Chunking metrics service.

Handles metrics collection, tracking, and reporting for chunking operations.
"""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

from prometheus_client import CollectorRegistry, Counter, Histogram, Summary

from packages.shared.metrics.prometheus import registry

MetricT = TypeVar("MetricT")

logger = logging.getLogger(__name__)


def _get_or_create_metric(
    metric_cls: type[MetricT],
    name: str,
    documentation: str,
    *,
    registry: CollectorRegistry,
    labelnames: tuple[str, ...] | list[str] | None = None,
    **kwargs,
    **kwargs: Any,
) -> MetricT:
    """Return existing metric from registry or create a new one."""

    labels = tuple(labelnames or ())
    existing: MetricT | None = None
    if hasattr(registry, "_names_to_collectors"):
        existing = cast(
            MetricT | None,
            getattr(registry, "_names_to_collectors", {}).get(name),  # type: ignore[attr-defined]
        )

    if existing is not None:
        return existing

    return metric_cls(name, documentation, labels, registry=registry, **kwargs)


class ChunkingMetrics:
    """Service responsible for chunking metrics collection."""

    def __init__(self, *, registry_override: CollectorRegistry | None = None) -> None:
        """Initialize metrics collectors."""
        self._registry = registry_override or registry
        # Operation metrics
        self.operation_counter = _get_or_create_metric(
            Counter,
            "chunking_operations_total",
            "Total number of chunking operations",
            registry=self._registry,
            labelnames=("strategy", "status"),
        )

        self.operation_duration = _get_or_create_metric(
            Histogram,
            "chunking_operation_duration_seconds",
            "Duration of chunking operations",
            registry=self._registry,
            labelnames=("strategy",),
        )

        self.chunk_count = _get_or_create_metric(
            Summary,
            "chunking_chunks_produced",
            "Number of chunks produced per operation",
            registry=self._registry,
            labelnames=("strategy",),
        )

        self.chunk_sizes = _get_or_create_metric(
            Summary,
            "chunking_chunk_sizes_bytes",
            "Size of chunks produced",
            registry=self._registry,
            labelnames=("strategy",),
        )

        self.fallback_counter = _get_or_create_metric(
            Counter,
            "chunking_fallbacks_total",
            "Number of times fallback strategy was used",
            registry=self._registry,
            labelnames=("original_strategy",),
        )

        # Cache metrics
        self.cache_hits = _get_or_create_metric(
            Counter,
            "chunking_cache_hits_total",
            "Number of cache hits",
            registry=self._registry,
            labelnames=("operation_type",),
        )

        self.cache_misses = _get_or_create_metric(
            Counter,
            "chunking_cache_misses_total",
            "Number of cache misses",
            registry=self._registry,
            labelnames=("operation_type",),
        )

        # Error metrics
        self.error_counter = _get_or_create_metric(
            Counter,
            "chunking_errors_total",
            "Total number of chunking errors",
            registry=self._registry,
            labelnames=("strategy", "error_type"),
        )

        # In-memory statistics for quick access
        self.statistics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_chunks_produced": 0,
            "strategy_usage": {},
            "average_duration": 0,
            "last_operation": None,
        }

    @asynccontextmanager
    async def measure_operation(
        self,
        strategy: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Context manager to measure operation duration and track metrics.

        Args:
            strategy: Strategy being used

        Yields:
            Metrics context dictionary
        """
        start_time = time.time()
        context = {
            "strategy": strategy,
            "start_time": start_time,
            "success": False,
            "chunks_produced": 0,
            "error": None,
        }

        try:
            yield context
            context["success"] = True
            self.operation_counter.labels(strategy=strategy, status="success").inc()
        except Exception as e:
            context["success"] = False
            context["error"] = str(e)
            self.operation_counter.labels(strategy=strategy, status="failure").inc()
            self.error_counter.labels(
                strategy=strategy,
                error_type=type(e).__name__,
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.operation_duration.labels(strategy=strategy).observe(duration)

            # Update statistics
            self._update_statistics(strategy, context, duration)

    def record_chunks_produced(
        self,
        strategy: str,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Record metrics about produced chunks.

        Args:
            strategy: Strategy used
            chunks: List of produced chunks
        """
        chunk_count = len(chunks)
        self.chunk_count.labels(strategy=strategy).observe(chunk_count)

        for chunk in chunks:
            content = chunk.get("content", "")
            size = len(content.encode("utf-8"))
            self.chunk_sizes.labels(strategy=strategy).observe(size)

        # Update total chunks in statistics
        total_chunks = self.statistics.get("total_chunks_produced", 0)
        if isinstance(total_chunks, int | float):
            self.statistics["total_chunks_produced"] = total_chunks + chunk_count

    def record_fallback(self, original_strategy: str) -> None:
        """
        Record when fallback strategy is used.

        Args:
            original_strategy: Strategy that failed
        """
        self.fallback_counter.labels(original_strategy=original_strategy).inc()

    def record_cache_hit(self, operation_type: str = "preview") -> None:
        """
        Record a cache hit.

        Args:
            operation_type: Type of operation (preview, etc.)
        """
        self.cache_hits.labels(operation_type=operation_type).inc()

    def record_cache_miss(self, operation_type: str = "preview") -> None:
        """
        Record a cache miss.

        Args:
            operation_type: Type of operation (preview, etc.)
        """
        self.cache_misses.labels(operation_type=operation_type).inc()

    def record_error(self, strategy: str, error: Exception) -> None:
        """
        Record an error occurrence.

        Args:
            strategy: Strategy that encountered error
            error: Exception that occurred
        """
        self.error_counter.labels(
            strategy=strategy,
            error_type=type(error).__name__,
        ).inc()

    def get_statistics(self) -> dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary of current statistics
        """
        stats = self.statistics.copy()

        # Calculate success rate
        total = stats.get("total_operations", 0)
        successful = stats.get("successful_operations", 0)
        if isinstance(total, int | float) and isinstance(successful, int | float) and total > 0:
            stats["success_rate"] = (successful / total) * 100
        else:
            stats["success_rate"] = 0

        return stats

    def get_strategy_metrics(self, strategy: str) -> dict[str, Any]:
        """
        Get metrics for a specific strategy.

        Args:
            strategy: Strategy name

        Returns:
            Dictionary of strategy-specific metrics
        """
        strategy_usage = self.statistics.get("strategy_usage", {})
        usage = strategy_usage.get(strategy, {}) if isinstance(strategy_usage, dict) else {}

        return {
            "strategy": strategy,
            "total_operations": usage.get("count", 0),
            "successful_operations": usage.get("success", 0),
            "failed_operations": usage.get("failure", 0),
            "average_duration": usage.get("avg_duration", 0),
            "total_chunks": usage.get("total_chunks", 0),
            "average_chunks": (
                usage.get("total_chunks", 0) / usage.get("count", 1) if usage.get("count", 0) > 0 else 0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset in-memory statistics."""
        self.statistics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_chunks_produced": 0,
            "strategy_usage": {},
            "average_duration": 0,
            "last_operation": None,
        }
        logger.info("Statistics reset")

    def _update_statistics(
        self,
        strategy: str,
        context: dict[str, Any],
        duration: float,
    ) -> None:
        """Update internal statistics."""
        # Update total operations
        total_ops = self.statistics.get("total_operations", 0)
        if isinstance(total_ops, int):
            self.statistics["total_operations"] = total_ops + 1

        if context["success"]:
            success_ops = self.statistics.get("successful_operations", 0)
            if isinstance(success_ops, int):
                self.statistics["successful_operations"] = success_ops + 1
        else:
            failed_ops = self.statistics.get("failed_operations", 0)
            if isinstance(failed_ops, int):
                self.statistics["failed_operations"] = failed_ops + 1

        # Update strategy-specific stats
        strategy_usage = self.statistics.get("strategy_usage", {})
        if not isinstance(strategy_usage, dict):
            strategy_usage = {}
            self.statistics["strategy_usage"] = strategy_usage

        if strategy not in strategy_usage:
            strategy_usage[strategy] = {
                "count": 0,
                "success": 0,
                "failure": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "total_chunks": 0,
            }

        strategy_stats = strategy_usage[strategy]
        strategy_stats["count"] += 1

        if context["success"]:
            strategy_stats["success"] += 1
        else:
            strategy_stats["failure"] += 1

        strategy_stats["total_duration"] += duration
        strategy_stats["avg_duration"] = strategy_stats["total_duration"] / strategy_stats["count"]
        strategy_stats["total_chunks"] += context.get("chunks_produced", 0)

        # Update overall average duration
        total_ops = self.statistics.get("total_operations", 0)
        if isinstance(total_ops, int) and total_ops > 0:
            current_avg = self.statistics.get("average_duration", 0)
            if isinstance(current_avg, int | float):
                self.statistics["average_duration"] = (current_avg * (total_ops - 1) + duration) / total_ops

        self.statistics["last_operation"] = {
            "strategy": strategy,
            "success": context["success"],
            "duration": duration,
            "timestamp": datetime.now(UTC).isoformat(),
        }
