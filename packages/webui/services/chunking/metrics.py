"""
Chunking metrics service.

Handles metrics collection, tracking, and reporting for chunking operations.
"""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from prometheus_client import Counter, Histogram, Summary

from packages.shared.metrics.prometheus import registry

logger = logging.getLogger(__name__)


class ChunkingMetrics:
    """Service responsible for chunking metrics collection."""

    def __init__(self):
        """Initialize metrics collectors."""
        # Operation metrics
        self.operation_counter = Counter(
            "chunking_operations_total",
            "Total number of chunking operations",
            ["strategy", "status"],
            registry=registry,
        )

        self.operation_duration = Histogram(
            "chunking_operation_duration_seconds",
            "Duration of chunking operations",
            ["strategy"],
            registry=registry,
        )

        self.chunk_count = Summary(
            "chunking_chunks_produced",
            "Number of chunks produced per operation",
            ["strategy"],
            registry=registry,
        )

        self.chunk_sizes = Summary(
            "chunking_chunk_sizes_bytes",
            "Size of chunks produced",
            ["strategy"],
            registry=registry,
        )

        self.fallback_counter = Counter(
            "chunking_fallbacks_total",
            "Number of times fallback strategy was used",
            ["original_strategy"],
            registry=registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "chunking_cache_hits_total",
            "Number of cache hits",
            ["operation_type"],
            registry=registry,
        )

        self.cache_misses = Counter(
            "chunking_cache_misses_total",
            "Number of cache misses",
            ["operation_type"],
            registry=registry,
        )

        # Error metrics
        self.error_counter = Counter(
            "chunking_errors_total",
            "Total number of chunking errors",
            ["strategy", "error_type"],
            registry=registry,
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
        self.statistics["total_chunks_produced"] += chunk_count

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
        total = stats["total_operations"]
        if total > 0:
            stats["success_rate"] = (stats["successful_operations"] / total) * 100
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
        usage = self.statistics["strategy_usage"].get(strategy, {})

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
        self.statistics["total_operations"] += 1

        if context["success"]:
            self.statistics["successful_operations"] += 1
        else:
            self.statistics["failed_operations"] += 1

        # Update strategy-specific stats
        if strategy not in self.statistics["strategy_usage"]:
            self.statistics["strategy_usage"][strategy] = {
                "count": 0,
                "success": 0,
                "failure": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "total_chunks": 0,
            }

        strategy_stats = self.statistics["strategy_usage"][strategy]
        strategy_stats["count"] += 1

        if context["success"]:
            strategy_stats["success"] += 1
        else:
            strategy_stats["failure"] += 1

        strategy_stats["total_duration"] += duration
        strategy_stats["avg_duration"] = strategy_stats["total_duration"] / strategy_stats["count"]
        strategy_stats["total_chunks"] += context.get("chunks_produced", 0)

        # Update overall average duration
        total_ops = self.statistics["total_operations"]
        if total_ops > 0:
            current_avg = self.statistics["average_duration"]
            self.statistics["average_duration"] = (current_avg * (total_ops - 1) + duration) / total_ops

        self.statistics["last_operation"] = {
            "strategy": strategy,
            "success": context["success"],
            "duration": duration,
            "timestamp": datetime.now(UTC).isoformat(),
        }
