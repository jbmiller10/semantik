"""Performance monitoring and metrics for chunking strategies.

This module provides utilities for tracking and logging performance metrics
across all chunking strategies.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChunkingMetrics:
    """Container for chunking performance metrics."""

    strategy: str
    doc_id: str
    input_chars: int
    output_chunks: int
    duration_seconds: float
    chunks_per_second: float
    chars_per_chunk: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ChunkingPerformanceMonitor:
    """Monitor and log performance metrics for chunking operations."""

    def __init__(self) -> None:
        """Initialize the performance monitor."""
        self.metrics_history: list[ChunkingMetrics] = []

    @contextmanager
    def measure_chunking(
        self,
        strategy: str,
        doc_id: str,
        text_length: int,
        metadata: dict[str, Any] | None = None,
    ):
        """Context manager to measure chunking performance.

        Args:
            strategy: Name of the chunking strategy
            doc_id: Document identifier
            text_length: Length of input text in characters
            metadata: Additional metadata to track

        Yields:
            metrics: ChunkingMetrics object to be updated

        Example:
            with monitor.measure_chunking("semantic", "doc123", len(text)) as metrics:
                chunks = chunker.chunk_text(text, doc_id)
                metrics.output_chunks = len(chunks)
        """
        start_time = time.time()
        metrics = ChunkingMetrics(
            strategy=strategy,
            doc_id=doc_id,
            input_chars=text_length,
            output_chunks=0,
            duration_seconds=0.0,
            chunks_per_second=0.0,
            chars_per_chunk=0.0,
            metadata=metadata or {},
        )

        try:
            yield metrics
        except Exception as e:
            metrics.error = str(e)
            logger.error(f"Error during {strategy} chunking: {e}")
            raise
        finally:
            # Calculate final metrics
            metrics.duration_seconds = time.time() - start_time

            if metrics.output_chunks > 0:
                metrics.chunks_per_second = metrics.output_chunks / max(metrics.duration_seconds, 0.001)
                metrics.chars_per_chunk = metrics.input_chars / metrics.output_chunks

            # Log performance
            self._log_metrics(metrics)

            # Store in history
            self.metrics_history.append(metrics)

    def _log_metrics(self, metrics: ChunkingMetrics) -> None:
        """Log performance metrics.

        Args:
            metrics: Metrics to log
        """
        if metrics.error:
            logger.error(
                f"Chunking failed - Strategy: {metrics.strategy}, " f"Doc: {metrics.doc_id}, Error: {metrics.error}"
            )
        else:
            # Log at INFO level for normal performance, WARN if unusually slow
            log_level = logging.INFO
            if metrics.chunks_per_second < 50:  # Below 50 chunks/sec is considered slow
                log_level = logging.WARNING

            logger.log(
                log_level,
                f"Chunking performance - Strategy: {metrics.strategy}, "
                f"Chunks: {metrics.output_chunks}, "
                f"Duration: {metrics.duration_seconds:.3f}s, "
                f"Speed: {metrics.chunks_per_second:.1f} chunks/s, "
                f"Avg size: {metrics.chars_per_chunk:.0f} chars/chunk",
            )

            # Log additional metadata if present
            if metrics.metadata:
                logger.debug(f"Chunking metadata: {metrics.metadata}")

    def get_strategy_summary(self, strategy: str) -> dict[str, Any]:
        """Get performance summary for a specific strategy.

        Args:
            strategy: Strategy name to summarize

        Returns:
            Summary statistics for the strategy
        """
        strategy_metrics = [m for m in self.metrics_history if m.strategy == strategy and not m.error]

        if not strategy_metrics:
            return {"strategy": strategy, "no_data": True}

        total_chunks = sum(m.output_chunks for m in strategy_metrics)
        total_time = sum(m.duration_seconds for m in strategy_metrics)
        avg_speed = total_chunks / max(total_time, 0.001)

        return {
            "strategy": strategy,
            "total_documents": len(strategy_metrics),
            "total_chunks": total_chunks,
            "total_time": total_time,
            "avg_chunks_per_second": avg_speed,
            "avg_chars_per_chunk": sum(m.chars_per_chunk for m in strategy_metrics) / len(strategy_metrics),
            "min_speed": min(m.chunks_per_second for m in strategy_metrics),
            "max_speed": max(m.chunks_per_second for m in strategy_metrics),
        }

    def log_summary(self) -> None:
        """Log a summary of all chunking performance."""
        strategies = {m.strategy for m in self.metrics_history}

        logger.info("=== Chunking Performance Summary ===")
        for strategy in sorted(strategies):
            summary = self.get_strategy_summary(strategy)
            if not summary.get("no_data"):
                logger.info(
                    f"{strategy}: {summary['total_documents']} docs, "
                    f"{summary['total_chunks']} chunks, "
                    f"avg speed: {summary['avg_chunks_per_second']:.1f} chunks/s"
                )


# Global performance monitor instance
performance_monitor = ChunkingPerformanceMonitor()


def log_chunking_performance(
    strategy: str,
    doc_id: str,
    input_chars: int,
    output_chunks: int,
    duration_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience function to log chunking performance.

    Args:
        strategy: Chunking strategy name
        doc_id: Document identifier
        input_chars: Number of input characters
        output_chunks: Number of output chunks
        duration_seconds: Time taken in seconds
        metadata: Additional metadata
    """
    metrics = ChunkingMetrics(
        strategy=strategy,
        doc_id=doc_id,
        input_chars=input_chars,
        output_chunks=output_chunks,
        duration_seconds=duration_seconds,
        chunks_per_second=output_chunks / max(duration_seconds, 0.001),
        chars_per_chunk=input_chars / max(output_chunks, 1),
        metadata=metadata or {},
    )

    performance_monitor._log_metrics(metrics)
    performance_monitor.metrics_history.append(metrics)
