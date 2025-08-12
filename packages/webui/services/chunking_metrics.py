"""
Prometheus metrics for chunking operations.

This module provides observability into chunking performance,
strategy usage, and fallback scenarios.
"""

from typing import Any, Dict, List, Union

from prometheus_client import Counter, Histogram, Summary

from packages.shared.metrics.prometheus import registry

# Chunking duration histogram - tracks time taken to chunk documents per strategy
ingestion_chunking_duration_seconds = Histogram(
    "ingestion_chunking_duration_seconds",
    "Duration of chunking operation per document in seconds",
    labelnames=["strategy"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    registry=registry,
)

# Chunking fallback counter - tracks when strategies fail and fallback is used
ingestion_chunking_fallback_total = Counter(
    "ingestion_chunking_fallback_total",
    "Total number of chunking fallbacks by strategy and reason",
    labelnames=["strategy", "reason"],
    registry=registry,
)

# Chunks produced counter - tracks total chunks created per strategy
ingestion_chunks_total = Counter(
    "ingestion_chunks_total",
    "Total number of chunks produced per strategy",
    labelnames=["strategy"],
    registry=registry,
)

# Average chunk size summary - tracks chunk size distribution per strategy
ingestion_avg_chunk_size_bytes = Summary(
    "ingestion_avg_chunk_size_bytes",
    "Average chunk size in bytes per strategy",
    labelnames=["strategy"],
    registry=registry,
)


# Helper functions for recording metrics
def record_chunking_duration(strategy: str, duration_seconds: float) -> None:
    """Record the duration of a chunking operation.

    Args:
        strategy: The chunking strategy used
        duration_seconds: Duration in seconds
    """
    ingestion_chunking_duration_seconds.labels(strategy=strategy).observe(duration_seconds)


def record_chunking_fallback(original_strategy: str, reason: str) -> None:
    """Record a chunking fallback event.

    Args:
        original_strategy: The strategy that failed
        reason: Reason for the fallback (e.g., 'invalid_config', 'runtime_error')
    """
    ingestion_chunking_fallback_total.labels(strategy=original_strategy, reason=reason).inc()


def record_chunks_produced(strategy: str, chunk_count: int) -> None:
    """Record the number of chunks produced.

    Args:
        strategy: The chunking strategy used
        chunk_count: Number of chunks produced
    """
    ingestion_chunks_total.labels(strategy=strategy).inc(chunk_count)


def record_chunk_sizes(strategy: str, chunks: List[Union[str, Dict[str, Any], Any]]) -> None:
    """Record chunk size statistics.

    Args:
        strategy: The chunking strategy used
        chunks: List of chunks (each chunk should have 'text' or be a string)
    """
    if not chunks:
        return

    # Calculate average chunk size
    total_size = 0
    for chunk in chunks:
        if isinstance(chunk, dict):
            # Handle dictionary chunks with 'text' field
            text = chunk.get('text', chunk.get('content', ''))
            total_size += len(text.encode('utf-8'))
        elif isinstance(chunk, str):
            # Handle string chunks
            total_size += len(chunk.encode('utf-8'))
        elif hasattr(chunk, 'content'):
            # Handle chunk objects with content attribute
            total_size += len(chunk.content.encode('utf-8'))

    if chunks:
        avg_size = total_size / len(chunks)
        ingestion_avg_chunk_size_bytes.labels(strategy=strategy).observe(avg_size)


# Export all metrics and helper functions
__all__ = [
    "ingestion_chunking_duration_seconds",
    "ingestion_chunking_fallback_total",
    "ingestion_chunks_total",
    "ingestion_avg_chunk_size_bytes",
    "record_chunking_duration",
    "record_chunking_fallback",
    "record_chunks_produced",
    "record_chunk_sizes",
]
