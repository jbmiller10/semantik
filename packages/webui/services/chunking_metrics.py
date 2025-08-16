"""
Prometheus metrics for chunking operations.

This module provides observability into chunking performance,
strategy usage, and fallback scenarios.
"""

from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, Summary

from packages.shared.metrics.prometheus import registry


def _get_or_create_metric(
    metric_class: type[Any], name: str, description: str, registry: CollectorRegistry, **kwargs: Any
) -> Any:
    """Get existing metric or create a new one if it doesn't exist."""
    # Check if metric already exists in registry by checking the names
    if hasattr(registry, "_collector_to_names"):
        for collector, names in registry._collector_to_names.items():
            # For Counter metrics, check for the _total suffix
            if (
                metric_class == Counter
                and f"{name}_total" in names
                or metric_class == Histogram
                and (f"{name}_bucket" in names or f"{name}_count" in names or f"{name}_sum" in names or name in names)
                or (
                    metric_class == Summary
                    and (f"{name}_count" in names or f"{name}_sum" in names or name in names)
                    or name in names
                )
            ):
                return collector

    # Create new metric
    try:
        return metric_class(name, description, registry=registry, **kwargs)
    except ValueError as e:
        # If we get a duplicate error, try to find and return the existing metric
        if "Duplicated timeseries" in str(e) or "Duplicated" in str(e):
            # Try to find the existing collector by iterating through all collectors
            if hasattr(registry, "_collector_to_names"):
                for collector in list(registry._collector_to_names.keys()):
                    if hasattr(collector, "_name") and collector._name == name:
                        return collector
            # If we still can't find it, return a mock that won't cause issues
            # This is a fallback for test environments
            import logging

            logging.warning(f"Could not find or create metric {name}, returning mock")

            # Create a simple mock that has the basic interface
            class MockMetric:
                def __init__(self) -> None:
                    self._name = name

                def labels(self, **_kwargs: Any) -> "MockMetric":
                    return self

                def observe(self, value: float) -> None:
                    pass

                def inc(self, amount: int = 1) -> None:
                    pass

                def set(self, value: float) -> None:
                    pass

            return MockMetric()
        raise


# Chunking duration histogram - tracks time taken to chunk documents per strategy
ingestion_chunking_duration_seconds = _get_or_create_metric(
    Histogram,
    "ingestion_chunking_duration_seconds",
    "Duration of chunking operation per document in seconds",
    registry,
    labelnames=["strategy"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

# Chunking fallback counter - tracks when strategies fail and fallback is used
ingestion_chunking_fallback_total = _get_or_create_metric(
    Counter,
    "ingestion_chunking_fallback_total",
    "Total number of chunking fallbacks by strategy and reason",
    registry,
    labelnames=["strategy", "reason"],
)

# Chunks produced counter - tracks total chunks created per strategy
ingestion_chunks_total = _get_or_create_metric(
    Counter,
    "ingestion_chunks_total",
    "Total number of chunks produced per strategy",
    registry,
    labelnames=["strategy"],
)

# Average chunk size summary - tracks chunk size distribution per strategy
ingestion_avg_chunk_size_bytes = _get_or_create_metric(
    Summary,
    "ingestion_avg_chunk_size_bytes",
    "Average chunk size in bytes per strategy",
    registry,
    labelnames=["strategy"],
)

# Segmentation metrics for Phase 3
ingestion_segmented_documents_total = _get_or_create_metric(
    Counter,
    "ingestion_segmented_documents_total",
    "Total number of documents that required segmentation",
    registry,
    labelnames=["strategy"],
)

ingestion_segments_total = _get_or_create_metric(
    Counter,
    "ingestion_segments_total",
    "Total number of segments created from large documents",
    registry,
    labelnames=["strategy"],
)

ingestion_segment_size_bytes = _get_or_create_metric(
    Histogram,
    "ingestion_segment_size_bytes",
    "Size distribution of document segments in bytes",
    registry,
    labelnames=["strategy"],
    buckets=(100000, 500000, 1000000, 2000000, 5000000, 10000000),  # 100KB to 10MB
)

ingestion_streaming_used_total = _get_or_create_metric(
    Counter,
    "ingestion_streaming_used_total",
    "Total number of documents processed with streaming strategies",
    registry,
    labelnames=["strategy"],
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


def record_chunk_sizes(strategy: str, chunks: list[str | dict[str, Any] | Any]) -> None:
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
            text = chunk.get("text", chunk.get("content", ""))
            if text is not None:
                total_size += len(text.encode("utf-8"))
        elif isinstance(chunk, str):
            # Handle string chunks
            total_size += len(chunk.encode("utf-8"))
        elif hasattr(chunk, "content"):
            # Handle chunk objects with content attribute
            total_size += len(chunk.content.encode("utf-8"))

    if chunks:
        avg_size = total_size / len(chunks)
        ingestion_avg_chunk_size_bytes.labels(strategy=strategy).observe(avg_size)


def record_document_segmented(strategy: str) -> None:
    """Record that a document required segmentation.

    Args:
        strategy: The chunking strategy used
    """
    ingestion_segmented_documents_total.labels(strategy=strategy).inc()


def record_segments_created(strategy: str, segment_count: int) -> None:
    """Record the number of segments created for a document.

    Args:
        strategy: The chunking strategy used
        segment_count: Number of segments created
    """
    ingestion_segments_total.labels(strategy=strategy).inc(segment_count)


def record_segment_size(strategy: str, segment_size: int) -> None:
    """Record the size of a document segment.

    Args:
        strategy: The chunking strategy used
        segment_size: Size of the segment in bytes
    """
    ingestion_segment_size_bytes.labels(strategy=strategy).observe(segment_size)


def record_streaming_used(strategy: str) -> None:
    """Record that a document was processed using streaming.

    Args:
        strategy: The chunking strategy used
    """
    ingestion_streaming_used_total.labels(strategy=strategy).inc()


# Export all metrics and helper functions
__all__ = [
    "ingestion_chunking_duration_seconds",
    "ingestion_chunking_fallback_total",
    "ingestion_chunks_total",
    "ingestion_avg_chunk_size_bytes",
    "ingestion_segmented_documents_total",
    "ingestion_segments_total",
    "ingestion_segment_size_bytes",
    "ingestion_streaming_used_total",
    "record_chunking_duration",
    "record_chunking_fallback",
    "record_chunks_produced",
    "record_chunk_sizes",
    "record_document_segmented",
    "record_segments_created",
    "record_segment_size",
    "record_streaming_used",
]
