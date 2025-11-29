"""Prometheus metrics setup for the vecpipe search service."""

from typing import Any

from prometheus_client import Counter, Histogram

from shared.metrics.prometheus import registry


def get_or_create_metric(
    metric_class: type,
    name: str,
    description: str,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a metric or return an existing one if already registered."""
    try:
        # Try to get existing collector from registry
        for collector in registry._collector_to_names:
            if hasattr(collector, "_name") and collector._name == name:  # pragma: no cover - registry internals
                return collector
    except AttributeError:
        pass

    if labels:
        return metric_class(name, description, labels, registry=registry, **kwargs)
    return metric_class(name, description, registry=registry, **kwargs)


search_latency = get_or_create_metric(
    Histogram,
    "search_api_latency_seconds",
    "Search API request latency",
    ["endpoint", "search_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
search_requests = get_or_create_metric(
    Counter, "search_api_requests_total", "Total search API requests", ["endpoint", "search_type"]
)
search_errors = get_or_create_metric(
    Counter, "search_api_errors_total", "Total search API errors", ["endpoint", "error_type"]
)
embedding_generation_latency = get_or_create_metric(
    Histogram,
    "search_api_embedding_latency_seconds",
    "Embedding generation latency for search queries",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
)


__all__ = [
    "get_or_create_metric",
    "search_latency",
    "search_requests",
    "search_errors",
    "embedding_generation_latency",
]
