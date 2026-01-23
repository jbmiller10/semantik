"""Prometheus metrics setup for the vecpipe search service."""

from typing import Any

from prometheus_client import Counter, Gauge, Histogram

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

# Sparse search metrics
sparse_encode_query_latency = get_or_create_metric(
    Histogram,
    "semantik_sparse_encode_query_seconds",
    "Sparse query encoding latency",
    ["sparse_type"],  # bm25 or splade
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
sparse_search_latency = get_or_create_metric(
    Histogram,
    "semantik_sparse_search_seconds",
    "Sparse Qdrant search latency",
    ["sparse_type"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
rrf_fusion_latency = get_or_create_metric(
    Histogram,
    "semantik_rrf_fusion_seconds",
    "RRF fusion computation latency",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05),
)
sparse_index_chunks = get_or_create_metric(
    Counter,
    "semantik_sparse_index_chunks_total",
    "Total chunks indexed in sparse collections",
    ["collection", "sparse_type"],
)
sparse_search_requests = get_or_create_metric(
    Counter,
    "semantik_sparse_search_requests_total",
    "Total sparse search requests",
    ["search_mode", "sparse_type"],  # search_mode: sparse or hybrid
)
sparse_search_fallbacks = get_or_create_metric(
    Counter,
    "semantik_sparse_search_fallbacks_total",
    "Total sparse search fallbacks to dense",
    ["reason"],  # sparse_not_enabled, plugin_not_found, etc.
)

# Dense search metrics
dense_search_fallbacks = get_or_create_metric(
    Counter,
    "semantik_dense_search_fallbacks_total",
    "Total dense search fallbacks",
    ["reason"],  # sdk_error, etc.
)

# GPU and Infrastructure Metrics
gpu_free_probe_latency = get_or_create_metric(
    Histogram,
    "semantik_gpu_free_probe_seconds",
    "Time to probe GPU free memory",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1),
)

qdrant_ad_hoc_client_total = get_or_create_metric(
    Counter,
    "semantik_qdrant_ad_hoc_client_total",
    "Count of ad-hoc Qdrant client creations",
    ["location"],
)

payload_fetch_latency = get_or_create_metric(
    Histogram,
    "semantik_payload_fetch_seconds",
    "Time to fetch dense payloads for sparse/hybrid results",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

rerank_content_fetch_latency = get_or_create_metric(
    Histogram,
    "semantik_rerank_content_fetch_seconds",
    "Time to scroll/fetch content for reranking",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

rerank_fallbacks = get_or_create_metric(
    Counter,
    "semantik_rerank_fallbacks_total",
    "Total rerank fallbacks",
    ["reason"],  # error, etc.
)

collection_metadata_fetch_latency = get_or_create_metric(
    Histogram,
    "semantik_collection_metadata_fetch_seconds",
    "Time to fetch collection metadata on cache miss",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)

# === Memory Governor Metrics ===

eviction_latency_seconds = get_or_create_metric(
    Histogram,
    "semantik_eviction_seconds",
    "Time to evict a model from GPU",
    ["action"],  # "offload" or "unload"
    buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)

evictions_total = get_or_create_metric(
    Counter,
    "semantik_evictions_total",
    "Total model evictions",
    ["model_type", "action"],
)

restore_latency_seconds = get_or_create_metric(
    Histogram,
    "semantik_restore_from_cpu_seconds",
    "Time to restore model from CPU to GPU",
    buckets=(0.5, 1, 2, 5, 10, 15, 30),
)

memory_request_latency_seconds = get_or_create_metric(
    Histogram,
    "semantik_memory_request_seconds",
    "Total time for request_memory including any eviction",
    ["outcome"],  # "success", "failed", "already_loaded"
    buckets=(0.001, 0.01, 0.1, 0.5, 1, 5, 10),
)

pressure_events_total = get_or_create_metric(
    Counter,
    "semantik_pressure_events_total",
    "Memory pressure events handled",
    ["level"],
)

models_evicted_per_event = get_or_create_metric(
    Histogram,
    "semantik_models_evicted_per_event",
    "Number of models evicted per pressure event",
    ["level"],
    buckets=(0, 1, 2, 3, 5, 10),
)

governor_degraded = get_or_create_metric(
    Gauge,
    "semantik_governor_degraded",
    "Memory governor degraded state (1=degraded, 0=healthy)",
)


__all__ = [
    "get_or_create_metric",
    "search_latency",
    "search_requests",
    "search_errors",
    "embedding_generation_latency",
    "sparse_encode_query_latency",
    "sparse_search_latency",
    "rrf_fusion_latency",
    "sparse_index_chunks",
    "sparse_search_requests",
    "sparse_search_fallbacks",
    "dense_search_fallbacks",
    "gpu_free_probe_latency",
    "qdrant_ad_hoc_client_total",
    "payload_fetch_latency",
    "rerank_content_fetch_latency",
    "rerank_fallbacks",
    "collection_metadata_fetch_latency",
    # Memory Governor Metrics
    "eviction_latency_seconds",
    "evictions_total",
    "restore_latency_seconds",
    "memory_request_latency_seconds",
    "pressure_events_total",
    "models_evicted_per_event",
    "governor_degraded",
]
