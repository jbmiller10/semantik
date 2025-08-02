#!/usr/bin/env python3
"""
Configuration constants for the chunking system.

This module centralizes all configuration values for the chunking error handling
and recovery framework, making them easily adjustable without code changes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingLimits:
    """Resource limits for chunking operations."""

    # Memory limits (in bytes)
    PREVIEW_MEMORY_LIMIT_BYTES: int = 512 * 1024 * 1024  # 512MB
    OPERATION_MEMORY_LIMIT_BYTES: int = 2 * 1024 * 1024 * 1024  # 2GB
    MIN_MEMORY_LIMIT_BYTES: int = 100 * 1024 * 1024  # 100MB

    # Document size limits (in bytes)
    MAX_DOCUMENT_SIZE_BYTES: int = 100 * 1024 * 1024  # 100MB
    MAX_PREVIEW_SIZE_BYTES: int = 1 * 1024 * 1024  # 1MB

    # Chunk limits
    MAX_CHUNKS_PER_DOCUMENT: int = 10000
    MAX_CHUNKS_PER_OPERATION: int = 100000
    MAX_CHUNK_SIZE_BYTES: int = 50000  # 50KB

    # Concurrent operation limits
    MAX_CONCURRENT_OPERATIONS: int = 10
    MAX_CONCURRENT_OPERATIONS_PER_USER: int = 3
    MAX_QUEUED_OPERATIONS: int = 50


@dataclass(frozen=True)
class ChunkingTimeouts:
    """Timeout configurations for chunking operations."""

    # Operation timeouts (in seconds)
    PREVIEW_TIMEOUT_SECONDS: float = 30.0
    OPERATION_SOFT_TIMEOUT_SECONDS: int = 3600  # 1 hour
    OPERATION_HARD_TIMEOUT_SECONDS: int = 7200  # 2 hours

    # Network timeouts
    EMBEDDING_SERVICE_TIMEOUT_SECONDS: float = 60.0
    QDRANT_TIMEOUT_SECONDS: float = 30.0

    # Retry delays
    MIN_RETRY_DELAY_SECONDS: int = 10
    MAX_RETRY_DELAY_SECONDS: int = 300  # 5 minutes
    RETRY_BACKOFF_FACTOR: float = 2.0


@dataclass(frozen=True)
class ChunkingCache:
    """Cache configuration for chunking operations."""

    # Cache TTLs (in seconds)
    PREVIEW_CACHE_TTL_SECONDS: int = 900  # 15 minutes
    OPERATION_STATE_TTL_SECONDS: int = 86400  # 24 hours
    ERROR_HISTORY_TTL_SECONDS: int = 604800  # 7 days

    # Cache limits
    MAX_CACHED_PREVIEWS: int = 1000
    MAX_ERROR_HISTORY_SIZE: int = 100


@dataclass(frozen=True)
class ChunkingRetry:
    """Retry configuration for error recovery."""

    # Retry counts
    DEFAULT_MAX_RETRIES: int = 3
    MEMORY_ERROR_MAX_RETRIES: int = 2
    TIMEOUT_ERROR_MAX_RETRIES: int = 3
    NETWORK_ERROR_MAX_RETRIES: int = 5

    # Backoff configuration
    RETRY_BACKOFF_ENABLED: bool = True
    RETRY_BACKOFF_MAX_SECONDS: int = 600  # 10 minutes
    RETRY_JITTER_ENABLED: bool = True


@dataclass(frozen=True)
class ChunkingBatches:
    """Batch processing configuration."""

    # Batch sizes
    DEFAULT_BATCH_SIZE: int = 32
    REDUCED_BATCH_SIZE: int = 16
    MIN_BATCH_SIZE: int = 4

    # Processing batches
    EMBEDDING_BATCH_SIZE: int = 100
    VECTOR_UPLOAD_BATCH_SIZE: int = 100
    DOCUMENT_REMOVAL_BATCH_SIZE: int = 100


@dataclass(frozen=True)
class ChunkingMetrics:
    """Metrics configuration for monitoring."""

    # Collection intervals
    METRICS_COLLECTION_INTERVAL_SECONDS: int = 60

    # Thresholds for alerts
    ERROR_RATE_THRESHOLD: float = 0.1  # 10% error rate
    MEMORY_USAGE_THRESHOLD: float = 0.8  # 80% of limit
    QUEUE_SIZE_THRESHOLD: int = 40  # 80% of max queue size


@dataclass(frozen=True)
class ChunkingCircuitBreaker:
    """Circuit breaker configuration."""

    # Circuit breaker settings
    FAILURE_THRESHOLD: int = 5
    SUCCESS_THRESHOLD: int = 2
    TIMEOUT_SECONDS: int = 60
    HALF_OPEN_REQUESTS: int = 3


# Singleton instances
CHUNKING_LIMITS = ChunkingLimits()
CHUNKING_TIMEOUTS = ChunkingTimeouts()
CHUNKING_CACHE = ChunkingCache()
CHUNKING_RETRY = ChunkingRetry()
CHUNKING_BATCHES = ChunkingBatches()
CHUNKING_METRICS = ChunkingMetrics()
CHUNKING_CIRCUIT_BREAKER = ChunkingCircuitBreaker()


# Validation thresholds
REINDEX_VECTOR_COUNT_VARIANCE = 0.1  # 10% variance allowed
REINDEX_SEARCH_MISMATCH_THRESHOLD = 0.3  # 30% mismatch threshold
REINDEX_SCORE_DIFF_THRESHOLD = 0.1  # 0.1 score difference threshold


# Resource tracking
CLEANUP_DELAY_SECONDS = 300  # 5 minutes default delay
CLEANUP_DELAY_MIN_SECONDS = 300  # 5 minutes minimum
CLEANUP_DELAY_MAX_SECONDS = 1800  # 30 minutes maximum
CLEANUP_DELAY_PER_10K_VECTORS = 60  # Additional 1 minute per 10k vectors


def get_memory_limit_for_strategy(strategy: str) -> int:
    """Get memory limit based on chunking strategy.

    Args:
        strategy: Chunking strategy name

    Returns:
        Memory limit in bytes
    """
    # Semantic and hierarchical strategies need more memory
    if strategy in ["semantic", "hierarchical"]:
        return CHUNKING_LIMITS.OPERATION_MEMORY_LIMIT_BYTES
    return CHUNKING_LIMITS.OPERATION_MEMORY_LIMIT_BYTES // 2


def get_timeout_for_strategy(strategy: str) -> float:
    """Get timeout based on chunking strategy.

    Args:
        strategy: Chunking strategy name

    Returns:
        Timeout in seconds
    """
    # Semantic strategy is slower due to embeddings
    if strategy == "semantic":
        return CHUNKING_TIMEOUTS.OPERATION_SOFT_TIMEOUT_SECONDS * 1.5
    return CHUNKING_TIMEOUTS.OPERATION_SOFT_TIMEOUT_SECONDS
