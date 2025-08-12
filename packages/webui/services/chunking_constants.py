"""
Constants and configuration for chunking operations.

This module centralizes all magic numbers and configuration values
to make them easily maintainable and configurable.
"""

from datetime import timedelta

# Size limits (in bytes)
MAX_PREVIEW_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB
MAX_DOCUMENT_SIZE = 100 * 1024 * 1024  # 100MB
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 4096
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
MAX_CHUNK_OVERLAP = 500

# Progressive segmentation configuration
SEGMENT_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5MB - Documents larger than this will be segmented
DEFAULT_SEGMENT_SIZE = 1 * 1024 * 1024  # 1MB - Size of each segment
DEFAULT_SEGMENT_OVERLAP = 10 * 1024  # 10KB - Overlap between segments to preserve context
MAX_SEGMENTS_PER_DOCUMENT = 100  # Maximum number of segments to prevent runaway processing

# Per-strategy segmentation thresholds (optional overrides)
STRATEGY_SEGMENT_THRESHOLDS = {
    "semantic": 2 * 1024 * 1024,  # 2MB - Semantic needs smaller segments for embedding
    "markdown": 10 * 1024 * 1024,  # 10MB - Markdown can handle larger segments
    "recursive": 8 * 1024 * 1024,  # 8MB - Recursive can handle moderately large segments
    "hierarchical": 5 * 1024 * 1024,  # 5MB - Hierarchical needs moderate segments
    "hybrid": 3 * 1024 * 1024,  # 3MB - Hybrid needs smaller segments for strategy switching
}

# Streaming configuration
STREAMING_ENABLED = True  # Enable streaming strategies for supported types
STREAMING_STRATEGIES = ["markdown", "recursive"]  # Strategies with streaming support
STREAMING_BUFFER_SIZE = 64 * 1024  # 64KB - Read buffer for streaming
STREAMING_WINDOW_SIZE = 256 * 1024  # 256KB - Processing window for streaming

# Preview limits
MAX_PREVIEW_CHUNKS = 50
DEFAULT_PREVIEW_CHUNKS = 10
MIN_PREVIEW_CHUNKS = 1
MAX_PREVIEW_LENGTH = 1000  # Characters per chunk in preview
PREVIEW_CACHE_TTL = timedelta(minutes=15)

# Rate limiting
PREVIEW_RATE_LIMIT = "10/minute"
COMPARE_RATE_LIMIT = "5/minute"
CHUNKING_OPERATION_RATE_LIMIT = "100/hour"

# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# Operation limits
MAX_CONCURRENT_OPERATIONS = 10
MAX_OPERATION_PRIORITY = 10
MIN_OPERATION_PRIORITY = 1
DEFAULT_OPERATION_PRIORITY = 5
OPERATION_TIMEOUT = timedelta(hours=1)

# WebSocket limits
MAX_WEBSOCKET_CONNECTIONS_PER_USER = 10
MAX_TOTAL_WEBSOCKET_CONNECTIONS = 1000
WEBSOCKET_PROGRESS_THROTTLE_MS = 500  # Minimum time between progress updates
WEBSOCKET_PING_INTERVAL = 30  # Seconds
WEBSOCKET_TIMEOUT = 60  # Seconds

# Strategy comparison
MAX_STRATEGIES_TO_COMPARE = 6
MIN_STRATEGIES_TO_COMPARE = 2
MAX_CHUNKS_PER_STRATEGY_COMPARE = 20

# Performance thresholds
HIGH_MEMORY_THRESHOLD = 0.8  # 80% memory usage
HIGH_CPU_THRESHOLD = 0.9  # 90% CPU usage
SLOW_OPERATION_THRESHOLD = timedelta(seconds=30)

# Caching
CACHE_KEY_PREFIX = "chunking:"
STRATEGY_CACHE_TTL = timedelta(hours=1)
METRICS_CACHE_TTL = timedelta(minutes=5)

# Quality thresholds
MIN_QUALITY_SCORE = 0.0
MAX_QUALITY_SCORE = 1.0
GOOD_QUALITY_THRESHOLD = 0.7
EXCELLENT_QUALITY_THRESHOLD = 0.9

# File type limits
MAX_FILE_TYPE_LENGTH = 10
MAX_FILE_TYPES_PER_REQUEST = 20

# Configuration limits
MAX_CONFIG_NAME_LENGTH = 100
MAX_CONFIG_DESCRIPTION_LENGTH = 500
MAX_CONFIG_TAGS = 10
MAX_SAVED_CONFIGS_PER_USER = 100

# Metrics periods
DEFAULT_METRICS_PERIOD_DAYS = 30
MAX_METRICS_PERIOD_DAYS = 365
MIN_METRICS_PERIOD_DAYS = 1

# Redis configuration
REDIS_MAX_CONNECTIONS = 50
REDIS_HEALTH_CHECK_INTERVAL = 30
REDIS_SOCKET_KEEPALIVE = True
REDIS_RETRY_ON_TIMEOUT = True

# Error message templates
ERROR_CONTENT_TOO_LARGE = "Content exceeds maximum size of {max_size}MB"
ERROR_INVALID_CHUNK_SIZE = "Chunk size must be between {min} and {max}"
ERROR_INVALID_OVERLAP = "Overlap must be less than chunk size"
ERROR_RATE_LIMIT_EXCEEDED = "Rate limit exceeded. Please try again later"
ERROR_OPERATION_TIMEOUT = "Operation timed out after {timeout} seconds"
ERROR_TOO_MANY_CONNECTIONS = "Maximum concurrent connections exceeded"

# Success message templates
SUCCESS_PREVIEW_CACHED = "Preview cached for {ttl} minutes"
SUCCESS_OPERATION_STARTED = "Chunking operation started successfully"
SUCCESS_STRATEGY_UPDATED = "Chunking strategy updated successfully"
