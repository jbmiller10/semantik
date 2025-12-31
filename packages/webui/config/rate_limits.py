"""
Rate limiting configuration for chunking API endpoints.

This module provides centralized configuration for rate limiting across
the chunking API, including per-operation limits and circuit breaker settings.
"""

import os
import secrets
import threading
from typing import Any


class RateLimitConfig:
    """Configuration for rate limiting."""

    # Environment variable overrides
    PREVIEW_LIMIT = int(os.getenv("CHUNKING_PREVIEW_RATE_LIMIT", "10"))
    COMPARE_LIMIT = int(os.getenv("CHUNKING_COMPARE_RATE_LIMIT", "5"))
    PROCESS_LIMIT = int(os.getenv("CHUNKING_PROCESS_RATE_LIMIT", "20"))
    READ_LIMIT = int(os.getenv("CHUNKING_READ_RATE_LIMIT", "60"))
    ANALYTICS_LIMIT = int(os.getenv("CHUNKING_ANALYTICS_RATE_LIMIT", "30"))

    # Admin bypass token
    BYPASS_TOKEN = os.getenv("RATE_LIMIT_BYPASS_TOKEN", None)

    # Redis configuration for distributed rate limiting
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/1")  # Use DB 1 for rate limits

    # Circuit breaker configuration
    CIRCUIT_BREAKER_FAILURES = int(os.getenv("CIRCUIT_BREAKER_FAILURES", "5"))
    CIRCUIT_BREAKER_TIMEOUT = int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60"))

    # Rate limit strings for slowapi
    PREVIEW_RATE = f"{PREVIEW_LIMIT}/minute"
    COMPARE_RATE = f"{COMPARE_LIMIT}/minute"
    PROCESS_RATE = f"{PROCESS_LIMIT}/hour"
    READ_RATE = f"{READ_LIMIT}/minute"
    ANALYTICS_RATE = f"{ANALYTICS_LIMIT}/minute"

    # Default global limit
    DEFAULT_LIMIT = "1000/hour"

    @classmethod
    def get_rate_limit_string(cls, operation: str) -> str:
        """Get rate limit string for a specific operation.

        Args:
            operation: The operation type (preview, compare, process, read, analytics)

        Returns:
            Rate limit string for slowapi
        """
        rate_map = {
            "preview": cls.PREVIEW_RATE,
            "compare": cls.COMPARE_RATE,
            "process": cls.PROCESS_RATE,
            "read": cls.READ_RATE,
            "analytics": cls.ANALYTICS_RATE,
        }
        return rate_map.get(operation, cls.DEFAULT_LIMIT)

    @classmethod
    def bypass_rate_limit(cls, request_headers: dict[str, Any]) -> bool:
        """Check if request should bypass rate limiting.

        Args:
            request_headers: Request headers dictionary

        Returns:
            True if rate limiting should be bypassed
        """
        if not cls.BYPASS_TOKEN:
            return False

        auth_header = request_headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            return secrets.compare_digest(token, cls.BYPASS_TOKEN)

        return False


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern.

    Thread-safe implementation using a lock to protect shared state.
    """

    def __init__(self) -> None:
        """Initialize circuit breaker configuration."""
        self.failure_threshold = RateLimitConfig.CIRCUIT_BREAKER_FAILURES
        self.timeout_seconds = RateLimitConfig.CIRCUIT_BREAKER_TIMEOUT
        self._lock = threading.Lock()
        self.failure_counts: dict[str, int] = {}
        self.blocked_until: dict[str, float] = {}

    def record_failure(self, key: str) -> bool:
        """Record a failure for the given key and check if circuit should open.

        Args:
            key: The rate limit key (user/IP identifier)

        Returns:
            True if circuit breaker is now open for this key
        """
        import time

        with self._lock:
            current_time = time.time()

            # Check if already blocked
            if key in self.blocked_until and current_time < self.blocked_until[key]:
                return True

            # Increment failure count
            self.failure_counts[key] = self.failure_counts.get(key, 0) + 1

            # Check if threshold reached
            if self.failure_counts[key] >= self.failure_threshold:
                self.blocked_until[key] = current_time + self.timeout_seconds
                self.failure_counts[key] = 0
                return True

            return False

    def is_blocked(self, key: str) -> tuple[bool, int]:
        """Check if the circuit breaker is open for the given key.

        Args:
            key: The rate limit key (user/IP identifier)

        Returns:
            Tuple of (is_blocked, remaining_seconds)
        """
        import time

        with self._lock:
            if key not in self.blocked_until:
                return False, 0

            current_time = time.time()
            blocked_time = self.blocked_until[key]

            if current_time >= blocked_time:
                # Block expired - reset state
                del self.blocked_until[key]
                self.failure_counts.pop(key, None)
                return False, 0

            remaining = int(blocked_time - current_time)
            return True, remaining

    def reset(self, key: str) -> None:
        """Reset failure count for a key (call on successful request).

        Args:
            key: The rate limit key (user/IP identifier)
        """
        with self._lock:
            self.failure_counts.pop(key, None)
