"""Consecutive failure tracking for pipeline execution.

This module provides the ConsecutiveFailureTracker class that tracks
consecutive failures to detect system-level issues that warrant halting
pipeline execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Record of a single failure.

    Attributes:
        file_uri: URI of the file that failed
        stage_id: Pipeline stage where failure occurred
        error_type: Category of the error
        error_message: Human-readable error description
        timestamp: When the failure occurred
    """

    file_uri: str
    stage_id: str
    error_type: str
    error_message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ConsecutiveFailureTracker:
    """Tracks consecutive failures to detect system-level issues.

    This tracker monitors failures during pipeline execution and determines
    when to halt processing due to repeated failures that indicate a
    systemic problem (e.g., database connection lost, storage unavailable,
    embedding service down).

    The tracker maintains a count of consecutive failures. Each success
    resets the count. When the count reaches the threshold, should_halt()
    returns True.

    Example:
        ```python
        tracker = ConsecutiveFailureTracker(threshold=10)

        for file in files:
            if tracker.should_halt():
                logger.error("Halting due to consecutive failures")
                break

            try:
                process(file)
                tracker.record_success()
            except Exception as e:
                tracker.record_failure(file.uri, "parser", str(e))
        ```

    Attributes:
        threshold: Number of consecutive failures before halt
        consecutive_count: Current consecutive failure count
        recent_failures: List of recent failure records
    """

    def __init__(self, threshold: int = 10, recent_limit: int = 50) -> None:
        """Initialize the failure tracker.

        Args:
            threshold: Number of consecutive failures to trigger halt (default: 10)
            recent_limit: Maximum number of recent failures to retain (default: 50)
        """
        if threshold < 1:
            raise ValueError("threshold must be at least 1")

        self.threshold = threshold
        self._recent_limit = recent_limit
        self._consecutive_count = 0
        self._recent_failures: list[FailureRecord] = []

    @property
    def consecutive_count(self) -> int:
        """Current count of consecutive failures."""
        return self._consecutive_count

    @property
    def recent_failures(self) -> list[FailureRecord]:
        """List of recent failure records."""
        return list(self._recent_failures)

    def record_success(self) -> None:
        """Record a successful file processing.

        Resets the consecutive failure count to 0.
        """
        if self._consecutive_count > 0:
            logger.debug(
                "Success after %d consecutive failures - resetting counter",
                self._consecutive_count,
            )
        self._consecutive_count = 0

    def record_failure(
        self,
        file_uri: str,
        stage_id: str,
        error_message: str,
        error_type: str = "unknown",
    ) -> None:
        """Record a file processing failure.

        Increments the consecutive failure count and stores the failure
        record for later analysis.

        Args:
            file_uri: URI of the file that failed
            stage_id: Pipeline stage ID where failure occurred
            error_message: Human-readable error description
            error_type: Category of the error (default: "unknown")
        """
        self._consecutive_count += 1

        record = FailureRecord(
            file_uri=file_uri,
            stage_id=stage_id,
            error_type=error_type,
            error_message=error_message,
        )
        self._recent_failures.append(record)

        # Trim to recent limit
        if len(self._recent_failures) > self._recent_limit:
            self._recent_failures = self._recent_failures[-self._recent_limit :]

        logger.debug(
            "Recorded failure #%d for %s at stage %s: %s",
            self._consecutive_count,
            file_uri,
            stage_id,
            error_message[:100],
        )

        if self._consecutive_count >= self.threshold:
            logger.warning(
                "Reached %d consecutive failures (threshold: %d) - pipeline will halt",
                self._consecutive_count,
                self.threshold,
            )

    def should_halt(self) -> bool:
        """Check if execution should halt due to consecutive failures.

        Returns:
            True if consecutive failures >= threshold
        """
        return self._consecutive_count >= self.threshold

    def get_halt_reason(self) -> str:
        """Get a human-readable halt reason message.

        Returns:
            Description of why the pipeline halted
        """
        if not self.should_halt():
            return ""

        # Get the most recent failures for context
        recent = self._recent_failures[-min(3, len(self._recent_failures)) :]
        stages = {f.stage_id for f in recent}
        error_types = {f.error_type for f in recent}

        return (
            f"Pipeline halted after {self._consecutive_count} consecutive failures. "
            f"Recent failures at stages: {', '.join(stages)}. "
            f"Error types: {', '.join(error_types)}."
        )

    def reset(self) -> None:
        """Reset the tracker to initial state.

        Clears all failure counts and records.
        """
        self._consecutive_count = 0
        self._recent_failures = []


__all__ = [
    "ConsecutiveFailureTracker",
    "FailureRecord",
]
