#!/usr/bin/env python3
"""Performance monitoring for regex operations."""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RegexMetrics:
    """Metrics for a single regex execution."""

    pattern: str
    execution_time: float
    input_size: int
    matched: bool
    timed_out: bool
    timestamp: float = field(default_factory=time.time)


class RegexPerformanceMonitor:
    """Monitor regex performance to detect issues."""

    def __init__(self, slow_threshold: float = 0.1, alert_threshold: int = 5):
        """Initialize the performance monitor.

        Args:
            slow_threshold: Time threshold (seconds) to consider a regex slow
            alert_threshold: Number of slow patterns before alerting
        """
        self.metrics: list[RegexMetrics] = []
        self.slow_patterns: dict[str, int] = {}
        self.slow_threshold = slow_threshold
        self.alert_threshold = alert_threshold
        self._total_executions = 0
        self._total_timeouts = 0
        self._total_time = 0.0

    def record_execution(
        self,
        pattern: str,
        execution_time: float,
        input_size: int,
        matched: bool = False,
        timed_out: bool = False,
    ) -> None:
        """Record regex execution metrics.

        Args:
            pattern: Regex pattern used
            execution_time: Time taken to execute (seconds)
            input_size: Size of input text
            matched: Whether pattern matched
            timed_out: Whether execution timed out
        """
        metric = RegexMetrics(
            pattern=pattern[:100],  # Truncate long patterns
            execution_time=execution_time,
            input_size=input_size,
            matched=matched,
            timed_out=timed_out,
        )

        self.metrics.append(metric)
        self._total_executions += 1
        self._total_time += execution_time

        if timed_out:
            self._total_timeouts += 1

        # Track slow patterns
        if execution_time > self.slow_threshold:
            self.slow_patterns[pattern] = self.slow_patterns.get(pattern, 0) + 1

            # Alert if pattern is consistently slow
            if self.slow_patterns[pattern] == self.alert_threshold:
                logger.warning(
                    f"Pattern consistently slow: '{pattern[:50]}...' "
                    f"({self.slow_patterns[pattern]} slow executions)"
                )

        # Log warnings for concerning metrics
        if timed_out:
            logger.error(f"Regex timeout: pattern='{pattern[:50]}...', " f"input_size={input_size}")
        elif execution_time > 1.0:
            logger.warning(
                f"Very slow regex: pattern='{pattern[:50]}...', " f"time={execution_time:.2f}s, input_size={input_size}"
            )

    def get_problematic_patterns(self) -> list[str]:
        """Get patterns that frequently cause issues.

        Returns:
            List of problematic pattern strings
        """
        return [pattern for pattern, count in self.slow_patterns.items() if count >= self.alert_threshold]

    def get_statistics(self) -> dict:
        """Get overall performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        if self._total_executions == 0:
            return {
                "total_executions": 0,
                "total_timeouts": 0,
                "timeout_rate": 0.0,
                "average_time": 0.0,
                "slow_patterns": 0,
                "problematic_patterns": [],
            }

        return {
            "total_executions": self._total_executions,
            "total_timeouts": self._total_timeouts,
            "timeout_rate": self._total_timeouts / self._total_executions,
            "average_time": self._total_time / self._total_executions,
            "slow_patterns": len(self.slow_patterns),
            "problematic_patterns": self.get_problematic_patterns(),
        }

    def get_recent_metrics(self, count: int = 100) -> list[RegexMetrics]:
        """Get the most recent metrics.

        Args:
            count: Number of recent metrics to return

        Returns:
            List of recent RegexMetrics
        """
        return self.metrics[-count:]

    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self.slow_patterns.clear()
        self._total_executions = 0
        self._total_timeouts = 0
        self._total_time = 0.0

    def analyze_pattern(self, pattern: str) -> dict:
        """Analyze metrics for a specific pattern.

        Args:
            pattern: Pattern to analyze

        Returns:
            Dictionary with pattern-specific metrics
        """
        pattern_metrics = [m for m in self.metrics if m.pattern == pattern[:100]]

        if not pattern_metrics:
            return {
                "pattern": pattern,
                "executions": 0,
                "average_time": 0.0,
                "max_time": 0.0,
                "min_time": 0.0,
                "timeouts": 0,
                "matches": 0,
            }

        execution_times = [m.execution_time for m in pattern_metrics]
        timeouts = sum(1 for m in pattern_metrics if m.timed_out)
        matches = sum(1 for m in pattern_metrics if m.matched)

        return {
            "pattern": pattern,
            "executions": len(pattern_metrics),
            "average_time": sum(execution_times) / len(execution_times),
            "max_time": max(execution_times),
            "min_time": min(execution_times),
            "timeouts": timeouts,
            "matches": matches,
        }

    def should_block_pattern(self, pattern: str) -> bool:
        """Determine if a pattern should be blocked based on history.

        Args:
            pattern: Pattern to check

        Returns:
            True if pattern should be blocked
        """
        # Block if pattern has caused multiple timeouts
        pattern_metrics = [m for m in self.metrics if m.pattern == pattern[:100]]
        timeout_count = sum(1 for m in pattern_metrics if m.timed_out)

        if timeout_count >= 3:
            return True

        # Block if pattern is consistently very slow
        if pattern in self.slow_patterns and self.slow_patterns[pattern] >= 10:
            pattern_analysis = self.analyze_pattern(pattern)
            if pattern_analysis["average_time"] > 0.5:  # 500ms average
                return True

        return False
