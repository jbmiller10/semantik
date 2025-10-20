#!/usr/bin/env python3
"""
Memory monitoring for streaming operations.

This module provides monitoring and alerting for memory usage in the streaming
processor to prevent OOM conditions and maintain system stability.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool

logger = logging.getLogger(__name__)


@dataclass
class MemoryAlert:
    """Memory usage alert information."""

    level: str  # "warning", "critical"
    message: str
    usage_percent: float
    details: dict[str, Any]


class MemoryMonitor:
    """
    Monitor memory usage and trigger alerts.

    This monitor tracks memory pool usage and system memory to prevent
    OOM conditions and provide early warnings of memory pressure.
    """

    def __init__(
        self,
        memory_pool: "MemoryPool",
        warning_threshold: float = 0.8,  # 80%
        critical_threshold: float = 0.95,  # 95%
        check_interval: int = 10,  # seconds
    ):
        """
        Initialize memory monitor.

        Args:
            memory_pool: Memory pool to monitor
            warning_threshold: Usage percentage to trigger warning
            critical_threshold: Usage percentage to trigger critical alert
            check_interval: Seconds between checks
        """
        self.memory_pool = memory_pool
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval

        self._monitor_task: asyncio.Task | None = None
        self._alert_callback: Callable | None = None
        self._last_alert_level: str | None = None

        # Statistics
        self.alerts_sent = 0
        self.max_usage_seen = 0.0

    async def start(self, alert_callback: Callable | None = None) -> None:
        """
        Start monitoring.

        Args:
            alert_callback: Optional callback for alerts
        """
        self._alert_callback = alert_callback
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Memory monitor started (warning={self.warning_threshold:.0%}, critical={self.critical_threshold:.0%})"
        )

    async def stop(self) -> None:
        """Stop monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            finally:
                self._monitor_task = None
                logger.info("Memory monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.check_interval)

                # Check memory usage
                stats = self.memory_pool.get_stats()
                usage_percent = stats["usage_percent"] / 100

                # Track maximum usage
                if usage_percent > self.max_usage_seen:
                    self.max_usage_seen = usage_percent

                # Get system memory info
                system_memory = self._get_system_memory_info()

                # Determine alert level
                alert_level = None
                if usage_percent >= self.critical_threshold:
                    alert_level = "critical"
                elif usage_percent >= self.warning_threshold:
                    alert_level = "warning"

                # Send alert if level changed or escalated
                if alert_level and alert_level != self._last_alert_level:
                    alert = MemoryAlert(
                        level=alert_level,
                        message=f"Memory pool usage at {usage_percent:.1%}",
                        usage_percent=usage_percent,
                        details={
                            **stats,
                            "system_memory": system_memory,
                        },
                    )

                    await self._send_alert(alert)
                    self._last_alert_level = alert_level

                elif not alert_level and self._last_alert_level:
                    # Clear alert
                    logger.info(f"Memory usage recovered to {usage_percent:.1%} (was {self._last_alert_level})")
                    self._last_alert_level = None

                # Log statistics periodically
                if alert_level:
                    logger.warning(
                        f"Memory pool {alert_level}: {usage_percent:.1%} used, "
                        f"active_buffers={stats['active_buffers']}, "
                        f"oldest_age={stats.get('oldest_allocation_age', 0):.1f}s",
                        extra=stats,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in memory monitor: {e}")

    async def _send_alert(self, alert: MemoryAlert) -> None:
        """
        Send alert to callback.

        Args:
            alert: Alert to send
        """
        self.alerts_sent += 1

        # Log alert
        log_method = logger.critical if alert.level == "critical" else logger.warning
        log_method(
            f"MEMORY ALERT [{alert.level.upper()}]: {alert.message}",
            extra=alert.details,
        )

        # Send to callback
        if self._alert_callback:
            try:
                if asyncio.iscoroutinefunction(self._alert_callback):
                    await self._alert_callback(alert)
                else:
                    self._alert_callback(alert)
            except Exception as e:
                logger.exception(f"Error sending alert: {e}")

    def _get_system_memory_info(self) -> dict[str, Any]:
        """
        Get system memory information.

        Returns:
            Dictionary with system memory stats
        """
        try:
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()
            process = psutil.Process()

            return {
                "system_total_mb": virtual_mem.total / 1024 / 1024,
                "system_available_mb": virtual_mem.available / 1024 / 1024,
                "system_used_percent": virtual_mem.percent,
                "swap_used_percent": swap_mem.percent,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "process_memory_percent": process.memory_percent(),
            }
        except Exception as e:
            logger.error(f"Failed to get system memory info: {e}")
            return {}

    def get_status(self) -> dict[str, Any]:
        """
        Get current monitor status.

        Returns:
            Dictionary with monitor status
        """
        pool_stats = self.memory_pool.get_stats()
        usage_percent = pool_stats["usage_percent"] / 100

        return {
            "is_monitoring": self._monitor_task is not None and not self._monitor_task.done(),
            "current_usage_percent": usage_percent,
            "max_usage_seen": self.max_usage_seen,
            "current_alert_level": self._last_alert_level,
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "alerts_sent": self.alerts_sent,
            "pool_stats": pool_stats,
            "system_memory": self._get_system_memory_info(),
        }

    async def check_health(self) -> tuple[bool, str]:
        """
        Check if memory usage is healthy.

        Returns:
            Tuple of (is_healthy, message)
        """
        stats = self.memory_pool.get_stats()
        usage_percent = stats["usage_percent"] / 100

        if usage_percent >= self.critical_threshold:
            return False, f"Critical: Memory usage at {usage_percent:.1%}"
        if usage_percent >= self.warning_threshold:
            return True, f"Warning: Memory usage at {usage_percent:.1%}"
        return True, f"Healthy: Memory usage at {usage_percent:.1%}"

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MemoryMonitor(warning={self.warning_threshold:.0%}, "
            f"critical={self.critical_threshold:.0%}, "
            f"monitoring={self._monitor_task is not None})"
        )
