"""
Configuration module for webui package.
"""

from .rate_limits import CircuitBreakerConfig, RateLimitConfig

__all__ = ["RateLimitConfig", "CircuitBreakerConfig"]
