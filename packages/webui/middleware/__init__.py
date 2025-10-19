"""Middleware modules for the WebUI application."""

from .correlation import (
    CorrelationMiddleware,
    configure_logging_with_correlation,
    get_correlation_id,
    get_or_generate_correlation_id,
    set_correlation_id,
)
from .exception_handlers import register_global_exception_handlers

__all__ = [
    "CorrelationMiddleware",
    "configure_logging_with_correlation",
    "get_correlation_id",
    "get_or_generate_correlation_id",
    "set_correlation_id",
    "register_global_exception_handlers",
]
