"""Reuse the v2 API fixtures for v1 endpoints without pytest_plugins."""

from tests.webui.api.v2.conftest import (
    api_auth_headers,
    api_client,
    _reset_redis_manager as reset_redis_manager,
)

__all__ = ["api_auth_headers", "api_client", "reset_redis_manager"]
