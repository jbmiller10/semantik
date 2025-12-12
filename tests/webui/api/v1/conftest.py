"""Reuse the v2 API fixtures for v1 endpoints without pytest_plugins."""

from tests.webui.api.v2.conftest import (
    _reset_redis_manager as reset_redis_manager,
    api_auth_headers,
    api_client,
    api_client_unauthenticated,
)

__all__ = ["api_auth_headers", "api_client", "api_client_unauthenticated", "reset_redis_manager"]
