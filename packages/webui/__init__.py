"""
Document Embedding Web UI Package
"""

# Test-time monkeypatch for fakeredis AsyncMock compatibility
import contextlib
import os as _os
import sys as _sys
from typing import Any

# Ensure absolute imports like ``import webui.auth`` resolve even when this
# module is first imported via the ``packages.webui`` name.
_sys.modules["webui"] = _sys.modules[__name__]

if _os.getenv("TESTING", "false").lower() in ("true", "1", "yes"):
    try:
        from unittest.mock import AsyncMock as _AsyncMock

        import fakeredis.aioredis as _fakeredis_aioredis

        _orig_init = _fakeredis_aioredis.FakeRedis.__init__

        def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            _orig_init(self, *args, **kwargs)
            # Replace selected coroutine methods with AsyncMocks so tests can set return_value/side_effect
            # Note: 'delete' is NOT included because it breaks FakeRedis lock/key operations
            for _name in (
                "ping",
                "xadd",
                "expire",
                "xrange",
                "xreadgroup",
                "xgroup_create",
                "xack",
                "xgroup_delconsumer",
                "xinfo_groups",
                "xgroup_destroy",
                "xinfo_stream",
                "close",
            ):
                with contextlib.suppress(Exception):
                    setattr(self, _name, _AsyncMock())

        _fakeredis_aioredis.FakeRedis.__init__ = _patched_init  # type: ignore[method-assign]

        # Note: ChunkingStrategy.MARKDOWN and ChunkingStrategy.HIERARCHICAL are already defined
        # as enum members in chunking_schemas.py, so no additional aliases are needed
    except Exception:
        # If fakeredis isn't available, skip patching
        pass

__all__ = ["app", "celery_app"]


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly in imports
    if name == "celery_app":
        try:
            from .celery_app import celery_app as _celery_app
        except Exception as exc:  # pragma: no cover - fail fast with guidance
            raise RuntimeError(
                "Failed to import webui.celery_app. Ensure required environment (e.g., JWT_SECRET_KEY) "
                "and dependencies are configured before importing webui."
            ) from exc
        return _celery_app
    if name == "app":
        try:
            from .main import app as _app
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import webui.main. Verify configuration (including JWT_SECRET_KEY) and dependencies."
            ) from exc
        return _app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
