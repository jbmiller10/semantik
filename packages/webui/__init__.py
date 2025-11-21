"""
Document Embedding Web UI Package
"""

# Test-time monkeypatch for fakeredis AsyncMock compatibility
import contextlib
import os as _os
import sys as _sys

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
            for _name in (
                "ping",
                "xadd",
                "expire",
                "xrange",
                "xreadgroup",
                "xgroup_create",
                "xack",
                "xgroup_delconsumer",
                "delete",
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

# Import Celery app
try:
    from .celery_app import celery_app
except Exception as exc:  # pragma: no cover - fail fast with guidance
    raise RuntimeError(
        "Failed to import webui.celery_app. Ensure required environment (e.g., JWT_SECRET_KEY) "
        "and dependencies are configured before importing webui."
    ) from exc

try:
    from .main import app
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import webui.main. Verify configuration (including JWT_SECRET_KEY) and dependencies."
    ) from exc

__all__ = ["app", "celery_app"]
