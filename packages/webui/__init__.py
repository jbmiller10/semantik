"""
Document Embedding Web UI Package
"""

# Test-time monkeypatch for fakeredis AsyncMock compatibility
import contextlib
import os as _os

if _os.getenv("TESTING", "false").lower() in ("true", "1", "yes"):
    try:
        from unittest.mock import AsyncMock as _AsyncMock

        import fakeredis.aioredis as _fakeredis_aioredis

        from packages.webui.api.v2 import chunking_schemas as _schemas

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

# Backward compatibility shim for old uvicorn command
# This allows "python -m uvicorn webui.app:app" to still work
try:
    from .main import app
except ImportError:
    # If main.py doesn't exist yet, try to import from app.py
    from .app import app

# Import Celery app
from .celery_app import celery_app

__all__ = ["app", "celery_app"]
