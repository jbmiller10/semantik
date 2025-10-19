"""Compatibility shim for legacy ``packages.webui.tasks`` imports.

The Celery task implementation now lives in the ``packages.webui.tasks`` package
directory. This module simply re-exports that package so import sites that rely on
``packages.webui.tasks`` as a module continue to function.
"""

# Mirror the package's ``__all__`` for tooling that introspects this module.
# Re-export everything from the package-level module.
from .tasks import *  # type: ignore  # noqa: F401,F403
from .tasks import __all__  # type: ignore  # noqa: F401
