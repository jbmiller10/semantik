"""Shim so ``import shared`` works from repo root without editable install.

This module aliases ``shared`` to the real package under ``packages/shared``
and adjusts ``sys.path`` so child imports resolve correctly. It keeps the
module object shared with ``packages.shared`` to avoid duplicate state.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PACKAGES_DIR = Path(__file__).resolve().parent.parent / "packages"
if _PACKAGES_DIR.is_dir():
    packages_path = str(_PACKAGES_DIR)
    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)

_module = importlib.import_module("packages.shared")
sys.modules[__name__] = _module
sys.modules.setdefault("packages.shared", _module)

