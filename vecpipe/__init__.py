"""Shim so ``import vecpipe`` works from the repo root without setup."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PACKAGES_DIR = Path(__file__).resolve().parent.parent / "packages"
if _PACKAGES_DIR.is_dir():
    packages_path = str(_PACKAGES_DIR)
    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)

_module = importlib.import_module("packages.vecpipe")
sys.modules[__name__] = _module
sys.modules.setdefault("packages.vecpipe", _module)

