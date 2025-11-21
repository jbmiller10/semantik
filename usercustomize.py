"""Make in-repo packages importable without setting PYTHONPATH.

Python automatically imports ``usercustomize`` (after ``sitecustomize``) if it
is importable on ``sys.path``. Because the repository root is on ``sys.path``
whenever commands are executed from the repo, this hook ensures ``shared``,
``webui``, and ``vecpipe`` resolve correctly even when the wheel isn't
installed and ``PYTHONPATH`` is unset.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PACKAGES_DIR = _ROOT / "packages"

if _PACKAGES_DIR.is_dir():
    packages_path = str(_PACKAGES_DIR)
    if packages_path not in sys.path:
        # Prepend so local sources win over any globally installed copy.
        sys.path.insert(0, packages_path)


def _alias_top_level(name: str) -> None:
    """Alias ``packages.<name>`` to ``<name>`` so both prefixes share modules."""

    qualified = f"packages.{name}"
    try:
        module = importlib.import_module(qualified)
    except Exception:  # pragma: no cover - aliasing best-effort
        return

    # Ensure both keys reference the same module object.
    sys.modules.setdefault(name, module)
    sys.modules.setdefault(qualified, module)


for _pkg in ("shared", "webui", "vecpipe"):
    _alias_top_level(_pkg)
