"""Make in-repo packages importable without setting PYTHONPATH.

Python automatically imports ``usercustomize`` (after ``sitecustomize``) if it
is importable on ``sys.path``. Because the repository root is on ``sys.path``
for repo-local commands, this hook ensures ``shared`` and ``vecpipe`` resolve
correctly even when the wheel isn't installed and ``PYTHONPATH`` is unset.
``webui`` stays importable via the ``packages`` path but is intentionally not
aliased here to avoid triggering its Celery bootstrap before callers set
environment flags (e.g., ``TESTING``).
"""

from __future__ import annotations

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
    """Alias ``packages.<name>`` to ``<name>`` if already loaded.

    Avoids importing modules eagerly (e.g., ``packages.webui`` spins up Celery).
    """

    qualified = f"packages.{name}"
    module = sys.modules.get(qualified) or sys.modules.get(name)
    if module is None:
        return

    sys.modules.setdefault(name, module)
    sys.modules.setdefault(qualified, module)


for _pkg in ("shared", "vecpipe"):
    _alias_top_level(_pkg)
