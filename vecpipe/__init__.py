"""Shim so ``import vecpipe`` works from the repo root without setup."""

from __future__ import annotations

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.util import find_spec, spec_from_loader
from pathlib import Path

_PACKAGES_DIR = Path(__file__).resolve().parent.parent / "packages"
if _PACKAGES_DIR.is_dir():
    packages_path = str(_PACKAGES_DIR)
    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)

_module = importlib.import_module("packages.vecpipe")
sys.modules[__name__] = _module
sys.modules.setdefault("packages.vecpipe", _module)


class _AliasLoader(Loader):
    def __init__(self, fullname: str, alias: str) -> None:
        self.fullname = fullname
        self.alias = alias

    def create_module(self, _spec):  # type: ignore[override]
        return None

    def exec_module(self, _module) -> None:  # type: ignore[override]
        target = importlib.import_module(self.alias)
        sys.modules[self.fullname] = target


class _VecpipeAliasFinder(MetaPathFinder):
    def find_spec(self, fullname: str, _path, _target=None):  # type: ignore[override]
        if not fullname.startswith("vecpipe."):
            return None

        alias = f"packages.{fullname}"
        if alias in sys.modules:
            sys.modules[fullname] = sys.modules[alias]
            alias_spec = find_spec(alias)
            is_pkg = bool(alias_spec and alias_spec.submodule_search_locations)
            return spec_from_loader(fullname, _AliasLoader(fullname, alias), is_package=is_pkg)

        alias_spec = find_spec(alias)
        if alias_spec is None:
            return None
        is_pkg = bool(alias_spec.submodule_search_locations)
        return spec_from_loader(fullname, _AliasLoader(fullname, alias), is_package=is_pkg)


if not any(isinstance(finder, _VecpipeAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _VecpipeAliasFinder())
