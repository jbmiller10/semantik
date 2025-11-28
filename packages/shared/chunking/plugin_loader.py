#!/usr/bin/env python3
"""Runtime loader for external chunking strategy plugins.

Plugins register an entry point under the group ``semantik.chunking_strategies``.
Each entry point should resolve to a ChunkingStrategy subclass (domain) or a
callable returning one. Minimal contract expected on the class:

    INTERNAL_NAME: str   # required internal identifier
    API_ID: str | None   # optional; defaults to INTERNAL_NAME
    METADATA: dict       # optional; see fields used below

METADATA fields (all optional):
    display_name, description, best_for, pros, cons,
    performance_characteristics, manager_defaults, builder_defaults,
    supported_file_types, aliases, factory_defaults
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from importlib.metadata import EntryPoints
from typing import Any

from webui.services.chunking.strategy_registry import register_strategy_definition
from webui.services.chunking_strategy_factory import ChunkingStrategyFactory

logger = logging.getLogger(__name__)


ENTRYPOINT_GROUP = "semantik.chunking_strategies"
ENV_FLAG = "SEMANTIK_ENABLE_PLUGINS"


def _should_enable_plugins() -> bool:
    """Check env flag to allow disabling plugin loading."""

    value = os.getenv(ENV_FLAG, "true").lower()
    return value not in {"0", "false", "no", "off"}


def _coerce_class(obj: Any) -> type | None:
    """Return a class object if obj is a class or a callable returning a class."""

    if isinstance(obj, type):
        return obj
    if callable(obj):
        maybe_cls = obj()
        if isinstance(maybe_cls, type):
            return maybe_cls
        if isinstance(maybe_cls, object):
            return maybe_cls.__class__
    return None


def load_chunking_plugins() -> list[str]:
    """
    Discover and register plugin strategies via entry points.

    Returns:
        List of api_ids successfully registered.
    """

    if not _should_enable_plugins():
        logger.info("Chunking plugins disabled via %s", ENV_FLAG)
        return []

    try:
        eps = metadata.entry_points()
        ep_group = eps.select(group=ENTRYPOINT_GROUP) if hasattr(eps, "select") else eps.get(ENTRYPOINT_GROUP, EntryPoints())
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to query entry points for chunking plugins: %s", exc)
        return []

    registered: list[str] = []

    for ep in ep_group:
        try:
            loaded = ep.load()
            plugin_cls = _coerce_class(loaded)
            if plugin_cls is None:
                raise TypeError(f"Entry point {ep.name} did not resolve to a class.")

            internal_name = getattr(plugin_cls, "INTERNAL_NAME", None) or getattr(plugin_cls, "name", None) or ep.name
            api_id = getattr(plugin_cls, "API_ID", None) or internal_name
            metadata_dict: dict[str, Any] = getattr(plugin_cls, "METADATA", {}) or {}

            # Require a visual example so the guide can render a preview consistently.
            visual_example = metadata_dict.get("visual_example")
            if not visual_example or not isinstance(visual_example, dict):
                logger.warning(
                    "Skipping plugin '%s': missing required visual_example (url + optional caption)",
                    api_id,
                )
                continue
            url = visual_example.get("url")
            if not isinstance(url, str) or not url.startswith("https://"):
                logger.warning(
                    "Skipping plugin '%s': visual_example.url must be an https:// URL",
                    api_id,
                )
                continue

            display_name = metadata_dict.get("display_name", api_id.replace("_", " ").title())
            description = metadata_dict.get("description", f"Plugin strategy {display_name}")

            # Register into the strategy factory / domain registry
            ChunkingStrategyFactory.register_strategy(internal_name, plugin_cls, api_enum=None)

            # Register metadata so the strategy surfaces in listings
            register_strategy_definition(
                api_id=str(api_id),
                internal_id=str(internal_name),
                display_name=display_name,
                description=description,
                best_for=tuple(metadata_dict.get("best_for", ())),
                pros=tuple(metadata_dict.get("pros", ())),
                cons=tuple(metadata_dict.get("cons", ())),
                performance_characteristics=metadata_dict.get("performance_characteristics", {}),
                manager_defaults=metadata_dict.get("manager_defaults"),
                builder_defaults=metadata_dict.get("builder_defaults"),
                supported_file_types=tuple(metadata_dict.get("supported_file_types", ())),
                aliases=tuple(metadata_dict.get("aliases", ())),
                factory_defaults=metadata_dict.get("factory_defaults"),
                is_plugin=True,
                visual_example=visual_example,
            )

            registered.append(str(api_id))
            logger.info("Registered chunking plugin '%s' (internal: %s)", api_id, internal_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load chunking plugin %s: %s", getattr(ep, "name", "unknown"), exc)
            continue

    return registered
