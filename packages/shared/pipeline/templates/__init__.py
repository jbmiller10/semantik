"""Pipeline templates module.

This module provides pre-configured pipeline templates for common use cases.
Templates are loaded lazily and validated against the plugin registry at startup.

Usage:
    from shared.pipeline.templates import list_templates, load_template

    # List all available templates
    templates = list_templates()
    for template in templates:
        print(f"{template.id}: {template.name}")

    # Load a specific template
    template = load_template("academic-papers")
    if template:
        dag = template.pipeline
"""

from __future__ import annotations

import logging

from .types import PipelineTemplate, TunableParameter, resolve_tunable_path

logger = logging.getLogger(__name__)

# Cache for loaded templates
_templates_cache: dict[str, PipelineTemplate] | None = None


def _resolve_tunable_path(
    template: PipelineTemplate, path: str
) -> tuple[str | None, str | None]:
    """Resolve a tunable parameter path to node_id and config_key.

    Args:
        template: The pipeline template to resolve against
        path: Dot-notation path (e.g., "nodes.chunker.config.max_tokens")

    Returns:
        Tuple of (node_id, config_key) if valid, (None, None) if invalid
    """
    parts = path.split(".")

    # Expected format: nodes.<node_id>.config.<param_name>
    if len(parts) != 4:
        return (None, None)

    if parts[0] != "nodes" or parts[2] != "config":
        return (None, None)

    node_id = parts[1]
    config_key = parts[3]

    # Verify node exists in the DAG
    node_ids = {node.id for node in template.pipeline.nodes}
    if node_id not in node_ids:
        return (None, None)

    return (node_id, config_key)


def _validate_tunable_paths(template: PipelineTemplate) -> list[str]:
    """Validate all tunable parameter paths in a template.

    Args:
        template: The template to validate

    Returns:
        List of error messages for invalid paths (empty if all valid)
    """
    errors: list[str] = []

    for tunable in template.tunable:
        node_id, config_key = _resolve_tunable_path(template, tunable.path)
        if node_id is None:
            errors.append(
                f"Invalid tunable path '{tunable.path}' in template '{template.id}'"
            )
            continue

        # Optionally validate that the config key exists in the node's config
        # This is a soft check since some configs may be dynamic
        for node in template.pipeline.nodes:
            if node.id == node_id:
                if config_key not in node.config:
                    logger.debug(
                        "Tunable path '%s' references config key '%s' not present "
                        "in node '%s' default config (may be set dynamically)",
                        tunable.path,
                        config_key,
                        node_id,
                    )
                break

    return errors


def _validate_template(template: PipelineTemplate) -> None:
    """Validate a template's DAG and tunable paths.

    Args:
        template: The template to validate

    Raises:
        ValueError: If the template has validation errors
    """
    from shared.plugins.registry import plugin_registry

    errors: list[str] = []

    # Validate the DAG structure
    known_plugins = set(plugin_registry.list_ids())
    dag_errors = template.pipeline.validate(known_plugins)

    for dag_error in dag_errors:
        errors.append(
            f"DAG validation error in template '{template.id}': "
            f"[{dag_error.rule}] {dag_error.message}"
        )

    # Validate tunable parameter paths
    tunable_errors = _validate_tunable_paths(template)
    errors.extend(tunable_errors)

    if errors:
        error_msg = f"Template '{template.id}' validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(error_msg)


def _load_all() -> dict[str, PipelineTemplate]:
    """Lazily load and validate all templates.

    Returns:
        Dictionary mapping template IDs to validated PipelineTemplate objects

    Raises:
        ValueError: If any template fails validation
    """
    global _templates_cache

    if _templates_cache is not None:
        return _templates_cache

    from shared.pipeline.templates.academic_papers import (
        TEMPLATE as ACADEMIC_PAPERS_TEMPLATE,
    )
    from shared.pipeline.templates.codebase import TEMPLATE as CODEBASE_TEMPLATE
    from shared.pipeline.templates.documentation import (
        TEMPLATE as DOCUMENTATION_TEMPLATE,
    )
    from shared.pipeline.templates.email_archive import (
        TEMPLATE as EMAIL_ARCHIVE_TEMPLATE,
    )
    from shared.pipeline.templates.mixed_documents import (
        TEMPLATE as MIXED_DOCUMENTS_TEMPLATE,
    )

    templates = [
        ACADEMIC_PAPERS_TEMPLATE,
        CODEBASE_TEMPLATE,
        DOCUMENTATION_TEMPLATE,
        EMAIL_ARCHIVE_TEMPLATE,
        MIXED_DOCUMENTS_TEMPLATE,
    ]

    result: dict[str, PipelineTemplate] = {}
    all_errors: list[str] = []

    for template in templates:
        try:
            _validate_template(template)
            result[template.id] = template
            logger.debug("Loaded template: %s", template.id)
        except ValueError as e:
            all_errors.append(str(e))

    if all_errors:
        raise ValueError(
            "Template validation errors:\n" + "\n".join(all_errors)
        )

    _templates_cache = result
    logger.info("Loaded %d pipeline templates", len(result))
    return result


def list_templates() -> list[PipelineTemplate]:
    """List all available pipeline templates.

    Returns:
        List of validated PipelineTemplate objects

    Raises:
        ValueError: If any template fails validation
    """
    templates_dict = _load_all()
    return list(templates_dict.values())


def load_template(template_id: str) -> PipelineTemplate | None:
    """Load a specific template by ID.

    Args:
        template_id: The unique identifier for the template

    Returns:
        The PipelineTemplate if found, None otherwise

    Raises:
        ValueError: If template validation fails during initial load
    """
    templates_dict = _load_all()
    return templates_dict.get(template_id)


def clear_cache() -> None:
    """Clear the templates cache (for testing purposes)."""
    global _templates_cache
    _templates_cache = None


__all__ = [
    "list_templates",
    "load_template",
    "clear_cache",
    "PipelineTemplate",
    "TunableParameter",
    "resolve_tunable_path",
]
