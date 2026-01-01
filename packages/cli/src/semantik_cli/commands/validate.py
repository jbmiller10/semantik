"""Validate a plugin project against the Semantik contract."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click


def _load_pyproject(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as fh:
        return tomllib.load(fh)


def _resolve_plugin_type(plugin_cls: type) -> str | None:
    try:
        import shared.embedding.plugin_base as embedding_plugin_base
    except Exception:
        embedding_plugin_base = None

    try:
        import shared.chunking.domain.services.chunking_strategies.base as chunking_strategy_base
    except Exception:
        chunking_strategy_base = None

    try:
        import shared.connectors.base as connector_base
    except Exception:
        connector_base = None

    if embedding_plugin_base is not None and issubclass(plugin_cls, embedding_plugin_base.BaseEmbeddingPlugin):
        return "embedding"
    if chunking_strategy_base is not None and issubclass(plugin_cls, chunking_strategy_base.ChunkingStrategy):
        return "chunking"
    if connector_base is not None and issubclass(plugin_cls, connector_base.BaseConnector):
        return "connector"
    return None


def _validate_embedding(plugin_cls: type) -> list[str]:
    errors: list[str] = []
    required_attrs = ["INTERNAL_NAME", "API_ID", "PROVIDER_TYPE"]
    for attr in required_attrs:
        if not getattr(plugin_cls, attr, None):
            errors.append(f"missing required attribute: {attr}")
    provider_type = getattr(plugin_cls, "PROVIDER_TYPE", "")
    if provider_type and provider_type not in ("local", "remote", "hybrid"):
        errors.append(f"invalid PROVIDER_TYPE: {provider_type}")
    if not callable(getattr(plugin_cls, "get_definition", None)):
        errors.append("missing get_definition()")
    if not callable(getattr(plugin_cls, "supports_model", None)):
        errors.append("missing supports_model()")
    return errors


def _validate_chunking(plugin_cls: type) -> list[str]:
    errors: list[str] = []
    if not getattr(plugin_cls, "INTERNAL_NAME", None):
        errors.append("missing INTERNAL_NAME")
    metadata = getattr(plugin_cls, "METADATA", {}) or {}
    visual_example = metadata.get("visual_example")
    if not visual_example or not isinstance(visual_example, dict):
        errors.append("missing METADATA.visual_example")
    else:
        url = visual_example.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            errors.append("visual_example.url must be https://")
    if not callable(getattr(plugin_cls, "chunk", None)):
        errors.append("missing chunk()")
    if not callable(getattr(plugin_cls, "validate_content", None)):
        errors.append("missing validate_content()")
    if not callable(getattr(plugin_cls, "estimate_chunks", None)):
        errors.append("missing estimate_chunks()")
    return errors


def _validate_connector(plugin_cls: type) -> list[str]:
    errors: list[str] = []
    if not getattr(plugin_cls, "PLUGIN_ID", None):
        errors.append("missing PLUGIN_ID")
    if not callable(getattr(plugin_cls, "get_config_fields", None)):
        errors.append("missing get_config_fields()")
    if not callable(getattr(plugin_cls, "get_secret_fields", None)):
        errors.append("missing get_secret_fields()")
    if not callable(getattr(plugin_cls, "authenticate", None)):
        errors.append("missing authenticate()")
    if not callable(getattr(plugin_cls, "load_documents", None)):
        errors.append("missing load_documents()")
    return errors


def _validate_plugin(plugin_cls: type, plugin_type: str) -> list[str]:
    if plugin_type == "embedding":
        return _validate_embedding(plugin_cls)
    if plugin_type == "chunking":
        return _validate_chunking(plugin_cls)
    if plugin_type == "connector":
        return _validate_connector(plugin_cls)
    return [f"unsupported plugin type: {plugin_type}"]


@click.command()
@click.argument("path", default=".")
def validate(path: str) -> None:
    """Validate a plugin project by inspecting its entry points."""
    project_dir = Path(path).resolve()
    pyproject_path = project_dir / "pyproject.toml"
    if not pyproject_path.exists():
        click.echo("pyproject.toml not found", err=True)
        raise SystemExit(1)

    data = _load_pyproject(pyproject_path)
    entry_points = data.get("project", {}).get("entry-points", {}).get("semantik.plugins", {})

    if not entry_points:
        click.echo("No [project.entry-points.'semantik.plugins'] found", err=True)
        raise SystemExit(1)

    errors_found = False
    for name, value in entry_points.items():
        if ":" not in value:
            click.echo(f"Entry point '{name}' is invalid (expected module:attr)", err=True)
            errors_found = True
            continue
        module_name, attr = value.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            plugin_cls = getattr(module, attr)
        except Exception as exc:
            click.echo(f"Failed to import '{value}': {exc}", err=True)
            errors_found = True
            continue

        plugin_type = _resolve_plugin_type(plugin_cls)
        if plugin_type is None:
            click.echo(f"Entry point '{name}' does not map to a known plugin type", err=True)
            errors_found = True
            continue

        errors = _validate_plugin(plugin_cls, plugin_type)
        if errors:
            errors_found = True
            click.echo(f"{name} ({plugin_type}) failed validation:")
            for err in errors:
                click.echo(f"  - {err}")
        else:
            click.echo(f"{name} ({plugin_type}) OK")

    if errors_found:
        raise SystemExit(1)

    click.echo("Validation complete")
