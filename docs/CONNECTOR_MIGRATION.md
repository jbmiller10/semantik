# Connector Plugin Migration Guide

This guide covers migrating connector plugins to the unified plugin system.

## Summary

- `connector_catalog.py` is removed.
- Connectors self-describe via class metadata and schema helpers.
- Register connectors under the unified entry point group: `semantik.plugins`.

## Required Connector API

Your connector must subclass `shared.connectors.base.BaseConnector` and define:

- `PLUGIN_ID`: unique connector ID (used as `source_type`).
- `METADATA`: dict with `name`, `description`, `icon`, and optional `supports_sync`, `preview_endpoint`.
- `get_config_fields()`: list of config fields for UI.
- `get_secret_fields()`: list of secret fields for UI.
- `authenticate()` and `load_documents()` methods.

## Entry Point Registration

In your plugin package `pyproject.toml`:

```toml
[project.entry-points."semantik.plugins"]
my_connector = "my_package.connector:MyConnector"
```

## Notes

- Only external plugins are listed under `/api/v2/plugins`.
- Connector discovery for UI remains via `/api/v2/connectors`.
- Plugin enable/disable requires a service restart to take effect.
