"""Pytest configuration for plugin testing tests."""

from __future__ import annotations

# Import all fixtures from the plugin testing module
# This makes them available to tests in this directory
pytest_plugins = ["shared.plugins.testing.fixtures"]
