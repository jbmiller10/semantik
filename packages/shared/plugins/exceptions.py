"""Plugin-specific exceptions."""


class PluginError(Exception):
    """Base error for plugin-related issues."""


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""


class PluginConfigError(PluginError):
    """Raised when plugin configuration is invalid."""


class PluginCompatibilityError(PluginError):
    """Raised when a plugin is incompatible with the current Semantik version."""
