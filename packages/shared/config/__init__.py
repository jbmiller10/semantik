# shared/config/__init__.py
"""
Configuration module for shared settings.
Provides backward compatibility by instantiating a default Settings object.
"""


from .base import BaseConfig
from .postgres import PostgresConfig, postgres_config
from .vecpipe import VecpipeConfig
from .webui import WebuiConfig


# For backward compatibility, create a unified Settings class that combines all configs
class Settings(VecpipeConfig, WebuiConfig):
    """
    Unified settings class that includes all configuration options.
    This maintains backward compatibility with the original shared.config.Settings
    """


# Instantiate settings once and export
settings = Settings()

# Expose in builtins for tests that refer to `settings` without import
try:  # pragma: no cover
    import builtins as _builtins

    _builtins.settings = settings  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    pass

# Export all config classes
__all__ = ["BaseConfig", "VecpipeConfig", "WebuiConfig", "PostgresConfig", "Settings", "settings", "postgres_config"]
