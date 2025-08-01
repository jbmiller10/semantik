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

# Export all config classes
__all__ = ["BaseConfig", "VecpipeConfig", "WebuiConfig", "PostgresConfig", "Settings", "settings", "postgres_config"]
