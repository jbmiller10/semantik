"""Shared library initialization.

This module adds the plugins directory to sys.path AFTER site-packages,
ensuring that app packages take precedence over plugin packages.
"""

import os
import sys
from pathlib import Path

# Add plugins directory to sys.path if it exists and isn't already there
# This is done here (not via PYTHONPATH) so plugins are searched AFTER site-packages
_plugins_dir = os.environ.get("SEMANTIK_PLUGINS_DIR", "/app/plugins")
if Path(_plugins_dir).is_dir() and _plugins_dir not in sys.path:
    sys.path.append(_plugins_dir)
