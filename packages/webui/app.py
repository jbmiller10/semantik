"""
Backward compatibility shim for webui.app module
This file ensures that existing scripts using "webui.app:app" continue to work
"""

# Use absolute import which works with both poetry and direct python
from .main import app

__all__ = ["app"]
