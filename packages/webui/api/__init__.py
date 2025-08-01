"""
API router package for Document Embedding Web UI
"""

from . import auth, health, internal, metrics, models, root, settings

__all__ = ["auth", "health", "internal", "metrics", "root", "settings", "models"]
