"""
API router package for Document Embedding Web UI
"""

from . import auth, files, jobs, metrics, models, root, search, settings

__all__ = ["auth", "jobs", "files", "metrics", "root", "settings", "models", "search"]
