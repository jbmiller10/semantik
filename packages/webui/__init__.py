"""
Document Embedding Web UI Package
"""

# Backward compatibility shim for old uvicorn command
# This allows "python -m uvicorn webui.app:app" to still work
try:
    from .main import app
except ImportError:
    # If main.py doesn't exist yet, try to import from app.py
    from .app import app

# Import Celery app
from .celery_app import celery_app

__all__ = ["app", "celery_app"]
