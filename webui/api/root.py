"""
Root and static file routes for the Web UI
"""

import os

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["root"])


@router.get("/")
async def root():
    """Serve the main UI"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(base_dir, "static", "index.html"))


@router.get("/login")
async def login_page():
    """Serve the login page (SvelteKit handles routing)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(base_dir, "static", "index.html"))


@router.get("/settings")
async def settings_page():
    """Serve the settings page"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return FileResponse(os.path.join(base_dir, "static", "settings.html"))
