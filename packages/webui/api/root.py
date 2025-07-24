"""
Root and static file routes for the Web UI
"""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["root"])


@router.get("/")
async def root() -> FileResponse:
    """Serve the main UI"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "index.html")


@router.get("/login")
async def login_page() -> FileResponse:
    """Serve the React app for login route"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "index.html")


@router.get("/settings")
async def settings_page() -> FileResponse:
    """Serve the React app for settings route"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "index.html")


@router.get("/vite.svg")
async def vite_svg() -> FileResponse:
    """Serve the Vite logo"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "vite.svg")


@router.get("/verification")
async def verification_page() -> FileResponse:
    """Serve the React app for verification route"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "index.html")


@router.get("/collections")
async def collections_list_page() -> FileResponse:
    """Serve the React app for collections list route"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "index.html")


@router.get("/collections/{collection_id}")
async def collection_detail_page(collection_id: str) -> FileResponse:  # noqa: ARG001
    """Serve the React app for collection detail route"""
    base_dir = Path(__file__).resolve().parent.parent
    return FileResponse(base_dir / "static" / "index.html")


# Removed catch-all route - it was interfering with static file serving
# Client-side routing is handled by explicit routes above
