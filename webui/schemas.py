"""
Shared Pydantic models for the Web UI
"""

from pydantic import BaseModel


class FileInfo(BaseModel):
    path: str
    size: int
    modified: str
    extension: str
    hash: str | None = None
