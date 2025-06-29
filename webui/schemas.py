"""
Shared Pydantic models for the Web UI
"""

from typing import Optional
from pydantic import BaseModel

class FileInfo(BaseModel):
    path: str
    size: int
    modified: str
    extension: str
    hash: Optional[str] = None