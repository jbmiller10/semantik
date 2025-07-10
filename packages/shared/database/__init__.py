"""
Shared database utilities and abstractions.

This module provides database-related functionality that's shared across
the vecpipe and webui packages.
"""

from .collection_metadata import ensure_metadata_collection, get_collection_metadata, store_collection_metadata

__all__ = [
    "ensure_metadata_collection",
    "get_collection_metadata",
    "store_collection_metadata",
]
