#!/usr/bin/env python3
"""
File change tracking functionality for maintenance service
Tracks files using SHA256 hashes and identifies removed files
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)


class FileChangeTracker:
    """Track file changes using SHA256 and SCD-like approach"""

    def __init__(self, db_path: str | None = None):
        """
        Initialize file change tracker

        Args:
            db_path: Path to JSON database file. If None, uses in-memory storage only.
        """
        self.db_path = db_path
        self.tracking_data: dict[str, Any] = {"files": {}, "metadata": {"created": datetime.now(UTC).isoformat()}}

        # Load existing data if db_path is provided
        if self.db_path:
            self._load()

    def _load(self) -> None:
        """Load tracking data from JSON file"""
        if not self.db_path:
            return

        db_file = Path(self.db_path)
        if db_file.exists():
            try:
                with db_file.open("r") as f:
                    self.tracking_data = json.load(f)
                logger.info(f"Loaded {len(self.tracking_data.get('files', {}))} tracked files from {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to load tracking data from {self.db_path}: {e}")
                # Keep default empty tracking data
        else:
            logger.info(f"No existing tracking database found at {self.db_path}, starting fresh")

    def save(self) -> None:
        """Save tracking data to JSON file"""
        if not self.db_path:
            logger.warning("No db_path specified, cannot save tracking data")
            return

        try:
            db_file = Path(self.db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            # Update metadata
            self.tracking_data.setdefault("metadata", {})
            self.tracking_data["metadata"]["last_updated"] = datetime.now(UTC).isoformat()

            with db_file.open("w") as f:
                json.dump(self.tracking_data, f, indent=2)
            logger.info(f"Saved tracking data to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save tracking data to {self.db_path}: {e}")

    def get_removed_documents(self, current_files: list[str]) -> list[dict[str, str]]:
        """
        Get list of documents that were tracked but are no longer in current files

        Args:
            current_files: List of currently existing file paths

        Returns:
            List of dicts with 'path' and 'doc_id' keys for removed documents
        """
        current_set = set(current_files)
        tracked_files = self.tracking_data.get("files", {})

        removed = []
        for file_path, file_info in tracked_files.items():
            if file_path not in current_set:
                removed.append(
                    {
                        "path": file_path,
                        "doc_id": file_info.get("doc_id", ""),
                    }
                )

        logger.info(f"Found {len(removed)} removed documents out of {len(tracked_files)} tracked files")
        return removed

    def remove_file(self, file_path: str) -> None:
        """
        Remove a file from tracking

        Args:
            file_path: Path of file to remove from tracking
        """
        files = self.tracking_data.get("files", {})
        if file_path in files:
            del files[file_path]
            logger.info(f"Removed {file_path} from tracking")
        else:
            logger.warning(f"File {file_path} not found in tracking data")

    def add_file(self, file_path: str, doc_id: str, file_hash: str) -> None:
        """
        Add or update a file in tracking

        Args:
            file_path: Path of file to track
            doc_id: Document ID for the file
            file_hash: SHA256 hash of the file
        """
        self.tracking_data.setdefault("files", {})[file_path] = {
            "doc_id": doc_id,
            "hash": file_hash,
            "last_seen": datetime.now(UTC).isoformat(),
        }
        logger.debug(f"Added/updated tracking for {file_path} with doc_id={doc_id}")

    def get_document_info(self, file_path: str) -> dict[str, str] | None:
        """
        Get tracking info for a specific document

        Args:
            file_path: Path of file to look up

        Returns:
            Dict with file info or None if not tracked
        """
        result = self.tracking_data.get("files", {}).get(file_path)
        if result is None or not isinstance(result, dict):
            return None
        # Type cast to satisfy mypy - we've verified it's a dict
        return cast(dict[str, str], result)

    def is_file_changed(self, file_path: str, new_hash: str) -> bool:
        """
        Check if a file has changed based on hash comparison

        Args:
            file_path: Path of file to check
            new_hash: New SHA256 hash to compare

        Returns:
            True if file has changed or is new, False otherwise
        """
        file_info = self.get_document_info(file_path)
        if not file_info:
            return True  # New file
        return file_info.get("hash") != new_hash
