#!/usr/bin/env python3
"""
Checkpoint management for resumable document processing.

This module handles saving and loading checkpoints to enable resuming
document processing after failures or interruptions.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiofiles  # type: ignore[import-untyped]
import aiofiles.os  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


@dataclass
class StreamingCheckpoint:
    """
    Checkpoint data for resuming stream processing.

    Contains all necessary state to resume processing from a specific point.
    """

    document_id: str
    file_path: str
    byte_position: int
    char_position: int
    chunks_processed: int
    total_chunks: int
    operation_id: str
    strategy_name: str
    timestamp: str
    metadata: dict[str, Any]

    # Processing state
    last_window_content: str = ""
    pending_bytes: bytes = b""
    processing_stats: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary for serialization."""
        data = asdict(self)
        # Convert bytes to base64 for JSON serialization
        if "pending_bytes" in data and data["pending_bytes"] is not None:
            import base64

            # Handle both non-empty and empty bytes
            if isinstance(data["pending_bytes"], bytes):
                data["pending_bytes"] = base64.b64encode(data["pending_bytes"]).decode("ascii")
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StreamingCheckpoint":
        """Create checkpoint from dictionary."""
        # Convert base64 back to bytes
        if "pending_bytes" in data:
            import base64

            if data["pending_bytes"]:
                data["pending_bytes"] = base64.b64decode(data["pending_bytes"])
            else:
                # Handle empty string (from empty bytes)
                data["pending_bytes"] = b""
        return cls(**data)


class CheckpointManager:
    """
    Manages checkpoints for streaming document processing.

    Handles saving, loading, and cleaning up checkpoints to enable
    resumable processing of large documents.
    """

    def __init__(self, checkpoint_dir: str = "/tmp/semantik_checkpoints"):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track active checkpoints
        self.active_checkpoints: dict[str, StreamingCheckpoint] = {}

        # Checkpoint interval (bytes)
        self.checkpoint_interval = 100 * 1024 * 1024  # 100MB

        # Maximum checkpoint age (hours)
        self.max_checkpoint_age = 24

    def _get_checkpoint_path(self, operation_id: str) -> Path:
        """
        Get the checkpoint file path for an operation.

        Args:
            operation_id: Unique operation identifier

        Returns:
            Path to checkpoint file

        Raises:
            ValueError: If operation_id contains invalid characters
        """
        # Sanitize operation_id to prevent path traversal attacks
        # Only allow alphanumeric characters, dash, and underscore
        if not re.match(r"^[a-zA-Z0-9_-]+$", operation_id):
            logger.error(f"Invalid operation_id: {operation_id}")
            raise ValueError(
                f"Invalid operation_id: '{operation_id}'. "
                "Only alphanumeric characters, dash, and underscore are allowed."
            )

        return self.checkpoint_dir / f"checkpoint_{operation_id}.json"

    async def save_checkpoint(
        self,
        operation_id: str,
        document_id: str,
        file_path: str,
        byte_position: int,
        char_position: int,
        chunks_processed: int,
        total_chunks: int,
        strategy_name: str,
        metadata: dict[str, Any] | None = None,
        last_window_content: str = "",
        pending_bytes: bytes = b"",
        processing_stats: dict[str, Any] | None = None,
    ) -> StreamingCheckpoint:
        """
        Save a checkpoint for the current processing state.

        Args:
            operation_id: Unique operation identifier
            document_id: Document being processed
            file_path: Path to the document file
            byte_position: Current byte position in file
            char_position: Current character position
            chunks_processed: Number of chunks processed so far
            total_chunks: Estimated total chunks
            strategy_name: Chunking strategy being used
            metadata: Additional metadata
            last_window_content: Content of last processing window
            pending_bytes: Unprocessed bytes from last read
            processing_stats: Processing statistics

        Returns:
            The saved checkpoint
        """
        checkpoint = StreamingCheckpoint(
            document_id=document_id,
            file_path=file_path,
            byte_position=byte_position,
            char_position=char_position,
            chunks_processed=chunks_processed,
            total_chunks=total_chunks,
            operation_id=operation_id,
            strategy_name=strategy_name,
            timestamp=datetime.now(tz=UTC).isoformat(),
            metadata=metadata or {},
            last_window_content=last_window_content,
            pending_bytes=pending_bytes,
            processing_stats=processing_stats or {},
        )

        # Save to disk using async I/O
        checkpoint_path = self._get_checkpoint_path(operation_id)

        try:
            async with aiofiles.open(checkpoint_path, "w") as f:
                await f.write(json.dumps(checkpoint.to_dict(), indent=2))
            logger.debug(f"Saved checkpoint for operation {operation_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint for operation {operation_id}: {e}")
            raise

        # Track in memory
        self.active_checkpoints[operation_id] = checkpoint

        return checkpoint

    async def load_checkpoint(self, operation_id: str) -> StreamingCheckpoint | None:
        """
        Load a checkpoint for an operation.

        Args:
            operation_id: Unique operation identifier

        Returns:
            The checkpoint if found, None otherwise
        """
        # Check memory cache first
        if operation_id in self.active_checkpoints:
            return self.active_checkpoints[operation_id]

        # Load from disk using async I/O
        checkpoint_path = self._get_checkpoint_path(operation_id)

        # Check existence asynchronously
        try:
            await aiofiles.os.stat(checkpoint_path)
        except FileNotFoundError:
            return None

        try:
            async with aiofiles.open(checkpoint_path, "r") as f:
                content = await f.read()
                data = json.loads(content)

            checkpoint = StreamingCheckpoint.from_dict(data)

            # Cache in memory
            self.active_checkpoints[operation_id] = checkpoint

            logger.debug(f"Loaded checkpoint for operation {operation_id}")
            return checkpoint
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # Log error but don't fail - treat as no checkpoint
            logger.warning(f"Failed to load checkpoint {operation_id}: {e}")
            return None

    async def delete_checkpoint(self, operation_id: str) -> bool:
        """
        Delete a checkpoint after successful completion.

        Args:
            operation_id: Unique operation identifier

        Returns:
            True if deleted, False if not found
        """
        # Remove from memory
        if operation_id in self.active_checkpoints:
            del self.active_checkpoints[operation_id]

        # Remove from disk using async I/O
        checkpoint_path = self._get_checkpoint_path(operation_id)

        try:
            await aiofiles.os.remove(checkpoint_path)
            logger.debug(f"Deleted checkpoint for operation {operation_id}")
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.error(f"Failed to delete checkpoint {operation_id}: {e}")
            return False

    async def cleanup_old_checkpoints(self) -> int:
        """
        Clean up old checkpoints that have expired.

        Returns:
            Number of checkpoints cleaned up
        """
        from datetime import UTC, datetime, timedelta

        cleanup_count = 0
        cutoff_time = datetime.now(tz=UTC) - timedelta(hours=self.max_checkpoint_age)

        # Scan checkpoint directory
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                # Check file modification time asynchronously
                stat = await aiofiles.os.stat(checkpoint_file)
                mtime = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
                if mtime < cutoff_time:
                    await aiofiles.os.remove(checkpoint_file)
                    cleanup_count += 1

                    # Remove from memory cache if present
                    operation_id = checkpoint_file.stem.replace("checkpoint_", "")
                    if operation_id in self.active_checkpoints:
                        del self.active_checkpoints[operation_id]

                    logger.debug(f"Cleaned up old checkpoint: {operation_id}")
            except OSError as e:
                # Skip files we can't access
                logger.warning(f"Could not clean up checkpoint {checkpoint_file}: {e}")
                continue

        return cleanup_count

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint summaries
        """
        checkpoints = []

        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                async with aiofiles.open(checkpoint_file, "r") as f:
                    content = await f.read()
                    data = json.loads(content)

                # Get file size asynchronously
                stat = await aiofiles.os.stat(checkpoint_file)

                # Create summary
                summary = {
                    "operation_id": data.get("operation_id"),
                    "document_id": data.get("document_id"),
                    "file_path": data.get("file_path"),
                    "byte_position": data.get("byte_position"),
                    "chunks_processed": data.get("chunks_processed"),
                    "timestamp": data.get("timestamp"),
                    "file_size": stat.st_size,
                }
                checkpoints.append(summary)
            except (json.JSONDecodeError, OSError, KeyError) as e:
                # Skip invalid checkpoints
                logger.warning(f"Skipping invalid checkpoint {checkpoint_file}: {e}")
                continue

        # Sort by timestamp (most recent first)
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        return checkpoints

    def should_checkpoint(self, bytes_processed: int, last_checkpoint_bytes: int) -> bool:
        """
        Determine if a checkpoint should be created.

        Args:
            bytes_processed: Total bytes processed so far
            last_checkpoint_bytes: Bytes processed at last checkpoint

        Returns:
            True if checkpoint should be created
        """
        return (bytes_processed - last_checkpoint_bytes) >= self.checkpoint_interval

    async def get_resume_position(self, operation_id: str) -> int | None:
        """
        Get the byte position to resume from.

        Args:
            operation_id: Unique operation identifier

        Returns:
            Byte position to resume from, or None if no checkpoint
        """
        checkpoint = await self.load_checkpoint(operation_id)
        if checkpoint:
            return checkpoint.byte_position
        return None

    def estimate_progress(self, checkpoint: StreamingCheckpoint) -> float:
        """
        Estimate processing progress from checkpoint.

        Args:
            checkpoint: The checkpoint to analyze

        Returns:
            Progress percentage (0.0 to 100.0)
        """
        if checkpoint.total_chunks > 0:
            return (checkpoint.chunks_processed / checkpoint.total_chunks) * 100.0
        return 0.0

    def __repr__(self) -> str:
        """String representation of the manager."""
        return f"CheckpointManager(dir='{self.checkpoint_dir}', active={len(self.active_checkpoints)})"
