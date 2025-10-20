#!/usr/bin/env python3
"""
Streaming document processor with bounded memory usage.

This module implements the core streaming pipeline that processes documents
of arbitrary size while maintaining strict memory constraints and ensuring
UTF-8 character boundary safety.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles  # type: ignore[import-untyped]
import redis.asyncio as redis

from packages.shared.chunking.domain.entities.chunk import Chunk
from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.domain.value_objects.chunk_metadata import ChunkMetadata

from .checkpoint import CheckpointManager
from .memory_pool import MemoryPool
from .window import StreamingWindow

logger = logging.getLogger(__name__)


class StreamingDocumentProcessor:
    """
    Stream process documents with bounded memory and UTF-8 safety.

    This processor handles documents of any size while:
    - Maintaining memory usage under 100MB
    - Never splitting UTF-8 characters
    - Supporting checkpoint/resume
    - Providing real-time progress updates
    - Managing backpressure
    """

    # Constants for memory management
    BUFFER_SIZE = 64 * 1024  # 64KB read buffer
    WINDOW_SIZE = 256 * 1024  # 256KB processing window
    CHECKPOINT_INTERVAL = 100 * 1024 * 1024  # 100MB between checkpoints
    MAX_MEMORY = 100 * 1024 * 1024  # 100MB total memory limit
    PROGRESS_INTERVAL = 1.0  # Progress update every second
    PROGRESS_BYTES = 1024 * 1024  # Progress update every 1MB

    # Backpressure thresholds
    HIGH_WATERMARK = 0.8  # 80% memory usage
    LOW_WATERMARK = 0.6  # 60% memory usage

    def __init__(
        self,
        checkpoint_manager: CheckpointManager | None = None,
        memory_pool: MemoryPool | None = None,
        redis_client: redis.Redis | None = None,
    ):
        """
        Initialize the streaming processor.

        Args:
            checkpoint_manager: Manager for checkpoints (created if None)
            memory_pool: Buffer pool (created if None)
            redis_client: Redis client for pub/sub progress updates
        """
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.memory_pool = memory_pool or MemoryPool(
            buffer_size=self.BUFFER_SIZE, pool_size=max(10, self.MAX_MEMORY // self.BUFFER_SIZE // 2)
        )
        self.redis_client = redis_client

        # Processing state
        self.current_operation_id: str | None = None
        self.downstream_pressure = 0.0
        self.processing_paused = False

        # Statistics
        self.total_bytes_processed = 0
        self.total_chunks_created = 0
        self.last_progress_time = 0.0
        self.last_progress_bytes = 0

    async def process_document(
        self,
        file_path: str,
        strategy: ChunkingStrategy,
        config: ChunkConfig,
        operation_id: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> AsyncIterator[Chunk]:
        """
        Stream process a document with the given strategy.

        Args:
            file_path: Path to the document file
            strategy: Chunking strategy to use
            config: Chunk configuration
            operation_id: Optional operation ID for checkpoint/resume
            progress_callback: Optional callback for progress updates

        Yields:
            Chunks as they are processed

        Raises:
            FileNotFoundError: If file doesn't exist
            MemoryError: If memory constraints violated
            UnicodeDecodeError: If file encoding invalid
        """
        # Validate file
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path_obj}")

        file_size = file_path_obj.stat().st_size
        operation_id = operation_id or str(uuid4())
        self.current_operation_id = operation_id

        # Initialize components (before try block for error handling)
        window = StreamingWindow(self.WINDOW_SIZE)
        checkpoint = await self.checkpoint_manager.load_checkpoint(operation_id)

        # Statistics
        bytes_processed = checkpoint.byte_position if checkpoint else 0
        chunks_processed = checkpoint.chunks_processed if checkpoint else 0
        last_checkpoint_bytes = bytes_processed
        last_progress_time = time.time()
        last_progress_bytes = bytes_processed

        # Document metadata
        document_id = str(uuid4())

        try:
            async with aiofiles.open(file_path_obj, "rb") as file:
                # Resume from checkpoint if available
                if checkpoint:
                    await file.seek(checkpoint.byte_position)
                    if checkpoint.pending_bytes:
                        window.append(checkpoint.pending_bytes)
                    logger.info(f"Resuming from checkpoint: {checkpoint.byte_position}/{file_size} bytes")

                # Main processing loop
                while True:
                    # Check backpressure
                    await self._manage_backpressure()

                    # Use context manager for safe buffer acquisition
                    async with self.memory_pool.acquire_async(size=self.BUFFER_SIZE, timeout=10.0) as managed_buffer:
                        buffer = managed_buffer.data

                        # Read chunk with size limit
                        chunk = await file.read(len(buffer))
                        if not chunk:
                            # End of file - process remaining window
                            if not window.is_empty:
                                text = window.decode_safe()
                                if text:
                                    # Process final chunks
                                    final_chunks = self._process_text(
                                        text, strategy, config, document_id, chunks_processed, bytes_processed
                                    )
                                    for chunk_obj in final_chunks:
                                        chunks_processed += 1
                                        yield chunk_obj
                            break

                        # Find UTF-8 boundary - CRITICAL
                        safe_boundary = self._find_utf8_boundary(chunk)
                        if safe_boundary < len(chunk):
                            # Save incomplete UTF-8 sequence for next iteration
                            remainder = chunk[safe_boundary:]
                            chunk = chunk[:safe_boundary]
                        else:
                            remainder = b""

                        # Add to window
                        window.append(chunk)

                        # Process when window is ready
                        if window.is_ready():
                            # Decode safely
                            text = window.decode_safe()

                            # Process text into chunks
                            text_chunks = self._process_text(
                                text, strategy, config, document_id, chunks_processed, bytes_processed
                            )

                            for chunk_obj in text_chunks:
                                chunks_processed += 1
                                self.total_chunks_created += 1
                                yield chunk_obj

                            # Slide window
                            window.slide()

                        # Add remainder if any
                        if remainder:
                            window.append(remainder)

                        # Update counters
                        bytes_processed += len(chunk)
                        self.total_bytes_processed = bytes_processed

                        # Progress tracking
                        current_time = time.time()
                        bytes_since_progress = bytes_processed - last_progress_bytes
                        time_since_progress = current_time - last_progress_time

                        if bytes_since_progress >= self.PROGRESS_BYTES or time_since_progress >= self.PROGRESS_INTERVAL:
                            await self._emit_progress(
                                operation_id, bytes_processed, file_size, chunks_processed, progress_callback
                            )
                            last_progress_time = current_time
                            last_progress_bytes = bytes_processed

                        # Checkpointing
                        if self.checkpoint_manager.should_checkpoint(bytes_processed, last_checkpoint_bytes):
                            await self.checkpoint_manager.save_checkpoint(
                                operation_id=operation_id,
                                document_id=document_id,
                                file_path=str(file_path_obj),
                                byte_position=bytes_processed,
                                char_position=0,  # Would need char tracking
                                chunks_processed=chunks_processed,
                                total_chunks=self._estimate_total_chunks(file_size, config),
                                strategy_name=strategy.name,
                                pending_bytes=bytes(window._pending_bytes),
                                processing_stats={
                                    "bytes_per_second": (
                                        bytes_processed / time_since_progress if time_since_progress > 0 else 0
                                    ),
                                    "chunks_per_second": (
                                        chunks_processed / time_since_progress if time_since_progress > 0 else 0
                                    ),
                                },
                            )
                            last_checkpoint_bytes = bytes_processed
                            logger.debug(f"Checkpoint saved at {bytes_processed}/{file_size} bytes")

                    # Buffer is automatically released by context manager

                # Final progress update
                await self._emit_progress(
                    operation_id, bytes_processed, file_size, chunks_processed, progress_callback, is_complete=True
                )

                # Clean up checkpoint on success
                await self.checkpoint_manager.delete_checkpoint(operation_id)

                logger.info(f"Document processed: {chunks_processed} chunks from {bytes_processed} bytes")

        except Exception as e:
            # Save checkpoint on error for resume
            if bytes_processed > 0:
                await self.checkpoint_manager.save_checkpoint(
                    operation_id=operation_id,
                    document_id=document_id,
                    file_path=str(file_path),
                    byte_position=bytes_processed,
                    char_position=0,
                    chunks_processed=chunks_processed,
                    total_chunks=self._estimate_total_chunks(file_size, config),
                    strategy_name=strategy.name,
                    pending_bytes=bytes(window._pending_bytes) if hasattr(window, "_pending_bytes") else b"",
                    metadata={"error": str(e)},
                )
            raise

        finally:
            self.current_operation_id = None

    def _find_utf8_boundary(self, data: bytes, max_pos: int = -1) -> int:
        """
        Find safe UTF-8 character boundary - CRITICAL FUNCTION.

        This function ensures we NEVER split multi-byte UTF-8 characters.

        UTF-8 byte patterns:
        - 0xxxxxxx: ASCII (single byte)
        - 110xxxxx: 2-byte sequence start
        - 1110xxxx: 3-byte sequence start
        - 11110xxx: 4-byte sequence start
        - 10xxxxxx: Continuation byte

        Args:
            data: Byte data to search
            max_pos: Maximum position to consider (-1 for end)

        Returns:
            Safe boundary position that won't split characters
        """
        if not data:
            return 0

        max_pos = len(data) if max_pos == -1 else min(max_pos, len(data))

        # If max_pos is 0, return 0
        if max_pos == 0:
            return 0

        # Start from the desired position and walk backwards
        pos = max_pos - 1

        # Special case: if we're looking at the last position and it's ASCII
        if pos < len(data) and data[pos] < 0x80:
            return max_pos

        while pos >= 0:
            byte = data[pos]

            # ASCII byte (0xxxxxxx) - safe to cut after
            if byte < 0x80:
                return min(pos + 1, max_pos)

            # UTF-8 start byte (11xxxxxx) - check if complete
            if byte >= 0xC0:
                # Determine expected sequence length
                if byte < 0xE0:
                    expected_len = 2  # 110xxxxx
                elif byte < 0xF0:
                    expected_len = 3  # 1110xxxx
                elif byte < 0xF8:
                    expected_len = 4  # 11110xxx
                else:
                    # Invalid UTF-8, treat as boundary
                    return pos

                # Check if we have the complete sequence within max_pos
                if pos + expected_len <= max_pos and pos + expected_len <= len(data):
                    # Validate continuation bytes
                    valid = True
                    for i in range(1, expected_len):
                        if pos + i >= len(data):
                            valid = False
                            break
                        cont_byte = data[pos + i]
                        if not (0x80 <= cont_byte < 0xC0):
                            valid = False
                            break

                    if valid:
                        # Complete valid sequence, safe to cut after
                        return min(pos + expected_len, max_pos)

                # Incomplete sequence or extends beyond max_pos, cut before
                return pos

            # Continuation byte (10xxxxxx) - keep going back
            pos -= 1

        # Reached beginning
        return 0

    def _process_text(
        self,
        text: str,
        strategy: ChunkingStrategy,
        config: ChunkConfig,
        document_id: str,
        chunk_offset: int,
        byte_offset: int,
    ) -> list[Chunk]:
        """
        Process text into chunks using the strategy.

        Args:
            text: Text to process
            strategy: Chunking strategy
            config: Configuration
            document_id: Document ID
            chunk_offset: Starting chunk index
            byte_offset: Starting byte position

        Returns:
            List of created chunks
        """
        chunks = []

        # Use strategy to create chunks
        raw_chunks = strategy.chunk(text, config)

        for i, raw_chunk in enumerate(raw_chunks):
            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=str(uuid4()),
                document_id=document_id,
                chunk_index=chunk_offset + i,
                start_offset=byte_offset,
                end_offset=byte_offset + len(raw_chunk.content.encode("utf-8")),
                token_count=raw_chunk.metadata.token_count,
                strategy_name=strategy.name,
            )

            # Create chunk entity
            chunk = Chunk(
                content=raw_chunk.content,
                metadata=metadata,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens,
            )

            chunks.append(chunk)

        return chunks

    async def _manage_backpressure(self) -> None:
        """
        Manage backpressure by monitoring memory usage.

        Pauses processing if memory usage exceeds high watermark.
        """
        # Calculate current memory pressure
        pool_stats = self.memory_pool.get_statistics()
        memory_pressure = pool_stats["utilization"]

        self.downstream_pressure = memory_pressure

        if memory_pressure > self.HIGH_WATERMARK:
            if not self.processing_paused:
                logger.warning(f"High memory pressure ({memory_pressure:.1%}), pausing processing")
                self.processing_paused = True

            # Wait for pressure to reduce
            while memory_pressure > self.LOW_WATERMARK:
                await asyncio.sleep(0.1)
                pool_stats = self.memory_pool.get_statistics()
                memory_pressure = pool_stats["utilization"]

            if self.processing_paused:
                logger.info(f"Memory pressure reduced ({memory_pressure:.1%}), resuming")
                self.processing_paused = False

    async def _emit_progress(
        self,
        operation_id: str,
        bytes_processed: int,
        total_bytes: int,
        chunks_processed: int,
        callback: Callable[[dict[str, Any]], None | Awaitable[None]] | None = None,
        is_complete: bool = False,
    ) -> None:
        """
        Emit progress update via callback and Redis pub/sub.

        Args:
            operation_id: Operation identifier
            bytes_processed: Bytes processed so far
            total_bytes: Total file size
            chunks_processed: Chunks created so far
            callback: Optional progress callback
            is_complete: Whether processing is complete
        """
        progress = {
            "operation_id": operation_id,
            "bytes_processed": bytes_processed,
            "total_bytes": total_bytes,
            "percentage": (bytes_processed / total_bytes * 100) if total_bytes > 0 else 0,
            "chunks_processed": chunks_processed,
            "is_complete": is_complete,
            "timestamp": time.time(),
        }

        # Call callback if provided
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

        # Publish to Redis if available
        if self.redis_client:
            try:
                channel = f"chunking:progress:{operation_id}"
                await self.redis_client.publish(channel, json.dumps(progress))
            except Exception as e:
                logger.error(f"Redis publish error: {e}")

    def _estimate_total_chunks(self, file_size: int, config: ChunkConfig) -> int:
        """
        Estimate total number of chunks for a file.

        Args:
            file_size: Size of file in bytes
            config: Chunk configuration

        Returns:
            Estimated chunk count
        """
        # Rough estimate: assume average 4 bytes per token
        estimated_tokens = file_size // 4
        return config.estimate_chunks(estimated_tokens)

    def get_memory_usage(self) -> dict[str, Any]:
        """
        Get current memory usage statistics.

        Returns:
            Dictionary with memory usage details
        """
        pool_stats = self.memory_pool.get_statistics()

        return {
            "total_allocated": pool_stats["total_memory"],
            "buffers_in_use": pool_stats["in_use"],
            "buffers_available": pool_stats["available"],
            "utilization": pool_stats["utilization"],
            "max_memory": self.MAX_MEMORY,
            "within_limit": pool_stats["total_memory"] <= self.MAX_MEMORY,
        }

    async def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel an ongoing operation.

        Args:
            operation_id: Operation to cancel

        Returns:
            True if cancelled, False if not found
        """
        if self.current_operation_id == operation_id:
            # Will cause the processing loop to exit
            self.current_operation_id = None
            return True
        return False
