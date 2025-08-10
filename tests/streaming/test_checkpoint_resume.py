#!/usr/bin/env python3
"""
Test checkpoint and resume functionality in streaming processor.

This module tests the ability to checkpoint processing state and resume
from failures or interruptions.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.infrastructure.streaming.checkpoint import (
    CheckpointManager,
    StreamingCheckpoint,
)
from packages.shared.chunking.infrastructure.streaming.processor import StreamingDocumentProcessor


class InterruptibleStrategy(ChunkingStrategy):
    """Chunking strategy that can be interrupted for testing."""
    
    def __init__(self, interrupt_after: Optional[int] = None):
        super().__init__("interruptible_strategy")  # Pass name to parent constructor
        self.chunks_processed = 0
        self.interrupt_after = interrupt_after
        self.total_chunks_returned = 0
    
    def chunk(self, text: str, config: ChunkConfig) -> List:
        """Chunk text and potentially interrupt."""
        chunks = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if sentence.strip():
                self.chunks_processed += 1
                
                chunk = MagicMock()
                chunk.content = sentence.strip()
                chunk.metadata = MagicMock()
                chunk.metadata.token_count = len(sentence.split())
                chunks.append(chunk)
                
                # Track total chunks we're about to return
                if self.interrupt_after and (self.total_chunks_returned + len(chunks)) >= self.interrupt_after:
                    # Return the chunks we have so far, then interrupt
                    self.total_chunks_returned += len(chunks)
                    # Truncate to exactly the number we want before interrupting
                    chunks_to_return = self.interrupt_after - (self.total_chunks_returned - len(chunks))
                    if chunks_to_return > 0:
                        return chunks[:chunks_to_return]
                    raise RuntimeError("Simulated interruption")
        
        self.total_chunks_returned += len(chunks)
        
        # Check if we should interrupt after returning these chunks
        if self.interrupt_after and self.total_chunks_returned >= self.interrupt_after:
            raise RuntimeError("Simulated interruption")
            
        return chunks
    
    def validate_content(self, content: str) -> tuple[bool, str | None]:
        """Validate content for testing."""
        if not content:
            return False, "Content is empty"
        return True, None
    
    def estimate_chunks(self, content_length: int, config: ChunkConfig) -> int:
        """Estimate number of chunks for testing."""
        # Rough estimate: one chunk per 100 characters
        return max(1, content_length // 100)


class TestCheckpointManager:
    """Test suite for checkpoint manager."""
    
    @pytest.fixture
    def checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    async def test_save_and_load_checkpoint(self, checkpoint_dir):
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(checkpoint_dir)
        
        operation_id = str(uuid4())
        document_id = str(uuid4())
        
        # Save checkpoint
        checkpoint = await manager.save_checkpoint(
            operation_id=operation_id,
            document_id=document_id,
            file_path="/test/file.txt",
            byte_position=1024,
            char_position=500,
            chunks_processed=10,
            total_chunks=100,
            strategy_name="test_strategy",
            metadata={"test": "data"},
            pending_bytes=b"incomplete",
            processing_stats={"speed": 1000},
        )
        
        assert checkpoint.operation_id == operation_id
        assert checkpoint.byte_position == 1024
        assert checkpoint.chunks_processed == 10
        
        # Load checkpoint
        loaded = await manager.load_checkpoint(operation_id)
        assert loaded is not None
        assert loaded.operation_id == operation_id
        assert loaded.byte_position == 1024
        assert loaded.chunks_processed == 10
        assert loaded.pending_bytes == b"incomplete"
        assert loaded.metadata["test"] == "data"
    
    async def test_delete_checkpoint(self, checkpoint_dir):
        """Test deleting checkpoints."""
        manager = CheckpointManager(checkpoint_dir)
        
        operation_id = str(uuid4())
        
        # Save checkpoint
        await manager.save_checkpoint(
            operation_id=operation_id,
            document_id=str(uuid4()),
            file_path="/test/file.txt",
            byte_position=1024,
            char_position=500,
            chunks_processed=10,
            total_chunks=100,
            strategy_name="test_strategy",
        )
        
        # Verify it exists
        loaded = await manager.load_checkpoint(operation_id)
        assert loaded is not None
        
        # Delete it
        deleted = await manager.delete_checkpoint(operation_id)
        assert deleted is True
        
        # Verify it's gone
        loaded = await manager.load_checkpoint(operation_id)
        assert loaded is None
    
    async def test_checkpoint_interval_logic(self, checkpoint_dir):
        """Test checkpoint interval determination."""
        manager = CheckpointManager(checkpoint_dir)
        
        # Default interval is 100MB
        assert manager.checkpoint_interval == 100 * 1024 * 1024
        
        # Should checkpoint after interval
        assert manager.should_checkpoint(100 * 1024 * 1024, 0) is True
        assert manager.should_checkpoint(50 * 1024 * 1024, 0) is False
        assert manager.should_checkpoint(150 * 1024 * 1024, 50 * 1024 * 1024) is True
    
    async def test_list_checkpoints(self, checkpoint_dir):
        """Test listing all checkpoints."""
        manager = CheckpointManager(checkpoint_dir)
        
        # Create multiple checkpoints
        operation_ids = [str(uuid4()) for _ in range(3)]
        
        for i, op_id in enumerate(operation_ids):
            await manager.save_checkpoint(
                operation_id=op_id,
                document_id=str(uuid4()),
                file_path=f"/test/file{i}.txt",
                byte_position=1024 * (i + 1),
                char_position=500 * (i + 1),
                chunks_processed=10 * (i + 1),
                total_chunks=100,
                strategy_name="test_strategy",
            )
        
        # List checkpoints
        checkpoints = await manager.list_checkpoints()
        assert len(checkpoints) == 3
        
        # Verify all operation IDs present
        listed_ids = [cp["operation_id"] for cp in checkpoints]
        assert all(op_id in listed_ids for op_id in operation_ids)
    
    async def test_cleanup_old_checkpoints(self, checkpoint_dir):
        """Test cleanup of old checkpoints."""
        manager = CheckpointManager(checkpoint_dir)
        manager.max_checkpoint_age = 0  # Set to 0 hours for immediate cleanup
        
        # Create checkpoint
        operation_id = str(uuid4())
        await manager.save_checkpoint(
            operation_id=operation_id,
            document_id=str(uuid4()),
            file_path="/test/file.txt",
            byte_position=1024,
            char_position=500,
            chunks_processed=10,
            total_chunks=100,
            strategy_name="test_strategy",
        )
        
        # Wait a moment
        await asyncio.sleep(0.1)
        
        # Cleanup
        cleaned = await manager.cleanup_old_checkpoints()
        assert cleaned == 1
        
        # Verify it's gone
        loaded = await manager.load_checkpoint(operation_id)
        assert loaded is None
    
    async def test_estimate_progress(self, checkpoint_dir):
        """Test progress estimation from checkpoint."""
        manager = CheckpointManager(checkpoint_dir)
        
        checkpoint = StreamingCheckpoint(
            document_id=str(uuid4()),
            file_path="/test/file.txt",
            byte_position=5000,
            char_position=2500,
            chunks_processed=50,
            total_chunks=100,
            operation_id=str(uuid4()),
            strategy_name="test",
            timestamp="2024-01-01T00:00:00",
            metadata={},
        )
        
        progress = manager.estimate_progress(checkpoint)
        assert progress == 50.0  # 50/100 = 50%

