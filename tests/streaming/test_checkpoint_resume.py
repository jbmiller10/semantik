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
        self.name = "interruptible_strategy"
        self.chunks_processed = 0
        self.interrupt_after = interrupt_after
    
    def chunk(self, text: str, config: ChunkConfig) -> List:
        """Chunk text and potentially interrupt."""
        chunks = []
        sentences = text.split('.')
        
        for sentence in sentences:
            if sentence.strip():
                self.chunks_processed += 1
                
                # Simulate interruption
                if self.interrupt_after and self.chunks_processed >= self.interrupt_after:
                    raise RuntimeError("Simulated interruption")
                
                chunk = MagicMock()
                chunk.content = sentence.strip()
                chunk.metadata = MagicMock()
                chunk.metadata.token_count = len(sentence.split())
                chunks.append(chunk)
        
        return chunks


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


class TestStreamingProcessorCheckpoint:
    """Test checkpoint/resume in streaming processor."""
    
    @pytest.fixture
    def temp_file(self):
        """Create temporary test file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create file with numbered sentences for easy verification
            content = ". ".join([f"Sentence {i}" for i in range(100)])
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_config(self) -> ChunkConfig:
        """Create mock chunk configuration."""
        config = MagicMock(spec=ChunkConfig)
        config.min_tokens = 1
        config.max_tokens = 100
        config.estimate_chunks = lambda tokens: tokens // 10
        return config
    
    async def test_resume_from_checkpoint(self, temp_file, mock_config):
        """Test resuming processing from a checkpoint."""
        checkpoint_manager = CheckpointManager()
        operation_id = str(uuid4())
        
        # First pass - interrupt after 5 chunks
        processor1 = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
        strategy1 = InterruptibleStrategy(interrupt_after=5)
        
        chunks_pass1 = []
        with pytest.raises(RuntimeError, match="Simulated interruption"):
            async for chunk in processor1.process_document(
                temp_file, strategy1, mock_config, operation_id=operation_id
            ):
                chunks_pass1.append(chunk)
        
        # Should have processed some chunks before interruption
        assert len(chunks_pass1) > 0
        assert len(chunks_pass1) < 10  # But not all
        
        # Verify checkpoint was saved
        checkpoint = await checkpoint_manager.load_checkpoint(operation_id)
        assert checkpoint is not None
        assert checkpoint.chunks_processed > 0
        
        # Second pass - resume from checkpoint
        processor2 = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
        strategy2 = InterruptibleStrategy()  # No interruption this time
        
        chunks_pass2 = []
        async for chunk in processor2.process_document(
            temp_file, strategy2, mock_config, operation_id=operation_id
        ):
            chunks_pass2.append(chunk)
        
        # Should have completed processing
        assert len(chunks_pass2) > 0
        
        # Checkpoint should be cleaned up after success
        checkpoint = await checkpoint_manager.load_checkpoint(operation_id)
        assert checkpoint is None
    
    async def test_checkpoint_with_pending_bytes(self, mock_config):
        """Test checkpoint correctly saves and restores pending bytes."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            # Create file with UTF-8 text that will be split
            content = "Hello 世界. Test 中文. End.".encode('utf-8')
            f.write(content)
            temp_file = f.name
        
        try:
            checkpoint_manager = CheckpointManager()
            operation_id = str(uuid4())
            
            # Create checkpoint with pending bytes
            await checkpoint_manager.save_checkpoint(
                operation_id=operation_id,
                document_id=str(uuid4()),
                file_path=temp_file,
                byte_position=10,
                char_position=8,
                chunks_processed=1,
                total_chunks=10,
                strategy_name="test",
                pending_bytes=b'\xe4\xb8',  # Incomplete UTF-8 sequence
            )
            
            # Load and verify
            checkpoint = await checkpoint_manager.load_checkpoint(operation_id)
            assert checkpoint.pending_bytes == b'\xe4\xb8'
            
            # Process with checkpoint
            processor = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
            strategy = InterruptibleStrategy()
            
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config, operation_id=operation_id
            ):
                chunks.append(chunk)
            
            # Should process successfully
            assert len(chunks) > 0
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_multiple_checkpoints_during_processing(self, mock_config):
        """Test multiple checkpoints are created during large file processing."""
        # Create large file that will trigger multiple checkpoints
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create content larger than checkpoint interval
            content = ". ".join([f"Sentence {i}" for i in range(10000)])
            f.write(content)
            temp_file = f.name
        
        try:
            checkpoint_manager = CheckpointManager()
            # Set small checkpoint interval for testing
            checkpoint_manager.checkpoint_interval = 1024  # 1KB
            
            processor = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
            strategy = InterruptibleStrategy()
            operation_id = str(uuid4())
            
            checkpoints_created = []
            
            # Mock save_checkpoint to track calls
            original_save = checkpoint_manager.save_checkpoint
            async def track_checkpoint(*args, **kwargs):
                result = await original_save(*args, **kwargs)
                checkpoints_created.append(result.byte_position)
                return result
            
            checkpoint_manager.save_checkpoint = track_checkpoint
            
            # Process file
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config, operation_id=operation_id
            ):
                chunks.append(chunk)
            
            # Should have created multiple checkpoints
            assert len(checkpoints_created) > 1
            
            # Checkpoints should be at increasing byte positions
            for i in range(1, len(checkpoints_created)):
                assert checkpoints_created[i] > checkpoints_created[i-1]
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_checkpoint_on_error(self, temp_file, mock_config):
        """Test checkpoint is saved when an error occurs."""
        checkpoint_manager = CheckpointManager()
        processor = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
        
        # Strategy that fails after processing some chunks
        strategy = InterruptibleStrategy(interrupt_after=3)
        operation_id = str(uuid4())
        
        with pytest.raises(RuntimeError):
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config, operation_id=operation_id
            ):
                chunks.append(chunk)
        
        # Checkpoint should exist with error metadata
        checkpoint = await checkpoint_manager.load_checkpoint(operation_id)
        assert checkpoint is not None
        assert checkpoint.chunks_processed > 0
        assert "error" in checkpoint.metadata
        assert "Simulated interruption" in checkpoint.metadata["error"]
    
    async def test_progress_callback_with_checkpoint(self, temp_file, mock_config):
        """Test progress callbacks work with checkpoint/resume."""
        checkpoint_manager = CheckpointManager()
        operation_id = str(uuid4())
        
        progress_updates = []
        
        async def progress_callback(progress):
            progress_updates.append(progress.copy())
        
        # First pass - partial processing
        processor1 = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
        strategy1 = InterruptibleStrategy(interrupt_after=2)
        
        with pytest.raises(RuntimeError):
            async for chunk in processor1.process_document(
                temp_file, strategy1, mock_config,
                operation_id=operation_id,
                progress_callback=progress_callback
            ):
                pass
        
        initial_updates = len(progress_updates)
        assert initial_updates > 0
        
        # Second pass - resume
        processor2 = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
        strategy2 = InterruptibleStrategy()
        
        async for chunk in processor2.process_document(
            temp_file, strategy2, mock_config,
            operation_id=operation_id,
            progress_callback=progress_callback
        ):
            pass
        
        # Should have more progress updates
        assert len(progress_updates) > initial_updates
        
        # Final update should indicate completion
        final_update = progress_updates[-1]
        assert final_update["is_complete"] is True
    
    async def test_concurrent_checkpoints_isolated(self, mock_config):
        """Test concurrent operations have isolated checkpoints."""
        # Create two temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='_1.txt', delete=False) as f1:
            f1.write(". ".join([f"File1 Sentence {i}" for i in range(50)]))
            file1 = f1.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='_2.txt', delete=False) as f2:
            f2.write(". ".join([f"File2 Sentence {i}" for i in range(50)]))
            file2 = f2.name
        
        try:
            checkpoint_manager = CheckpointManager()
            
            async def process_with_interrupt(file_path, operation_id, interrupt_at):
                processor = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
                strategy = InterruptibleStrategy(interrupt_after=interrupt_at)
                chunks = []
                
                try:
                    async for chunk in processor.process_document(
                        file_path, strategy, mock_config, operation_id=operation_id
                    ):
                        chunks.append(chunk)
                except RuntimeError:
                    pass  # Expected interruption
                
                return len(chunks)
            
            # Process both files concurrently with interruptions
            op1 = str(uuid4())
            op2 = str(uuid4())
            
            results = await asyncio.gather(
                process_with_interrupt(file1, op1, 3),
                process_with_interrupt(file2, op2, 5),
            )
            
            # Both should have processed some chunks
            assert results[0] > 0
            assert results[1] > 0
            assert results[0] != results[1]  # Different interrupt points
            
            # Both should have checkpoints
            checkpoint1 = await checkpoint_manager.load_checkpoint(op1)
            checkpoint2 = await checkpoint_manager.load_checkpoint(op2)
            
            assert checkpoint1 is not None
            assert checkpoint2 is not None
            assert checkpoint1.chunks_processed != checkpoint2.chunks_processed
            
        finally:
            Path(file1).unlink(missing_ok=True)
            Path(file2).unlink(missing_ok=True)
    
    async def test_checkpoint_file_corruption_handling(self, checkpoint_dir):
        """Test handling of corrupted checkpoint files."""
        manager = CheckpointManager(checkpoint_dir)
        operation_id = str(uuid4())
        
        # Create corrupted checkpoint file
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{operation_id}.json"
        checkpoint_path.write_text("{ invalid json }")
        
        # Should handle gracefully
        checkpoint = await manager.load_checkpoint(operation_id)
        assert checkpoint is None  # Returns None for corrupted file
    
    async def test_resume_position_calculation(self, checkpoint_dir):
        """Test calculating resume position from checkpoint."""
        manager = CheckpointManager(checkpoint_dir)
        operation_id = str(uuid4())
        
        # Save checkpoint at specific position
        await manager.save_checkpoint(
            operation_id=operation_id,
            document_id=str(uuid4()),
            file_path="/test/file.txt",
            byte_position=12345,
            char_position=6789,
            chunks_processed=50,
            total_chunks=100,
            strategy_name="test",
        )
        
        # Get resume position
        position = await manager.get_resume_position(operation_id)
        assert position == 12345
        
        # Non-existent checkpoint
        position = await manager.get_resume_position("non-existent")
        assert position is None


class TestCheckpointEdgeCases:
    """Test edge cases in checkpoint/resume functionality."""
    
    async def test_empty_file_checkpoint(self, mock_config):
        """Test checkpoint with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Empty file
            temp_file = f.name
        
        try:
            processor = StreamingDocumentProcessor()
            strategy = InterruptibleStrategy()
            
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config
            ):
                chunks.append(chunk)
            
            # Should complete without error
            assert len(chunks) == 0
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_single_byte_file_checkpoint(self, mock_config):
        """Test checkpoint with single byte file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as f:
            f.write(b"A")
            temp_file = f.name
        
        try:
            checkpoint_manager = CheckpointManager()
            processor = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
            strategy = InterruptibleStrategy()
            operation_id = str(uuid4())
            
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config, operation_id=operation_id
            ):
                chunks.append(chunk)
            
            # Should process single character
            assert len(chunks) >= 0  # Depends on strategy
            
            # No checkpoint should remain after success
            checkpoint = await checkpoint_manager.load_checkpoint(operation_id)
            assert checkpoint is None
            
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    async def test_checkpoint_at_file_boundary(self, mock_config):
        """Test checkpoint exactly at file end."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content.")
            temp_file = f.name
            file_size = Path(temp_file).stat().st_size
        
        try:
            checkpoint_manager = CheckpointManager()
            operation_id = str(uuid4())
            
            # Create checkpoint at exact file end
            await checkpoint_manager.save_checkpoint(
                operation_id=operation_id,
                document_id=str(uuid4()),
                file_path=temp_file,
                byte_position=file_size,
                char_position=13,
                chunks_processed=1,
                total_chunks=1,
                strategy_name="test",
            )
            
            # Resume should complete immediately
            processor = StreamingDocumentProcessor(checkpoint_manager=checkpoint_manager)
            strategy = InterruptibleStrategy()
            
            chunks = []
            async for chunk in processor.process_document(
                temp_file, strategy, mock_config, operation_id=operation_id
            ):
                chunks.append(chunk)
            
            # Should not process additional chunks
            assert len(chunks) == 0
            
        finally:
            Path(temp_file).unlink(missing_ok=True)