"""
Tests for embedding generation module
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from vecpipe.embed_chunks_parallel import (
    ParallelEmbeddingService,
    EmbeddingTask,
    read_parquet_async,
    write_parquet_async,
    process_file_async,
)


class TestParallelEmbeddingService:
    """Test parallel embedding service"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock sentence transformer model"""
        mock = MagicMock()
        mock.encode.return_value = np.random.rand(10, 1024).astype(np.float32)
        return mock
    
    @patch('vecpipe.embed_chunks_parallel.SentenceTransformer')
    def test_initialization(self, mock_st_class):
        """Test service initialization"""
        mock_st_class.return_value = Mock()
        
        service = ParallelEmbeddingService(
            model_name="test-model",
            device="cpu",
            batch_size=32
        )
        
        assert service.model_name == "test-model"
        assert service.device == "cpu"
        assert service.batch_size == 32
        assert service.gpu_thread is not None
        
        service.shutdown()
    
    @patch('vecpipe.embed_chunks_parallel.SentenceTransformer')
    def test_oom_handling(self, mock_st_class):
        """Test OOM error handling"""
        mock_model = Mock()
        
        # First call raises OOM, second succeeds
        oom_error = Exception("CUDA out of memory")
        oom_error.__class__ = type('OutOfMemoryError', (Exception,), {})
        oom_error.__class__.__module__ = 'torch.cuda'
        oom_error.__class__.__name__ = 'OutOfMemoryError'
        
        mock_model.encode.side_effect = [
            oom_error,
            np.random.rand(5, 1024).astype(np.float32)
        ]
        
        mock_st_class.return_value = mock_model
        
        service = ParallelEmbeddingService(batch_size=10)
        service.model = mock_model
        
        # Should succeed after reducing batch size
        embeddings = service._generate_embeddings_batch(["text"] * 5)
        
        assert embeddings.shape == (5, 1024)
        assert service.batch_size < 10  # Batch size should be reduced
        
        service.shutdown()
    
    @pytest.mark.asyncio
    async def test_async_embedding_generation(self):
        """Test async embedding generation"""
        with patch('vecpipe.embed_chunks_parallel.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.random.rand(3, 1024).astype(np.float32)
            mock_st.return_value = mock_model
            
            service = ParallelEmbeddingService()
            service.model = mock_model
            
            texts = ["text1", "text2", "text3"]
            embeddings = await service.generate_embeddings_async(texts)
            
            assert embeddings.shape == (3, 1024)
            
            service.shutdown()


class TestAsyncFileOperations:
    """Test async file operations"""
    
    @pytest.mark.asyncio
    async def test_read_parquet_async(self, tmp_path):
        """Test async parquet reading"""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create test parquet file
        test_file = tmp_path / "test.parquet"
        table = pa.table({
            'doc_id': ['doc1', 'doc2'],
            'chunk_id': ['chunk1', 'chunk2'],
            'path': ['/path1', '/path2'],
            'text': ['text1', 'text2']
        })
        pq.write_table(table, str(test_file))
        
        # Read async
        result = await read_parquet_async(str(test_file))
        
        assert result['doc_ids'] == ['doc1', 'doc2']
        assert result['chunk_ids'] == ['chunk1', 'chunk2']
        assert result['texts'] == ['text1', 'text2']
        assert result['file_path'] == str(test_file)
    
    @pytest.mark.asyncio
    async def test_write_parquet_async(self, tmp_path):
        """Test async parquet writing"""
        import pyarrow.parquet as pq
        
        output_file = tmp_path / "output.parquet"
        
        data = {
            'id': ['id1', 'id2'],
            'vector': [[1.0, 2.0], [3.0, 4.0]],
            'payload': [{'test': 1}, {'test': 2}]
        }
        
        await write_parquet_async(str(output_file), data)
        
        # Verify file was written
        assert output_file.exists()
        
        # Read back and verify
        table = pq.read_table(str(output_file))
        assert len(table) == 2
        assert table.column('id').to_pylist() == ['id1', 'id2']


class TestProcessingPipeline:
    """Test full processing pipeline"""
    
    @pytest.mark.asyncio
    async def test_process_file_async_skip_existing(self, tmp_path):
        """Test that existing files are skipped"""
        # Create existing output file
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing_file = output_dir / "test_embedded.parquet"
        existing_file.touch()
        
        with patch('vecpipe.embed_chunks_parallel.ParallelEmbeddingService'):
            service = Mock()
            
            result = await process_file_async(
                str(tmp_path / "test.parquet"),
                str(output_dir),
                service
            )
            
            assert result == str(existing_file)
            # Should not call embedding service
            service.generate_embeddings_async.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_process_file_async_error_handling(self, tmp_path):
        """Test error handling in file processing"""
        with patch('vecpipe.embed_chunks_parallel.read_parquet_async') as mock_read:
            mock_read.side_effect = Exception("Read error")
            
            service = Mock()
            
            result = await process_file_async(
                str(tmp_path / "test.parquet"),
                str(tmp_path),
                service
            )
            
            assert result is None  # Should return None on error


class TestBatchProcessing:
    """Test batch processing logic"""
    
    def test_batch_size_adjustment(self):
        """Test that batch size adjusts on OOM"""
        with patch('vecpipe.embed_chunks_parallel.SentenceTransformer') as mock_st:
            service = ParallelEmbeddingService(batch_size=100)
            
            # Simulate OOM on large batch
            def encode_with_oom(*args, batch_size=100, **kwargs):
                if batch_size > 50:
                    oom = Exception("CUDA out of memory")
                    oom.__class__.__name__ = 'OutOfMemoryError'
                    raise oom
                return np.random.rand(len(args[0]), 1024)
            
            service.model = Mock()
            service.model.encode.side_effect = encode_with_oom
            
            # Process large batch
            texts = ["text"] * 200
            result = service._generate_embeddings_batch(texts)
            
            assert result.shape == (200, 1024)
            assert service.batch_size <= 50
            
            service.shutdown()
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches"""
        with patch('vecpipe.embed_chunks_parallel.SentenceTransformer'):
            service = ParallelEmbeddingService()
            service.model = Mock()
            service.model.encode.return_value = np.array([])
            
            result = service._generate_embeddings_batch([])
            
            assert result.shape == (0,)
            
            service.shutdown()


# Integration tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring actual models"""
    
    @pytest.mark.skipif(True, reason="GPU not available")
    @pytest.mark.asyncio
    async def test_gpu_memory_management(self):
        """Test GPU memory is properly managed"""
        import torch
        
        initial_memory = torch.cuda.memory_allocated()
        
        service = ParallelEmbeddingService(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda"
        )
        
        # Generate embeddings
        texts = ["test text"] * 1000
        embeddings = await service.generate_embeddings_async(texts)
        
        assert embeddings.shape == (1000, 384)  # MiniLM produces 384d vectors
        
        service.shutdown()
        
        # Check memory is released
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should be mostly released
        assert final_memory < initial_memory + 100_000_000  # Allow 100MB overhead