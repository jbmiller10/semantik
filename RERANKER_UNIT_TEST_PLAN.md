# Unit Testing Implementation Plan for CrossEncoderReranker

## Overview
This document provides a comprehensive plan for implementing unit tests for the `CrossEncoderReranker` class located in `packages/vecpipe/reranker.py`. The reranker uses Qwen3-Reranker models to perform cross-encoder reranking for improved search relevance.

## Context and Background

### What is the CrossEncoderReranker?
The `CrossEncoderReranker` is a class that implements two-stage retrieval reranking using Qwen3-Reranker models. It:
- Takes a query and a list of documents
- Uses a cross-encoder model to score each query-document pair
- Returns the top-k documents sorted by relevance score
- Uses special "yes/no" token prediction for relevance scoring

### Key Implementation Details
1. **Models**: Supports Qwen3-Reranker-0.6B, 4B, and 8B variants
2. **Quantization**: Supports float32, float16, and int8 quantization
3. **Device**: Can run on CUDA or CPU (auto-fallback)
4. **Batch Processing**: Processes documents in configurable batches
5. **Thread Safety**: Uses locks for concurrent access
6. **Memory Management**: Explicit model loading/unloading

### Current File Structure
```
/root/document-embedding-project/
├── packages/
│   └── vecpipe/
│       ├── reranker.py          # The module to test
│       └── qwen3_search_config.py  # Configuration
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_reranking_e2e.py   # Existing e2e tests
│   └── test_reranker.py         # TO BE CREATED
```

## Detailed Test Implementation Plan

### 1. Test File Setup

Create `/root/document-embedding-project/tests/test_reranker.py`:

```python
"""
Unit tests for CrossEncoderReranker class
Tests the reranking functionality with mocked models
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import numpy as np
from typing import List, Tuple

# Import the class to test
from packages.vecpipe.reranker import CrossEncoderReranker

# Test constants
TEST_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
YES_TOKEN_ID = 9454  # Mock token ID for "Yes"
NO_TOKEN_ID = 2753   # Mock token ID for "No"
```

### 2. Fixtures to Implement

```python
@pytest.fixture
def mock_torch_cuda():
    """Mock torch.cuda availability"""
    with patch('torch.cuda.is_available', return_value=True):
        yield

@pytest.fixture
def mock_transformers():
    """Mock transformers imports"""
    with patch('packages.vecpipe.reranker.AutoModelForCausalLM') as mock_model_class, \
         patch('packages.vecpipe.reranker.AutoTokenizer') as mock_tokenizer_class:
        
        # Mock model instance
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock tokenizer instance
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Configure tokenizer behavior
        mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: {
            "Yes": [YES_TOKEN_ID],
            "No": [NO_TOKEN_ID],
            "yes": [YES_TOKEN_ID],
            "no": [NO_TOKEN_ID]
        }.get(text, [1, 2, 3])  # Default tokens for other text
        
        yield mock_model_class, mock_tokenizer_class, mock_model, mock_tokenizer

@pytest.fixture
def reranker_unloaded(mock_transformers):
    """Create reranker instance without loading model"""
    _, _, _, _ = mock_transformers
    return CrossEncoderReranker(
        model_name=TEST_MODEL_NAME,
        device="cuda",
        quantization="float16"
    )

@pytest.fixture
def reranker_loaded(reranker_unloaded, mock_transformers):
    """Create reranker instance with model loaded"""
    reranker_unloaded.load_model()
    return reranker_unloaded

@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language for data science.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information."
    ]

@pytest.fixture
def mock_model_output():
    """Mock model output with logits"""
    def create_output(batch_size):
        # Create mock logits tensor (batch_size, seq_len, vocab_size)
        vocab_size = 10000
        seq_len = 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Set specific logits for yes/no tokens in last position
        for i in range(batch_size):
            # Vary scores to test ranking
            yes_score = 2.0 - (i * 0.5)  # Decreasing relevance
            no_score = -1.0
            logits[i, -1, YES_TOKEN_ID] = yes_score
            logits[i, -1, NO_TOKEN_ID] = no_score
        
        output = MagicMock()
        output.logits = logits
        return output
    
    return create_output
```

### 3. Test Categories and Implementations

#### A. Initialization Tests

```python
class TestInitialization:
    def test_default_initialization(self):
        """Test reranker initialization with default parameters"""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == "Qwen/Qwen3-Reranker-0.6B"
        assert reranker.quantization == "float16"
        assert reranker.max_length == 512
        assert reranker.model is None
        assert reranker.tokenizer is None
    
    def test_custom_initialization(self):
        """Test reranker initialization with custom parameters"""
        reranker = CrossEncoderReranker(
            model_name="Qwen/Qwen3-Reranker-4B",
            device="cpu",
            quantization="int8",
            max_length=1024
        )
        assert reranker.model_name == "Qwen/Qwen3-Reranker-4B"
        assert reranker.device == "cpu"
        assert reranker.quantization == "int8"
        assert reranker.max_length == 1024
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_fallback(self):
        """Test automatic fallback to CPU when CUDA unavailable"""
        reranker = CrossEncoderReranker(device="cuda")
        assert reranker.device == "cpu"
```

#### B. Model Loading/Unloading Tests

```python
class TestModelLoading:
    def test_load_model_success(self, reranker_unloaded, mock_transformers):
        """Test successful model loading"""
        model_class, tokenizer_class, mock_model, mock_tokenizer = mock_transformers
        
        reranker_unloaded.load_model()
        
        # Verify model loading
        assert reranker_unloaded.model is not None
        assert reranker_unloaded.tokenizer is not None
        
        # Verify correct loading parameters
        model_class.from_pretrained.assert_called_once()
        tokenizer_class.from_pretrained.assert_called_once_with(TEST_MODEL_NAME)
    
    def test_load_model_with_quantization(self, mock_transformers):
        """Test model loading with int8 quantization"""
        model_class, _, _, _ = mock_transformers
        
        reranker = CrossEncoderReranker(quantization="int8")
        with patch('packages.vecpipe.reranker.BitsAndBytesConfig') as mock_bnb:
            reranker.load_model()
            mock_bnb.assert_called_once_with(load_in_8bit=True)
    
    def test_unload_model(self, reranker_loaded):
        """Test model unloading and cleanup"""
        reranker_loaded.unload_model()
        
        assert reranker_loaded.model is None
        assert reranker_loaded.tokenizer is None
    
    def test_load_model_thread_safety(self, reranker_unloaded, mock_transformers):
        """Test thread-safe model loading"""
        import threading
        
        def load_model():
            reranker_unloaded.load_model()
        
        threads = [threading.Thread(target=load_model) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Model should only be loaded once
        model_class, _, _, _ = mock_transformers
        assert model_class.from_pretrained.call_count == 1
```

#### C. Input Formatting Tests

```python
class TestInputFormatting:
    def test_format_input_default_instruction(self, reranker_loaded):
        """Test input formatting with default instruction"""
        query = "What is machine learning?"
        document = "Machine learning is a subset of AI."
        
        formatted = reranker_loaded.format_input(query, document)
        
        assert "<Instruct>" in formatted
        assert "<Query>" in formatted
        assert "<Document>" in formatted
        assert query in formatted
        assert document in formatted
    
    def test_format_input_custom_instruction(self, reranker_loaded):
        """Test input formatting with custom instruction"""
        query = "Python programming"
        document = "Python is a high-level language."
        instruction = "Check if document is about programming"
        
        formatted = reranker_loaded.format_input(query, document, instruction)
        
        assert instruction in formatted
        assert query in formatted
        assert document in formatted
    
    def test_format_input_empty_values(self, reranker_loaded):
        """Test input formatting with empty values"""
        # Should not raise errors
        formatted = reranker_loaded.format_input("", "")
        assert formatted is not None
```

#### D. Relevance Scoring Tests

```python
class TestRelevanceScoring:
    def test_compute_relevance_scores_basic(self, reranker_loaded, mock_transformers, 
                                           sample_documents, mock_model_output):
        """Test basic relevance score computation"""
        _, _, mock_model, mock_tokenizer = mock_transformers
        query = "machine learning algorithms"
        
        # Configure mock
        mock_model.return_value = mock_model_output(len(sample_documents))
        mock_tokenizer.return_value = MagicMock(
            input_ids=torch.randint(0, 1000, (len(sample_documents), 10))
        )
        
        scores = reranker_loaded.compute_relevance_scores(query, sample_documents)
        
        assert len(scores) == len(sample_documents)
        assert all(isinstance(s, float) for s in scores)
        assert all(0 <= s <= 1 for s in scores)
    
    def test_compute_relevance_scores_empty_documents(self, reranker_loaded):
        """Test handling of empty documents"""
        query = "test query"
        documents = ["", "   ", "valid document", None]
        
        with patch.object(reranker_loaded, 'tokenizer') as mock_tokenizer:
            mock_tokenizer.encode.return_value = [YES_TOKEN_ID]
            # Should handle gracefully without errors
            scores = reranker_loaded.compute_relevance_scores(
                query, 
                [d if d else "" for d in documents]
            )
            assert len(scores) == len(documents)
    
    def test_compute_relevance_scores_batching(self, reranker_loaded, mock_transformers):
        """Test batch processing of documents"""
        _, _, mock_model, _ = mock_transformers
        
        # Create many documents to force batching
        many_documents = ["Document " + str(i) for i in range(100)]
        query = "test query"
        
        # Mock batch processing
        def batch_output(encoded):
            batch_size = encoded['input_ids'].shape[0]
            return mock_model_output(batch_size)
        
        mock_model.side_effect = batch_output
        
        scores = reranker_loaded.compute_relevance_scores(query, many_documents)
        
        assert len(scores) == len(many_documents)
        # Verify multiple batches were processed
        assert mock_model.call_count > 1
```

#### E. Reranking Tests

```python
class TestReranking:
    def test_rerank_basic(self, reranker_loaded, sample_documents):
        """Test basic reranking functionality"""
        query = "artificial intelligence"
        
        with patch.object(reranker_loaded, 'compute_relevance_scores') as mock_compute:
            # Mock scores in non-sorted order
            mock_compute.return_value = [0.3, 0.9, 0.1, 0.7, 0.5]
            
            results = reranker_loaded.rerank(query, sample_documents, top_k=3)
            
            assert len(results) == 3
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
            
            # Check sorting (highest score first)
            indices, scores = zip(*results)
            assert indices == (1, 3, 4)  # Indices of top 3 scores
            assert scores == (0.9, 0.7, 0.5)
    
    def test_rerank_top_k_larger_than_docs(self, reranker_loaded, sample_documents):
        """Test reranking when top_k > number of documents"""
        query = "test"
        
        with patch.object(reranker_loaded, 'compute_relevance_scores') as mock_compute:
            mock_compute.return_value = [0.5] * len(sample_documents)
            
            results = reranker_loaded.rerank(query, sample_documents, top_k=100)
            
            assert len(results) == len(sample_documents)
    
    def test_rerank_empty_documents(self, reranker_loaded):
        """Test reranking with empty document list"""
        results = reranker_loaded.rerank("query", [], top_k=10)
        assert results == []
    
    def test_rerank_with_scores(self, reranker_loaded, sample_documents):
        """Test reranking with return_scores=True"""
        query = "machine learning"
        
        with patch.object(reranker_loaded, 'compute_relevance_scores') as mock_compute:
            mock_compute.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            results = reranker_loaded.rerank(
                query, sample_documents, top_k=3, return_scores=True
            )
            
            # Should still return (index, score) tuples
            assert all(isinstance(r[1], float) for r in results)
```

#### F. Edge Cases and Error Handling

```python
class TestEdgeCases:
    def test_very_long_documents(self, reranker_loaded):
        """Test handling of documents exceeding max_length"""
        query = "test"
        very_long_doc = "word " * 10000  # Very long document
        
        with patch.object(reranker_loaded, 'tokenizer') as mock_tokenizer:
            mock_tokenizer.return_value = MagicMock(
                input_ids=torch.ones((1, reranker_loaded.max_length))
            )
            mock_tokenizer.encode.return_value = [YES_TOKEN_ID]
            
            # Should truncate without error
            scores = reranker_loaded.compute_relevance_scores(query, [very_long_doc])
            assert len(scores) == 1
    
    def test_unicode_content(self, reranker_loaded):
        """Test handling of Unicode/multilingual content"""
        query = "机器学习"  # Chinese
        documents = [
            "Machine learning in 中文",
            "🤖 AI and ML 📚",
            "Überraschung für KI-Forschung"
        ]
        
        # Should handle without encoding errors
        with patch.object(reranker_loaded, 'compute_relevance_scores') as mock_compute:
            mock_compute.return_value = [0.5, 0.6, 0.7]
            results = reranker_loaded.rerank(query, documents, top_k=2)
            assert len(results) == 2
    
    def test_concurrent_reranking(self, reranker_loaded, sample_documents):
        """Test thread safety during concurrent reranking"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def rerank_task(query_id):
            with patch.object(reranker_loaded, 'compute_relevance_scores') as mock:
                mock.return_value = [0.1 * i for i in range(len(sample_documents))]
                results = reranker_loaded.rerank(
                    f"query {query_id}", 
                    sample_documents, 
                    top_k=3
                )
                results_queue.put((query_id, results))
        
        threads = [threading.Thread(target=rerank_task, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # All threads should complete successfully
        assert results_queue.qsize() == 10
```

#### G. Performance and Resource Tests

```python
class TestPerformance:
    def test_batch_size_configuration(self):
        """Test batch size configuration for different models"""
        test_cases = [
            ("Qwen/Qwen3-Reranker-0.6B", "float16", 128),
            ("Qwen/Qwen3-Reranker-4B", "int8", 64),
            ("Qwen/Qwen3-Reranker-8B", "float32", 8),
        ]
        
        for model_name, quantization, expected_batch_size in test_cases:
            reranker = CrossEncoderReranker(
                model_name=model_name,
                quantization=quantization
            )
            assert reranker.get_batch_size() == expected_batch_size
    
    def test_model_info(self, reranker_loaded):
        """Test model info retrieval"""
        info = reranker_loaded.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "quantization" in info
        assert "loaded" in info
        assert info["loaded"] is True
    
    @patch('torch.cuda.empty_cache')
    def test_memory_cleanup(self, mock_empty_cache, reranker_loaded):
        """Test GPU memory cleanup on unload"""
        reranker_loaded.unload_model()
        
        # Should call empty_cache if on CUDA
        if reranker_loaded.device == "cuda":
            mock_empty_cache.assert_called_once()
```

### 4. Test Execution Strategy

```bash
# Run all reranker tests
pytest tests/test_reranker.py -v

# Run with coverage
pytest tests/test_reranker.py --cov=packages.vecpipe.reranker --cov-report=html

# Run specific test class
pytest tests/test_reranker.py::TestReranking -v

# Run with markers (if we add any)
pytest tests/test_reranker.py -m "not slow"
```

### 5. Integration with CI/CD

Add to `.github/workflows/test.yml`:
```yaml
- name: Run reranker unit tests
  env:
    USE_MOCK_EMBEDDINGS: "true"
    PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"
  run: |
    pytest tests/test_reranker.py -v --cov=packages.vecpipe.reranker
```

### 6. Additional Test Utilities

```python
# Add to tests/test_reranker.py

def assert_valid_rerank_results(results: List[Tuple[int, float]], 
                               expected_length: int,
                               num_documents: int):
    """Helper to validate reranking results"""
    assert len(results) == expected_length
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(idx, int) and isinstance(score, float) for idx, score in results)
    assert all(0 <= idx < num_documents for idx, _ in results)
    assert all(0 <= score <= 1 for _, score in results)
    
    # Check descending order
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

def create_mock_tokenizer_encode(special_tokens: dict):
    """Create mock tokenizer encode function"""
    def encode(text, add_special_tokens=True):
        return special_tokens.get(text, [1, 2, 3])  # Default tokens
    return encode
```

## Implementation Checklist

- [ ] Create `tests/test_reranker.py` file
- [ ] Implement all fixtures
- [ ] Implement initialization tests
- [ ] Implement model loading/unloading tests
- [ ] Implement input formatting tests
- [ ] Implement relevance scoring tests
- [ ] Implement reranking tests
- [ ] Implement edge case tests
- [ ] Implement performance tests
- [ ] Add test utilities
- [ ] Run tests with coverage report
- [ ] Fix any failing tests
- [ ] Add to CI/CD pipeline
- [ ] Document any special test requirements

## Notes for Tomorrow

1. **Environment Setup**: Ensure `USE_MOCK_EMBEDDINGS=true` is set
2. **Dependencies**: May need to install `pytest-mock` if not already available
3. **Model Mocking**: The key is properly mocking the transformers library
4. **Thread Safety**: Pay special attention to testing concurrent access
5. **Coverage Goal**: Aim for 90%+ coverage of reranker.py

## Potential Issues to Watch

1. **Import Errors**: May need to adjust import paths based on test execution context
2. **Tensor Operations**: Ensure torch tensors are properly mocked
3. **Memory Tests**: GPU memory tests may need special handling in CI
4. **Tokenizer Behavior**: The yes/no token extraction is critical to mock correctly

This comprehensive plan should provide all the context needed to implement the unit tests successfully.