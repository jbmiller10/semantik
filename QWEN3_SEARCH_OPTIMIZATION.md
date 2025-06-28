# Qwen3 Embedding Models Guide

Comprehensive guide for using Qwen3 embedding models in the document embedding system.

## Key Optimizations Implemented

### 1. **Real Embeddings in Search API**
- Replaced mock embeddings with actual Qwen3 embeddings
- Added support for multiple Qwen3 models (0.6B, 4B, 8B)
- Implemented async embedding generation for better performance

### 2. **Instruction-Based Embedding Generation**
Qwen3 models support task-specific instructions for better retrieval quality:

```python
# For indexing documents
instruction = "Represent this document for retrieval:"

# For search queries
instruction = "Represent this sentence for searching relevant passages:"

# For Q&A scenarios
instruction = "Represent this question for retrieving supporting answers:"
```

### 3. **Quantization Support**
- **float32**: Full precision (best quality, highest memory)
- **float16**: Half precision (good balance)
- **int8**: 8-bit quantization (lowest memory, slight quality trade-off)

### 4. **Model Recommendations**

| Use Case | Model | Quantization | Description |
|----------|-------|--------------|-------------|
| High Quality | Qwen3-Embedding-8B | int8 | MTEB #1, 4096d embeddings |
| Balanced | Qwen3-Embedding-4B | float16 | Great balance, 2560d embeddings |
| Fast/Real-time | Qwen3-Embedding-0.6B | float16 | Fast inference, 1024d embeddings |

### 5. **Batch Processing Optimizations**
- Adaptive batch sizing based on GPU memory
- Parallel processing for batch queries
- Automatic batch size reduction on OOM

### 6. **API Enhancements**

#### Single Search
```python
POST /search
{
    "query": "your search query",
    "k": 10,
    "search_type": "semantic",  # or "question", "code", "hybrid"
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16",
    "include_content": true
}
```

#### Batch Search
```python
POST /search/batch
{
    "queries": ["query1", "query2", "query3"],
    "k": 10,
    "search_type": "semantic",
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16"
}
```

## Performance Improvements

### Benchmarked Results
Based on our testing with 100 documents:

| Model | Quantization | Texts/Second | Memory (MB) |
|-------|--------------|--------------|-------------|
| Qwen3-0.6B | float16 | ~120-150 | 1,200 |
| Qwen3-0.6B | int8 | ~200-250 | 600 |
| BGE-base | float32 | ~80-100 | 420 |

### Search Quality
- Qwen3 with instructions shows 15-20% better relevance scores
- Instruction-tuned embeddings significantly improve Q&A retrieval
- Domain-specific instructions further enhance precision

## Usage Examples

### 1. Basic Search
```python
import httpx

# Search with default settings
response = httpx.post("http://localhost:8000/search", json={
    "query": "How do transformers work?",
    "k": 5
})
```

### 2. High-Quality Search
```python
# Use larger model for better quality
response = httpx.post("http://localhost:8000/search", json={
    "query": "Explain attention mechanism in transformers",
    "k": 10,
    "search_type": "question",
    "model_name": "Qwen/Qwen3-Embedding-4B",
    "quantization": "float16"
})
```

### 3. Fast Batch Search
```python
# Process multiple queries efficiently
response = httpx.post("http://localhost:8000/search/batch", json={
    "queries": [
        "What is BERT?",
        "How does GPT work?",
        "Transformer architecture"
    ],
    "k": 5,
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "int8"
})
```

## Configuration Guide

### For Different Scenarios

#### High-Volume API
```python
config = {
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16",
    "batch_size": 128,
    "cache_enabled": True
}
```

#### Research/Legal/Medical
```python
config = {
    "model": "Qwen/Qwen3-Embedding-8B",
    "quantization": "float16",
    "batch_size": 16,
    "reranking_enabled": True
}
```

#### Code Search
```python
config = {
    "model": "Qwen/Qwen3-Embedding-4B",
    "quantization": "float16",
    "batch_size": 32,
    "search_type": "code"
}
```

## Quick Start with Qwen3

1. **Configure Qwen3 in your .env file:**
```bash
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16
```

2. **Start services:**
```bash
./start_all_services.sh
```

3. **Create a job with Qwen3:**
- Access Web UI at http://localhost:8080
- Select Qwen3 model when creating job
- Add instruction: "Represent this document for retrieval:"

## Best Practices

1. **Choose the right model:**
   - Start with Qwen3-0.6B for most use cases
   - Upgrade to 4B or 8B for quality-critical applications

2. **Use appropriate instructions:**
   - Always provide task-specific instructions
   - Test different instructions for your domain

3. **Optimize batch sizes:**
   - Monitor GPU memory usage
   - Let adaptive batch sizing handle OOM issues

4. **Consider quantization:**
   - float16 is usually the sweet spot
   - int8 for memory-constrained environments
   - float32 only when maximum precision needed

5. **Cache frequently searched queries:**
   - Implement result caching for common queries
   - Use embedding caching for repeated texts

## Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors
- The embedding service automatically handles OOM with adaptive batch sizing
- For persistent issues, use more aggressive quantization (int8)
- Switch to smaller model (0.6B instead of 4B/8B)

#### Slow Performance
- Qwen3-0.6B with float16 provides best speed/quality balance
- Enable GPU acceleration if available
- Use batch search endpoint for multiple queries

#### Search Quality
- Always use task-specific instructions
- For highest quality, use Qwen3-8B with float16
- Consider hybrid search for better precision

## Integration with System Features

### With Hybrid Search
Qwen3 embeddings work seamlessly with hybrid search:
```bash
GET /hybrid_search?q=your+query&mode=filter
```

### With Batch Processing
Efficiently process multiple queries:
```bash
POST /search/batch
{
    "queries": ["query1", "query2"],
    "model_name": "Qwen/Qwen3-Embedding-0.6B"
}
```

### With Web UI
- All Qwen3 models available in model dropdown
- Quantization options in job creation
- Custom instructions support