# Search API with Qwen3 Support

The production search API (`search_api.py`) has been updated to support real Qwen3 embeddings while maintaining backward compatibility with mock embeddings.

## Configuration

The API behavior is controlled by environment variables:

```bash
# Use mock embeddings (default: false)
USE_MOCK_EMBEDDINGS=false

# Default embedding model (default: Qwen/Qwen3-Embedding-0.6B)
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B

# Default quantization (default: float16)
DEFAULT_QUANTIZATION=float16

# Qdrant configuration
QDRANT_HOST=192.168.1.173
QDRANT_PORT=6333
```

## Running the API

### With Real Qwen3 Embeddings (Recommended)
```bash
# This is the default - uses real embeddings
python vecpipe/search_api.py
```

### With Mock Embeddings (Fallback/Testing)
```bash
USE_MOCK_EMBEDDINGS=true python vecpipe/search_api.py
```

### With Custom Model
```bash
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B DEFAULT_QUANTIZATION=int8 python vecpipe/search_api.py
```

## API Endpoints

### Health Check
```
GET /
```
Returns API status and embedding configuration.

### Search
```
GET /search?q=<query>&k=<num_results>
```
Performs vector similarity search using configured embeddings.

### Embedding Info
```
GET /embedding/info
```
Returns detailed information about the current embedding configuration.

### Collection Info
```
GET /collection/info
```
Returns Qdrant collection statistics.

## Features

1. **No Silent Failures**: If real embeddings are configured but fail, the API will error explicitly rather than falling back to mock embeddings.

2. **Real Embeddings**: When enabled, uses Qwen3 embeddings with:
   - Task-specific instruction: "Represent this sentence for searching relevant passages:"
   - Support for float32, float16, and int8 quantization
   - GPU acceleration when available

3. **Explicit Configuration**: Must explicitly set `USE_MOCK_EMBEDDINGS=true` to use mock embeddings.

4. **Performance**: 
   - Mock embeddings: Instant (hash-based)
   - Qwen3-0.6B float16: ~10-20ms per query
   - Qwen3-4B float16: ~30-50ms per query

## Testing

Run the integration test:
```bash
python test_search_integration.py
```

## Migration Notes

To migrate from mock to real embeddings:

1. Ensure GPU is available (or use CPU with longer latencies)
2. Ensure required dependencies: `pip install transformers>=4.51.0 torch accelerate`
3. Set `USE_MOCK_EMBEDDINGS=false` (or remove it, as false is default)
4. Restart the API
5. The API will fail to start if the model cannot be loaded
6. Once started successfully, the first query will be slightly slower as the model warms up

## Troubleshooting

### Model fails to load
- Check GPU memory (need ~1.2GB for Qwen3-0.6B float16)
- Try int8 quantization: `DEFAULT_QUANTIZATION=int8`
- Check transformers version: `pip install transformers>=4.51.0`

### Slow performance
- Use float16 or int8 quantization
- Consider using smaller model (Qwen3-0.6B)
- Enable GPU if available

### Out of memory
- Use int8 quantization
- Reduce batch size in embedding service
- Use smaller model