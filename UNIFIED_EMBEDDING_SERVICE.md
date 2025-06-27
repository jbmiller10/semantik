# Unified Embedding Service Refactoring

## Summary

This document summarizes the consolidation of two parallel embedding implementations into a single unified service.

## Changes Made

### 1. Enhanced webui/embedding_service.py
- Added adaptive batch sizing with restoration after successful batches
- Added mock mode support with deterministic embeddings based on text hash
- Added minimum batch size configuration (default: 4)
- Enhanced OOM handling with automatic batch size reduction and restoration
- Added success counter tracking for batch size restoration

### 2. Created vecpipe/embed_chunks_unified.py
- New CLI entry point that uses webui.embedding_service.EmbeddingService
- Maintains backward compatibility with old embed_chunks.py interface
- Supports all features: mock mode, quantization, batch processing
- Uses async/await for parallel file processing

### 3. Updated deploy/systemd/embed.service
- Changed to use embed_chunks_unified.py instead of embed_chunks_simple.py
- No other configuration changes needed

### 4. Deleted vecpipe/embed_chunks.py
- Removed deprecated implementation to avoid confusion
- All functionality now available through unified service

## Benefits

1. **Single Source of Truth**: All embedding generation now goes through webui.embedding_service.EmbeddingService
2. **Consistent Behavior**: Embeddings are generated identically regardless of entry point (WebUI or CLI)
3. **Reduced Maintenance**: Only one codebase to maintain and update
4. **Feature Parity**: All features from both implementations are preserved
5. **Better Resource Management**: Adaptive batch sizing prevents OOM errors and maximizes GPU utilization

## Usage

### CLI Usage
```bash
python3 vecpipe/embed_chunks_unified.py \
    --input /opt/vecpipe/extract \
    --output /var/embeddings/ingest \
    --model BAAI/bge-large-en-v1.5 \
    --quantization float32 \
    --batch-size 96
```

### WebUI/API Usage
```python
from webui.embedding_service import EmbeddingService

service = EmbeddingService()
embeddings = service.generate_embeddings(
    texts=["text1", "text2"],
    model_name="BAAI/bge-large-en-v1.5",
    quantization="float32",
    batch_size=96
)
```

### Mock Mode (for testing)
```bash
# CLI
python3 vecpipe/embed_chunks_unified.py --mock

# Python
service = EmbeddingService(mock_mode=True)
```

## Configuration

The unified service supports:
- Multiple quantization modes: float32, float16, int8
- Adaptive batch sizing with automatic restoration
- Mock mode for testing without GPU
- All popular embedding models including Qwen3 models