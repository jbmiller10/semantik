# Pipeline Consolidation Summary

## Overview
Consolidated multiple pipeline variants into canonical implementations, reducing maintenance overhead and code duplication.

## Changes Made

### 1. Extract Chunks Pipeline
- **Kept**: `extract_chunks_v2.py` → `extract_chunks.py`
- **Archived**: `extract_chunks.py` → `archive/extract_chunks_original.py`
- **Key Features Preserved**:
  - Token-based chunking with tiktoken
  - File change tracking with SHA256
  - Resume capability
  - Removed file detection

### 2. Embed Chunks Pipeline  
- **Kept**: `embed_chunks_parallel.py` → `embed_chunks.py`
- **Archived**: 
  - `embed_chunks.py` → `archive/embed_chunks_original.py`
  - `embed_chunks_simple.py` → `archive/embed_chunks_simple.py`
- **Key Features Preserved**:
  - Asyncio-based parallel processing
  - Dedicated GPU worker thread
  - Advanced OOM handling
  - Added `--mock` flag for testing without GPU

### 3. Web UI Embedding Service
- **Kept**: `embedding_service_v2.py` → `embedding_service.py`
- **Archived**: `embedding_service.py` → `archive/embedding_service_original.py`
- **Key Features Preserved**:
  - Quantization support (float32, float16, int8)
  - Qwen3 model support
  - Instruction-aware embeddings
  - Bitsandbytes integration

## Benefits
- Single canonical implementation for each component
- Reduced maintenance overhead
- Consistent behavior across the pipeline
- Best features from all variants preserved
- Mock mode for testing without GPU resources

## Usage
The consolidated versions maintain backward compatibility while offering enhanced features:
- Use `--mock` flag with embed_chunks.py for testing
- All existing command-line arguments are preserved
- Enhanced features available through configuration