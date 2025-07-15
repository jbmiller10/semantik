# Docker Embedding Generation Investigation Plan

## UPDATE: Bitsandbytes INT8 Quantization Issue

### New Problem
User reports issues with INT8 quantization through bitsandbytes in Docker container.

### Investigation Summary

#### Root Cause Analysis
1. **Docker Image Mismatch**: The default docker-compose.yml uses the regular Dockerfile which lacks CUDA support
2. **Missing CUDA Libraries**: Bitsandbytes requires specific CUDA libraries (libcudart, libcublas, libcusparse)
3. **Dockerfile.cuda Issues**:
   - Uses Python 3.11 (vs 3.12 in regular Dockerfile)
   - Wrong Python package paths (dist-packages vs site-packages)
   - Missing critical environment variables for bitsandbytes
4. **Configuration Issue**: Users must explicitly use docker-compose.cuda.yml override

### Solution Plan

#### Immediate Fix - Update Dockerfile.cuda
1. Fix Python package paths (use site-packages for Python 3.11)
2. Add CUDA environment variables:
   - CUDA_HOME=/usr/local/cuda
   - LD_LIBRARY_PATH with CUDA lib paths
3. Install bitsandbytes with proper CUDA detection
4. Add build-time test for bitsandbytes

#### Configuration Updates
1. Update docker-compose.cuda.yml with required environment variables
2. Document proper usage for INT8 quantization
3. Add troubleshooting script

### Files to Modify
- Dockerfile.cuda (critical fixes)
- docker-compose.cuda.yml (env vars)
- docs/docker-cuda-fixes.md (new documentation)
- .env.docker.example (update comments)

---

## Previous Investigation (RESOLVED)

### Root Cause Identified
The embedding generation was failing due to **undefined variables** in the `embedding_service.py` file:
- `max_retries` variable was not defined
- `retry_delay` variable was not defined  
- `time` module was not imported

These variables were used in the Qwen3 model loading retry logic (lines 163-198).

### Fix Applied
1. Added `import time` to the imports section
2. Defined `max_retries = 3` before the retry loops
3. Defined `retry_delay = 2` before the retry loops

### Verification
After rebuilding the Docker containers with the fixed code:
- The WebUI service started successfully with CUDA enabled
- Successfully created an embedding job via API
- Job processed 13+ files out of 138 total files
- Embeddings were generated using Qwen3-Embedding-0.6B model with float16 quantization
- Points were successfully uploaded to Qdrant vector database

### Additional Findings
- Docker configuration is correct with proper GPU passthrough
- Volume mounts are working correctly
- The UI has some minor issues with the registration flow but the API works perfectly
- The system successfully uses GPU acceleration for embedding generation

The embedding generation is now working correctly in the Docker container!