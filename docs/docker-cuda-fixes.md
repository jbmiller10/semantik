# Docker CUDA and Model Loading Fixes

This comprehensive guide explains how to properly set up and troubleshoot Semantik in Docker with GPU support, particularly for INT8 quantization using bitsandbytes.

> **Note**: As of the latest release, the standard Docker build includes gcc/g++ and supports INT8 quantization. However, the CUDA build is still recommended for GPU users as it provides better performance and GPU-specific optimizations.

## Overview

Semantik supports three quantization modes:
- **float32**: Full precision (CPU/GPU compatible, highest quality)
- **float16**: Half precision (GPU only, 50% memory reduction)
- **int8**: 8-bit quantization via bitsandbytes (GPU only, 75% memory reduction)

## Issues Addressed

1. **Bitsandbytes INT8 quantization errors**: `Could not import module 'validate_bnb_backend_availability'`
2. **CUDA library errors**: `libcudart.so not found`, `libcusparse.so.11: cannot open shared object file`
3. **HuggingFace rate limiting**: `HTTP Error 429` when downloading models
4. **Python version compatibility**: Issues with bitsandbytes on different Python versions

## Prerequisites

1. **NVIDIA GPU** with compute capability 7.0+ (for INT8)
2. **NVIDIA Docker runtime**:
   ```bash
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## Solution

### Using the CUDA-enabled Docker Image

The standard Docker image now includes gcc for basic INT8 support, but for full GPU acceleration and optimal performance, use the CUDA-enabled build:

```bash
# Build and run with CUDA support
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

### What's Different in the CUDA Build?

1. **Base Image**: Uses `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` instead of `python:3.12-slim`
2. **CUDA Libraries**: Includes libcusparse11, libcublas11, and proper library paths for GPU acceleration
3. **Environment Setup**: Configures CUDA_HOME, LD_LIBRARY_PATH, and BNB_CUDA_VERSION
4. **Optimized GPU Support**: Better GPU memory management and CUDA-optimized operations
5. **Python Version**: Uses Python 3.11 (vs 3.12 in standard) for better CUDA compatibility

### Quick Setup for INT8 Quantization

1. **Copy and configure environment**:
   ```bash
   cp .env.docker.example .env
   # Edit .env and set:
   # DEFAULT_QUANTIZATION=int8
   ```

2. **Build and run with CUDA support**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
   ```

3. **Verify the setup**:
   ```bash
   # Check GPU access
   docker exec semantik-webui nvidia-smi
   
   # Test bitsandbytes
   docker exec semantik-webui python -c "import bitsandbytes; print('✓ Bitsandbytes loaded')"
   ```

### Configuration Options

#### Model Caching

Models are cached in `/app/data/.cache/huggingface`. To avoid repeated downloads:

1. **First Run**: Download models normally
2. **Subsequent Runs**: Enable offline mode in your `.env`:
   ```
   HF_HUB_OFFLINE=true
   ```

#### Fallback Behavior

If INT8 quantization fails, the system automatically falls back to float16:
- Qwen3-Embedding-0.6B: Works well with float16 (1.2GB VRAM)
- Qwen3-Embedding-4B: May require INT8 on smaller GPUs
- Qwen3-Embedding-8B: Requires INT8 or multiple GPUs

### Troubleshooting

#### Permission Errors?

If you see `PermissionError at /app/.cache/huggingface/hub`:

```bash
# The container runs as UID 1000, fix directory permissions:
./scripts/fix-permissions.sh

# Or manually:
sudo chown -R 1000:1000 ./models ./data ./logs
```

#### Still Getting Bitsandbytes Errors?

1. Ensure you're using the CUDA override:
   ```bash
   docker compose down
   docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
   ```

2. Check CUDA is available in container:
   ```bash
   docker exec semantik-webui nvidia-smi
   ```

3. Verify bitsandbytes installation:
   ```bash
   docker exec semantik-webui python -c "import bitsandbytes; print('Bitsandbytes OK')"
   ```

#### Rate Limiting Issues?

1. The updated code includes automatic retries with exponential backoff
2. Models are cached after first download
3. Consider using `HF_HUB_OFFLINE=true` after initial setup

#### Memory Issues?

Adjust quantization in your `.env`:
```
# For smaller GPUs (4-6GB)
DEFAULT_QUANTIZATION=int8

# For larger GPUs (8GB+)  
DEFAULT_QUANTIZATION=float16

# CPU only
DEFAULT_QUANTIZATION=float32
```

### Performance Tips

1. **Model Selection by GPU Memory**:
   - 4GB: Use Qwen3-Embedding-0.6B with int8
   - 6GB: Use Qwen3-Embedding-0.6B with float16 or 4B with int8
   - 8GB+: Any model with appropriate quantization

2. **Batch Size**: The system automatically adjusts batch size on OOM errors

3. **Multi-GPU**: Set `CUDA_VISIBLE_DEVICES=0,1` for multiple GPUs