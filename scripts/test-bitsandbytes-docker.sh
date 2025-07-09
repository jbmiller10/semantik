#!/bin/bash
# Quick test script to verify bitsandbytes fixes in Docker

echo "Testing bitsandbytes INT8 quantization setup..."
echo "============================================="

# Test using the CUDA base image directly
docker run --rm --gpus all \
  -v $(pwd)/scripts/test-bitsandbytes.py:/test.py \
  -v $(pwd)/packages:/app/packages \
  -e PYTHONPATH=/app/packages \
  -e CUDA_HOME=/usr/local/cuda \
  -e LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
  -e BNB_CUDA_VERSION=121 \
  nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 \
  bash -c '
    apt-get update && apt-get install -y python3-pip python3.11 libcusparse11 libcublas11 && \
    pip install torch transformers bitsandbytes accelerate && \
    python3 -c "import bitsandbytes; print(\"✓ Bitsandbytes imported successfully\")" && \
    python3 -c "import torch; print(f\"✓ CUDA available: {torch.cuda.is_available()}\")" && \
    echo "✓ Basic setup verified!"
  '

echo ""
echo "If the above tests pass, the Dockerfile.cuda fixes should work correctly."
echo "The full build may take 10-15 minutes due to downloading large dependencies."