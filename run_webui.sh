#!/bin/bash
# Script to run the Web UI with proper CUDA setup

# Create symlink for CUDA library if it doesn't exist
CUDA_LIB_PATH="/root/.pyenv/versions/3.12.11/envs/embeddingdocs/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if [ -f "$CUDA_LIB_PATH/libcudart.so.12" ] && [ ! -f "$CUDA_LIB_PATH/libcudart.so" ]; then
    echo "Creating symlink for libcudart.so..."
    ln -sf "$CUDA_LIB_PATH/libcudart.so.12" "$CUDA_LIB_PATH/libcudart.so"
fi

# Set CUDA library path
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:$LD_LIBRARY_PATH"

# Also add the parent nvidia directory for other CUDA libraries
export LD_LIBRARY_PATH="/root/.pyenv/versions/3.12.11/envs/embeddingdocs/lib/python3.12/site-packages/nvidia:$LD_LIBRARY_PATH"

# Run the web UI with poetry
cd /root/document-embedding-project
poetry run python -m uvicorn webui.app:app --host 0.0.0.0 --port 8080 --reload