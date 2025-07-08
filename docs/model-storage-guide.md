# Model Storage and Offline Mode Guide

## Overview

HuggingFace models can be quite large (0.6GB to 16GB+). To avoid re-downloading them every time you rebuild containers or hit rate limits, Semantik supports persistent model storage outside containers.

## How It Works

1. **Models are stored on your host machine** in `./models` by default
2. **This directory is mounted into containers** at `/app/.cache/huggingface`  
3. **Models persist across container rebuilds, restarts, and removals**
4. **Offline mode prevents any network requests** to HuggingFace

## Setup

### 1. First Time Setup (Download Models)

```bash
# Create models directory
mkdir -p ./models

# Start services (models will be downloaded on first use)
docker compose up -d

# Watch logs to see download progress
docker compose logs -f webui
```

### 2. Enable Offline Mode (After Download)

Once models are downloaded, enable offline mode in your `.env`:

```bash
# Prevent any HuggingFace network requests
HF_HUB_OFFLINE=true
```

### 3. Custom Model Storage Location

To store models elsewhere, set in your `.env`:

```bash
# Store models in a different location
HF_CACHE_DIR=/mnt/shared/ai-models
```

## Benefits

1. **No Re-downloads**: Models persist across container lifecycle
2. **No Rate Limits**: Offline mode prevents 429 errors
3. **Faster Startup**: No download delays
4. **Shared Models**: Multiple Semantik instances can share the same model cache
5. **Backup Friendly**: Easy to backup/restore model files

## Model Sizes

Approximate download sizes:
- `Qwen/Qwen3-Embedding-0.6B`: ~1.2GB
- `Qwen/Qwen3-Embedding-4B`: ~8GB  
- `Qwen/Qwen3-Embedding-8B`: ~16GB

## Troubleshooting

### Permission Errors

If you see `PermissionError at /app/.cache/huggingface/hub`, the container user (UID 1000) doesn't have write access:

```bash
# Quick fix
./scripts/fix-permissions.sh

# Or manually
sudo chown -R 1000:1000 ./models ./data ./logs
```

### Verify Models Are Cached

```bash
# Check model files
ls -la ./models/hub/models--Qwen--Qwen3-Embedding-0.6B/
```

### Clear Model Cache

```bash
# Remove all cached models
rm -rf ./models/*
```

### Download Models Manually

```bash
# Run a one-off container to download models
docker run --rm -it \
  -v ./models:/app/.cache/huggingface \
  -e HF_HOME=/app/.cache/huggingface \
  semantik-webui \
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')"
```

## Best Practices

1. **Download Once**: Let first run complete model downloads
2. **Enable Offline**: Set `HF_HUB_OFFLINE=true` after download
3. **Backup Models**: Include `./models` in your backup strategy
4. **Share Cache**: Use network storage for multi-host deployments