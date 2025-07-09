# Quick Fixes for Common Issues

## 1. Permission Errors When Downloading Models

**Error**: `PermissionError at /app/.cache/huggingface/hub when downloading`

**Solution**:
```bash
# Run the fix script
./scripts/fix-permissions.sh

# Or manually fix permissions (container runs as UID 1000)
sudo chown -R 1000:1000 ./models ./data ./logs
```

## 2. Bitsandbytes INT8 Quantization Errors

**Error**: `Could not import module 'validate_bnb_backend_availability'`

**Solution**: This usually means bitsandbytes isn't installed. Rebuild your containers:
```bash
# Standard build (now includes gcc for INT8 support)
docker compose up -d --build

# Or use CUDA build for better GPU performance
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

**Error**: `Failed to find C compiler. Please specify via CC environment variable`

**Solution**: This error should not occur with the latest builds. If you see it:
```bash
# Your container may be outdated - rebuild:
docker compose down
docker compose up -d --build
```

## 3. HuggingFace Rate Limiting (429 Errors)

**Solution**: Enable offline mode after first download:
```bash
# In your .env file:
HF_HUB_OFFLINE=true
```

## 4. Pre-download Models to Avoid Issues

```bash
# Download models before starting containers
./scripts/download-models.sh --model Qwen/Qwen3-Embedding-0.6B
```

## Complete Fresh Start

```bash
# 1. Stop everything
docker compose down

# 2. Fix permissions
./scripts/fix-permissions.sh

# 3. Pre-download models (optional but recommended)
./scripts/download-models.sh

# 4. Start with CUDA support
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build

# 5. Watch logs
docker compose logs -f webui
```