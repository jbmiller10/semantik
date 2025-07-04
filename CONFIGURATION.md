# Configuration Guide

This guide covers all configuration options for the Document Embedding System.

## Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env` and customize as needed.

### Core Settings

#### Qdrant Configuration
```bash
# Qdrant vector database connection
QDRANT_HOST=localhost          # Qdrant server hostname/IP
QDRANT_PORT=6333              # Qdrant server port
DEFAULT_COLLECTION=work_docs   # Default collection name
```

#### Embedding Model Configuration
```bash
# Embedding service settings
USE_MOCK_EMBEDDINGS=false     # Use mock embeddings for testing (true/false)
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B  # Default model
DEFAULT_QUANTIZATION=float16  # Quantization mode: float32, float16, int8
MODEL_UNLOAD_AFTER_SECONDS=300  # Unload models after N seconds of inactivity (default: 300)
```

#### Authentication
```bash
# JWT authentication for Web UI
JWT_SECRET_KEY=your-secret-key-here  # Generate with: openssl rand -hex 32
ACCESS_TOKEN_EXPIRE_MINUTES=1440     # Token expiration (24 hours default)
```

### Optional Path Configuration

These paths have sensible defaults but can be overridden:

```bash
# Data directories
FILE_TRACKING_DB=/var/embeddings/file_tracking.json
WEBUI_DB=/var/embeddings/webui.db
EXTRACT_DIR=/opt/semantik/extract
INGEST_DIR=/var/embeddings/ingest
LOADED_DIR=/var/embeddings/loaded
REJECT_DIR=/var/embeddings/rejects

# File lists and logs
MANIFEST_FILE=/var/embeddings/filelist.null
ERROR_LOG=/var/embeddings/error_extract.log
CLEANUP_LOG=/var/embeddings/cleanup.log
```

## Model Configuration

### Available Models

The system supports various embedding models:

| Model | Environment Variable | Dimensions | Notes |
|-------|---------------------|------------|-------|
| BAAI/bge-large-en-v1.5 | `DEFAULT_EMBEDDING_MODEL` | 1024 | High quality, general purpose |
| BAAI/bge-base-en-v1.5 | `DEFAULT_EMBEDDING_MODEL` | 768 | Balanced performance |
| Qwen/Qwen3-Embedding-0.6B | `DEFAULT_EMBEDDING_MODEL` | 1024 | Fast, good quality (default) |
| Qwen/Qwen3-Embedding-4B | `DEFAULT_EMBEDDING_MODEL` | 2560 | Higher quality, slower |
| Qwen/Qwen3-Embedding-8B | `DEFAULT_EMBEDDING_MODEL` | 4096 | Best quality, requires more GPU |
| sentence-transformers/all-MiniLM-L6-v2 | `DEFAULT_EMBEDDING_MODEL` | 384 | Very fast, lower quality |
| sentence-transformers/all-mpnet-base-v2 | `DEFAULT_EMBEDDING_MODEL` | 768 | Good balance |

### Quantization Modes

Control memory usage and performance:

- `float32`: Full precision (best quality, highest memory use)
- `float16`: Half precision (recommended - good balance)
- `int8`: 8-bit quantization (lowest memory, slight quality loss)

### Task Instructions

For Qwen3 models, use task-specific instructions:

```python
# Document indexing
"Represent this document for retrieval:"

# Search queries
"Represent this sentence for searching relevant passages:"

# Q&A scenarios
"Represent this question for retrieving supporting answers:"
```

## Service Configuration

### Search API

The Search API (`vecpipe/search_api.py`) uses these environment variables:

- `QDRANT_HOST`, `QDRANT_PORT`: Database connection
- `DEFAULT_COLLECTION`: Default search collection
- `USE_MOCK_EMBEDDINGS`: Enable mock mode
- `DEFAULT_EMBEDDING_MODEL`: Model for query embeddings
- `DEFAULT_QUANTIZATION`: Quantization mode

### Web UI

The Web UI (`webui/main.py`) additionally uses:

- `JWT_SECRET_KEY`: Required for authentication
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Session duration
- Database paths for job management

## Runtime Configuration

### Embedding Service Options

When creating jobs or using the API, you can override defaults:

```python
{
  "model_name": "Qwen/Qwen3-Embedding-4B",
  "quantization": "int8",
  "batch_size": 32,
  "instruction": "Custom instruction"
}
```

### Search Options

Configure search behavior per request:

```python
{
  "search_type": "semantic",  # or "question", "code", "hybrid"
  "include_content": true,    # Include full text in results
  "metadata_filter": {},      # Filter by metadata fields
  "score_threshold": 0.7      # Minimum similarity score
}
```

### Hybrid Search Modes

- `filter`: Use keywords as pre-filter (faster)
- `rerank`: Retrieve more candidates and re-score (more accurate)

## Performance Tuning

### GPU Configuration

The system automatically detects and uses available GPUs. For specific GPU selection:

```bash
# Select specific GPU
CUDA_VISIBLE_DEVICES=0 python vecpipe/search_api.py

# Use CPU only
CUDA_VISIBLE_DEVICES="" python vecpipe/search_api.py
```

### Batch Sizes

Adjust based on available memory:

- GPU with 8GB VRAM: batch_size=96
- GPU with 4GB VRAM: batch_size=32
- CPU mode: batch_size=8

### Adaptive Batch Sizing

The embedding service automatically adjusts batch size on OOM errors. Configure:

```python
# In webui/embedding_service.py
MIN_BATCH_SIZE = 4          # Minimum batch size
BATCH_RESTORE_THRESHOLD = 5  # Successful batches before increasing size
```

## Logging Configuration

### Log Levels

Set logging verbosity:

```bash
# In Python scripts
import logging
logging.basicConfig(level=logging.INFO)  # or DEBUG, WARNING, ERROR
```

### Log Files

Services write to these log files:

- `search_api.log`: Search API logs
- `webui.log`: Web UI logs
- `data/jobs/job_*/output.log`: Individual job logs

## Security Configuration

### Authentication

1. **Generate secure JWT secret**:
   ```bash
   openssl rand -hex 32
   ```

2. **Set in .env**:
   ```bash
   JWT_SECRET_KEY=your-generated-secret
   ```

3. **Configure token expiration**:
   ```bash
   ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours
   ```

### CORS Settings

Configure allowed origins in `webui/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Development Configuration

### Mock Mode

For testing without GPU:

```bash
# Enable mock embeddings
USE_MOCK_EMBEDDINGS=true

# Or use CLI flag
python vecpipe/embed_chunks_unified.py --mock
```

### Debug Mode

Enable detailed logging:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG

# Enable FastAPI debug mode
export DEBUG=true
```

## Production Recommendations

1. **Security**:
   - Use strong JWT secret
   - Enable HTTPS in production
   - Restrict CORS origins

2. **Performance**:
   - Use float16 quantization
   - Enable GPU acceleration
   - Configure appropriate batch sizes

3. **Reliability**:
   - Set up proper logging
   - Monitor disk space for embeddings
   - Regular Qdrant backups

4. **Scalability**:
   - Use external Qdrant cluster
   - Consider load balancing for API
   - Implement caching layer