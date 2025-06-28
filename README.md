# Document Embedding Project - Vector Search MVP

A production-ready vector search system for document retrieval using Qdrant 1.8.

## Architecture Overview

- **Document Storage**: ZFS-backed document archive on Unraid server
- **Vector Database**: Qdrant 1.8 with HNSW indexing and disk-based storage
- **Embedding Model**: BAAI/bge-large-en-v1.5 (1024-dimensional vectors)
- **Processing**: Multi-stage pipeline with extraction, embedding, and ingestion
- **API**: FastAPI-based search service

## Quick Start

### Prerequisites

- Python 3.12+
- Qdrant instance running (default: 192.168.1.173:6333)
- Access to document directories

### Installation

```bash
# Clone the repository
cd /root/document-embedding-project

# Install dependencies
poetry install

# Copy environment configuration
cp .env.example .env
# Edit .env with your local configuration

# Create required directories
mkdir -p /opt/vecpipe/{extract,embed,tmp}
mkdir -p /var/embeddings/{ingest,loaded,rejects}
```

### Initial Setup

1. **Create Qdrant Collection**:
```bash
python3 scripts/test_qdrant_simple.py
```

2. **Test Pipeline on Sample Data**:
```bash
# Create test file list
echo -n "test_data/sample.txt" > /tmp/test_filelist.null

# Extract chunks
python3 vecpipe/extract_chunks.py -i /tmp/test_filelist.null -o /opt/vecpipe/extract

# Generate embeddings
python3 vecpipe/embed_chunks_simple.py -i /opt/vecpipe/extract -o /var/embeddings/ingest

# Ingest to Qdrant
python3 vecpipe/ingest_qdrant.py
```

3. **Start Search API**:
```bash
python3 vecpipe/search_api.py
```

### Search API Examples

Basic search:
```bash
curl "http://localhost:8000/search?q=machine+learning&k=5"
```

Advanced search with options:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are transformers in AI?",
    "k": 10,
    "search_type": "question",
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16"
  }'
```

Batch search:
```bash
curl -X POST "http://localhost:8000/search/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is BERT?", "How does GPT work?", "Explain attention mechanism"],
    "k": 5,
    "search_type": "question"
  }'
```

## Components

### 1. Manifest Generation (`scripts/build_manifest.sh`)
- Scans configured directories for PDF, DOCX, and TXT files
- Creates null-delimited file list for processing

### 2. Document Extraction (`vecpipe/extract_chunks.py`)
- Parses documents using pypdf and python-docx
- Chunks text with 600-token segments and 200-token overlap
- Outputs parquet files with document metadata

### 3. Embedding Generation (`vecpipe/embed_chunks_simple.py`)
- Generates vector embeddings for text chunks
- Currently using mock embeddings for testing
- For production: Replace with actual BGE model on GPU

### 4. Qdrant Ingestion (`vecpipe/ingest_qdrant.py`)
- Bulk uploads vectors to Qdrant collection
- Handles retries and error recovery
- Moves processed files to loaded/rejects directories

### 5. Search API (`vecpipe/search_api.py`)
- Unified FastAPI service for vector similarity search with advanced features
- Endpoints:
  - `GET /` - Detailed health check with service status
  - `GET /search?q=query&k=10` - Basic search (GET for compatibility)
  - `POST /search` - Advanced search with options:
    - Multiple search types: semantic, question, code, hybrid
    - Model and quantization selection
    - Metadata filters
    - Include content in results
  - `POST /search/batch` - Batch search for multiple queries
  - `GET /hybrid_search` - Hybrid search combining vector and keyword matching
  - `GET /keyword_search` - Keyword-only search
  - `GET /collection/info` - Collection statistics
  - `GET /models` - List available embedding models
  - `POST /models/load` - Dynamically load different models
  - `GET /embedding/info` - Current embedding configuration

## Deployment

### Running Services

For development, use the provided shell scripts:
```bash
# Start all services
poetry run ./start_all_services.sh

# Check service status
./status_services.sh

# Stop all services
./stop_all_services.sh
```

### Environment Variables

Configure your environment by editing the `.env` file:
- `QDRANT_HOST`: Qdrant server address (default: 192.168.1.173)
- `QDRANT_PORT`: Qdrant server port (default: 6333)
- `JWT_SECRET_KEY`: Secret key for JWT authentication
- See `.env.example` for all available configuration options

## Monitoring

When running services with the shell scripts:

```bash
# Check service status
./status_services.sh

# View logs (services log to console when run via shell scripts)
# Each service runs in its own terminal/screen session
```

## Performance Tuning

### Qdrant Configuration
- HNSW index: m=32, ef_construct=200
- Disk-based storage with memmap_threshold=0
- On-disk payload storage

### Processing Parameters
- Extraction: Batch size 100, multiprocessing
- Embedding: Batch size 96 (GPU) or 8 (CPU)
- Ingestion: Batch size 4000, 4 parallel workers

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Check Qdrant is running: `curl http://192.168.1.173:6333/collections`
   - Verify network connectivity

2. **Extraction Errors**
   - Check error log: `/tmp/error_extract.log`
   - Verify file permissions and formats

3. **Ingestion Failures**
   - Check rejected files: `/var/embeddings/rejects/`
   - Review Qdrant logs for capacity issues

### Manual Operations

Re-process failed files:
```bash
# Move rejects back to ingest
mv /var/embeddings/rejects/*.parquet /var/embeddings/ingest/

# Run ingestion manually
python3 vecpipe/ingest_qdrant.py
```

## Web UI

The project includes a comprehensive web interface for managing embeddings.

### Features

- **Directory Scanning**: Browse and select directories containing documents
- **Model Selection**: Choose from popular HuggingFace embedding models or specify custom ones
- **Job Management**: Create, monitor, and manage embedding jobs
- **Progress Tracking**: Real-time updates via WebSocket
- **Search Interface**: Search across different embedding collections
- **Resume Capability**: Jobs can be resumed if interrupted

### Running the Web UI

```bash
cd /root/document-embedding-project
python -m uvicorn webui.app:app --host 0.0.0.0 --port 8080
```


### Accessing the UI

Open your browser to `http://localhost:8080`

### Web UI Endpoints

- `GET /` - Main web interface
- `POST /api/scan-directory` - Scan directory for documents
- `POST /api/jobs` - Create new embedding job
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/search` - Search documents
- `GET /api/models` - Get available embedding models
- `WS /ws/{job_id}` - WebSocket for real-time job updates

### Supported Embedding Models

The web UI supports popular models from HuggingFace:

- **BAAI/bge-large-en-v1.5** (1024d) - High quality general purpose
- **BAAI/bge-base-en-v1.5** (768d) - Balanced quality and speed
- **sentence-transformers/all-MiniLM-L6-v2** (384d) - Very fast
- **sentence-transformers/all-mpnet-base-v2** (768d) - High quality
- **thenlper/gte-large** (1024d) - State-of-the-art
- **intfloat/e5-large-v2** (1024d) - Excellent for semantic search
- Custom models - Any HuggingFace sentence transformer

## Next Steps

1. **GPU Support**: Enable CUDA for faster embedding generation
2. **Authentication**: Add JWT/OIDC to web UI and APIs
3. **Monitoring**: Integrate Prometheus metrics and Grafana dashboards
4. **Backup**: Implement automated Qdrant snapshots
5. **Clustering**: Support for distributed processing across multiple nodes

## License

This project is proprietary and confidential.