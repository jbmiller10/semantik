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

- Python 3.8+
- Qdrant instance running (default: 192.168.1.173:6333)
- Access to document directories

### Installation

```bash
# Clone the repository
cd /root/document-embedding-project

# Install dependencies
pip install -r requirements.txt

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

## Components

### 1. Manifest Generation (`scripts/build_manifest.sh`)
- Scans configured directories for PDF, DOCX, and TXT files
- Creates null-delimited file list for processing
- Runs hourly via systemd timer

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
- FastAPI service for vector similarity search
- Endpoints:
  - `GET /` - Health check
  - `GET /search?q=query&k=10` - Search documents
  - `GET /collection/info` - Collection statistics

## Deployment

### Systemd Services

Copy service files to systemd directory:
```bash
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo cp deploy/systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable and start services:
```bash
# Enable timers
sudo systemctl enable --now manifest.timer
sudo systemctl enable --now extract.timer
sudo systemctl enable --now embed.timer
sudo systemctl enable --now ingest.timer

# Enable API service
sudo systemctl enable --now search-api.service
```

### Environment Variables

- `QDRANT_HOST`: Qdrant server address (default: 192.168.1.173)
- `QDRANT_PORT`: Qdrant server port (default: 6333)

## Monitoring

Check service status:
```bash
systemctl status manifest.timer extract.timer embed.timer ingest.timer
systemctl status search-api.service
```

View logs:
```bash
journalctl -u manifest.service -f
journalctl -u extract.service -f
journalctl -u embed.service -f
journalctl -u ingest.service -f
journalctl -u search-api.service -f
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

#### Option 1: Direct Python
```bash
cd /root/document-embedding-project
python -m uvicorn webui.app:app --host 0.0.0.0 --port 8080
```

#### Option 2: Docker
```bash
# Build and run with Docker Compose
docker-compose up -d webui

# View logs
docker-compose logs -f webui
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