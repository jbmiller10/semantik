# Document Embedding System

A production-ready document embedding and vector search system with web interface, REST API, and advanced search capabilities.

## Features

- 🚀 **High-Performance Vector Search**: Powered by Qdrant with support for semantic, keyword, and hybrid search
- 🤖 **Multiple Embedding Models**: Support for BAAI/BGE, Qwen3, and custom HuggingFace models
- 🎯 **Quantization Support**: float32, float16, and int8 modes for optimal performance/quality balance
- 🌐 **Web Interface**: Full-featured UI for job management, directory scanning, and search
- 🔧 **REST API**: Comprehensive API for programmatic access
- 📊 **Real-time Progress**: WebSocket-based progress tracking for embedding jobs
- 🔐 **Authentication**: JWT-based authentication system
- 🧪 **Mock Mode**: Test without GPU resources

## Quick Start

### Prerequisites

- Python 3.12+
- Qdrant instance running
- GPU with CUDA support (optional, can run in CPU mode)
- Poetry for dependency management

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd document-embedding-project

# Install dependencies
poetry install

# Copy environment configuration
cp .env.example .env
# Edit .env with your configuration

# Create required directories (if not exists)
mkdir -p data/{jobs,output}
mkdir -p logs
```

### Running the System

#### Option 1: Start All Services (Recommended)

```bash
# Start both Search API and Web UI
./start_all_services.sh

# Check service status
./status_services.sh

# Stop all services
./stop_all_services.sh
```

Services will be available at:
- **Web UI**: http://localhost:8080
- **Search API**: http://localhost:8000

#### Option 2: Run Services Individually

```bash
# Start Search API only
poetry run python vecpipe/search_api.py

# Start Web UI only
poetry run python webui/app.py
```

## Using the Web Interface

1. **Access the UI**: Navigate to http://localhost:8080
2. **Login**: Use the authentication system (create account on first use)
3. **Create Embedding Job**:
   - Go to "Create Job" tab
   - Select directories containing documents
   - Choose embedding model (e.g., Qwen/Qwen3-Embedding-0.6B)
   - Select quantization (float16 recommended)
   - Optionally add task instruction
   - Click "Create Job"
4. **Monitor Progress**: Real-time updates via WebSocket
5. **Search Documents**: Use the "Search" tab once embedding is complete

## Architecture

### Core Components

#### 1. **Document Processing Pipeline**
- **Extract** (`vecpipe/extract_chunks.py`): Token-based text extraction and chunking
- **Embed** (`vecpipe/embed_chunks_unified.py`): Unified embedding generation service
- **Ingest** (`vecpipe/ingest_qdrant.py`): Vector database ingestion with error handling

#### 2. **Search Services**
- **REST API** (`vecpipe/search_api.py`): FastAPI service for programmatic access
- **Web UI** (`webui/app.py`): Full-featured web interface with job management
- **Unified Search** (`vecpipe/search_utils.py`): Core search implementation

#### 3. **Embedding Service**
- **Unified Implementation** (`webui/embedding_service.py`): Single service for all embedding needs
- **Adaptive Batch Sizing**: Automatic OOM handling and recovery
- **Multi-Model Support**: BAAI/BGE, Qwen3, Sentence Transformers, custom models
- **Quantization**: float32, float16, int8 support with bitsandbytes

## API Documentation

### Search API Endpoints

#### Basic Search
```bash
curl "http://localhost:8000/search?q=machine+learning&k=5"
```

#### Advanced Search
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

#### Batch Search
```bash
curl -X POST "http://localhost:8000/search/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is BERT?", "How does GPT work?", "Explain attention mechanism"],
    "k": 5,
    "search_type": "question"
  }'
```

#### Hybrid Search
```bash
curl "http://localhost:8000/hybrid_search?q=python+docker&k=10&mode=filter&keyword_mode=any"
```

### Web UI API Endpoints

- `POST /api/scan-directory` - Scan directory for documents
- `POST /api/jobs` - Create new embedding job
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/search` - Search documents
- `POST /api/hybrid_search` - Hybrid search
- `GET /api/models` - Get available embedding models
- `WS /ws/{job_id}` - WebSocket for real-time job updates

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
DEFAULT_COLLECTION=work_docs

# Embedding Model Configuration
USE_MOCK_EMBEDDINGS=false
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_QUANTIZATION=float16

# Authentication (generate with: openssl rand -hex 32)
JWT_SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=1440
```

### Supported Embedding Models

| Model | Dimensions | Description |
|-------|------------|-------------|
| BAAI/bge-large-en-v1.5 | 1024 | High quality general purpose |
| BAAI/bge-base-en-v1.5 | 768 | Balanced quality and speed |
| Qwen/Qwen3-Embedding-0.6B | 1024 | Fast, good quality |
| Qwen/Qwen3-Embedding-4B | 2560 | Balanced performance |
| Qwen/Qwen3-Embedding-8B | 4096 | Highest quality |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | Very fast |
| sentence-transformers/all-mpnet-base-v2 | 768 | High quality |
| Custom models | Varies | Any HuggingFace sentence transformer |

## Advanced Features

### Hybrid Search

Combines vector similarity with keyword matching for improved accuracy:

```python
# Filter mode - uses Qdrant's text filtering
GET /hybrid_search?q=python+docker&mode=filter&keyword_mode=any

# Rerank mode - retrieves more candidates and re-scores
GET /hybrid_search?q=machine+learning&mode=rerank&keyword_mode=all
```

### Batch Processing

Process multiple queries efficiently:

```python
POST /search/batch
{
    "queries": ["query1", "query2", "query3"],
    "k": 10,
    "search_type": "semantic"
}
```

### Task-Specific Instructions

Improve search quality with custom instructions:

```python
# For document indexing
"Represent this document for retrieval:"

# For search queries
"Represent this sentence for searching relevant passages:"

# For Q&A scenarios
"Represent this question for retrieving supporting answers:"
```

## Testing

### Run Test Suite

```bash
# Run all tests
poetry run pytest

# Run specific test module
poetry run pytest tests/test_embedding.py

# Run with coverage
poetry run pytest --cov=vecpipe --cov=webui
```

### Test Scripts

- `test_qdrant_connection.py` - Verify Qdrant connectivity
- `test_hybrid_search.py` - Test hybrid search functionality
- `test_search_unification.py` - Verify search API consistency
- `test_qwen3_search.py` - Test Qwen3 model integration

### Mock Mode Testing

```bash
# Run embedding service in mock mode (no GPU required)
python vecpipe/embed_chunks_unified.py --mock

# Run search API with mock embeddings
USE_MOCK_EMBEDDINGS=true python vecpipe/search_api.py
```

## Troubleshooting

### Common Issues

1. **Qdrant Connection Failed**
   - Verify Qdrant is running and accessible
   - Check QDRANT_HOST and QDRANT_PORT in .env
   - Test connection: `curl http://localhost:6333/collections`

2. **Out of Memory Errors**
   - Use more aggressive quantization (int8)
   - Reduce batch size in embedding service
   - Switch to smaller model (e.g., Qwen3-0.6B)

3. **Slow Performance**
   - Enable GPU acceleration if available
   - Use float16 quantization
   - Increase batch size if GPU memory allows

4. **Authentication Issues**
   - Ensure JWT_SECRET_KEY is set in .env
   - Clear browser cookies and retry
   - Check webui.log for detailed errors

### Logs and Monitoring

```bash
# View service logs
tail -f search_api.log
tail -f webui.log

# Check service status
./status_services.sh

# Monitor GPU usage (if available)
nvidia-smi -l 1
```

## Development

### Code Quality

```bash
# Run linter
poetry run ruff check .

# Format code
poetry run black .

# Type checking
poetry run mypy .
```

### Project Structure

```
├── vecpipe/           # Core pipeline components
│   ├── extract_chunks.py
│   ├── embed_chunks_unified.py
│   ├── ingest_qdrant.py
│   ├── search_api.py
│   ├── search_utils.py
│   └── hybrid_search.py
├── webui/             # Web interface
│   ├── app.py
│   ├── embedding_service.py
│   ├── auth.py
│   └── static/
├── tests/             # Test suite
├── scripts/           # Utility scripts
└── data/              # Runtime data
    ├── jobs/
    └── output/
```

## License

This project is proprietary and confidential.