# Semantik 🚀 - Production-Ready Semantic Search Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Semantik** is the only self-hosted semantic search engine with enterprise-grade GPU memory management and true microservices architecture. Built for technical users who need **control**, **performance**, and **privacy**.

![Semantik Dashboard](docs/images/semantik-dashboard.png)

## 🎯 Why Semantik?

### The Problem
- **Cloud services** (Algolia, Pinecone) lock your data in their infrastructure
- **Libraries** (FAISS, ChromaDB) require you to build the application layer  
- **Existing solutions** lack production features like GPU management and quantization
- **RAG demos** aren't ready for real workloads

### The Solution
Semantik bridges the gap with:
- 🧠 **Adaptive GPU Memory Management** - Automatic model loading/unloading
- 🏗️ **True Microservices** - Use the search engine without the UI
- 📊 **Production Metrics** - Prometheus-ready monitoring out of the box
- 🔒 **Complete Data Control** - Self-hosted with no external dependencies

## ⚡ Quick Start (5 minutes)

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- 8GB+ RAM (16GB+ recommended)
- GPU with CUDA support (optional)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/semantik.git
cd semantik

# Start all services with Docker Compose
docker-compose up -d

# Services will be available at:
# - Web UI: http://localhost:8080
# - Search API: http://localhost:8000
# - Qdrant: http://localhost:6333
```

### 2. Create Your First Search Index

```bash
# Option A: Use the included demo script
./scripts/setup_demo_data.sh

# Option B: Use the API directly
curl -X POST http://localhost:8080/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-docs",
    "directory_path": "/path/to/your/documents",
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "quantization": "float16"
  }'
```

### 3. Search Your Documents

```bash
# Semantic search
curl "http://localhost:8000/search?q=how+to+configure+nginx&k=5"

# Hybrid search (semantic + keyword)
curl "http://localhost:8000/hybrid_search?q=python+docker&mode=filter"
```

## 🌟 Key Features

### 🧠 Intelligent Resource Management
- **Adaptive Batch Sizing**: Automatically reduces batch size on OOM errors
- **Model Lazy Loading**: Models load on-demand and unload after 5 minutes
- **GPU Memory Tracking**: Real-time monitoring and optimization
- **Quantization Support**: float32, float16, and int8 modes

### 🔍 Advanced Search Capabilities
- **Semantic Search**: State-of-the-art embedding models
- **Hybrid Search**: Combines vector similarity with keyword matching
- **Question-Answering**: Optimized prompts for Q&A scenarios
- **Batch Processing**: Efficient multi-query operations
- **Collections Management**: Organize and search across multiple data sources

### 📊 Production-Ready Features
- **Prometheus Metrics**: Complete observability
- **JWT Authentication**: Secure multi-user support
- **Rate Limiting**: API protection built-in
- **WebSocket Progress**: Real-time job monitoring
- **Error Recovery**: Automatic retry and fallback

### 🏗️ Clean Architecture
```
┌─────────────────┐     ┌──────────────────┐
│    Web UI       │────▶│   Control Plane   │
│  (React 19)     │     │   (FastAPI)       │
└─────────────────┘     └──────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │   Search API      │
                        │   (Semantik Core) │
                        └──────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │    Qdrant DB      │
                        │  (Vector Store)   │
                        └──────────────────┘
```

## 🚀 Deployment Options

### Docker Compose (Coming Soon)

> **Note**: Docker deployment is currently under development. For now, please use the manual installation method above.

<!-- Example configuration (not yet available):
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  semantik:
    build: ./packages/vecpipe
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant
      - USE_CUDA=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  webui:
    build: ./packages/webui
    ports:
      - "8080:8080"
    environment:
      - SEARCH_API_URL=http://semantik:8000
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}

volumes:
  qdrant_data:
```
-->

### Kubernetes (Helm Chart - Coming Soon)

> **Note**: Kubernetes deployment via Helm chart is currently under development.

### Manual Installation

Follow the Quick Start guide above for manual installation instructions.

## 🔧 Configuration

### Embedding Models

Semantik supports any Sentence Transformer model. Popular choices:

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|-----------|
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | Good | Fast | General purpose |
| `BAAI/bge-large-en-v1.5` | 0.3B | Excellent | Medium | English documents |
| `intfloat/multilingual-e5-large` | 0.6B | Excellent | Medium | Multi-language |
| `sentence-transformers/all-MiniLM-L6-v2` | 0.02B | Decent | Very Fast | Quick prototypes |

### Quantization Modes

| Mode | Memory Usage | Quality | Speed |
|------|--------------|---------|-------|
| float32 | 100% | Best | Baseline |
| float16 | 50% | 99.9% | 1.5x faster |
| int8 | 25% | 98% | 2x faster |

## 🔌 API Examples

### Python SDK

```python
from semantik import SemantikClient

client = SemantikClient("http://localhost:8000")

# Create embedding job
job = client.create_job(
    name="technical-docs",
    directory="/docs",
    model="Qwen/Qwen3-Embedding-0.6B",
    quantization="float16"
)

# Search documents
results = client.search(
    query="How to deploy with Docker?",
    k=5,
    search_type="hybrid"
)

for result in results:
    print(f"Score: {result.score:.3f} - {result.file_path}")
    print(f"Content: {result.text[:200]}...\n")
```

### REST API

```bash
# Create embedding job
curl -X POST http://localhost:8080/api/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @- << EOF
{
  "name": "my-knowledge-base",
  "directory_path": "/data/documents",
  "model_name": "BAAI/bge-large-en-v1.5",
  "quantization": "float16",
  "chunk_size": 600,
  "chunk_overlap": 200
}
EOF

# Monitor job progress via WebSocket
wscat -c ws://localhost:8080/ws/job-id

# Batch search
curl -X POST http://localhost:8000/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is transformer architecture?",
      "How does attention mechanism work?",
      "Explain BERT vs GPT"
    ],
    "k": 5,
    "search_type": "semantic"
  }'
```

## 📊 Monitoring & Metrics

Semantik exposes Prometheus metrics at `/metrics`:

```prometheus
# Search latency by operation
search_api_latency_seconds{endpoint="/search",search_type="semantic"} 0.096

# Embedding generation performance  
embedding_duration_seconds{model="Qwen/Qwen3-Embedding-0.6B",quantization="float16"} 0.251

# GPU memory usage
gpu_memory_used_mb{device="0"} 2150

# OOM recovery events
embedding_oom_errors_total{model="Qwen/Qwen3-Embedding-8B",quantization="float32"} 3
batch_size_reductions_total{model="Qwen/Qwen3-Embedding-8B"} 3
```

## 🤝 Semantik vs Alternatives

| Feature | Semantik | Elasticsearch | Pinecone | ChromaDB | Weaviate |
|---------|---------|---------------|----------|----------|----------|
| Self-hosted | ✅ | ✅ | ❌ | ✅ | ✅ |
| GPU Management | ✅ Adaptive | ❌ | N/A | ❌ | ❌ |
| Quantization | ✅ 3 modes | ❌ | ❌ | ❌ | ❌ |
| Microservices | ✅ | ❌ | N/A | ❌ | ❌ |
| Production UI | ✅ | ✅ | ✅ | ❌ | ✅ |
| Hybrid Search | ✅ | ✅ | ❌ | ❌ | ✅ |
| Model Hot-swap | ✅ | ❌ | ❌ | ❌ | ❌ |
| Prometheus Metrics | ✅ | ✅ | ❌ | ❌ | ✅ |
| True Open Source | ✅ AGPL | ✅ Elastic | ❌ | ✅ Apache | ✅ BSD |

## 🛠️ Troubleshooting

### Common Issues

<details>
<summary><b>🔴 Authentication Error: "Invalid credentials"</b></summary>

```bash
# Clear browser cookies and local storage
# Generate new JWT secret
openssl rand -hex 32 > jwt_secret
# Update .env file
echo "JWT_SECRET_KEY=$(cat jwt_secret)" >> .env
# Restart services
docker-compose restart webui
```
</details>

<details>
<summary><b>🟡 GPU Out of Memory</b></summary>

```bash
# Option 1: Use more aggressive quantization
curl -X POST http://localhost:8080/api/jobs \
  -d '{"quantization": "int8", ...}'

# Option 2: Use smaller model
curl -X POST http://localhost:8080/api/jobs \
  -d '{"model_name": "sentence-transformers/all-MiniLM-L6-v2", ...}'

# Option 3: Reduce batch size (automatic in Semantik)
```
</details>

<details>
<summary><b>🟡 Slow Search Performance</b></summary>

```bash
# Enable hybrid search for better results
curl "http://localhost:8000/hybrid_search?q=your+query&mode=filter"

# Check metrics for bottlenecks
curl http://localhost:8000/metrics | grep search_api_latency

# Ensure GPU is being used
curl http://localhost:8000/api/health | jq .gpu_available
```
</details>

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Development setup
poetry install
make format  # Format code
make lint    # Run linters
make test    # Run tests
```

## 📚 Documentation

- [Architecture Overview](docs/ARCH.md)
- [API Reference](API_REFERENCE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Collections Management](docs/COLLECTIONS.md)
- [Hybrid Search](HYBRID_SEARCH.md)
- [Database Architecture](docs/DATABASE_ARCH.md)
- [WebUI Backend](docs/WEBUI_BACKEND.md)
- [Frontend Architecture](docs/FRONTEND_ARCH.md)

## 🚀 Roadmap

- [ ] **v2.1** - Multi-modal search (images, PDFs)
- [ ] **v2.2** - Streaming API for real-time indexing
- [ ] **v2.3** - Distributed computing support
- [ ] **v2.4** - AutoML for model selection
- [ ] **v3.0** - Native cloud integrations (S3, GCS)

## 📄 License

Semantik is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with excellent open source projects:
- [Qdrant](https://qdrant.tech) - Vector database
- [FastAPI](https://fastapi.tiangolo.com) - API framework
- [Sentence Transformers](https://sbert.net) - Embedding models
- [React](https://react.dev) - UI framework

---

<p align="center">
  <a href="https://github.com/jbmiller10/semantik">⭐ Star us on GitHub</a> •
</p>

