# Semantik Document Embedding System - Comprehensive Architecture Documentation

## Executive Summary

Semantik is a production-ready, high-performance document embedding and vector search system designed for technical users who prioritize performance and control. The system features a largely clean separation between its core search engine (vecpipe) and control plane (WebUI), with a pragmatic exception for the embedding service to avoid code duplication. This architecture enables both standalone usage and user-friendly management through a modern React interface.

### Key Features
- 🚀 High-performance vector search powered by Qdrant
- 🤖 Support for multiple embedding models with quantization (float32, float16, int8)
- 🎯 Hybrid search combining vector similarity and keyword matching
- 🌐 Full-featured web interface with real-time progress tracking
- 🔧 Comprehensive REST API for programmatic access
- 🔐 JWT-based authentication system
- 📊 Prometheus metrics for monitoring
- 🧪 Mock mode for testing without GPU resources

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Interface                              │
│                        React Frontend (Port 8080)                        │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │ HTTP/WebSocket
┌────────────────────────────────┴───────────────────────────────────────┐
│                           WebUI Backend                                 │
│                     FastAPI Control Plane (Port 8080)                   │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │Auth Service │  │Job Management│  │Search Proxy │  │ WebSockets  │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────┘ │
└────────┬───────────────────┬──────────────┬────────────────────────────┘
         │                   │              │
         │ SQLite           │              │ HTTP Proxy
┌────────┴─────────┐        │              │
│   User Database  │        │              │
│  Jobs, Files,    │        │              │
│  Users, Tokens   │        │              │
└──────────────────┘        │              │
                           │              │
┌──────────────────────────┴──────────────┴─────────────────────────────┐
│                          Semantik Core Engine                          │
│                      Search API (Port 8000)                            │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Extract    │  │   Embed      │  │   Ingest    │  │   Search   │ │
│  │  Chunks     │  │   Chunks     │  │   Qdrant    │  │   Utils    │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └────────────┘ │
└────────────────────────────┬──────────────────────────────────────────┘
                            │
                            │ gRPC/HTTP
┌───────────────────────────┴────────────────────────────────────────────┐
│                         Qdrant Vector Database                          │
│                     Vector Storage & Similarity Search                  │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Semantik Core Engine (`packages/vecpipe/`)

The headless data processing and search API that forms the heart of the system.

**Key Components:**
- **extract_chunks.py**: Document parsing and token-based chunking
- **embed_chunks_unified.py**: Unified embedding generation with adaptive batching
- **ingest_qdrant.py**: Vector database ingestion with retry mechanisms
- **search_api.py**: FastAPI service exposing search functionality
- **model_manager.py**: GPU memory management with lazy loading and automatic unloading
- **hybrid_search.py**: Combined vector and keyword search implementation
- **reranker.py**: Cross-encoder reranking for improved search accuracy

**ModelManager Architecture:**
The ModelManager provides intelligent model lifecycle management:
- **Lazy Loading**: Models are loaded only when needed
- **Automatic Unloading**: Models are unloaded after configurable inactivity (default: 300s)
- **Memory Tracking**: Monitors GPU memory usage and prevents OOM errors
- **Multi-Model Support**: Manages both embedding and reranker models independently
- **Thread-Safe**: Handles concurrent requests with proper locking

**Design Principles:**
- Modular pipeline architecture
- Resource-efficient with automatic GPU memory management
- Supports multiple embedding models and quantization levels
- Robust error handling and recovery

For detailed documentation, see [SEMANTIK_CORE.md](./SEMANTIK_CORE.md)

### 2. WebUI Control Plane (`packages/webui/`)

User-facing application for job management and search interface.

**Backend Components:**
- **main.py**: FastAPI application with modular router architecture
- **api/**: RESTful API routers for auth, jobs, search, files, metrics
- **database.py**: SQLite database management
- **auth.py**: JWT-based authentication
- **embedding_service.py**: Embedding generation service (see Architectural Note below)

**Architectural Note - embedding_service.py:**
While the architecture emphasizes separation between vecpipe and webui packages, `embedding_service.py` is an intentional exception:
- **Location**: `packages/webui/embedding_service.py`
- **Used by**: Both webui (for job processing) and vecpipe (for search)
- **Rationale**: Avoids code duplication for complex embedding logic
- **Impact**: Creates coupling between packages but ensures consistency
- **Future**: May be refactored into a separate shared package

**Frontend (React):**
- Modern React 19 with TypeScript
- Zustand for state management
- Real-time WebSocket updates
- Tailwind CSS for styling

For detailed documentation, see:
- [WEBUI_BACKEND.md](./WEBUI_BACKEND.md)
- [FRONTEND_ARCH.md](./FRONTEND_ARCH.md)

### 3. Database Architecture

**Hybrid Database Design:**
- **SQLite**: Relational data (jobs, files, users, auth tokens)
- **Qdrant**: Vector storage and similarity search

**Key Design Decisions:**
- Clear separation between metadata and vectors
- Optimized for single-instance deployments
- Comprehensive indexing for performance

For detailed documentation, see [DATABASE_ARCH.md](./DATABASE_ARCH.md)

## Data Flow

### Document Processing Pipeline

```
1. User uploads documents via WebUI
   ↓
2. WebUI creates job record in SQLite
   ↓
3. Directory scan identifies processable files
   ↓
4. Extract service chunks documents by tokens
   ↓
5. Embed service generates vectors (with batching)
   ↓
6. Ingest service stores vectors in Qdrant
   ↓
7. WebSocket updates UI with progress
```

### Search Flow

```
1. User enters search query in UI
   ↓
2. WebUI proxies request to Search API
   ↓
3. Search API generates query embedding
   ↓
4. Qdrant performs similarity search
   ↓
5. Results enriched with metadata
   ↓
6. Response returned through proxy
```

## API Architecture

### Search API (Port 8000)
- Pure REST API for vector/hybrid search
- No authentication required
- Supports batch operations
- Model lazy-loading for efficiency

### WebUI API (Port 8080)
- JWT-authenticated endpoints
- Job management and monitoring
- Search proxy with authentication
- WebSocket for real-time updates

For complete API documentation, see [API_ARCHITECTURE.md](./API_ARCHITECTURE.md)

## Search Capabilities

### Search Types
1. **Vector Search**: Semantic similarity using embeddings
   - Optional cross-encoder reranking for improved accuracy
2. **Hybrid Search**: Combines vector + keyword matching
   - Filter mode: Uses Qdrant's text filtering
   - Rerank mode: Post-processes with weighted scoring
3. **Keyword Search**: Pure text-based without embeddings
4. **Batch Search**: Process multiple queries in parallel

### Cross-Encoder Reranking
When enabled, the system uses a two-stage retrieval process:
1. **First Stage**: Retrieve more candidates (5x requested) using vector search
2. **Second Stage**: Re-score candidates with cross-encoder model

**Automatic Model Selection**:
- BAAI embeddings → BAAI reranker
- Sentence transformers → MS MARCO cross-encoder
- Qwen3 embeddings → Matching Qwen3 reranker size

**Benefits**:
- Significantly improved relevance for complex queries
- Better handling of nuanced questions
- Trade-off: Slightly increased latency for better accuracy

### Supported Models
- BAAI/bge series (base, large)
- Qwen3 series (0.6B, 4B, 8B)
- Sentence Transformers models
- Custom HuggingFace models

For detailed documentation, see [SEARCH_SYSTEM.md](./SEARCH_SYSTEM.md)

## Infrastructure & DevOps

### Development Environment
- Python 3.12+ with Poetry
- Make commands for common tasks
- Comprehensive test suite with pytest
- Mock mode for GPU-less testing

### Service Management
- Start/stop scripts for orchestration
- Status monitoring
- Prometheus metrics integration
- Structured logging

### CI/CD Pipeline
- GitHub Actions workflow
- Automated linting and testing
- Build verification

For detailed documentation, see [INFRASTRUCTURE.md](./INFRASTRUCTURE.md)

## Security Architecture

### Authentication
- JWT tokens (access: 30min, refresh: 30 days)
- Bcrypt password hashing
- Secure token storage and rotation

### Input Validation
- Path traversal prevention
- Request size limits
- Rate limiting on all endpoints

### Data Security
- No secrets in code or logs
- Environment-based configuration
- Secure file access validation

## Performance Considerations

### Optimization Strategies
1. **Model Management**
   - Lazy loading with configurable timeout (default: 5 minutes)
   - Automatic GPU memory cleanup
   - Quantization support (int8, float16)
   - Intelligent model selection based on available GPU memory

2. **Batch Processing**
   - Adaptive batch sizing
   - Parallel file processing
   - Efficient vector ingestion (4000/batch)

3. **Search Performance**
   - HNSW indexing in Qdrant
   - Result caching
   - Batch query support

### Monitoring
- Prometheus metrics for all operations
- Request duration tracking
- GPU memory monitoring
- Error rate tracking

## Deployment Guide

### System Requirements
- Python 3.12+
- CUDA-capable GPU (optional)
- 8GB+ RAM recommended
- Qdrant instance

### Quick Start
```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env

# Start all services
./start_all_services.sh
```

### Production Deployment
- Use systemd service files
- Enable Prometheus monitoring
- Configure reverse proxy (nginx)
- Set up SSL/TLS termination

## Future Enhancements

### Planned Features
1. **Horizontal Scaling**
   - Distributed job processing
   - Multi-instance WebUI
   - Qdrant cluster support

2. **Enhanced Search**
   - Cross-lingual search
   - Faceted search
   - Query expansion

3. **Advanced Features**
   - Incremental updates
   - Real-time indexing
   - Custom model fine-tuning

### Architecture Evolution
- Microservices decomposition
- Event-driven architecture
- Cloud-native deployment

## Related Documentation

For deep dives into specific components:

- **Core Engine**: [SEMANTIK_CORE.md](./SEMANTIK_CORE.md)
- **Backend**: [WEBUI_BACKEND.md](./WEBUI_BACKEND.md)
- **Frontend**: [FRONTEND_ARCH.md](./FRONTEND_ARCH.md)
- **Database**: [DATABASE_ARCH.md](./DATABASE_ARCH.md)
- **APIs**: [API_ARCHITECTURE.md](./API_ARCHITECTURE.md)
- **Search**: [SEARCH_SYSTEM.md](./SEARCH_SYSTEM.md)
- **Infrastructure**: [INFRASTRUCTURE.md](./INFRASTRUCTURE.md)

## Conclusion

Semantik demonstrates a well-architected system with clear separation of concerns, robust error handling, and excellent performance characteristics. The modular design enables both standalone usage of the search engine and full-featured operation through the web interface, making it suitable for a wide range of deployment scenarios from development laptops to production servers.