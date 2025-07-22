# Semantik Document Embedding System - Comprehensive Architecture Documentation

## Executive Summary

Semantik is a production-ready, high-performance document embedding and vector search system designed for technical users who prioritize performance and control. The system features a clean three-package architecture with clear separation between its core search engine (vecpipe), control plane (webui), and shared components. This modular design enables both standalone usage and user-friendly management through a modern React interface.

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
│                        WebUI Package (Port 8080)                        │
│                         FastAPI Control Plane                           │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │Auth Service │  │Job Management│  │Search Proxy │  │ WebSockets  │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └─────────────┘ │
└────────┬──────────────────────────────┬────────────────────────────────┘
         │                              │
         │                              │ HTTP Proxy
         │ ┌────────────────────────────┴──────────────────────────────┐
         │ │                    Shared Package                          │
         │ │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
         │ │  │   Database    │  │   Contracts  │  │    Config     │  │
         │ │  │ Repositories  │  │   & Models   │  │  Management   │  │
         │ │  └──────────────┘  └──────────────┘  └───────────────┘  │
         │ │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
         │ │  │  Embedding    │  │    Metrics   │  │     Text      │  │
         │ │  │   Service     │  │   Tracking   │  │  Processing   │  │
         │ │  └──────────────┘  └──────────────┘  └───────────────┘  │
         │ └────────────────────────────────────────────────────────────┘
         │                              │
┌────────┴─────────┐                   │
│  PostgreSQL DB   │                   │
│  (WebUI-owned)   │                   │
│  Jobs, Files,    │                   │
│  Users, Tokens,  │                   │
│  Collections     │                   │
└──────────────────┘                   │
                                      │
┌──────────────────────────────────────┴─────────────────────────────────┐
│                         VecPipe Package (Port 8000)                    │
│                          Search & Processing API                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Extract    │  │  Maintenance │  │   Ingest    │  │   Search   │  │
│  │  Service    │  │   Service    │  │   Service   │  │    API     │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └────────────┘  │
└────────────────────────────┬───────────────────────────────────────────┘
                            │
                            │ gRPC/HTTP
┌───────────────────────────┴────────────────────────────────────────────┐
│                         Qdrant Vector Database                          │
│                     Vector Storage & Similarity Search                  │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. VecPipe Package (`packages/vecpipe/`)

The headless data processing and search API that forms the heart of the system. This package is completely independent and has no dependencies on the webui package.

**Key Services:**
- **Extract Service** (`extract_service.py`): Document parsing and intelligent chunking
- **Maintenance Service** (`maintenance_service.py`): Vector database operations and health monitoring
- **Ingest Service** (`ingest_service.py`): Efficient vector database population
- **Search API** (`search_api.py`): FastAPI service exposing search functionality

**Core Components:**
- **model_manager.py**: GPU memory management with lazy loading and automatic unloading
- **hybrid_search.py**: Combined vector and keyword search implementation
- **reranker.py**: Cross-encoder reranking for improved search accuracy

**Design Principles:**
- Completely standalone operation
- No awareness of users, jobs, or authentication
- Direct interaction with Qdrant vector database
- Resource-efficient GPU memory management

### 2. WebUI Package (`packages/webui/`)

User-facing application providing authentication, job management, and search interface. Owns and manages the PostgreSQL database containing user data, jobs, files, and collections.

**Backend Components:**
- **main.py**: FastAPI application with modular router architecture
- **api/**: RESTful API routers for auth, jobs, search, files, metrics
- **auth.py**: JWT-based authentication system
- **job_processing.py**: Orchestrates document processing pipeline

**Frontend (React):**
- Modern React 19 with TypeScript
- Zustand for state management
- Real-time WebSocket updates
- Tailwind CSS for styling

**Key Responsibilities:**
- User authentication and authorization
- Job creation and management
- File tracking and status updates
- Search request proxying with authentication
- Real-time progress updates via WebSocket

### 3. Shared Package (`packages/shared/`)

Common components and utilities used by both webui and vecpipe packages. This package ensures consistency and avoids code duplication.

**Core Modules:**

**Database Module (`shared/database/`):**
- **Repository Pattern**: Clean data access layer with type-safe interfaces
- **PostgreSQL Implementation**: Concrete implementations of repository interfaces
- **Legacy Wrappers**: Deprecated direct database functions for backward compatibility
- **Schema Management**: Centralized database schema definitions

**Contracts Module (`shared/contracts/`):**
- **API Models**: Pydantic models for request/response validation
- **Search Models**: Unified search request and response structures
- **Job Models**: Shared job and file status definitions

**Config Module (`shared/config/`):**
- **Settings Management**: Environment-based configuration
- **Model Configuration**: Embedding and reranker model settings
- **Service URLs**: Centralized service endpoint configuration

**Embedding Module (`shared/embedding/`):**
- **Embedding Service**: Core embedding generation logic
- **Model Management**: Shared model loading and caching
- **Batch Processing**: Efficient batch embedding generation

**Other Utilities:**
- **Metrics** (`shared/metrics/`): Prometheus metrics collection
- **Text Processing** (`shared/text_processing/`): Document parsing and chunking
- **Logging** (`shared/logging_config.py`): Structured logging configuration

### 4. Database Architecture

**Hybrid Database Design:**
- **PostgreSQL**: Relational data (jobs, files, users, auth tokens, collections)
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
2. WebUI creates job record in PostgreSQL
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