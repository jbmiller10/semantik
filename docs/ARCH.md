# Semantik Architecture

Self-hosted semantic search with three packages: vecpipe (search engine), webui (control plane), and shared (common utilities). Collection-centric design for multi-model support.

## Key Features
- Vector search powered by Qdrant
- Collection-based document organization with multi-model support
- Support for multiple embedding models with quantization (float32, float16, int8)
- Hybrid search combining vector similarity and keyword matching
- Full-featured web interface with real-time progress tracking
- Comprehensive REST API v2 for programmatic access
- JWT-based authentication system with refresh tokens
- Prometheus metrics for monitoring
- Async operations with WebSocket progress updates
- Mock mode for testing without GPU resources

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
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │Auth Service │  │Collection Mgmt  │  │Search Proxy │  │WebSockets│ │
│  │   (JWT)     │  │& Operations     │  │   (v2 API)  │  │(Progress)│ │
│  └─────────────┘  └─────────────────┘  └─────────────┘  └──────────┘ │
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
┌────────┴─────────┐                   │            ┌─────────────────┐
│  PostgreSQL DB   │                   │            │   Celery Worker │
│  (WebUI-owned)   │                   │            │  (Background    │
│  Collections,    │                   │            │   Operations)   │
│  Operations,     │←──────────────────┼────────────┤                 │
│  Documents,      │                   │            └─────────────────┘
│  Users, Tokens   │                   │
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
│                 (Collection naming: {uuid}_{model}_{quant})             │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. VecPipe Package (`packages/vecpipe/`)

Headless data processing and search API. Runs on port 8000, completely independent of webui.

**Key Services:**
- `search_api.py` - FastAPI endpoints for search, embed, and upsert
- `embed_chunks_unified.py` + `ingest_qdrant.py` - Embedding pipeline
- `maintenance.py` - Collection cleanup
- `model_manager.py` - GPU memory management with lazy loading
- `hybrid_search.py` - Vector + keyword search
- `reranker.py` - Cross-encoder reranking

**Design:**
- Standalone, no user auth; internal API key for service-to-service endpoints
- Talks directly to Qdrant
- Efficient GPU memory management
- Multi-model support

### 2. WebUI Package (`packages/webui/`)

User-facing application providing authentication, collection management, and search interface. Owns and manages the PostgreSQL database containing user data, collections, operations, and documents.

**Backend Components:**
- **main.py**: FastAPI application with modular router architecture.
- **api/v2/**: REST routers for collections, operations, documents, search, chunking, embedding.
- **api/auth.py**: JWT‑based authentication with refresh tokens.
- **services/**: Service layer with business logic (collection_service, operation_service, search_service, etc.).
- **celery_app.py**, **tasks/**, **chunking_tasks.py**: Celery worker entrypoint and task implementations.

**Frontend (React):**
- Modern React 19 with TypeScript
- Zustand for state management
- Real-time WebSocket updates for operation progress
- Tailwind CSS for styling
- React Query for data fetching

**Key Responsibilities:**
- User authentication and authorization
- Collection lifecycle management
- Operation orchestration and tracking
- Document management and deduplication
- Search request proxying with multi-collection support
- Real-time progress updates via WebSocket

### 3. Shared Package (`packages/shared/`)

Common components and utilities used by both webui and vecpipe packages. This package ensures consistency and avoids code duplication.

**Core Modules:**

**Database Module (`shared/database/`):**
- **Repository Pattern**: Clean data access layer with type-safe interfaces
- **PostgreSQL Implementation**: Concrete implementations of repository interfaces
- **Models**: SQLAlchemy models for collections, operations, documents, users
- **Schema Management**: Centralized database schema definitions
- **Exceptions**: Custom database exceptions for error handling

**Contracts Module (`shared/contracts/`):**
- **API Models**: Pydantic models for request/response validation
- **Search Models**: Unified search request and response structures
- **Collection Models**: Shared collection and operation status definitions
- **Document Models**: Document metadata and status tracking

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
- **PostgreSQL**: Relational data (collections, operations, documents, users, auth tokens)
- **Qdrant**: Vector storage and similarity search

**Collection-Centric Schema:**
- **Collections**: Primary organizational unit with UUID identifiers
- **Operations**: Async tasks (index, reindex, append, remove_source)
- **Documents**: Individual files within collections
- **Sources**: Data sources that provide documents to collections

**Key Design Decisions:**
- UUID-based identifiers for collections and documents
- Clear separation between metadata and vectors
- Operation tracking for async task management
- Content hash-based duplicate detection
- Comprehensive indexing for performance

For detailed documentation, see [DATABASE_ARCH.md](./DATABASE_ARCH.md)

## Collection-Centric Architecture

Collections replaced the old job-based system. Each collection is a logical grouping of documents with its own embedding model and configuration.

**Benefits:**
- Different models per collection type
- Incremental updates (no full reprocessing)
- Clear ownership and access control
- Multiple sources per collection

### Operation Types

- **INDEX** - Initial population of a new collection
- **APPEND** - Add new documents, with deduplication
- **REINDEX** - Regenerate all embeddings (for model upgrades)
- **REMOVE_SOURCE** - Delete all documents from a specific source path

### Sources

Collections can have multiple sources (directories, files, glob patterns). Add/remove incrementally with content-hash deduplication.

### Multi-Model Support

Each collection uses its own embedding model and quantization. Search across multiple collections with different models simultaneously.

## Data Flow

### Collection Creation and Document Processing

```
1. User creates collection with embedding configuration
   ↓
2. WebUI creates collection record in PostgreSQL
   ↓
3. Qdrant collection created with deterministic naming
   ↓
4. User adds source (directory/file) to collection
   ↓
5. Operation created and queued to Celery worker
   ↓
6. Worker scans source and creates document records
   ↓
7. Extract service chunks documents by tokens
   ↓
8. Embed service generates vectors (with batching)
   ↓
9. Ingest service stores vectors in Qdrant
   ↓
10. WebSocket updates UI with real-time progress
```

### Connector-Based Ingestion

The indexing pipeline now uses a connector + registry abstraction:

- **IngestedDocument DTO** (`shared.dtos.ingestion.IngestedDocument`) is the unified contract that all connectors emit.
- **BaseConnector / LocalFileConnector** (`shared.connectors.base` / `shared.connectors.local`) encapsulate source-specific loading
  (e.g., walking directories for `"directory"` sources) while returning in-memory content.
- **ConnectorFactory** (`webui.services.connector_factory.ConnectorFactory`) resolves a connector implementation from `source_type`.
- **DocumentRegistryService** (`webui.services.document_registry_service.DocumentRegistryService`) owns registration and
  deduplication, using `content_hash` and the new `documents.uri` / `documents.source_metadata` fields to keep ingestion
  consistent across all source types.

Celery APPEND tasks call `ConnectorFactory.get_connector(...)` and then `DocumentRegistryService.register(...)` for each
`IngestedDocument`, so adding a new source (for example, `"web"` or `"slack"`) only requires a new connector; the core
pipeline does not change.

### Multi-Collection Search Flow

```
1. User selects collections and enters search query
   ↓
2. WebUI validates access permissions
   ↓
3. Request proxied to Search API with collection UUIDs
   ↓
4. Search API maps UUIDs to Qdrant collection names
   ↓
5. Query embedding generated for each model type
   ↓
6. Parallel search across multiple Qdrant collections
   ↓
7. Results normalized and optionally reranked
   ↓
8. Response enriched with collection metadata
   ↓
9. Results returned through proxy with scores
```

## API Architecture

### Search API (Port 8000)
- Pure REST API for vector/hybrid search
- No authentication required
- Supports batch search and per-request collection selection
- Model lazy-loading for efficiency
- Collection-aware with Qdrant naming convention

### WebUI API v2 (Port 8080)
- JWT-authenticated endpoints with refresh tokens
- Collection lifecycle management
- Operation tracking and monitoring
- Document management with deduplication
- Search proxy with multi-collection support
- WebSocket for real-time operation progress

**Key v2 API Endpoints:**
- `/api/v2/collections` - Collection CRUD operations
- `/api/v2/operations` - Operation tracking and management
- `/api/v2/search` - Multi-collection semantic search
- `/api/v2/documents` - Document access and metadata

### Collection-Centric API Examples

The v2 API design reflects the collection-centric architecture:

```python
# Create a new collection with specific embedding configuration
POST /api/v2/collections
{
    "name": "Technical Documentation",
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "embedding_size": 1024,
    "quantization": "float16",
    "chunk_size": 512,
    "chunk_overlap": 50
}

# Add a source to the collection (triggers APPEND operation)
POST /api/v2/collections/{collection_id}/sources
{
    "source_type": "directory",
    "source_config": {
        "path": "/data/docs/api-reference/"
    }
}

# Search across multiple collections
POST /api/v2/search
{
    "query": "How to authenticate API requests?",
    "collection_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "limit": 10,
    "search_type": "hybrid",
    "rerank": true
}

# Monitor operation progress
GET /api/v2/operations/{operation_id}
WebSocket: /ws/operations/{operation_id}/progress

# Reindex a collection with updated configuration
POST /api/v2/collections/{collection_id}/reindex
```

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
- Python 3.11+ with Poetry
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
- Python 3.11+
- CUDA-capable GPU (optional)
- 8GB+ RAM recommended
- Qdrant instance

### Quick Start
```bash
# Install dependencies
uv sync

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

## Future Ideas

- Collection templates and presets
- Hierarchical collections
- Collection-aware search routing
- Event-driven operation updates
- Better multi-region support

## Related Docs

- [DATABASE_ARCH.md](./DATABASE_ARCH.md) - Database schema
- [API_ARCHITECTURE.md](./API_ARCHITECTURE.md) - API design
- [SEARCH_SYSTEM.md](./SEARCH_SYSTEM.md) - Search implementation
- [INFRASTRUCTURE.md](./INFRASTRUCTURE.md) - Deployment
