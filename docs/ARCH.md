# Semantik Document Embedding System - Comprehensive Architecture Documentation

## Executive Summary

Semantik is a production-ready, high-performance document embedding and vector search system designed for technical users who prioritize performance and control. The system features a clean three-package architecture with clear separation between its core search engine (vecpipe), control plane (webui), and shared components. This modular design enables both standalone usage and user-friendly management through a modern React interface.

The system has undergone a major architectural refactoring, transitioning from a job-centric model to a collection-centric architecture. This new design provides better organization, scalability, and multi-model support through collections that group related documents with shared embedding configurations.

### Key Features
- High-performance vector search powered by Qdrant
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
│                         VecPipe Package (Port 8001)                    │
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

The headless data processing and search API that forms the heart of the system. This package is completely independent and has no dependencies on the webui package. Operates on port 8001 in the refactored architecture.

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
- No awareness of users, collections, or authentication
- Direct interaction with Qdrant vector database using collection naming convention
- Resource-efficient GPU memory management
- Supports multiple concurrent collections with different embedding models

### 2. WebUI Package (`packages/webui/`)

User-facing application providing authentication, collection management, and search interface. Owns and manages the PostgreSQL database containing user data, collections, operations, and documents.

**Backend Components:**
- **main.py**: FastAPI application with modular router architecture
- **api/v2/**: Modern RESTful API routers for collections, operations, documents, search
- **api/auth/**: JWT-based authentication with refresh tokens
- **services/**: Service layer with business logic (collection_service, operation_service)
- **worker/**: Celery worker for asynchronous operation processing

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

## Collection-Centric Architecture Benefits

The collection-centric architecture represents a fundamental improvement over the previous job-centric design:

### Key Advantages

1. **Better Organization**: Collections provide a logical grouping of related documents with shared configuration
2. **Multi-Model Support**: Each collection can use different embedding models and configurations
3. **Incremental Updates**: Add or remove documents without re-processing the entire dataset
4. **Resource Efficiency**: Operations are scoped to collections, reducing unnecessary reprocessing
5. **Clear Ownership**: Collections have explicit ownership and access control
6. **Flexible Source Management**: Collections can aggregate documents from multiple sources

### Collection Lifecycle Management

Collections progress through several states during their lifecycle:

1. **Creation**: Collection initialized with embedding configuration
2. **Initial Indexing**: First operation populates the collection with documents
3. **Active Management**: Ongoing operations to append, remove, or reindex documents
4. **Maintenance**: Periodic reindexing to incorporate model improvements
5. **Archival/Deletion**: Clean removal of collection and associated resources

### Operation Types and Purposes

The system supports four primary operation types, each serving a specific purpose:

#### INDEX Operation
- **Purpose**: Initial population of a new collection
- **Use Case**: First-time document processing after collection creation
- **Process**: Scans sources, extracts text, generates embeddings, stores in Qdrant
- **Result**: Collection ready for search with initial document set

#### APPEND Operation
- **Purpose**: Add new documents to an existing collection
- **Use Case**: Incremental updates as new documents become available
- **Process**: Scans only new sources, deduplicates against existing documents
- **Result**: Collection expanded with new documents without disrupting existing ones

#### REINDEX Operation
- **Purpose**: Regenerate all embeddings for a collection
- **Use Case**: Model upgrades, configuration changes, or quality improvements
- **Process**: Re-processes all documents with current model configuration
- **Result**: Updated embeddings reflecting latest model capabilities

#### REMOVE_SOURCE Operation
- **Purpose**: Remove all documents from a specific source path
- **Use Case**: Clean up outdated documents or remove specific directories
- **Process**: Identifies and removes documents matching the source path
- **Result**: Collection updated with documents from specified source removed

### Collection Source Management

Collections support flexible source management to aggregate documents from multiple locations:

```
Collection
    ├── Source 1: /data/technical-docs/
    │   ├── manual.pdf
    │   └── guide.md
    ├── Source 2: /data/api-docs/
    │   └── openapi.yaml
    └── Source 3: /shared/knowledge-base/
        ├── faq.md
        └── troubleshooting.pdf
```

**Source Features**:
- Add sources incrementally without reprocessing existing documents
- Remove specific sources while preserving others
- Track source metadata for document provenance
- Support for directories, individual files, and glob patterns
- Automatic deduplication based on content hash

### Multi-Model Support Architecture

The architecture enables sophisticated multi-model deployments:

```
User Search Query
        ↓
┌─────────────────────────────────────┐
│   Collection 1: Technical Docs      │
│   Model: BAAI/bge-large-en-v1.5     │
│   Quantization: float16             │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Collection 2: Code Documentation  │
│   Model: Qwen/Qwen3-0.6B           │
│   Quantization: int8                │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│   Collection 3: Research Papers     │
│   Model: sentence-transformers/...  │
│   Quantization: float32             │
└─────────────────────────────────────┘
        ↓
    Unified Search Results
```

**Multi-Model Benefits**:
- Optimal model selection per document type
- Resource optimization through quantization choices
- Parallel search across heterogeneous collections
- Model-specific reranking for best relevance

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

### Search API (Port 8001)
- Pure REST API for vector/hybrid search
- No authentication required
- Supports batch operations and multi-collection search
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
    "source_path": "/data/docs/api-reference/"
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

### Collection-Centric Evolution

The collection-centric architecture provides a foundation for advanced features:

#### Advanced Collection Management
1. **Collection Templates**
   - Pre-configured templates for common use cases (legal docs, code, research papers)
   - Shareable configuration profiles across organizations
   - Automatic model recommendations based on content type

2. **Collection Relationships**
   - Hierarchical collections with inheritance
   - Cross-collection linking and references
   - Collection groups for unified search contexts

3. **Smart Operations**
   - Intelligent operation scheduling based on resource availability
   - Batch operation optimization across multiple collections
   - Predictive reindexing based on model updates
   - Operation templates for complex workflows

#### Enhanced Search Capabilities
1. **Collection-Aware Search**
   - Dynamic collection selection based on query intent
   - Collection-specific ranking algorithms
   - Inter-collection result fusion strategies
   - Collection authority scoring

2. **Advanced Features**
   - Real-time collection updates with streaming ingestion
   - Collection-specific knowledge graphs
   - Multi-stage retrieval pipelines per collection
   - Collection-based query routing

#### Enterprise Features
1. **Collection Governance**
   - Collection lifecycle policies
   - Automated collection archival and retention
   - Collection compliance tracking
   - Usage analytics per collection

2. **Scalability**
   - Collection sharding for massive datasets
   - Federated search across distributed collections
   - Collection-level resource quotas
   - Dynamic collection migration

### Architecture Evolution

The collection-centric design enables future architectural improvements:

1. **Collection Microservices**
   - Dedicated services per high-volume collection
   - Collection-specific optimization strategies
   - Independent scaling per collection type

2. **Event-Driven Collections**
   - Collection state change events
   - Operation completion notifications
   - Real-time collection synchronization
   - Event sourcing for collection history

3. **Cloud-Native Collections**
   - Kubernetes operators for collection management
   - Collection-as-a-Service (CaaS) abstractions
   - Serverless operation processing
   - Multi-region collection replication

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

Semantik's collection-centric architecture represents a significant advancement in semantic search system design. By organizing documents into collections with dedicated configurations, the system provides:

- **Flexibility**: Each collection can be optimized for its specific content type and use case
- **Scalability**: Operations are scoped to collections, enabling efficient resource utilization
- **Maintainability**: Clear separation between collections simplifies management and troubleshooting
- **Extensibility**: The architecture naturally supports advanced features like multi-model search and incremental updates

The transition from a job-centric to collection-centric design has transformed Semantik from a simple document processing pipeline into a sophisticated knowledge management platform. The modular architecture, combined with the collection-based organization, enables deployment scenarios ranging from personal knowledge bases on development laptops to enterprise-scale semantic search infrastructure.

This architecture positions Semantik as a production-ready solution for organizations that value data privacy, require fine-grained control over their search infrastructure, and need the flexibility to optimize for diverse document types and use cases.