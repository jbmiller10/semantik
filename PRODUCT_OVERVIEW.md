# Semantik - Product Overview
## Self-Hosted Semantic Search Engine

**Version:** Pre-release (Active Development)
**Last Updated:** 2025-10-20

---

## What Is Semantik?

Semantik is a **self-hosted semantic search engine** that transforms your private file servers into AI-powered knowledge bases **without data ever leaving your hardware**. Unlike cloud-based solutions (OpenAI, Anthropic, etc.), Semantik runs entirely on your own infrastructure, giving you complete control over your data.

Think of it as "Google for your private documents" - but instead of keyword matching, it understands the *meaning* of your content using AI embeddings.

---

## Core Value Proposition

**Privacy-First Semantic Search**
- All processing happens on your hardware (local or self-hosted)
- No data sent to external APIs
- GPU-accelerated embedding generation using open-source models
- Full ownership and control of your knowledge base

**For Whom:**
- Privacy-conscious organizations (legal, healthcare, finance)
- Researchers with proprietary data
- Companies with sensitive internal documentation
- Self-hosters and privacy advocates
- Anyone who wants semantic search without cloud dependencies

---

## How It Works

### 1. **Collection Creation**
Users create "collections" - logical groups of documents to be searched together.

**Configuration Options:**
- **Embedding Model**: Choose from multiple models (e.g., BAAI/bge-small-en-v1.5, sentence-transformers)
- **Quantization**: Select precision (float32, float16, int8) to balance quality vs memory
- **Chunking Strategy**: Choose how documents are split:
  - `CHARACTER` - Simple character-based splitting
  - `RECURSIVE` - Intelligent hierarchical splitting
  - `MARKDOWN` - Markdown-aware preservation
  - `SEMANTIC` - Meaning-based boundaries
  - `HIERARCHICAL` - Document structure-aware
  - `HYBRID` - Combined approach
- **Chunk Size & Overlap**: Configure how text is segmented

### 2. **Document Ingestion**
Point Semantik at a directory on your filesystem.

**What Happens:**
1. **Scan**: Discovers all supported files (PDF, TXT, DOCX, MD, etc.)
2. **Parse**: Extracts text content from documents
3. **Chunk**: Splits documents using selected chunking strategy
4. **Embed**: Generates vector embeddings using chosen model (runs on GPU if available)
5. **Index**: Stores vectors in Qdrant vector database
6. **Track**: Real-time progress via WebSocket updates

**Background Processing:**
- All heavy lifting happens asynchronously via Celery workers
- You can continue using the app while indexing happens
- Progress tracked in operations dashboard

### 3. **Semantic Search**
Search across collections using natural language.

**Search Types:**
- **Semantic Search**: Find content by meaning (not exact keywords)
- **Hybrid Search**: Combine semantic similarity with keyword matching
- **Question Answering Mode**: Optimized for Q&A queries
- **Code Search**: Specialized for source code

**Advanced Features:**
- **Multi-Collection Search**: Search across up to 10 collections simultaneously
- **Result Reranking**: Optional cross-encoder reranking for better precision
- **Score Thresholding**: Filter results by relevance score
- **Adjustable Result Count**: Control number of results (k=1-100)

### 4. **Collection Management**
Full lifecycle management of your knowledge bases.

**Operations:**
- **Append**: Add new documents to existing collection
- **Reindex**: Update with new embedding model or chunking strategy (zero-downtime blue-green reindexing)
- **Remove Source**: Remove specific directories from collection
- **Delete**: Remove collection and all associated data

**Real-time Monitoring:**
- Live operation progress via WebSockets
- Operation history tracking
- Detailed error messages and recovery

---

## Architecture

### Microservices Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                      │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   WebUI      │   VecPipe    │   Worker     │ Infrastructure │
│  (Port 8080) │  (Port 8000) │   (Celery)   │                │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ • React UI   │ • Embeddings │ • Indexing   │ • PostgreSQL   │
│ • REST API   │ • Search     │ • Background │ • Redis        │
│ • WebSockets │ • Parsing    │   Tasks      │ • Qdrant       │
│ • Auth (JWT) │ • Reranking  │ • Cleanup    │                │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

### Tech Stack

**Backend:**
- **Language**: Python 3.11+
- **Web Framework**: FastAPI (REST + WebSocket)
- **Database**: PostgreSQL 15 with table partitioning (100 partitions on chunks table)
- **Vector Store**: Qdrant (dedicated vector database)
- **Cache/Queue**: Redis
- **Task Queue**: Celery
- **ORM**: SQLAlchemy (async)
- **Migrations**: Alembic

**Frontend:**
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand + React Query
- **Styling**: TailwindCSS
- **HTTP Client**: Axios with interceptors

**ML/AI:**
- **Embeddings**: Sentence Transformers / BAAI BGE models
- **Reranking**: Qwen3-Reranker cross-encoder models
- **Quantization**: Support for float32, float16, int8
- **Device**: CUDA GPU (if available) or CPU fallback

**DevOps:**
- **Containerization**: Docker + Docker Compose
- **Reverse Proxy**: Nginx (implicit)
- **Monitoring**: Prometheus metrics export
- **Health Checks**: Kubernetes-style probes (liveness, readiness, startup)

---

## Current Features (What Actually Works)

### ✅ User Authentication
- JWT-based authentication (24h access tokens, 30d refresh tokens)
- User registration and login
- Session management
- Automatic token refresh

### ✅ Collection Management
- Create collections with configurable embedding models
- List/view/update/delete collections
- Collection status tracking (pending, ready, processing, error, degraded)
- Document count and vector count statistics
- Owner-based access control

### ✅ Document Ingestion
- Directory scanning with file type detection
- Supported formats: PDF, TXT, DOCX, MD, HTML, and more
- Async background processing with Celery
- Real-time progress updates via WebSocket
- Duplicate detection
- Error handling and retry logic

### ✅ Chunking System
- 6 different chunking strategies (character, recursive, markdown, semantic, hierarchical, hybrid)
- Configurable chunk size and overlap
- Chunking preview (see chunks before committing)
- Strategy comparison (compare multiple strategies side-by-side)
- Custom presets (save and reuse chunking configurations)

### ✅ Semantic Search
- Natural language search across collections
- Multiple search modes (semantic, hybrid, question, code)
- Multi-collection search (up to 10 at once)
- Optional reranking with cross-encoder models
- Score thresholding
- Partial failure handling (some collections can fail without breaking search)
- GPU memory error detection and handling

### ✅ Operations Tracking
- Real-time operation progress (WebSocket-based)
- Operation types: INDEX, APPEND, REINDEX, REMOVE_SOURCE, DELETE
- Operation history per collection
- Cancel pending/processing operations
- Detailed error messages
- Operation filtering by status/type

### ✅ Reindexing (Blue-Green)
- Zero-downtime reindexing
- Change embedding model without service interruption
- Change chunking strategy
- Validation phase (compares old vs new)
- Automatic atomic swap
- Cleanup of old collections

### ✅ System Features
- Database statistics (collection count, file count, size)
- Database reset (admin)
- System status checks (GPU availability, reranking support)
- Health check endpoints (liveness, readiness, startup)
- Prometheus metrics export

### ✅ Real-Time Features
- WebSocket connections for operation progress
- Directory scan progress updates
- Redis Pub/Sub for horizontal scaling
- Per-user connection limits (max 10 per user)
- Automatic reconnection with backoff

### ✅ Developer Features
- Comprehensive REST API (documented with OpenAPI/Swagger)
- Interactive API docs at `/docs`
- Development mode (hot reload for webui)
- Docker-based development environment
- Extensive test suite (unit, integration, e2e, security tests)

---

## User Journey Example

### Scenario: Research Team Wants to Search Academic Papers

1. **Setup** (One-time)
   - Run `make wizard` to set up configuration
   - Run `make docker-up` to start services
   - Access UI at `http://localhost:8080`

2. **Create Account**
   - Register with username/email/password
   - Login and receive JWT token

3. **Create Collection**
   - Click "New Collection"
   - Name: "Research Papers 2024"
   - Choose model: `BAAI/bge-small-en-v1.5` (good balance of speed/quality)
   - Quantization: `int8` (save memory)
   - Chunking: `SEMANTIC` (preserve meaning boundaries)
   - Chunk size: 512 tokens

4. **Add Documents**
   - Select directory: `/data/papers/2024/`
   - System scans and finds 150 PDF files
   - Real-time progress: "Processing document 47 of 150..."
   - Indexing completes in ~10 minutes (GPU-accelerated)

5. **Search**
   - Query: "What are the latest approaches to few-shot learning?"
   - Select collections: "Research Papers 2024"
   - Search type: Semantic
   - Enable reranking for better precision
   - Get 10 most relevant results with scores
   - Click result to view document content

6. **Refine Collection**
   - Add new papers: Click "Add Data Source" → select new directory
   - System appends without full reindex
   - Or: Click "Reindex" to try different chunking strategy
   - Blue-green reindex ensures no downtime

---

## What Makes Semantik Different?

### vs. Cloud Solutions (OpenAI, Anthropic, Pinecone)
- ✅ **Privacy**: No data leaves your infrastructure
- ✅ **Cost**: No per-query pricing or API fees
- ✅ **Control**: Full control over models and configuration
- ❌ **Convenience**: Requires self-hosting and GPU hardware

### vs. ElasticSearch / OpenSearch
- ✅ **Semantic Understanding**: Finds content by meaning, not keywords
- ✅ **Modern Stack**: Built for embeddings from ground up
- ✅ **Simpler Setup**: Docker Compose vs complex ES cluster
- ❌ **Maturity**: Pre-release vs battle-tested ES

### vs. txtai, LlamaIndex, LangChain
- ✅ **Full Application**: Complete UI + API, not just a library
- ✅ **Multi-User**: Authentication, collections, permissions
- ✅ **Production-Ready Architecture**: Microservices, monitoring, health checks
- ❌ **Flexibility**: Less flexible than programming libraries

---

## Technical Highlights

### Performance
- **Embedding Generation**: 100+ texts/second on GPU
- **Search Latency**: <500ms p95 for single queries
- **Reranking**: <200ms for 100 candidates
- **Batch Processing**: Parallel embedding generation

### Scalability
- **Partitioned Database**: 100 LIST partitions on chunks table for query optimization
- **Horizontal Scaling**: Redis Pub/Sub enables multi-instance deployments
- **Connection Limits**: 10 per user, 10,000 total WebSocket connections
- **Efficient Queries**: Partition pruning ensures queries only scan relevant data

### Reliability
- **Health Checks**: Liveness, readiness, startup probes
- **Retry Logic**: Automatic retries with exponential backoff
- **Graceful Degradation**: Partial search failures don't break entire query
- **Background Tasks**: All heavy processing async via Celery
- **Transaction Safety**: Database commits before Celery dispatch (prevents race conditions)

### Security
- **Authentication**: JWT with refresh tokens
- **Authorization**: Owner-based collection access
- **Input Validation**: Pydantic models for all API inputs
- **Path Traversal Prevention**: All file paths validated
- **SQL Injection Prevention**: Parameterized queries via SQLAlchemy
- **Rate Limiting**: Per-endpoint rate limits with circuit breakers
- **Security Tests**: Dedicated test suite for vulnerabilities

### Observability
- **Prometheus Metrics**: Request latency, counts, errors, resource usage
- **Structured Logging**: JSON logs with correlation IDs
- **Operation Tracking**: Full audit trail of all operations
- **Resource Monitoring**: CPU, RAM, GPU tracking via psutil/gputil

---

## Current Limitations / Known Issues

### What's Not Implemented Yet
- ❌ Model selection UI (backend exists, no frontend)
- ❌ Chunking quality metrics (backend exists, no frontend)
- ❌ Performance metrics dashboard (metrics exported, no UI)
- ❌ Public/private collections (field exists, no UI toggle)
- ❌ Advanced search filters (metadata filtering backend ready, no UI)
- ❌ Batch search (backend complete, no UI)
- ❌ Keyword-only search (backend complete, not exposed in UI)

### Technical Debt
- Migration from "job" to "operation" terminology (ongoing refactor)
- Some legacy code in chunking module (being replaced with new system)
- Limited frontend test coverage (backend well-tested)

### Resource Requirements
- **Minimum**: 4GB RAM, 20GB disk (CPU-only mode)
- **Recommended**: 8GB+ RAM, NVIDIA GPU with 4GB+ VRAM, 50GB+ disk
- **Heavy Usage**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM, 100GB+ disk

---

## Development Status

**Current Phase:** Pre-release / Active Development

**Active Branch:** `fix/fix-chunking-bug` (main development)

**Recent Work:**
- Chunking system refactor (domain-driven design)
- Security hardening
- Test coverage improvements
- Bug fixes in partition queries

**Near-Term Roadmap:**
- Complete chunking quality metrics frontend
- Model management UI
- Performance dashboard
- Public collection sharing
- Documentation improvements

---

## Use Cases

### 1. **Internal Knowledge Base**
Company wikis, documentation, policies, training materials

### 2. **Research Literature Review**
Academic papers, research notes, citations - search by concept instead of keywords

### 3. **Legal Document Search**
Contracts, case law, regulations - privacy-sensitive content

### 4. **Code Search**
Search internal codebases by functionality, not just function names

### 5. **Personal Knowledge Management**
Notes, articles, books - build your second brain

### 6. **Customer Support**
Internal documentation, troubleshooting guides, product manuals

---

## Getting Started (Quick Summary)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/semantik.git
cd semantik

# 2. Run interactive setup
make wizard

# 3. Start all services
make docker-up

# 4. Access UI
open http://localhost:8080

# 5. Create account and start searching!
```

**Detailed Documentation:**
- Setup: `docs/SETUP.md`
- API Reference: `docs/API_REFERENCE.md`
- Architecture: `docs/ARCH.md`
- Development: `CLAUDE.md`

---

## Summary

**Semantik is a privacy-first, self-hosted semantic search engine** that brings AI-powered search to your private documents without sending data to the cloud. It's a **complete application** (not a library) with a modern React UI, robust FastAPI backend, and production-grade architecture including microservices, background processing, and real-time updates.

**Current State:** Fully functional for core use cases (create collections, ingest documents, semantic search), with several advanced features already implemented in the backend but not yet exposed in the frontend UI.

**Target Users:** Privacy-conscious individuals and organizations who want semantic search capabilities without cloud dependencies.

**Technical Level:** Production-grade architecture with proper security, monitoring, testing, and operational features. Suitable for deployment in real environments, though currently pre-release with active development.

---

**Questions for Feedback:**
1. Is the core value proposition (privacy-first semantic search) compelling?
2. Is the UX clear and intuitive?
3. What features would you expect that are currently missing?
4. How would you use this in your workflow?
5. What concerns you most about self-hosting an AI application?
6. Is the performance acceptable for your use case?
7. Would you prefer this over cloud solutions? Why or why not?

---

**Contact:** [Your contact info]
**License:** [Your license]
**Repository:** [Your GitHub URL]
