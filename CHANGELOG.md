# Changelog

## [0.8.0] - 2026-01-20

### Added
- **MCP server** — stdio transport exposing collections via `semantik-mcp serve` 
- **MCP profiles** — scoped collection access with per-profile search configuration and client config generation
- **BM25 sparse indexer** — TF-IDF keyword indexing with IDF statistics persistence (~1000 docs/sec)
- **SPLADE sparse indexer** — neural sparse indexing via HuggingFace models (10-50 docs/sec GPU)
- **Hybrid search** — RRF fusion combining dense and sparse results with tunable k parameter
- **Plugin protocols** — runtime-checkable interfaces for all 6 plugin types enabling external plugin development
- **DTO adapters** — type-safe conversion layer between plugin outputs and internal types
- **API Key Management** — programmatic API access with SHA-256 hashed keys, expiration, revocation, and per-user limits
- **LLM integration** — multi-provider support (Anthropic, OpenAI, local GPU) with quality tiers and encrypted API key storage
- **HyDE search** — Hypothetical Document Embeddings for query expansion via LLM-generated hypothetical documents
- **Benchmarking system** — search quality evaluation with Precision@K, Recall@K, MRR, nDCG metrics and ground truth datasets
- **Model Manager** — centralized LLM lifecycle management with memory estimation and quantization support (int4, int8, float16)
- **Settings overhaul** — admin panel for deployment-wide settings and user defaults management for search/collections
- WebSocket streaming for agent responses
- Search mode selector UI (dense, sparse, hybrid)
- Sparse index configuration UI per collection
- Parser refactoring for groundwork on multiple file parser support

### Changed
- Search API accepts `search_mode`, `rrf_k`, and `use_hyde` parameters
- Collection stats update in real-time during indexing
- Settings menu reorganized with Database, Plugins, MCP Profiles, LLM, and API Keys tabs

### Fixed
- Collection stats not refreshing after indexing completes
- Chunk numbering display and Qdrant payload construction
- Source directories showing empty despite indexed files
- Continuous sync tasks not being consumed by worker
- WebSocket `operation_completed` message missing `collection_id`
- Dashboard polling frequency during processing

## [0.7.7] - 2026-01-05

### Added
- **GPUMemoryGovernor** for dynamic VRAM management — LRU-based model tracking, CPU offloading for warm models, background pressure monitoring
- Memory API endpoints: `/memory/stats`, `/memory/models`, `/memory/evictions`
- Parallel document processing with producer-consumer architecture
- Internal service-to-service authentication

## [0.7.6] - 2026-01-03

### Added
- Database indexes for plugin config queries
- Sync error propagation in plugin management endpoints

## [0.7.5] - 2026-01-02

### Added
- **Reranker plugins** with built-in Qwen3 Reranker
- **Extractor plugins** for metadata extraction during ingestion
- Remote plugin registry with in-app installation UI
- Audit logging for plugin operations

## [0.7.4] - 2026-01-01

### Added
- Unified plugin registry using `semantik.plugins` entry points
- `/api/v2/plugins` management API
- `semantik-plugin` CLI for scaffolding and validation

## [0.7.3] - 2025-12-31

### Added
- Thread-safe plugin registries
- Connector self-description metadata

### Security
- WebSocket JWT tokens moved from URL to protocol header
- XSS and path traversal fixes
- Docker hardening (required credentials, localhost-bound ports)
- CORS rejects wildcard origins

## [0.7.0] - 2025-12-14

### Added
- **Source management** — CRUD API for data sources with continuous sync support
- **GitConnector** — SSH/HTTPS, branch filtering, shallow clones
- **IMAPConnector** — Email indexing with folder filtering and attachment extraction
- Encrypted credential storage (Fernet)
- Document staleness tracking

### Breaking Changes
- `CONNECTOR_SECRETS_KEY` required for credentialed connectors
- 5 new database migrations

## [0.6.0] - 2025-12-12

### Added
- **Embedding visualization** — 2D projections via TMAP, t-SNE, PCA
- **Plugin system** for chunking strategies and embedding providers
- Connector base classes and factory pattern
- Collection metadata caching

### Changed
- Search page redesign
- Modularized vecpipe into dedicated service/router/schema modules

### Fixed
- Qdrant client connection leaks
- Collection metadata retrieval
- VecPipe API contract alignment

## [0.5.0] - 2025-10-20

### Added
- **6 chunking strategies**: Character, Recursive, Markdown, Semantic, Hierarchical, Hybrid
- Chunking REST API with preview, compare, and analytics endpoints
- WebSocket progress updates
- Database partitioning for chunks table
- Redis caching for previews

### Security
- ReDoS protection in regex operations
- Input validation limits
