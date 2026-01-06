# Changelog

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
