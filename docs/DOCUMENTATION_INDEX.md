# Semantik Documentation Index

This index lists all project‑owned documentation currently in `docs/` and suggests a few reading paths. If a file is not linked here, it either lives outside `docs/` (e.g., app/package READMEs) or is not part of the maintained documentation set.

## Quick Links

- [Getting Started](#getting-started-path) — New to Semantik? Start here.
- [Developer Path](#developer-path) — Working on Semantik or integrating deeply.
- [Operations Path](#operations-path) — Deploying and running Semantik in production.

## Documentation Overview

### Core System

**Architecture**
- **[ARCH.md](./ARCH.md)** — Full system architecture, components, and design decisions.
- **[API_ARCHITECTURE.md](./API_ARCHITECTURE.md)** — API design, versioning, and security patterns.
- **[DATABASE_ARCH.md](./DATABASE_ARCH.md)** — Postgres schema and persistence strategy.
- **[INFRASTRUCTURE.md](./INFRASTRUCTURE.md)** — Service topology, container layout, and scaling notes.
- **[FRONTEND_ARCH.md](./FRONTEND_ARCH.md)** — React/Zustand architecture and UI design notes.

**APIs**
- **[API_REFERENCE.md](./API_REFERENCE.md)** — REST API reference (primarily `/api/v2/*`).
- **[WEBSOCKET_API.md](./WEBSOCKET_API.md)** — WebSocket events for operation progress and live updates.
- **[SEARCH.md](./SEARCH.md)** — Search endpoint guide and examples.
- **[api/CHUNKING_API.md](./api/CHUNKING_API.md)** — Chunking API endpoints.
- **[api/CHUNKING_EXAMPLES.md](./api/CHUNKING_EXAMPLES.md)** — Chunking API usage examples.

**Features**
- **[COLLECTIONS.md](./COLLECTIONS.md)** — Collection model and lifecycle.
- **[COLLECTION_MANAGEMENT.md](./COLLECTION_MANAGEMENT.md)** — UI/API guide for creating and managing collections.
- **[SEARCH_SYSTEM.md](./SEARCH_SYSTEM.md)** — Semantic/keyword/hybrid search implementation details.
- **[RERANKING.md](./RERANKING.md)** — Cross‑encoder reranking flow and tuning.
- **[CHUNKING_FEATURE_OVERVIEW.md](./CHUNKING_FEATURE_OVERVIEW.md)** — Built‑in chunkers and when to use them.
- **[CHUNKING_PLUGINS.md](./CHUNKING_PLUGINS.md)** — Plugin system for custom chunking strategies.
- **[EMBEDDING_PLUGINS.md](./EMBEDDING_PLUGINS.md)** — Embedding provider plugins and configuration.
- **[EMBEDDING_VISUALIZATION.md](./EMBEDDING_VISUALIZATION.md)** — UMAP/projection visualization features.
- **[DATA_ACCESS_CATALOG.md](./DATA_ACCESS_CATALOG.md)** — Data access patterns and repository catalog.

### Deployment & Operations

- **[DOCKER.md](./DOCKER.md)** — Docker/Compose setup, CPU vs GPU profiles, and common fixes.
- **[DEPLOYMENT.md](./DEPLOYMENT.md)** — Production deployment checklist and hardening.
- **[CONFIGURATION.md](./CONFIGURATION.md)** — Environment variable reference and tuning.
- **[partition-monitoring.md](./partition-monitoring.md)** — Partition/collection monitoring and health workflows.
- **[security.md](./security.md)** — Auth, secrets, and security considerations.
- **[TESTING.md](./TESTING.md)** — Test strategy, commands, and CI notes.

## Reading Paths

### Getting Started Path

1. **[README.md](../README.md)** — Overview and quickstart.
2. **[DOCKER.md](./DOCKER.md)** — Bring up the stack.
3. **[CONFIGURATION.md](./CONFIGURATION.md)** — Core environment settings.
4. **[COLLECTION_MANAGEMENT.md](./COLLECTION_MANAGEMENT.md)** — Create your first collection.
5. **[SEARCH.md](./SEARCH.md)** — Run searches and interpret results.

### Developer Path

1. **[ARCH.md](./ARCH.md)** and **[API_ARCHITECTURE.md](./API_ARCHITECTURE.md)** — Big picture.
2. **[API_REFERENCE.md](./API_REFERENCE.md)** and **[WEBSOCKET_API.md](./WEBSOCKET_API.md)** — Contracts.
3. **[DATABASE_ARCH.md](./DATABASE_ARCH.md)** and **[DATA_ACCESS_CATALOG.md](./DATA_ACCESS_CATALOG.md)** — Persistence and repos.
4. **[CHUNKING_FEATURE_OVERVIEW.md](./CHUNKING_FEATURE_OVERVIEW.md)**, **[CHUNKING_PLUGINS.md](./CHUNKING_PLUGINS.md)** — Chunking internals.
5. **[EMBEDDING_PLUGINS.md](./EMBEDDING_PLUGINS.md)** and **[RERANKING.md](./RERANKING.md)** — Retrieval tuning.
6. **[FRONTEND_ARCH.md](./FRONTEND_ARCH.md)** — UI implementation.
7. **[TESTING.md](./TESTING.md)** — Running and extending tests.

### Operations Path

1. **[DEPLOYMENT.md](./DEPLOYMENT.md)** — Production setup.
2. **[INFRASTRUCTURE.md](./INFRASTRUCTURE.md)** and **[DOCKER.md](./DOCKER.md)** — Service layout.
3. **[CONFIGURATION.md](./CONFIGURATION.md)** — Environment hardening and tuning.
4. **[partition-monitoring.md](./partition-monitoring.md)** — Monitoring and remediation.
5. **[security.md](./security.md)** — Auth and secrets.

## Version Information

Current as of:
- **Semantik Version**: `v0.6` (current branch; collection‑centric architecture)
- **Documentation Update**: 2025‑12‑12
- **API Version**: v2 (`/api/v2/*`)

---

If you spot a mismatch between docs and code, please open an issue or PR so the index stays aligned.
