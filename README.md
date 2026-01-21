# Semantik

Private, self‑hosted semantic search for your documents.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Semantik is a self‑hosted semantic search stack for local documents. I built it to make searching my own unstructured corpus less painful and to keep a clean playground for experimenting with various retrieval, chunking, and reranking strategies. It runs fully on your hardware: FastAPI control plane + Celery workers + a dedicated embedding/search service over Postgres, Redis, and Qdrant. GPU is recommended for indexing speed, but CPU works for modest collections.

By default, models run locally and no document content leaves your machine.

This is a personal project and still pre‑release — expect rough edges and API churn.

<p align="center">
  <a href="docs/images/dashboard.png"><img src="docs/images/dashboard.png" width="32%" alt="Collections Dashboard" /></a>
  <a href="docs/images/search.png"><img src="docs/images/search.png" width="32%" alt="Search Interface" /></a>
  <a href="docs/images/models.png"><img src="docs/images/models.png" width="32%" alt="Model Manager" /></a>
</p>

<details>
<summary>More screenshots</summary>
<p align="center">
  <a href="docs/images/benchmarks.png"><img src="docs/images/benchmarks.png" width="45%" alt="Benchmarking" /></a>
  <a href="docs/images/mcp-profiles.png"><img src="docs/images/mcp-profiles.png" width="45%" alt="MCP Profiles" /></a>
</p>
</details>

## What It Does
- **Collections**: group documents with their own embedding + chunking config.
- **Ingestion pipeline**: scan → extract → chunk → embed → upsert, all async.
- **Connectors**: pluggable ingestion sources — built-ins include directories, Git repos, and IMAP mailboxes (credentials encrypted at rest; see `docs/CONNECTORS.md`).
- **Formats** include PDF, DOCX, Markdown, HTML, plain text, and more (via `unstructured`).
- **Search**: dense (semantic), sparse (BM25/SPLADE), and hybrid modes with RRF fusion. Optional cross‑encoder reranking and HyDE query expansion.
- **Sparse indexing**: BM25 for fast TF-IDF keyword search (~1000 docs/sec), SPLADE for neural sparse vectors (10-50 docs/sec on GPU).
- **LLM integration**: multi-provider support (Anthropic, OpenAI, local GPU) for HyDE query expansion and future features. Quality tiers let you balance cost vs capability.
- **MCP Integration**: connect AI assistants (Claude Desktop, Cursor) to search your collections via the Model Context Protocol. See `docs/MCP.md`.
- **API keys**: programmatic access with scoped, revocable keys for integrations and automation.
- **Model Manager**: download, track, and delete models (embedding, LLM, reranker, SPLADE) with cache visibility and usage tracking.
- **Benchmarking**: evaluate search quality with standard IR metrics (Precision@K, Recall@K, MRR, nDCG) against ground truth datasets.
- **Live progress** is streamed to the UI over Redis + WebSockets.
- **Zero‑downtime reindexing**: blue/green staging + swap + cleanup.
- **Chunking lab**: 6 built‑in strategies (character, recursive, markdown, semantic, hierarchical, hybrid) plus a plugin system for adding additional strategies.
- **Embeddings lab**: swap models/quantization per collection; mock mode for testing.
- **Visualize:** project embeddings into 2D space & visualize relationships.
- **Continuous sync**: keep collections up-to-date automatically with configurable sync intervals for your data sources.

## Sources & Continuous Sync

Collections support multiple data sources with optional continuous sync:

- **Sync Modes**: `one_time` (manual) or `continuous` (automatic at intervals)
- **Interval**: Minimum 15 minutes for continuous sync
- **Encrypted Credentials**: Git tokens, SSH keys, and IMAP passwords stored encrypted at rest

## Architecture
Three Python packages, one frontend:
- `packages/webui/` – FastAPI app serving REST + WebSocket APIs and the React UI; owns Postgres state.
- `packages/vecpipe/` – embedding + search HTTP service that talks to Qdrant.
- `packages/shared/` – shared configs, DB models/repos, chunking + embedding utilities.
- `apps/webui-react/` – React/Vite UI, built into `packages/webui/static/`.

Data flow:
1. Client creates/updates a collection.
2. WebUI writes an **operation** row to Postgres and dispatches a Celery task.
3. Worker extracts text, chunks, requests embeddings from vecpipe, and upserts to Qdrant.
4. Progress goes to Redis → WebSockets → UI; final status is persisted in Postgres.

```mermaid
flowchart LR
  UI[React Web UI] -->|HTTP/WebSocket| WebUI[FastAPI WebUI]
  WebUI <--> PG[(Postgres)]
  WebUI <--> R[Redis]
  WebUI -->|Celery task| W[Worker]
  W -->|/embed /search| V[Vecpipe]
  V <--> Q[(Qdrant)]
```

## Quickstart (Docker)
Prereqs: Docker + Compose. NVIDIA runtime if you want GPU acceleration.

```bash
git clone https://github.com/jbmiller10/semantik.git
cd semantik

# Guided setup: generates .env, checks GPU/paths, starts stack
make wizard

# Manual setup:
cp .env.docker.example .env
# IMPORTANT: Replace ALL placeholder credentials in .env before starting:
#   JWT_SECRET_KEY, POSTGRES_PASSWORD, REDIS_PASSWORD, QDRANT_API_KEY,
#   FLOWER_USERNAME, FLOWER_PASSWORD
# Generate secrets with: openssl rand -hex 32
# If you are not using connectors with credentials, set CONNECTOR_SECRETS_KEY="" to disable secrets encryption.
make docker-up
```

Endpoints after boot:
- Web UI + API: `http://localhost:8080`
- Vecpipe API (internal-only by default): `http://vecpipe:8000` (expose port 8000 if you need host access)
- Qdrant: `http://localhost:6333`

First run note: open the UI and create an account. The very first user is made admin/superuser automatically.

Stop: `make docker-down` (keep volumes) or `make docker-down-clean` (wipe volumes).

## Usage
1. Open the UI at `http://localhost:8080` and create a collection.
2. Add one or more sources (directories; optionally Git repos / IMAP mailboxes). Semantik will index in the background.
3. Search across one or more collections.

If you prefer the API, the v2 endpoints are under `/api/v2/*` — see `docs/API_REFERENCE.md`.

## Configuration
`make wizard` or `.env.docker.example` covers the common knobs. 

See:`docs/CONFIGURATION.md`.

## Extensibility
Semantik is meant to be a sandbox for trying retrieval ideas:
- **Plugins** for 6 extension points: embedding providers, chunking strategies, connectors, rerankers, sparse indexers, and extractors. Load via Python entry points (`semantik.plugins`).
- **Runtime-checkable protocols** let you build external plugins without importing Semantik internals. See `docs/external-plugins.md` and `docs/plugin-protocols.md`.

Plugin loading is idempotent and safe to run without plugins installed.

## Dev Notes
Backend:
```bash
make dev-install
make run        # FastAPI hot reload on :8080
```

Frontend:
```bash
make frontend-install
make frontend-dev   # Vite on :5173, proxies /api and /ws to backend
```

Integrated dev stack (API + worker + vecpipe):
```bash
make dev
```

## Testing
- Backend: `make test` (Pytest) or `make test-coverage`.
- E2E (requires running stack): `make test-e2e`.
- Frontend: `npm test --prefix apps/webui-react`.
- Test Postgres profile: `docker compose --profile testing up -d postgres_test` (port 55432).


## Roadmap 
Next up:
- Built-in agentic search
- Improvements to the memory governor system
- bugfixes

In no particular order:
- Knowledge graph builder for GraphRAG
- Expanded benchmarking tools
- Support for Text+Image embedding models
- Support for custom embedding models/any sentence-transformers compatible model
- Expanded visualization options for exploring the embedding space
- Additional connectors, parsers, chunkers

## License
AGPL‑3.0. See `LICENSE`.

_Active development; breaking changes are possible._
