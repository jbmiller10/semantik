# Semantik

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg?logo=docker)](https://www.docker.com)

**Semantik** is a self-hosted, privacy-first semantic search engine.

I built Semantik to explore advanced RAG (Retrieval-Augmented Generation) techniques and to have a private, local way to search my document archives without relying on the cloud. It has since grown into a full-featured system.

> ⚠️ **Note**: This is a personal project in active development (pre-release). While functional, expect rough edges.

---

## Why Semantik?

Most semantic search tools are either simple libraries or cloud-dependent SaaS. Semantik is designed to be:
*   **Private**: Runs 100% on local hardware (Docker). Zero external API calls.
*   **Transparent**: Full control over chunking strategies, embedding models, and search parameters.
*   **Educational**: A "deep dive" into how modern vector search systems are architected.

## Architecture Deep Dive

I designed Semantik with a strict separation of concerns to mimic production-grade microservices:

*   **`vecpipe` (Headless Vector Engine)**: A standalone service solely responsible for heavy lifting—extracting text, chunking, generating embeddings, and interfacing with Qdrant. It knows nothing about users or permissions.
*   **`webui` (Management Console)**: The user-facing application. It handles authentication, collection management, and orchestrates tasks via Celery.
*   **`worker`**: Background workers that handle long-running ingestion jobs, ensuring the UI remains responsive.

```mermaid
flowchart LR
  subgraph Client
    UI[Web UI (React 19)]
  end
  subgraph API
    WebUI[FastAPI Control Plane]
  end
  subgraph Infra
    PG[(Postgres)]
    R[Redis]
    Q[Qdrant]
  end
  subgraph Worker
    C[Celery Tasks]
  end
  subgraph Vector Engine
    V[Vecpipe Service]
  end

  UI -->|HTTP/WebSocket| WebUI
  WebUI <--> PG
  WebUI <--> R
  WebUI -->|dispatch ops| C
  C <--> PG
  C <--> R
  C -->|embed/search| V
  V <--> Q
```

### Key Technical Features

*   **Collection-Centric Design**: Documents are grouped into "Collections," each with its own isolated configuration (embedding model, quantization level, chunking strategy).
*   **Advanced RAG Techniques**:
    *   **Hybrid Search**: Combines dense vector retrieval with keyword matching.
    *   **Reranking**: Optional cross-encoder step to re-score results for higher precision.
    *   **Configurable Chunking**: Supports Recursive, Markdown, and Semantic chunking strategies.
*   **Interactive Embedding Visualization**:
    *   Project high-dimensional vectors into 2D using UMAP, t-SNE, or PCA.
    *   Visually explore document clusters to find patterns or outliers.
    *   WebGPU-accelerated rendering for smooth interaction with large datasets.
*   **Real-Time Feedback**: Uses WebSockets to stream detailed progress of ingestion tasks (e.g., "Processing file 45/100") directly to the UI.
*   **Blue/Green Reindexing**: Reindexing a collection builds a shadow index and swaps it atomically upon completion, ensuring zero downtime.

## Extensibility

Semantik is designed to be extended without forking the core codebase. It uses Python entry points to discover plugins:

*   **Chunking Strategies**: Register custom chunking strategies via `semantik.chunking_strategies`.
*   **Embedding Providers**: Add support for custom embedding models via `semantik.embedding_providers`.

## Tech Stack

*   **Backend**: Python 3.11, FastAPI, SQLAlchemy (Async), Celery, Pydantic.
*   **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Zustand, React Query.
*   **Data & Infra**:
    *   **Qdrant**: Vector database.
    *   **Postgres 16**: Relational data and state.
    *   **Redis 7**: Task broker and pub/sub for real-time updates.
    *   **Docker Compose**: Orchestration.

## Getting Started

You will need **Docker** and **Docker Compose**. An NVIDIA GPU is highly recommended for reasonable indexing speeds, but CPU mode is supported for small datasets.

### 1. Setup
Use the included wizard to generate your configuration (`.env`):

```bash
make wizard
```

### 2. Run
Start the full stack:

```bash
make docker-up
```

Access the UI at **http://localhost:8080**.

### 3. Development
If you want to poke around the code:
*   **Backend**: `packages/` (managed with `uv`)
*   **Frontend**: `apps/webui-react/` (managed with `npm`)

Run the backend locally while keeping infra in Docker:
```bash
make docker-dev-up  # Starts Postgres, Redis, Qdrant
make run            # Starts FastAPI locally
```

## Roadmap & Limitations

*   **Status**: Pre-release. Core search and ingestion work well, but UI for some advanced features (like model management) is still in progress.
*   **Hardware**: Embedding models are heavy. A GPU with at least 6GB VRAM is recommended for the default models (Qwen/BAAI).
*   **Future**:
    *   Better support for non-text formats (OCR).
    *   Public/Private collection sharing.
    *   MCP (Model Context Protocol) integration.

## License

AGPL v3.0
