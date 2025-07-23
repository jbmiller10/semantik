# Semantik Project Overview

## Purpose
Semantik is a self-hosted semantic search engine that transforms private file servers into powerful, AI-powered knowledge bases without data ever leaving the user's hardware. It's currently in pre-release state and undergoing a critical refactoring from a "job-centric" to a "collection-centric" architecture.

## Key Features
- 100% Private & Self-Hosted (zero external API calls)
- State-of-the-Art Search Intelligence (semantic search, cross-encoder reranking, hybrid search)
- Complete Control & Tunability (model selection, resource management, GPU handling)
- Easy-to-Use Interface (intuitive UI, real-time monitoring, document viewer)

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI
- **Database**: PostgreSQL (metadata), Qdrant (vectors)
- **ORM**: SQLAlchemy (async)
- **Task Queue**: Celery with Redis
- **Dependencies**: sentence-transformers, torch, transformers

### Frontend  
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand
- **API Client**: React Query, Axios
- **UI Library**: TailwindCSS
- **Icons**: Lucide React

### DevOps
- **Containerization**: Docker & Docker Compose
- **Database Migrations**: Alembic
- **Process Management**: Supervisor (in containers)

## Architecture Components
1. **webui**: FastAPI backend for user auth, collection management, and serving frontend
2. **vecpipe**: Dedicated FastAPI service for document parsing, embedding, and search queries
3. **worker**: Celery worker for asynchronous background tasks (indexing, re-indexing)
4. **webui-react**: React SPA providing the user interface
5. **shared**: Shared Python library with database models, configs, and utilities