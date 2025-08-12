# Semantik Cleanroom Documentation

## Overview

This directory contains comprehensive cleanroom documentation for all major components of the Semantik application. These documents are designed to provide LLM agents with precise, accurate context when working on specific parts of the codebase.

**Last Updated:** 2025-08-12  
**Total Documents:** 19  
**Coverage:** Complete system architecture from database to frontend

## Purpose

These documents serve as authoritative references that:
- Provide deep technical context without requiring full codebase analysis
- Enable accurate code generation and modification
- Document architectural decisions and patterns
- Specify security requirements and best practices
- Define testing requirements and patterns

## Document Organization

### Backend Services (`backend/`)

#### Core Services
- **[WEBUI_SERVICE.md](backend/WEBUI_SERVICE.md)** - FastAPI backend service
  - API routing, service layer, WebSocket management, rate limiting
  - Key files: `packages/webui/main.py`, `packages/webui/app.py`
  
- **[VECPIPE_SERVICE.md](backend/VECPIPE_SERVICE.md)** - Vector pipeline service
  - Embedding generation, Qdrant integration, search, reranking
  - Key files: `packages/vecpipe/*.py`
  
- **[WORKER_SERVICE.md](backend/WORKER_SERVICE.md)** - Celery worker service
  - Async task processing, progress reporting, cancellation
  - Key files: `packages/webui/celery_app.py`, `packages/webui/tasks.py`

#### Shared Infrastructure
- **[SHARED_LIBRARY.md](backend/SHARED_LIBRARY.md)** - Shared components
  - Database models, repositories, configuration, utilities
  - Key directory: `packages/shared/`
  
- **[DATABASE_LAYER.md](backend/DATABASE_LAYER.md)** - Data persistence
  - PostgreSQL schema, partitioning, migrations, repositories
  - Key files: `packages/shared/database/`, `alembic/`

### Domain Systems (`domain/`)

#### Core Functionality
- **[CHUNKING_SYSTEM.md](domain/CHUNKING_SYSTEM.md)** - Document chunking
  - 6 strategies, streaming processing, memory management
  - Key directory: `packages/shared/chunking/`
  
- **[SEARCH_SYSTEM.md](domain/SEARCH_SYSTEM.md)** - Search functionality
  - Hybrid search, reranking, query optimization
  - Key files: `packages/vecpipe/search_api.py`, `packages/webui/services/search_service.py`
  
- **[COLLECTION_MANAGEMENT.md](domain/COLLECTION_MANAGEMENT.md)** - Collection operations
  - CRUD operations, status management, permissions
  - Key files: `packages/webui/services/collection_service.py`

#### System Operations
- **[OPERATION_MANAGEMENT.md](domain/OPERATION_MANAGEMENT.md)** - Async operations
  - Operation lifecycle, progress tracking, WebSocket notifications
  - Key files: `packages/webui/services/operation_service.py`
  
- **[AUTH_SYSTEM.md](domain/AUTH_SYSTEM.md)** - Authentication & authorization
  - JWT tokens, session management, permissions
  - Key files: `packages/webui/auth.py`, `packages/webui/dependencies.py`

### Frontend Components (`frontend/`)

#### Architecture & State
- **[REACT_ARCHITECTURE.md](frontend/REACT_ARCHITECTURE.md)** - Frontend structure
  - React 19 SPA, Vite build, TypeScript configuration
  - Key directory: `apps/webui-react/src/`
  
- **[STATE_MANAGEMENT.md](frontend/STATE_MANAGEMENT.md)** - Zustand stores
  - State architecture, persistence, WebSocket sync
  - Key directory: `apps/webui-react/src/stores/`

#### UI & Communication
- **[UI_COMPONENTS.md](frontend/UI_COMPONENTS.md)** - Component library
  - 30+ React components, modals, real-time updates
  - Key directory: `apps/webui-react/src/components/`
  
- **[API_CLIENT_LAYER.md](frontend/API_CLIENT_LAYER.md)** - Backend communication
  - Axios client, WebSocket client, type definitions
  - Key directory: `apps/webui-react/src/services/`
  
- **[CHUNKING_UI.md](frontend/CHUNKING_UI.md)** - Chunking interface
  - Strategy selector, preview, comparison, analytics
  - Key directory: `apps/webui-react/src/components/chunking/`

### Infrastructure (`infrastructure/`)

- **[DOCKER_INFRASTRUCTURE.md](infrastructure/DOCKER_INFRASTRUCTURE.md)** - Containerization
  - Service definitions, networking, volumes, CUDA support
  - Key files: `docker-compose*.yml`, `Dockerfile`
  
- **[DEPLOYMENT_CONFIGURATION.md](infrastructure/DEPLOYMENT_CONFIGURATION.md)** - Deployment setup
  - Environment configuration, migrations, model setup
  - Key files: `Makefile`, deployment scripts
  
- **[TESTING_INFRASTRUCTURE.md](infrastructure/TESTING_INFRASTRUCTURE.md)** - Test framework
  - pytest, Vitest, MSW, coverage requirements
  - Key directories: `tests/`, component `__tests__/` directories

## Quick Reference Guide

### By Task Type

#### Working on API Endpoints
- Primary: [WEBUI_SERVICE.md](backend/WEBUI_SERVICE.md)
- Supporting: [AUTH_SYSTEM.md](domain/AUTH_SYSTEM.md), [DATABASE_LAYER.md](backend/DATABASE_LAYER.md)

#### Implementing New Features
- Backend: [SHARED_LIBRARY.md](backend/SHARED_LIBRARY.md), [WORKER_SERVICE.md](backend/WORKER_SERVICE.md)
- Frontend: [UI_COMPONENTS.md](frontend/UI_COMPONENTS.md), [STATE_MANAGEMENT.md](frontend/STATE_MANAGEMENT.md)

#### Search & Embeddings
- Primary: [SEARCH_SYSTEM.md](domain/SEARCH_SYSTEM.md), [VECPIPE_SERVICE.md](backend/VECPIPE_SERVICE.md)
- Supporting: [CHUNKING_SYSTEM.md](domain/CHUNKING_SYSTEM.md)

#### Collections & Documents
- Primary: [COLLECTION_MANAGEMENT.md](domain/COLLECTION_MANAGEMENT.md)
- Supporting: [OPERATION_MANAGEMENT.md](domain/OPERATION_MANAGEMENT.md)

#### Frontend Development
- Architecture: [REACT_ARCHITECTURE.md](frontend/REACT_ARCHITECTURE.md)
- Components: [UI_COMPONENTS.md](frontend/UI_COMPONENTS.md)
- API Integration: [API_CLIENT_LAYER.md](frontend/API_CLIENT_LAYER.md)

#### DevOps & Deployment
- Docker: [DOCKER_INFRASTRUCTURE.md](infrastructure/DOCKER_INFRASTRUCTURE.md)
- Deployment: [DEPLOYMENT_CONFIGURATION.md](infrastructure/DEPLOYMENT_CONFIGURATION.md)
- Testing: [TESTING_INFRASTRUCTURE.md](infrastructure/TESTING_INFRASTRUCTURE.md)

## Document Standards

Each document follows a consistent structure:

1. **Component Overview** - Purpose, responsibilities, key features
2. **Architecture & Design Patterns** - Patterns used, architectural decisions
3. **Key Interfaces & Contracts** - API definitions, type contracts
4. **Data Flow & Dependencies** - How data moves through the system
5. **Critical Implementation Details** - Core algorithms, important code
6. **Security Considerations** - Security features and requirements
7. **Testing Requirements** - Test patterns, coverage requirements
8. **Common Pitfalls & Best Practices** - Mistakes to avoid, patterns to follow
9. **Configuration & Environment** - Settings, environment variables
10. **Integration Points** - How component connects with others

## Usage Guidelines

### For LLM Agents

1. **Start with the index** - Use this README to identify relevant documents
2. **Read primary documents first** - Focus on the main document for your task
3. **Check integration points** - Review how your component interacts with others
4. **Follow security guidelines** - Always check security considerations
5. **Maintain patterns** - Follow established patterns documented in each file
6. **Test requirements** - Ensure you meet testing requirements specified

### For Developers

1. **Keep documentation updated** - Update relevant docs when making significant changes
2. **Maintain accuracy** - Documentation must reflect actual implementation
3. **Add version stamps** - Include date when making major updates
4. **Cross-reference** - Link between related documents
5. **Include examples** - Provide code examples for complex patterns

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   UI     │ │  State   │ │   API    │ │ Chunking │      │
│  │Components│ │Management│ │  Client  │ │    UI    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   WebUI Service   │
                    │    (FastAPI)      │
                    └───────┬───────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐ ┌────────▼────────┐ ┌───────▼────────┐
│ Worker Service │ │ VecPipe Service │ │  Auth System   │
│   (Celery)     │ │  (Embeddings)   │ │     (JWT)      │
└────────────────┘ └─────────────────┘ └────────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                ┌───────────▼───────────┐
                │   Shared Library      │
                │  (Models, Repos)      │
                └───────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐ ┌────────▼────────┐ ┌───────▼────────┐
│   PostgreSQL   │ │     Redis       │ │    Qdrant      │
│   (Metadata)   │ │ (Cache/Queue)   │ │   (Vectors)    │
└────────────────┘ └─────────────────┘ └────────────────┘
```

## Maintenance

- **Review Frequency:** Quarterly or after major refactoring
- **Update Trigger:** Any breaking API changes or architectural shifts
- **Validation:** Use backend/frontend code reviewers to verify accuracy
- **Version Control:** Track all documentation changes in git

## Contact

For questions about the documentation or to report inaccuracies, please:
1. Check the relevant document's "Common Pitfalls" section
2. Review integration points with other components
3. Consult the CLAUDE.md file for project-wide guidelines
4. Create an issue if documentation needs updating

---

*This cleanroom documentation represents the current state of the Semantik application architecture as of 2025-08-12. All file paths, interfaces, and implementation details have been verified against the actual codebase.*