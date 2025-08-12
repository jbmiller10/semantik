# Semantik - Complete System Specification

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Database Architecture](#database-architecture)
4. [Backend API Architecture](#backend-api-architecture)
5. [Frontend Architecture](#frontend-architecture)
6. [Document Processing Pipeline](#document-processing-pipeline)
7. [Security Architecture](#security-architecture)
8. [Testing Architecture](#testing-architecture)
9. [DevOps & Infrastructure](#devops--infrastructure)
10. [Shared Libraries](#shared-libraries)
11. [Business Logic & Workflows](#business-logic--workflows)
12. [Refactoring Status](#refactoring-status)
13. [Quick Reference](#quick-reference)

---

## Executive Summary

Semantik is a self-hosted semantic search engine that transforms private file servers into AI-powered knowledge bases without data ever leaving the user's hardware. The system is currently in pre-release, undergoing a critical refactoring from a "job-centric" to a "collection-centric" architecture (approximately 85% complete).

### Key Technologies
- **Backend**: Python 3.11+, FastAPI, SQLAlchemy, PostgreSQL, Celery, Redis
- **Frontend**: React 19, TypeScript, Vite, Zustand, React Query, TailwindCSS
- **Vector Database**: Qdrant for semantic embeddings
- **Infrastructure**: Docker, Docker Compose, Alembic migrations

### Architecture Highlights
- **Microservices**: webui, vecpipe, worker services with clear separation
- **Scalability**: PostgreSQL partitioning with 100 partitions for chunks
- **Real-time**: WebSocket support for live updates
- **Security**: JWT authentication, RBAC, comprehensive input validation
- **Performance**: Streaming processing, memory-bounded operations

---

## System Overview

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│                    Zustand + React Query + WS                │
└─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│                      WebUI Service (FastAPI)                 │
│              Authentication, API, WebSocket Server           │
└─────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   VecPipe Service        │    │      Celery Worker           │
│  Document Processing     │    │    Async Task Processing     │
│  Embedding Generation    │    │    Background Operations     │
└──────────────────────────┘    └──────────────────────────────┘
                    │                           │
        ┌───────────┴───────────┬───────────────┴────────┐
        ▼                       ▼                        ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│  PostgreSQL  │    │     Qdrant       │    │    Redis     │
│   Metadata   │    │  Vector Store    │    │  Cache/Queue │
└──────────────┘    └──────────────────┘    └──────────────┘
```

### Service Responsibilities

- **WebUI**: User authentication, API endpoints, frontend serving
- **VecPipe**: Document parsing, embedding generation, search queries
- **Worker**: Asynchronous indexing, reindexing, maintenance tasks
- **Shared**: Common models, configurations, utilities

---

## Database Architecture

### PostgreSQL Schema

#### Partitioning Strategy
The `chunks` table uses LIST partitioning with 100 partitions for scalability:

```sql
-- Partition key: abs(hashtext(collection_id)) % 100
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    collection_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding_id TEXT,
    chunk_index INTEGER,
    metadata JSONB,
    partition_key INTEGER GENERATED ALWAYS AS (abs(hashtext(collection_id::text)) % 100) STORED
) PARTITION BY LIST (partition_key);

-- 100 partitions: chunks_part_00 through chunks_part_99
```

#### Core Tables

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `users` | User authentication | bcrypt passwords, JWT tokens |
| `collections` | Document collections | Qdrant integration, config storage |
| `documents` | Document metadata | Chunking status, source tracking |
| `chunks` | Text chunks | Partitioned, embedding references |
| `operations` | Async operations | Status tracking, progress monitoring |
| `collection_permissions` | RBAC | Owner, read, write, admin levels |

#### Performance Optimizations
- **Partition Pruning**: ~100x query improvement for collection queries
- **Indexes**: Strategic indexes on foreign keys and search fields  
- **Connection Pooling**: 20 base connections, 40 overflow
- **Materialized Views**: Pre-computed statistics

### Qdrant Integration

#### Blue-Green Reindexing
```python
# Zero-downtime reindexing strategy
1. Create staging collection (col_uuid_staging_timestamp)
2. Process all documents with new configuration
3. Validate (10% variance allowed)
4. Atomic swap to production
5. Schedule cleanup of old collection
```

#### Collection Naming
- Format: `col_{uuid_with_underscores}`
- Staging: `col_{uuid}_staging_{timestamp}`
- Consistent mapping with PostgreSQL collections

### Monitoring Infrastructure

```sql
-- Health monitoring views
CREATE VIEW partition_health AS
SELECT partition_name, collection_count, chunk_count, avg_chunks_per_collection;

CREATE VIEW partition_distribution AS  
SELECT collection_id, partition_key, chunk_count, percentage_of_partition;

-- Monitoring functions
SELECT * FROM analyze_partition_skew();  -- Detect imbalances
SELECT * FROM get_partition_statistics('collection_uuid');
```

---

## Backend API Architecture

### FastAPI Application Structure

#### Middleware Stack (Order Matters)
1. **CorrelationIdMiddleware**: Request tracing
2. **RateLimitMiddleware**: DDoS protection  
3. **CORSMiddleware**: Cross-origin requests

#### API Version Structure
```
/api/v2/
├── auth/          # Authentication endpoints
├── collections/   # Collection CRUD
├── operations/    # Operation management
├── documents/     # Document operations
├── chunking/      # Chunking strategies
├── search/        # Semantic search
└── ws/           # WebSocket connections
```

### Service Layer Architecture

#### Service Pattern
```python
class CollectionService:
    def __init__(self, db: AsyncSession, repos: RepositoryFactory):
        self.db = db
        self.repos = repos
    
    async def create_collection(self, request: CreateCollectionRequest) -> Collection:
        # 1. Validate request
        # 2. Begin transaction
        # 3. Create collection via repository
        # 4. Create initial operation
        # 5. Dispatch async task
        # 6. Commit transaction
        # 7. Return result
```

#### Key Services
- **CollectionService**: Collection lifecycle management
- **OperationService**: Async operation tracking
- **SearchService**: Query processing and reranking
- **ChunkingService**: Document chunking strategies
- **AuthService**: User authentication/authorization

### Repository Pattern

```python
class BaseRepository(ABC):
    def __init__(self, db: AsyncSession):
        self.db = db
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Model | None:
        pass
    
    @abstractmethod
    async def create(self, data: dict) -> Model:
        pass
```

### Celery Worker Architecture

#### Task Configuration
```python
CELERY_CONFIG = {
    'task_soft_time_limit': 3600,  # 1 hour
    'task_time_limit': 7200,        # 2 hours
    'task_acks_late': True,         # Late acknowledgment
    'worker_max_tasks_per_child': 100,
    'worker_prefetch_multiplier': 1
}
```

#### Primary Tasks
- `process_collection_operation`: Central task router
- `cleanup_old_collections`: Scheduled cleanup
- `monitor_partition_health`: Health checks
- `refresh_collection_chunking_stats`: Statistics updates

### WebSocket Architecture

```python
# Scalable WebSocket with Redis Pub/Sub
class WebSocketManager:
    def __init__(self):
        self.connections: dict[str, WebSocket] = {}
        self.redis_client = Redis(db=2)  # Dedicated DB
    
    async def broadcast(self, channel: str, message: dict):
        await self.redis_client.publish(channel, json.dumps(message))
```

---

## Frontend Architecture

### Component Architecture

```
src/
├── components/
│   ├── layout/         # Layout components
│   ├── collections/    # Collection management
│   ├── search/         # Search interface
│   ├── chunking/       # Chunking configuration
│   └── common/         # Reusable components
├── stores/            # Zustand state management
├── services/          # API clients
├── hooks/            # Custom React hooks
└── utils/            # Utility functions
```

### State Management (Zustand)

#### Store Architecture
```typescript
// Five specialized stores
interface StoreArchitecture {
  authStore: AuthState;      // Persisted, JWT management
  collectionStore: UIState;   // Minimal UI state
  searchStore: SearchState;   // Search parameters
  chunkingStore: ChunkingState; // Strategy management
  uiStore: GlobalUIState;     // Toasts, modals, tabs
}
```

#### Optimistic Updates Pattern
```typescript
updateCollection: async (id, updates) => {
  // 1. Optimistic update
  get().optimisticUpdateCollection(id, updates);
  try {
    // 2. API call
    await collectionsV2Api.update(id, updates);
    // 3. Re-fetch canonical state
    await get().fetchCollectionById(id);
  } catch (error) {
    // 4. Rollback on failure
    await get().fetchCollectionById(id);
    toast.error('Update failed');
  }
}
```

### API Integration

#### Axios Client Configuration
```typescript
const apiClient = axios.create({
  baseURL: '/api/v2',
  timeout: 30000,
  withCredentials: true
});

// Interceptors for auth and error handling
apiClient.interceptors.request.use(attachToken);
apiClient.interceptors.response.use(handleSuccess, handleError);
```

#### React Query Integration
```typescript
// Query key factory pattern
const queryKeys = {
  collections: {
    all: ['collections'],
    byId: (id: string) => ['collections', id],
    search: (params) => ['collections', 'search', params]
  }
};

// Usage with caching
const { data } = useQuery({
  queryKey: queryKeys.collections.byId(id),
  queryFn: () => collectionsApi.getById(id),
  staleTime: 5 * 60 * 1000, // 5 minutes
});
```

### WebSocket Integration

```typescript
class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  
  connect() {
    this.ws = new WebSocket(`ws://localhost:8000/api/v2/ws`);
    this.setupEventHandlers();
    this.startHeartbeat();
  }
  
  private setupAutoReconnect() {
    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, 30s max
  }
}
```

### UI/UX Patterns

#### TailwindCSS Utilities
```typescript
// Consistent form styling
export const inputStyles = cn(
  "w-full px-3 py-2",
  "border border-gray-300 rounded-md",
  "focus:outline-none focus:ring-2 focus:ring-blue-500",
  "disabled:opacity-50 disabled:cursor-not-allowed"
);
```

#### Toast Notifications
```typescript
interface ToastOptions {
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration?: number;  // Default: 5000ms
  dismissible?: boolean;
}
```

---

## Document Processing Pipeline

### Document Parsing Architecture

#### Supported Formats (20+)
- **Office**: DOCX, XLSX, PPTX, ODT, ODS, ODP
- **Documents**: PDF, RTF, HTML, XML, Markdown
- **Images**: PNG, JPG, GIF (with OCR via Tesseract)
- **Code**: Python, JavaScript, Java, C++, etc.
- **Data**: CSV, JSON, YAML, TOML
- **Archives**: ZIP, TAR, GZ (recursive extraction)

#### Parsing Pipeline
```python
class UnstructuredExtractor:
    def extract(self, file_path: str) -> ExtractedDocument:
        # 1. Detect file type
        # 2. Select appropriate strategy
        # 3. Extract text and metadata
        # 4. Preserve structure (tables, lists)
        # 5. Return normalized document
```

### Chunking Strategies

#### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Character** | Fixed-size chunks | Simple documents |
| **Recursive** | Respects paragraph boundaries | General text |
| **Semantic** | Embedding-based boundaries | High-quality search |
| **Markdown** | Preserves markdown structure | Technical docs |
| **Hierarchical** | Tree-based chunking | Structured content |
| **Hybrid** | Combines multiple strategies | Complex documents |

#### Semantic Chunking Algorithm
```python
def semantic_chunk(text: str, max_chunk_size: int) -> List[str]:
    # 1. Split into sentences
    # 2. Generate embeddings for sentences
    # 3. Calculate semantic similarity
    # 4. Find optimal breakpoints (similarity < threshold)
    # 5. Merge sentences into chunks
    # 6. Ensure size constraints
```

### Embedding Generation

#### Model Configuration
```python
EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B": {
        "dimensions": 896,
        "max_tokens": 8192,
        "quantization": ["float32", "float16", "int8"]
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dimensions": 1536,
        "max_tokens": 8192
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dimensions": 2048,
        "max_tokens": 8192
    }
}
```

#### Memory-Bounded Processing
```python
class StreamingProcessor:
    def __init__(self, memory_limit_mb: int = 100):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.buffer_pool = BufferPool(size=10)
    
    async def process_stream(self, stream: AsyncIterator):
        async for chunk in stream:
            if self.get_memory_usage() > self.memory_limit:
                await self.flush_buffers()
            await self.process_chunk(chunk)
```

### Vector Storage & Indexing

#### Qdrant Configuration
```python
QDRANT_CONFIG = {
    "vector_config": {
        "size": 896,  # Depends on model
        "distance": "Cosine",
        "hnsw_config": {
            "m": 16,
            "ef_construct": 100,
            "full_scan_threshold": 10000
        }
    },
    "optimizers_config": {
        "indexing_threshold": 20000,
        "memmap_threshold": 50000
    }
}
```

#### Batch Upload Strategy
```python
async def upload_vectors(vectors: List[Vector], batch_size: int = 4000):
    # Optimal batch size for Qdrant
    for batch in chunked(vectors, batch_size):
        await qdrant_client.upsert(
            collection_name=collection_id,
            points=batch,
            wait=True  # Ensure consistency
        )
```

### Search & Retrieval

#### Multi-Stage Search Pipeline
```python
async def search(query: str, collection_id: str, k: int = 10):
    # Stage 1: Query enhancement
    enhanced_query = await enhance_query(query)
    
    # Stage 2: Vector search
    vector_results = await vector_search(enhanced_query, k * 3)
    
    # Stage 3: Keyword matching (hybrid)
    keyword_results = await keyword_search(query, k * 2)
    
    # Stage 4: Fusion & deduplication
    combined = fusion_algorithm(vector_results, keyword_results)
    
    # Stage 5: Reranking
    reranked = await rerank_results(query, combined, k)
    
    return reranked
```

#### Reranking with Cross-Encoder
```python
class Reranker:
    def __init__(self, model: str = "Qwen/Qwen3-Reranker-4B"):
        self.model = load_model(model)
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        # Score query-document pairs
        scores = self.model.score_pairs(
            [(query, r.content) for r in results]
        )
        # Sort by relevance
        return sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
```

---

## Security Architecture

### Authentication System

#### JWT Token Architecture
```python
TOKEN_CONFIG = {
    "ACCESS_TOKEN_EXPIRE": timedelta(hours=24),
    "REFRESH_TOKEN_EXPIRE": timedelta(days=30),
    "ALGORITHM": "HS256",
    "TOKEN_URL": "/api/v2/auth/token"
}

# Dual-token system
def create_tokens(user_id: str) -> TokenPair:
    access_token = create_jwt(
        {"sub": user_id, "type": "access"},
        expires_delta=TOKEN_CONFIG["ACCESS_TOKEN_EXPIRE"]
    )
    refresh_token = create_jwt(
        {"sub": user_id, "type": "refresh"},
        expires_delta=TOKEN_CONFIG["REFRESH_TOKEN_EXPIRE"]
    )
    return TokenPair(access=access_token, refresh=refresh_token)
```

#### Password Security
```python
# Bcrypt with salt
def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt(rounds=12)
    ).decode('utf-8')
```

### Authorization Patterns

#### Role-Based Access Control
```python
class CollectionPermission(Enum):
    OWNER = "owner"    # Full control
    ADMIN = "admin"    # Manage collection
    WRITE = "write"    # Add/remove documents
    READ = "read"      # View and search

# Permission checking at repository level
async def get_accessible_collections(user_id: str, permission: CollectionPermission):
    return await db.execute(
        select(Collection)
        .join(CollectionPermissions)
        .where(
            CollectionPermissions.user_id == user_id,
            CollectionPermissions.permission >= permission
        )
    )
```

### Input Validation & Sanitization

#### Comprehensive Validation
```python
class CreateCollectionRequest(BaseModel):
    name: constr(min_length=1, max_length=255, regex="^[a-zA-Z0-9_-]+$")
    description: constr(max_length=1000) | None
    embedding_model: Literal[*ALLOWED_MODELS]
    chunk_size: conint(ge=100, le=10000)
    
    @validator('name')
    def sanitize_name(cls, v):
        # Remove any potential SQL/NoSQL injection attempts
        return html.escape(v.strip())
```

#### Path Traversal Prevention
```python
def validate_file_path(path: str) -> Path:
    # Normalize and resolve path
    normalized = Path(path).resolve()
    
    # Check for null bytes
    if '\x00' in str(normalized):
        raise SecurityError("Null byte in path")
    
    # Ensure within allowed directory
    if not normalized.is_relative_to(ALLOWED_BASE_PATH):
        raise SecurityError("Path traversal attempt")
    
    return normalized
```

### API Security

#### Rate Limiting
```python
RATE_LIMITS = {
    "default": "100/minute",
    "search": "30/minute",
    "auth": "5/minute",
    "upload": "10/hour"
}

class RateLimiter:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.limits = RATE_LIMITS
    
    async def check_rate_limit(self, key: str, endpoint: str) -> bool:
        # Sliding window algorithm
        # Returns True if within limits
```

#### CORS Configuration
```python
CORS_CONFIG = {
    "allow_origins": ["http://localhost:3000"],  # Frontend dev
    "allow_credentials": True,
    "allow_methods": ["GET", "POST", "PUT", "DELETE"],
    "allow_headers": ["*"],
    "expose_headers": ["X-Total-Count", "X-Correlation-Id"]
}
```

### Data Security

#### Sensitive Data Handling
```python
class PIISanitizer:
    PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    }
    
    def sanitize(self, text: str) -> str:
        for pattern_name, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f"[REDACTED_{pattern_name.upper()}]", text)
        return text
```

### Infrastructure Security

#### Docker Security Hardening
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
cap_add:
  - CHOWN
  - SETUID
  - SETGID
read_only: true
tmpfs:
  - /tmp
  - /var/run
```

#### Network Isolation
```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access
```

---

## Testing Architecture

### Backend Testing

#### Test Structure
```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── api/           # API endpoint tests
├── domain/        # Domain logic tests
└── fixtures/      # Test data and mocks
```

#### Key Testing Patterns
```python
# Service layer testing
@pytest.mark.asyncio
async def test_create_collection(db_session, mock_qdrant):
    service = CollectionService(db_session)
    
    result = await service.create_collection(
        CreateCollectionRequest(name="test", ...)
    )
    
    assert result.status == CollectionStatus.PENDING
    mock_qdrant.create_collection.assert_called_once()
```

#### Test Fixtures
```python
@pytest.fixture
async def db_session():
    """Provide isolated database session for tests"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        async with AsyncSession(conn) as session:
            yield session
        await conn.rollback()
```

### Frontend Testing

#### Test Setup
```typescript
// Custom render with providers
export function renderWithProviders(
  ui: React.ReactElement,
  options?: RenderOptions
) {
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {ui}
      </BrowserRouter>
    </QueryClientProvider>,
    options
  );
}
```

#### Component Testing
```typescript
describe('CollectionCard', () => {
  it('should display collection information', () => {
    const { getByText } = renderWithProviders(
      <CollectionCard collection={mockCollection} />
    );
    
    expect(getByText(mockCollection.name)).toBeInTheDocument();
    expect(getByText(mockCollection.document_count)).toBeInTheDocument();
  });
});
```

#### Store Testing
```typescript
describe('collectionStore', () => {
  it('should handle optimistic updates', async () => {
    const store = useCollectionStore.getState();
    
    // Trigger optimistic update
    await store.updateCollection('id', { name: 'new-name' });
    
    // Verify immediate UI update
    expect(store.collections[0].name).toBe('new-name');
  });
});
```

### Test Coverage

| Component | Coverage | Target |
|-----------|----------|--------|
| Backend API | 85% | 90% |
| Services | 80% | 85% |
| Repositories | 75% | 80% |
| Frontend Components | 70% | 80% |
| Stores | 75% | 85% |
| Utils | 90% | 95% |

### CI/CD Testing

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
      redis:
        image: redis:7
      qdrant:
        image: qdrant/qdrant
    steps:
      - uses: actions/checkout@v3
      - run: make test-backend
      - uses: codecov/codecov-action@v3

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: npm test -- --coverage
      - run: npm run test:e2e
```

---

## DevOps & Infrastructure

### Docker Architecture

#### Multi-Stage Build Pattern
```dockerfile
# Stage 1: Frontend build
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# Stage 2: Python dependencies
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS python-builder
RUN apt-get update && apt-get install -y python3.11 python3-pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Production runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=python-builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=frontend-builder /app/dist /app/static
USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

### Docker Compose Configuration

#### Service Definitions
```yaml
version: '3.9'

services:
  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: semantik
      POSTGRES_USER: semantik
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "semantik"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  webui:
    build:
      context: .
      dockerfile: packages/webui/Dockerfile
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql+asyncpg://semantik:password@postgres/semantik
      - REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'

  vecpipe:
    build:
      context: .
      dockerfile: packages/vecpipe/Dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker:
    build:
      context: .
      dockerfile: packages/worker/Dockerfile
    command: celery -A worker.celery_app worker --loglevel=info
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 2  # Horizontal scaling
```

### Development Workflow

#### Makefile Automation
```makefile
# Development commands
.PHONY: dev
dev: ## Start development environment
	@docker compose --profile backend up -d
	@npm run dev

.PHONY: test
test: ## Run all tests
	@make test-backend
	@make test-frontend

.PHONY: check
check: ## Run format, lint, and tests
	@poetry run black packages/ tests/
	@poetry run ruff packages/ tests/
	@poetry run mypy packages/
	@make test

# Database management
.PHONY: migrate
migrate: ## Run database migrations
	@poetry run alembic upgrade head

.PHONY: migration-create
migration-create: ## Create new migration
	@poetry run alembic revision --autogenerate -m "$(MSG)"

# Docker management
.PHONY: docker-up
docker-up: ## Start all services
	@docker compose up -d

.PHONY: docker-logs
docker-logs: ## Show service logs
	@docker compose logs -f $(SERVICE)

.PHONY: docker-clean
docker-clean: ## Clean up containers and volumes
	@docker compose down -v
```

### Production Deployment

#### Environment Configuration
```bash
# Production .env
NODE_ENV=production
DATABASE_URL=postgresql+asyncpg://user:pass@db/semantik
REDIS_URL=redis://redis:6379
JWT_SECRET_KEY=<generated-secret>
QDRANT_URL=http://qdrant:6333
CUDA_VISIBLE_DEVICES=0
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
QUANTIZATION=int8  # Memory optimization
```

#### Nginx Configuration
```nginx
upstream webui {
    server webui:8000;
}

server {
    listen 443 ssl http2;
    server_name semantik.example.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN";
    add_header X-Content-Type-Options "nosniff";
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://webui;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://webui;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Database Management

#### Backup Strategy
```bash
#!/bin/bash
# Automated backup script
BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec postgres pg_dump -U semantik semantik | gzip > "${BACKUP_DIR}/backup_${TIMESTAMP}.sql.gz"

# Rotate old backups (keep 30 days)
find "${BACKUP_DIR}" -name "backup_*.sql.gz" -mtime +30 -delete
```

#### Partition Maintenance
```sql
-- Monitor partition health
SELECT * FROM analyze_partition_skew();

-- Rebalance if needed
SELECT rebalance_partitions() WHERE skew_factor > 2.0;

-- Vacuum partitions
DO $$
DECLARE
    partition_name text;
BEGIN
    FOR partition_name IN 
        SELECT tablename FROM pg_tables 
        WHERE tablename LIKE 'chunks_part_%'
    LOOP
        EXECUTE format('VACUUM ANALYZE %I', partition_name);
    END LOOP;
END $$;
```

### Monitoring & Observability

#### Health Checks
```python
@router.get("/health")
async def health_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "qdrant": await check_qdrant(),
        "disk_space": check_disk_space(),
        "memory": check_memory()
    }
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    return {"status": status, "checks": checks}
```

#### Prometheus Metrics
```python
# Custom metrics
collection_operations = Counter(
    'semantik_collection_operations_total',
    'Total collection operations',
    ['operation_type', 'status']
)

search_duration = Histogram(
    'semantik_search_duration_seconds',
    'Search request duration',
    ['collection_id']
)

active_connections = Gauge(
    'semantik_websocket_connections',
    'Active WebSocket connections'
)
```

---

## Shared Libraries

### Package Structure

```
packages/shared/
├── chunking/         # Domain-driven chunking implementation
├── config/          # Configuration management
├── contracts/       # API contracts and DTOs
├── database/        # Models and repositories
├── embedding/       # Embedding service
├── managers/        # External service managers
├── metrics/         # Prometheus metrics
└── text_processing/ # Document extraction
```

### Database Models

#### Base Model Pattern
```python
class BaseModel(DeclarativeBase):
    __abstract__ = True
    
    id: Mapped[UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
```

#### Key Models
```python
class Collection(BaseModel):
    __tablename__ = "collections"
    
    name: Mapped[str] = mapped_column(String(255), unique=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"))
    status: Mapped[CollectionStatus] = mapped_column(
        Enum(CollectionStatus),
        default=CollectionStatus.PENDING
    )
    config: Mapped[dict] = mapped_column(JSONB, default={})
    vector_store_name: Mapped[str] = mapped_column(String(255), unique=True)
    
    # Relationships
    owner: Mapped["User"] = relationship(back_populates="collections")
    documents: Mapped[List["Document"]] = relationship(
        back_populates="collection",
        cascade="all, delete-orphan"
    )
```

### Configuration Management

```python
class BaseConfig(BaseSettings):
    """Base configuration with common settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Database
    database_url: PostgresDsn
    database_pool_size: int = 20
    database_max_overflow: int = 40
    
    # Redis
    redis_url: RedisDsn
    redis_pool_size: int = 10
    
    # Security
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"

class VecpipeConfig(BaseConfig):
    """Vecpipe-specific configuration"""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    quantization: Literal["float32", "float16", "int8"] = "float16"
    batch_size: int = 100
    max_memory_mb: int = 1000
```

### Repository Pattern

```python
class AbstractRepository(ABC):
    def __init__(self, session: AsyncSession):
        self.session = session
    
    @abstractmethod
    async def get(self, id: UUID) -> Model | None:
        pass
    
    @abstractmethod
    async def create(self, **kwargs) -> Model:
        pass
    
    @abstractmethod
    async def update(self, id: UUID, **kwargs) -> Model:
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        pass

class CollectionRepository(AbstractRepository):
    async def get_with_permissions(
        self,
        collection_id: UUID,
        user_id: UUID
    ) -> Collection | None:
        stmt = (
            select(Collection)
            .join(CollectionPermissions)
            .where(
                Collection.id == collection_id,
                CollectionPermissions.user_id == user_id
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
```

### Domain-Driven Design (Chunking Module)

```python
# Domain entities
@dataclass
class ChunkingStrategy:
    name: str
    chunk_size: int
    overlap: int
    separator: str | None = None
    
    def validate(self):
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk size")

# Domain services
class ChunkingService:
    def __init__(self, strategy: ChunkingStrategy):
        self.strategy = strategy
        self.strategy.validate()
    
    def chunk_text(self, text: str) -> List[Chunk]:
        chunker = ChunkerFactory.create(self.strategy)
        return chunker.chunk(text)

# Value objects
@dataclass(frozen=True)
class Chunk:
    content: str
    index: int
    metadata: dict
    
    def __post_init__(self):
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
```

### Cross-Service Communication

```python
# Standardized API contracts
class OperationStatusUpdate(BaseModel):
    operation_id: UUID
    status: OperationStatus
    progress: float = Field(ge=0, le=100)
    message: str | None = None
    error: str | None = None
    completed_at: datetime | None = None

# Event schemas for async communication
class CollectionIndexedEvent(BaseModel):
    collection_id: UUID
    document_count: int
    chunk_count: int
    vector_count: int
    duration_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Internal API authentication
class InternalAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.session = aiohttp.ClientSession(
            headers={"X-Internal-API-Key": api_key}
        )
        self.base_url = base_url
    
    async def call_service(self, endpoint: str, data: dict) -> dict:
        async with self.session.post(
            f"{self.base_url}/{endpoint}",
            json=data
        ) as response:
            response.raise_for_status()
            return await response.json()
```

---

## Business Logic & Workflows

### Collection Management

#### Collection Lifecycle
```mermaid
stateDiagram-v2
    [*] --> PENDING: Create
    PENDING --> PROCESSING: Start Operation
    PROCESSING --> READY: Success
    PROCESSING --> ERROR: Failure
    READY --> PROCESSING: New Operation
    ERROR --> PROCESSING: Retry
    READY --> [*]: Delete
```

#### Collection Operations
| Operation | Description | Duration | Resumable |
|-----------|-------------|----------|-----------|
| INDEX | Initial document indexing | Hours | Yes |
| REINDEX | Complete reprocessing | Hours | Yes |
| APPEND | Add new documents | Minutes | Yes |
| REMOVE | Remove documents | Seconds | No |
| UPDATE_CONFIG | Change settings | Seconds | No |

### Document Processing

#### Indexing Pipeline
```python
async def index_documents(collection_id: UUID, source_config: dict):
    # 1. Discovery
    files = await discover_files(source_config)
    
    # 2. Deduplication
    new_files = await filter_existing(files, collection_id)
    
    # 3. Processing loop
    for batch in chunk_files(new_files, batch_size=100):
        # Extract text
        documents = await extract_text_batch(batch)
        
        # Chunk documents
        chunks = await chunk_documents(documents, collection.config)
        
        # Generate embeddings
        embeddings = await generate_embeddings(chunks)
        
        # Store in databases
        await store_documents(documents)
        await store_chunks(chunks)
        await store_vectors(embeddings)
        
        # Update progress
        await update_progress(collection_id, processed/total)
```

### Search Workflows

#### Search Algorithm
```python
async def semantic_search(
    query: str,
    collection_id: UUID,
    k: int = 10,
    filters: dict = None
) -> List[SearchResult]:
    # 1. Access check
    if not await user_has_access(collection_id):
        raise PermissionError()
    
    # 2. Query embedding
    query_vector = await embed_text(query)
    
    # 3. Vector search
    vector_results = await qdrant.search(
        collection_name=f"col_{collection_id}",
        query_vector=query_vector,
        limit=k * 3,  # Over-fetch for reranking
        query_filter=filters
    )
    
    # 4. Reranking (optional)
    if collection.config.get("use_reranker"):
        results = await rerank(query, vector_results, k)
    else:
        results = vector_results[:k]
    
    # 5. Enrich with metadata
    return await enrich_results(results)
```

### Background Tasks

#### Task Orchestration
```python
@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True
)
def process_collection_operation(
    self,
    operation_id: str,
    operation_type: str,
    **kwargs
):
    try:
        # Setup
        operation = get_operation(operation_id)
        collection = get_collection(operation.collection_id)
        
        # Route to handler
        handlers = {
            "INDEX": handle_index,
            "REINDEX": handle_reindex,
            "APPEND": handle_append,
            "REMOVE": handle_remove
        }
        
        handler = handlers[operation_type]
        result = handler(collection, **kwargs)
        
        # Update status
        operation.status = OperationStatus.COMPLETED
        operation.result = result
        
    except Exception as e:
        # Retry logic
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        # Final failure
        operation.status = OperationStatus.FAILED
        operation.error = str(e)
        
    finally:
        # Always update operation
        save_operation(operation)
        
        # Send notification
        notify_operation_complete(operation)
```

### Error Handling

#### Structured Exception Hierarchy
```python
class SemantikException(Exception):
    """Base exception for all Semantik errors"""
    status_code = 500
    error_code = "INTERNAL_ERROR"

class ValidationError(SemantikException):
    status_code = 400
    error_code = "VALIDATION_ERROR"

class AuthenticationError(SemantikException):
    status_code = 401
    error_code = "AUTHENTICATION_ERROR"

class PermissionError(SemantikException):
    status_code = 403
    error_code = "PERMISSION_ERROR"

class ResourceNotFoundError(SemantikException):
    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"

class ConflictError(SemantikException):
    status_code = 409
    error_code = "CONFLICT_ERROR"

class RateLimitError(SemantikException):
    status_code = 429
    error_code = "RATE_LIMIT_ERROR"
```

#### Global Error Handler
```python
@app.exception_handler(SemantikException)
async def handle_semantik_exception(request: Request, exc: SemantikException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": str(exc),
            "correlation_id": request.state.correlation_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
```

---

## Refactoring Status

### Current Progress: 85% Complete

#### ✅ Completed (100%)
- Database schema migration
- PostgreSQL partitioning
- Backend models and repositories
- Service layer architecture
- API v2 endpoints
- WebSocket infrastructure
- Celery task refactoring

#### 🚧 In Progress (60%)
- Frontend API integration
- WebSocket client connection
- Real-time operation tracking
- UI terminology updates

#### ❌ Not Started (0%)
- UI component refactoring
- Progress indicators
- Migration documentation
- Performance benchmarking

### Critical Issues to Address

#### 1. Mock API Calls (BLOCKER)
**Location**: `apps/webui-react/src/stores/chunkingStore.ts`
```typescript
// CURRENT (Mock)
const mockApi = {
  async compareStrategies(request) {
    return new Promise(resolve => {
      setTimeout(() => resolve(mockData), 1000);
    });
  }
};

// NEEDED (Real)
const api = {
  async compareStrategies(request) {
    return axios.post('/api/v2/chunking/compare', request);
  }
};
```

#### 2. UI Terminology
**Location**: `apps/webui-react/src/components/collections/CollectionDetailsModal.tsx`
```typescript
// Line 267 - CURRENT
<Tab value="jobs">Jobs</Tab>

// NEEDED
<Tab value="operations">Operations</Tab>
```

#### 3. Permission Checks
**Location**: Multiple repository files
```python
# TODO: Check CollectionPermission table for access
# This needs immediate implementation for security
```

### Migration Path

#### Week 1: Critical Fixes
- Replace mock API calls
- Fix UI terminology
- Implement permission checks

#### Week 2: Integration
- Connect WebSocket client
- Implement real-time updates
- Add progress indicators

#### Week 3: Testing
- Integration tests for new features
- Load testing
- Security audit

#### Week 4: Documentation & Deployment
- Update user documentation
- Migration guide for existing deployments
- Production deployment preparation

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| API v2 Adoption | 60% | 100% |
| Real-time Updates | 0% | 100% |
| Permission Checks | 20% | 100% |
| UI Consistency | 70% | 100% |
| Test Coverage | 75% | 90% |

---

## Quick Reference

### Common Commands

```bash
# Development
make dev                    # Start development environment
make test                   # Run all tests
make check                  # Format, lint, and test

# Docker
make docker-up              # Start all services
make docker-dev-up          # Backend services only
make docker-logs SERVICE=x  # View service logs
make docker-clean           # Clean everything

# Database
make migrate                # Run migrations
make migration-create MSG=x # Create migration

# Production
make deploy                 # Deploy to production
make backup                 # Backup database
make monitor                # View monitoring dashboard
```

### API Endpoints

```python
# Authentication
POST   /api/v2/auth/register
POST   /api/v2/auth/login
POST   /api/v2/auth/refresh
POST   /api/v2/auth/logout

# Collections
GET    /api/v2/collections
POST   /api/v2/collections
GET    /api/v2/collections/{id}
PUT    /api/v2/collections/{id}
DELETE /api/v2/collections/{id}

# Operations
GET    /api/v2/operations
GET    /api/v2/operations/{id}
POST   /api/v2/collections/{id}/index
POST   /api/v2/collections/{id}/reindex

# Search
POST   /api/v2/search
POST   /api/v2/search/multi

# WebSocket
WS     /api/v2/ws
```

### Environment Variables

```bash
# Required
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
REDIS_URL=redis://host:6379
JWT_SECRET_KEY=your-secret-key
QDRANT_URL=http://qdrant:6333

# Optional
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
QUANTIZATION=float16
MAX_WORKERS=4
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
RATE_LIMIT=100/minute
```

### Database Queries

```sql
-- Check partition health
SELECT * FROM partition_health;

-- Get collection stats
SELECT * FROM collection_chunking_stats WHERE collection_id = ?;

-- Monitor operations
SELECT * FROM operations 
WHERE status = 'PROCESSING' 
ORDER BY created_at DESC;

-- User activity
SELECT u.username, COUNT(c.id) as collection_count
FROM users u
LEFT JOIN collections c ON u.id = c.owner_id
GROUP BY u.id;
```

### Monitoring Queries

```python
# Prometheus queries
rate(semantik_collection_operations_total[5m])
histogram_quantile(0.95, semantik_search_duration_seconds)
semantik_websocket_connections
rate(semantik_errors_total[1m])

# Redis monitoring
redis-cli INFO stats
redis-cli CLIENT LIST
redis-cli MONITOR

# PostgreSQL monitoring
SELECT * FROM pg_stat_activity;
SELECT * FROM pg_stat_user_tables;
```

---

## Appendix: File Structure

```
semantik/
├── apps/
│   └── webui-react/          # React frontend
├── packages/
│   ├── shared/               # Shared libraries
│   ├── webui/                # FastAPI backend
│   ├── vecpipe/              # Document processing
│   └── worker/               # Celery workers
├── scripts/                  # Utility scripts
├── tests/                    # Test suites
├── docker-compose.yml        # Service orchestration
├── Makefile                  # Automation commands
├── pyproject.toml           # Python dependencies
└── README.md                # Project documentation
```

---

*This master specification represents the complete technical documentation of the Semantik system as of the current codebase state. It serves as the authoritative reference for understanding, maintaining, and extending the application.*