# WebUI Services Architecture

> **Location:** `packages/webui/services/`

## Overview

The service layer implements business logic and orchestration between API endpoints and data repositories. Services follow the single responsibility principle and handle cross-cutting concerns.

## Service Structure

```
packages/webui/services/
├── collection_service.py    # Collection lifecycle management
├── search_service.py        # Search orchestration
├── operation_service.py     # Operation tracking
├── document_service.py      # Document management
├── source_service.py        # Data source handling
├── sync_service.py          # Continuous sync management
├── chunking_service.py      # Chunking preview/comparison
└── vecpipe_client.py        # VecPipe API client
```

## Core Services

### CollectionService

Manages collection lifecycle including creation, deletion, and source management.

```python
class CollectionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.collection_repo = CollectionRepository(db)
        self.operation_repo = OperationRepository(db)
        self.vecpipe = VecPipeClient()

    async def create(
        self,
        name: str,
        owner_id: int,
        embedding_model: str,
        chunking_config: ChunkingConfig | None = None,
        source_config: SourceConfig | None = None,
        sync_mode: str = "one_time"
    ) -> Collection:
        """Create a new collection with optional initial source."""
        # Generate unique Qdrant collection name
        vector_store_name = f"col_{uuid4().hex}"

        # Get embedding dimension from model
        dimension = await self.vecpipe.get_model_dimension(embedding_model)

        collection = Collection(
            id=str(uuid4()),
            name=name,
            owner_id=owner_id,
            vector_store_name=vector_store_name,
            embedding_model=embedding_model,
            embedding_dimension=dimension,
            status=CollectionStatus.PENDING,
            sync_mode=sync_mode
        )

        self.db.add(collection)
        await self.db.commit()

        # Create Qdrant collection
        await self.vecpipe.create_collection(
            name=vector_store_name,
            dimension=dimension
        )

        return collection

    async def add_source(
        self,
        collection_id: str,
        user_id: int,
        source_config: SourceConfig,
        secrets: dict | None = None
    ) -> Operation:
        """Add a data source to existing collection."""
        collection = await self.collection_repo.get_by_id(
            collection_id, user_id
        )
        if not collection:
            raise ResourceNotFoundError("Collection not found")

        # Encrypt and store secrets
        if secrets:
            await self._store_connector_secrets(collection_id, secrets)

        # Create and queue operation
        operation = await self._create_operation(
            collection_id=collection_id,
            user_id=user_id,
            type=OperationType.APPEND,
            config={"source_config": source_config.dict()}
        )

        # Queue Celery task
        celery_app.send_task(
            "webui.tasks.indexing.process_source",
            args=[operation.uuid]
        )

        return operation

    async def delete(self, collection_id: str, user_id: int) -> None:
        """Delete collection and all associated data."""
        collection = await self.collection_repo.get_by_id(
            collection_id, user_id
        )
        if not collection:
            raise ResourceNotFoundError("Collection not found")

        # Mark collection for deletion
        collection.status = CollectionStatus.PENDING

        # Queue deletion task
        celery_app.send_task(
            "webui.tasks.deletion.delete_collection",
            args=[collection_id]
        )
```

### SearchService

Orchestrates search across collections with permission checking.

```python
class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.collection_repo = CollectionRepository(db)
        self.vecpipe = VecPipeClient()

    async def search(
        self,
        query: str,
        collection_ids: list[str],
        user_id: int,
        top_k: int = 10,
        search_type: str = "semantic",
        use_reranker: bool = False,
        **kwargs
    ) -> SearchResponse:
        """Execute search across accessible collections."""
        # Verify user access to all collections
        accessible = await self.collection_repo.get_accessible(
            collection_ids=collection_ids,
            user_id=user_id
        )

        if len(accessible) != len(collection_ids):
            missing = set(collection_ids) - {c.id for c in accessible}
            raise PermissionError(f"Access denied to: {missing}")

        # Map to Qdrant collection names
        qdrant_names = [c.vector_store_name for c in accessible]

        # Execute search via VecPipe
        vecpipe_response = await self.vecpipe.search(
            query=query,
            collections=qdrant_names,
            top_k=top_k,
            search_type=search_type,
            use_reranker=use_reranker,
            **kwargs
        )

        # Map results back to collection metadata
        return self._map_results(vecpipe_response, accessible)

    def _map_results(
        self,
        response: VecPipeSearchResponse,
        collections: list[Collection]
    ) -> SearchResponse:
        """Map VecPipe results to include collection metadata."""
        name_to_collection = {c.vector_store_name: c for c in collections}

        results = []
        for result in response.results:
            collection = name_to_collection.get(result.collection)
            results.append(SearchResult(
                **result.dict(),
                collection_id=collection.id if collection else None,
                collection_name=collection.name if collection else None
            ))

        return SearchResponse(
            results=results,
            query=response.query,
            reranking_used=response.reranking_used
        )
```

### OperationService

Tracks and manages long-running operations.

```python
class OperationService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.operation_repo = OperationRepository(db)
        self.redis = get_redis_client()

    async def create(
        self,
        collection_id: str,
        user_id: int,
        type: OperationType,
        config: dict | None = None
    ) -> Operation:
        """Create a new operation."""
        operation = Operation(
            uuid=str(uuid4()),
            collection_id=collection_id,
            user_id=user_id,
            type=type,
            status=OperationStatus.PENDING,
            config=config
        )

        self.db.add(operation)
        await self.db.commit()

        return operation

    async def update_progress(
        self,
        operation_uuid: str,
        progress: int,
        message: str | None = None
    ) -> None:
        """Update operation progress and broadcast via WebSocket."""
        operation = await self.operation_repo.get_by_uuid(operation_uuid)
        if not operation:
            return

        operation.progress = progress
        await self.db.commit()

        # Broadcast progress update
        await self._broadcast_progress(operation, message)

    async def _broadcast_progress(
        self,
        operation: Operation,
        message: str | None = None
    ) -> None:
        """Publish progress to Redis for WebSocket broadcast."""
        await self.redis.xadd(
            f"operations:{operation.uuid}",
            {
                "status": operation.status.value,
                "progress": str(operation.progress),
                "message": message or ""
            }
        )

    async def cancel(
        self,
        operation_uuid: str,
        user_id: int
    ) -> Operation:
        """Cancel a pending or processing operation."""
        operation = await self.operation_repo.get_by_uuid(operation_uuid)

        if not operation:
            raise ResourceNotFoundError("Operation not found")
        if operation.user_id != user_id:
            raise PermissionError("Not authorized")
        if operation.status not in [OperationStatus.PENDING, OperationStatus.PROCESSING]:
            raise ValidationError("Cannot cancel completed operation")

        operation.status = OperationStatus.CANCELLED
        await self.db.commit()

        # Signal Celery task to stop
        celery_app.control.revoke(operation_uuid, terminate=True)

        return operation
```

### SourceService

Handles data source operations (connectors).

```python
class SourceService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.fernet = Fernet(os.environ["CONNECTOR_SECRETS_KEY"])

    async def process_source(
        self,
        collection_id: str,
        source_config: SourceConfig,
        operation_uuid: str
    ) -> int:
        """Process all documents from a data source."""
        connector = self._get_connector(source_config.type)

        documents_processed = 0
        async for document in connector.iterate_documents(source_config):
            # Extract text
            content = await self._extract_content(document)

            # Chunk content
            chunks = await self._chunk_content(content, collection_id)

            # Store document and chunks
            await self._store_document(collection_id, document, chunks)

            documents_processed += 1

            # Update progress
            await self._update_progress(operation_uuid, documents_processed)

        return documents_processed

    def _get_connector(self, connector_type: str) -> BaseConnector:
        """Get connector instance by type."""
        connectors = {
            "directory": DirectoryConnector,
            "git": GitConnector,
            "imap": ImapConnector,
        }
        return connectors[connector_type]()

    async def encrypt_secrets(self, secrets: dict) -> str:
        """Encrypt connector secrets for storage."""
        json_bytes = json.dumps(secrets).encode()
        return self.fernet.encrypt(json_bytes).decode()

    async def decrypt_secrets(self, encrypted: str) -> dict:
        """Decrypt stored connector secrets."""
        decrypted = self.fernet.decrypt(encrypted.encode())
        return json.loads(decrypted)
```

### VecPipeClient

HTTP client for VecPipe service communication.

```python
class VecPipeClient:
    def __init__(self):
        self.base_url = os.environ.get("VECPIPE_URL", "http://vecpipe:8000")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 10,
        search_type: str = "semantic",
        **kwargs
    ) -> VecPipeSearchResponse:
        """Execute search via VecPipe API."""
        response = await self.client.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "collections": collections,
                "top_k": top_k,
                "search_type": search_type,
                **kwargs
            }
        )
        response.raise_for_status()
        return VecPipeSearchResponse(**response.json())

    async def embed(
        self,
        texts: list[str],
        model: str,
        mode: str = "document"
    ) -> list[list[float]]:
        """Generate embeddings via VecPipe."""
        response = await self.client.post(
            f"{self.base_url}/embed",
            json={
                "texts": texts,
                "model": model,
                "mode": mode
            }
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    async def create_collection(
        self,
        name: str,
        dimension: int
    ) -> None:
        """Create Qdrant collection via VecPipe."""
        response = await self.client.post(
            f"{self.base_url}/collections",
            json={
                "name": name,
                "dimension": dimension
            }
        )
        response.raise_for_status()

    async def get_model_dimension(self, model_name: str) -> int:
        """Get embedding dimension for a model."""
        response = await self.client.get(
            f"{self.base_url}/models/{model_name}"
        )
        response.raise_for_status()
        return response.json()["dimension"]

    async def health(self) -> bool:
        """Check VecPipe health."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
```

## Service Patterns

### Dependency Injection
```python
# In API layer
@router.post("/collections")
async def create_collection(
    request: CreateCollectionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    service = CollectionService(db)  # Inject session
    return await service.create(...)
```

### Error Translation
```python
async def create_collection(...):
    try:
        return await self._do_create(...)
    except IntegrityError as e:
        if "unique constraint" in str(e):
            raise ValidationError("Collection name already exists")
        raise
    except QdrantException as e:
        raise ServiceUnavailableError("Vector database unavailable")
```

### Transaction Management
```python
async def complex_operation(self):
    async with self.db.begin():  # Start transaction
        await self.collection_repo.update(...)
        await self.document_repo.delete(...)
        # Both committed together or both rolled back
```

## Extension Points

### Adding a New Service
1. Create `services/my_service.py`
2. Initialize with `AsyncSession`
3. Use repositories for data access
4. Handle errors appropriately
5. Add to service layer tests

### Adding External Service Integration
1. Create client class (like `VecPipeClient`)
2. Use httpx for async HTTP
3. Handle timeouts and retries
4. Add health check method
5. Configure via environment variables
