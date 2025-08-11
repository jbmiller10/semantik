# BE-002: Move Business Logic from Routers to Services

## Ticket Information
- **Priority**: CRITICAL
- **Estimated Time**: 4 hours
- **Dependencies**: BE-001 (Redis client fixes should be complete)
- **Risk Level**: MEDIUM - Architectural refactoring
- **Affected Files**:
  - `packages/webui/api/v2/chunking.py`
  - `packages/webui/services/chunking_service.py`
  - `packages/webui/services/chunking_config_builder.py` (new)
  - `packages/webui/services/chunking_strategy_factory.py` (new)

## Context

The API routers currently contain significant business logic that belongs in the service layer. This violates separation of concerns and makes the code harder to test and maintain.

### Current Problems

```python
# packages/webui/api/v2/chunking.py - BAD
@router.post("/preview")
async def preview_chunks(request: PreviewRequest):
    # Business logic in router - WRONG!
    if request.strategy == "semantic":
        config = SemanticConfig(
            chunk_size=request.chunk_size or 512,
            similarity_threshold=request.similarity_threshold or 0.7
        )
    elif request.strategy == "markdown":
        config = MarkdownConfig(
            chunk_size=request.chunk_size or 1000,
            preserve_headers=True
        )
    # ... more strategy logic ...
    
    # Validation logic in router - WRONG!
    if request.document_id and request.content:
        raise HTTPException(400, "Cannot provide both document_id and content")
```

## Requirements

1. Move ALL business logic from routers to service layer
2. Routers should only handle HTTP concerns (request/response)
3. Create dedicated builder and factory classes for configuration
4. Implement proper validation in service layer
5. Ensure routers are thin controllers
6. Maintain backward compatibility of API responses

## Technical Details

### 1. Create Configuration Builder

```python
# packages/webui/services/chunking_config_builder.py

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ChunkingStrategy(Enum):
    CHARACTER = "character"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"

@dataclass
class ChunkingConfigResult:
    strategy: ChunkingStrategy
    config: Dict[str, Any]
    validation_errors: Optional[List[str]] = None

class ChunkingConfigBuilder:
    """Builds and validates chunking configurations"""
    
    # Default configurations per strategy
    DEFAULT_CONFIGS = {
        ChunkingStrategy.CHARACTER: {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separator": None,
            "keep_separator": False
        },
        ChunkingStrategy.RECURSIVE: {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "separators": ["\n\n", "\n", " ", ""],
            "keep_separator": True
        },
        ChunkingStrategy.MARKDOWN: {
            "chunk_size": 1000,
            "chunk_overlap": 100,
            "preserve_headers": True,
            "preserve_code_blocks": True,
            "min_header_level": 1,
            "max_header_level": 6
        },
        ChunkingStrategy.SEMANTIC: {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "similarity_threshold": 0.7,
            "min_chunk_size": 100,
            "max_chunk_size": 1000,
            "embedding_model": "default"
        },
        ChunkingStrategy.HIERARCHICAL: {
            "chunk_sizes": [2000, 1000, 500],
            "chunk_overlaps": [200, 100, 50],
            "level_names": ["section", "subsection", "paragraph"],
            "preserve_hierarchy": True
        },
        ChunkingStrategy.HYBRID: {
            "primary_strategy": "semantic",
            "fallback_strategy": "recursive",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "switch_threshold": 0.5
        }
    }
    
    def build_config(
        self,
        strategy: str,
        user_config: Optional[Dict[str, Any]] = None
    ) -> ChunkingConfigResult:
        """
        Build configuration for a chunking strategy
        
        Args:
            strategy: Strategy name
            user_config: User-provided configuration overrides
            
        Returns:
            ChunkingConfigResult with validated configuration
        """
        # Validate strategy
        try:
            strategy_enum = ChunkingStrategy(strategy.lower())
        except ValueError:
            return ChunkingConfigResult(
                strategy=ChunkingStrategy.CHARACTER,
                config={},
                validation_errors=[f"Unknown strategy: {strategy}"]
            )
        
        # Get default config
        config = self.DEFAULT_CONFIGS[strategy_enum].copy()
        
        # Apply user overrides
        if user_config:
            config = self._merge_configs(config, user_config)
        
        # Validate configuration
        errors = self._validate_config(strategy_enum, config)
        
        if errors:
            return ChunkingConfigResult(
                strategy=strategy_enum,
                config=config,
                validation_errors=errors
            )
        
        return ChunkingConfigResult(
            strategy=strategy_enum,
            config=config
        )
    
    def _merge_configs(
        self,
        default: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge user config with defaults"""
        result = default.copy()
        
        for key, value in override.items():
            if key in result:
                # Type checking
                if type(result[key]) != type(value):
                    # Try to convert
                    try:
                        if isinstance(result[key], int):
                            value = int(value)
                        elif isinstance(result[key], float):
                            value = float(value)
                        elif isinstance(result[key], bool):
                            value = bool(value)
                    except (ValueError, TypeError):
                        continue  # Skip invalid types
                
                result[key] = value
        
        return result
    
    def _validate_config(
        self,
        strategy: ChunkingStrategy,
        config: Dict[str, Any]
    ) -> List[str]:
        """Validate configuration for a strategy"""
        errors = []
        
        # Common validations
        if "chunk_size" in config:
            if config["chunk_size"] < 10:
                errors.append("chunk_size must be at least 10")
            if config["chunk_size"] > 100000:
                errors.append("chunk_size cannot exceed 100000")
        
        if "chunk_overlap" in config:
            if config["chunk_overlap"] < 0:
                errors.append("chunk_overlap cannot be negative")
            if config.get("chunk_size") and config["chunk_overlap"] >= config["chunk_size"]:
                errors.append("chunk_overlap must be less than chunk_size")
        
        # Strategy-specific validations
        if strategy == ChunkingStrategy.SEMANTIC:
            if config.get("similarity_threshold"):
                if not 0 <= config["similarity_threshold"] <= 1:
                    errors.append("similarity_threshold must be between 0 and 1")
        
        elif strategy == ChunkingStrategy.HIERARCHICAL:
            if len(config.get("chunk_sizes", [])) != len(config.get("chunk_overlaps", [])):
                errors.append("chunk_sizes and chunk_overlaps must have same length")
        
        return errors
```

### 2. Create Strategy Factory

```python
# packages/webui/services/chunking_strategy_factory.py

from typing import Optional, Dict, Any
from packages.shared.chunking.domain.services.chunking_strategies import (
    ChunkingStrategy,
    CharacterChunker,
    RecursiveChunker,
    MarkdownChunker,
    SemanticChunker,
    HierarchicalChunker,
    HybridChunker
)

class ChunkingStrategyFactory:
    """Factory for creating chunking strategy instances"""
    
    _strategies = {
        "character": CharacterChunker,
        "recursive": RecursiveChunker,
        "markdown": MarkdownChunker,
        "semantic": SemanticChunker,
        "hierarchical": HierarchicalChunker,
        "hybrid": HybridChunker
    }
    
    @classmethod
    def create_strategy(
        cls,
        strategy_name: str,
        config: Dict[str, Any]
    ) -> ChunkingStrategy:
        """
        Create a chunking strategy instance
        
        Args:
            strategy_name: Name of the strategy
            config: Configuration for the strategy
            
        Returns:
            Configured strategy instance
            
        Raises:
            ValueError: If strategy name is unknown
        """
        strategy_class = cls._strategies.get(strategy_name.lower())
        
        if not strategy_class:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Available: {', '.join(cls._strategies.keys())}"
            )
        
        return strategy_class(**config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available strategy names"""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(
        cls,
        name: str,
        strategy_class: type[ChunkingStrategy]
    ):
        """Register a custom strategy"""
        cls._strategies[name] = strategy_class
```

### 3. Refactor Service Layer

```python
# packages/webui/services/chunking_service.py

class ChunkingService:
    """Service layer for all chunking operations"""
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        chunk_repository: ChunkRepository,
        error_handler: ChunkingErrorHandler,
        user_id: Optional[str] = None
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.chunk_repository = chunk_repository
        self.error_handler = error_handler
        self.user_id = user_id
        self.config_builder = ChunkingConfigBuilder()
        self.strategy_factory = ChunkingStrategyFactory()
    
    async def preview_chunks(
        self,
        strategy: str,
        content: Optional[str] = None,
        document_id: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Preview chunking results
        
        All business logic is here, not in the router!
        """
        # Validation
        if not content and not document_id:
            raise ChunkingValidationError(
                "Either content or document_id must be provided"
            )
        
        if content and document_id:
            raise ChunkingValidationError(
                "Cannot provide both content and document_id"
            )
        
        # Document access validation
        if document_id:
            await self._validate_document_access(document_id)
            content = await self._load_document_content(document_id)
        
        # Build configuration
        config_result = self.config_builder.build_config(
            strategy=strategy,
            user_config=config_overrides
        )
        
        if config_result.validation_errors:
            raise ChunkingValidationError(
                f"Invalid configuration: {', '.join(config_result.validation_errors)}"
            )
        
        # Create strategy
        try:
            chunking_strategy = self.strategy_factory.create_strategy(
                strategy_name=config_result.strategy.value,
                config=config_result.config
            )
        except ValueError as e:
            raise ChunkingValidationError(str(e))
        
        # Perform chunking
        try:
            chunks = await self._execute_chunking(
                strategy=chunking_strategy,
                content=content,
                config=config_result.config
            )
        except Exception as e:
            self.error_handler.handle_chunking_error(e, strategy, self.user_id)
            raise ChunkingExecutionError(f"Chunking failed: {str(e)}")
        
        # Cache preview result
        preview_id = await self._cache_preview(chunks, strategy)
        
        return {
            "preview_id": preview_id,
            "strategy": strategy,
            "config": config_result.config,
            "chunks": chunks,
            "statistics": self._calculate_statistics(chunks)
        }
    
    async def _validate_document_access(self, document_id: str):
        """Validate user has access to document"""
        if not self.user_id:
            raise PermissionDeniedError("Authentication required")
        
        document = await self.db_session.get(Document, document_id)
        if not document:
            raise DocumentNotFoundError(f"Document {document_id} not found")
        
        # Check collection permissions
        permission = await self.db_session.execute(
            select(UserCollectionPermission)
            .where(
                UserCollectionPermission.user_id == self.user_id,
                UserCollectionPermission.collection_id == document.collection_id,
                UserCollectionPermission.permission.in_(['read', 'write', 'admin'])
            )
        )
        
        if not permission.scalar_one_or_none():
            raise PermissionDeniedError(
                f"No permission to access document {document_id}"
            )
    
    async def _load_document_content(self, document_id: str) -> str:
        """Load document content from storage"""
        document = await self.db_session.get(Document, document_id)
        
        # Load from appropriate storage
        if document.storage_type == "database":
            return document.content
        elif document.storage_type == "s3":
            return await self._load_from_s3(document.storage_path)
        elif document.storage_type == "filesystem":
            return await self._load_from_filesystem(document.storage_path)
        else:
            raise ValueError(f"Unknown storage type: {document.storage_type}")
    
    async def _execute_chunking(
        self,
        strategy: ChunkingStrategy,
        content: str,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute chunking with resource limits"""
        # Apply resource limits
        max_content_size = 10_000_000  # 10MB
        if len(content) > max_content_size:
            raise ChunkingValidationError(
                f"Content too large: {len(content)} > {max_content_size}"
            )
        
        # Execute with timeout
        try:
            chunks = await asyncio.wait_for(
                asyncio.to_thread(
                    strategy.chunk,
                    content,
                    config
                ),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise ChunkingExecutionError("Chunking operation timed out")
        
        return chunks
    
    def _calculate_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate chunking statistics"""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        sizes = [len(chunk["content"]) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes)
        }
```

### 4. Refactor Router to Thin Controller

```python
# packages/webui/api/v2/chunking.py

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any

router = APIRouter(prefix="/api/v2/chunking", tags=["chunking"])

@router.post("/preview")
async def preview_chunks(
    request: PreviewRequest,
    service: ChunkingService = Depends(get_chunking_service),
    current_user: User = Depends(get_current_user)
) -> PreviewResponse:
    """
    Preview chunking results
    
    Router is now a thin controller - all logic in service!
    """
    try:
        # Simply delegate to service
        result = await service.preview_chunks(
            strategy=request.strategy,
            content=request.content,
            document_id=request.document_id,
            config_overrides=request.config
        )
        
        # Transform to response model
        return PreviewResponse(**result)
        
    except ChunkingValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ChunkingExecutionError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/strategies")
async def list_strategies(
    service: ChunkingService = Depends(get_chunking_service)
) -> StrategiesResponse:
    """List available chunking strategies"""
    strategies = await service.get_available_strategies()
    return StrategiesResponse(strategies=strategies)

@router.post("/apply")
async def apply_chunking(
    request: ApplyChunkingRequest,
    service: ChunkingService = Depends(get_chunking_service),
    current_user: User = Depends(get_current_user)
) -> ApplyChunkingResponse:
    """Apply chunking to a document"""
    try:
        operation_id = await service.apply_chunking(
            document_id=request.document_id,
            strategy=request.strategy,
            config_overrides=request.config
        )
        
        return ApplyChunkingResponse(operation_id=operation_id)
        
    except ChunkingValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionDeniedError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except DocumentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

## Acceptance Criteria

1. **Clean Architecture**
   - [ ] Routers contain NO business logic
   - [ ] All validation logic in service layer
   - [ ] Configuration building extracted to dedicated class
   - [ ] Strategy creation handled by factory

2. **Service Layer**
   - [ ] All business logic in ChunkingService
   - [ ] Document access validation in service
   - [ ] Resource limits enforced in service
   - [ ] Error handling in service

3. **Testability**
   - [ ] Services can be unit tested without HTTP
   - [ ] Configuration builder can be tested independently
   - [ ] Strategy factory can be tested independently
   - [ ] Routers only need integration tests

4. **Maintainability**
   - [ ] Adding new strategy doesn't require router changes
   - [ ] Configuration changes don't affect routers
   - [ ] Business rules centralized in one place

## Testing Requirements

1. **Unit Tests**
   ```python
   async def test_service_validates_document_access():
       service = ChunkingService(user_id="user1")
       
       with pytest.raises(PermissionDeniedError):
           await service.preview_chunks(
               strategy="semantic",
               document_id="unauthorized_doc"
           )
   
   def test_config_builder_merges_configs():
       builder = ChunkingConfigBuilder()
       result = builder.build_config(
           "semantic",
           {"chunk_size": 1000}
       )
       
       assert result.config["chunk_size"] == 1000
       assert result.config["similarity_threshold"] == 0.7  # Default
   
   def test_strategy_factory_creates_strategies():
       factory = ChunkingStrategyFactory()
       strategy = factory.create_strategy(
           "markdown",
           {"chunk_size": 500}
       )
       
       assert isinstance(strategy, MarkdownChunker)
   ```

2. **Integration Tests**
   - Test complete flow from router to service
   - Test error propagation
   - Test authentication/authorization

## Rollback Plan

1. Keep backup of original router file
2. If issues found, can temporarily move logic back
3. Ensure API responses remain identical
4. Monitor for any behavioral changes

## Success Metrics

- Routers reduced to < 50 lines each
- Service layer methods properly isolated
- 100% unit test coverage for service logic
- No business logic in routers
- Configuration changes don't require router updates

## Notes for LLM Agent

- Focus on moving logic, not changing behavior
- Maintain exact same API responses
- Ensure all error codes remain the same
- Don't change any database queries yet
- Keep same authentication/authorization checks
- Test that API behavior is identical before/after