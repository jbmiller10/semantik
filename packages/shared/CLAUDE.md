<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding/removing models or repositories
     - Changing database schema or partitioning
     - Modifying chunking strategies
     - Altering exception hierarchy
     - Changing configuration patterns
     - Adding/modifying plugin types
     - Adding/removing connectors
     Keep this documentation in sync with the actual implementation! -->


<component>
  <name>Shared Library</name>
  <purpose>Core models, repositories, utilities, and plugin system used by all services</purpose>
  <location>packages/shared/</location>
</component>

<architecture>
  <pattern>Repository Pattern with Domain-Driven Design for chunking</pattern>
  <key-principle>All database access through repositories; plugin-based extensibility</key-principle>
</architecture>

<modules>
  <module path="database/">
    <purpose>SQLAlchemy models, repositories, and database utilities</purpose>
    <key-models>
      - User, Collection, Document, Chunk (partitioned), Operation
      - CollectionSource, DocumentArtifact, ConnectorSecret
      - MCPProfile, MCPProfileCollection (MCP search profiles)
      - ProjectionRun (embedding visualization)
      - ChunkingConfig, ChunkingConfigProfile, ChunkingStrategy
      - PluginConfig (external plugin configuration)
    </key-models>
    <enums>
      - CollectionStatus: PENDING -> READY -> PROCESSING -> ERROR/DEGRADED
      - OperationStatus: PENDING -> PROCESSING -> COMPLETED/FAILED/CANCELLED
      - OperationType: INDEX, APPEND, REINDEX, DELETE, REMOVE_SOURCE, PROJECTION_BUILD
      - DocumentStatus: PENDING -> PROCESSING -> COMPLETED/FAILED/DELETED
      - ProjectionRunStatus: PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED
    </enums>
    <partitioning>
      Chunks table uses LIST partitioning with 100 partitions.
      partition_key = abs(hashtext(collection_id)) % 100 (computed via trigger)
      CRITICAL: Always include collection_id in chunk queries for partition pruning.
    </partitioning>
  </module>

  <module path="database/repositories/">
    <purpose>Async repository implementations with session management</purpose>
    <repositories>
      - CollectionRepository: Collection CRUD, permissions, sync scheduling
      - DocumentRepository: Document management with source tracking
      - ChunkRepository: Partition-aware chunk operations (PartitionAwareMixin)
      - OperationRepository: Async operation tracking
      - PluginConfigRepository: Plugin enable/disable and config storage
      - ProjectionRunRepository: Embedding projection lifecycle
      - ChunkingConfigProfileRepository: User-scoped chunking presets
      - CollectionSyncRunRepository: Sync run tracking
    </repositories>
  </module>

  <module path="plugins/">
    <purpose>Unified plugin system with registry, loader, and type-specific bases</purpose>
    <structure>
      - base.py: SemanticPlugin abstract base class
      - manifest.py: PluginManifest dataclass for discovery/UI
      - registry.py: Thread-safe PluginRegistry (plugin_registry singleton)
      - loader.py: Entry point discovery (semantik.plugins group)
      - state.py: Plugin state file I/O for cross-service config
      - security.py: Audit logging, environment sanitization
      - types/: Type-specific plugin bases (embedding, chunking, connector, reranker, extractor)
      - builtins/: Built-in plugins (keyword_extractor, qwen3_reranker)
    </structure>
    <plugin-types>
      - EmbeddingPlugin: Model providers (local, remote, hybrid)
      - ChunkingPlugin: Text chunking strategies
      - ConnectorPlugin: Document source connectors
      - RerankerPlugin: Search result reranking
      - ExtractorPlugin: Entity/metadata extraction
    </plugin-types>
    <state-file>
      Plugin state persisted to /data/plugin_state.json (shared volume).
      WebUI writes state, VecPipe reads for enable/disable and config.
    </state-file>
  </module>

  <module path="embedding/">
    <purpose>Plugin-based embedding provider system with GPU memory management</purpose>
    <architecture>
      - plugin_base.py: BaseEmbeddingPlugin + EmbeddingProviderDefinition
      - factory.py: EmbeddingProviderFactory with model auto-detection
      - provider_registry.py: Provider metadata registry (LRU cached)
      - types.py: EmbeddingMode enum (QUERY, DOCUMENT)
      - batch_manager.py: AdaptiveBatchSizeManager for OOM recovery
      - context.py: ManagedEmbeddingService, context managers
      - validation.py: Dimension validation utilities
      - providers/: Built-in providers (dense_local, mock)
    </architecture>
    <asymmetric-mode>
      Many retrieval models need different processing for queries vs documents:
      - EmbeddingMode.QUERY: Search queries (applies prefixes/instructions)
      - EmbeddingMode.DOCUMENT: Document indexing (typically no prefix)
      ModelConfig fields: is_asymmetric, query_prefix, document_prefix
    </asymmetric-mode>
    <usage>
      from shared.embedding.factory import EmbeddingProviderFactory
      from shared.embedding.types import EmbeddingMode

      provider = EmbeddingProviderFactory.create_provider("model-name")
      embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)
    </usage>
  </module>

  <module path="chunking/">
    <purpose>Domain-driven chunking with unified strategy interface</purpose>
    <structure>
      - domain/: Core business logic, entities, value objects
        - entities/: Chunk, ChunkCollection, ChunkingOperation
        - value_objects/: ChunkConfig, ChunkMetadata, OperationStatus
        - services/chunking_strategies/: Strategy implementations
        - services/streaming_strategies/: Memory-efficient streaming
      - application/: Use cases, DTOs, interfaces
        - use_cases/: preview_chunking, process_document, compare_strategies
        - dto/: Request/response DTOs
        - interfaces/: Repository and service abstractions
      - infrastructure/: Exception handling, streaming support
        - streaming/: checkpoint, memory_pool, memory_monitor, processor
        - exception_translator.py: Layer boundary exception conversion
      - unified/: Single interface for all strategies
        - base.py: UnifiedChunkingStrategy abstract base
        - factory.py: UnifiedChunkingFactory
        - *_strategy.py: Concrete implementations
      - utils/: Input validation, regex safety, ReDoS protection
    </structure>
    <strategies>
      CHARACTER, RECURSIVE, MARKDOWN, SEMANTIC, HIERARCHICAL, HYBRID
    </strategies>
    <factory-usage>
      from shared.chunking.unified.factory import UnifiedChunkingFactory
      strategy = UnifiedChunkingFactory.create_strategy("semantic", use_llama_index=False)
      chunks = strategy.chunk(text, config)
    </factory-usage>
  </module>

  <module path="connectors/">
    <purpose>Document source connectors for ingestion pipeline</purpose>
    <connectors>
      - BaseConnector: Abstract base with authenticate() and load_documents()
      - LocalFileConnector: Local filesystem sources
      - GitConnector: Git repository cloning (git.py)
      - IMAPConnector: Email inbox sources (imap.py)
    </connectors>
    <contract>
      Connectors yield IngestedDocument DTOs with:
      - content: Parsed text content
      - unique_id: Source-specific identifier (URI, path, message ID)
      - source_type: Connector type identifier
      - metadata: Raw source metadata
      - content_hash: SHA-256 hash (64 lowercase hex chars)
    </contract>
  </module>

  <module path="config/">
    <purpose>Pydantic-based configuration with environment variable support</purpose>
    <files>
      - base.py: BaseConfig with paths, Qdrant, plugin state
      - postgres.py: PostgresConfig for database connection
      - vecpipe.py: VecpipeConfig for embedding service (GPU memory governor settings)
      - webui.py: WebuiConfig for API (JWT, Redis, document roots)
      - runtime.py: Runtime initialization helpers
    </files>
    <pattern>
      Pydantic BaseSettings with env_file=".env"
      Use properties for Docker path detection (data_dir, logs_dir)
    </pattern>
    <vecpipe-gpu-settings>
      ENABLE_MEMORY_GOVERNOR, GPU_MEMORY_RESERVE_PERCENT, GPU_MEMORY_MAX_PERCENT
      ENABLE_CPU_OFFLOAD, EVICTION_IDLE_THRESHOLD_SECONDS
      ENABLE_ADAPTIVE_BATCH_SIZE, MIN_BATCH_SIZE, MAX_BATCH_SIZE
    </vecpipe-gpu-settings>
  </module>

  <module path="managers/">
    <purpose>Qdrant vector database management with blue-green deployment</purpose>
    <class>QdrantManager</class>
    <features>
      - create_staging_collection(): Zero-downtime reindexing
      - cleanup_orphaned_collections(): Post-migration cleanup
      - rename_collection(): Clone-and-swap with point migration
      - validate_collection_health(): Health checks
      - get_collection_usage(): Async usage metrics
    </features>
  </module>

  <module path="text_processing/">
    <purpose>Legacy chunking (use chunking/ module for new features)</purpose>
    <note>Maintained for backward compatibility; prefer unified chunking</note>
  </module>

  <module path="contracts/">
    <purpose>Shared Pydantic models for API error responses</purpose>
    <classes>
      ErrorResponse, ValidationErrorResponse, NotFoundErrorResponse,
      InsufficientResourcesErrorResponse, ServiceUnavailableError, RateLimitError
    </classes>
  </module>

  <module path="dtos/">
    <purpose>Data transfer objects for ingestion pipeline</purpose>
    <classes>
      - IngestedDocument: Unified contract from all connectors
    </classes>
  </module>

  <module path="metrics/">
    <purpose>Prometheus metrics and collection statistics</purpose>
    <files>
      - prometheus.py: Metric definitions
      - collection_metrics.py: QdrantOperationTimer context manager
    </files>
  </module>

  <module path="utils/">
    <purpose>Utility functions</purpose>
    <files>
      - hashing.py: Content hash utilities
      - encryption.py: Fernet encryption for connector secrets
      - regex_safety.py: ReDoS-safe regex patterns
      - testing_utils.py: Test helpers
    </files>
  </module>
</modules>

<repository-pattern>
  <principle>All DB access through repositories with async sessions</principle>
  <location>database/repositories/</location>
  <critical-rules>
    1. Always include collection_id in chunk queries for partition pruning
    2. Use PartitionAwareMixin for chunk operations
    3. Group bulk operations by collection_id for efficiency
    4. Cross-partition queries are expensive - avoid when possible
  </critical-rules>
  <session-pattern>
    async with get_async_session() as session:
        repo = CollectionRepository(session)
        collection = await repo.get_by_uuid(uuid)
        await session.commit()
  </session-pattern>
</repository-pattern>

<exceptions>
  <database-hierarchy path="database/exceptions.py">
    RepositoryError (base)
    +-- EntityNotFoundError
    +-- EntityAlreadyExistsError
    +-- InvalidUserIdError
    +-- AccessDeniedError
    +-- ValidationError
    +-- DatabaseOperationError
    +-- TransactionError
    +-- ConcurrencyError
    +-- InvalidStateError
    +-- DimensionMismatchError
  </database-hierarchy>
  <plugin-hierarchy path="plugins/exceptions.py">
    PluginError (base)
    +-- PluginLoadError
    +-- PluginDuplicateError
    +-- PluginNotFoundError
    +-- PluginValidationError
  </plugin-hierarchy>
  <chunking-hierarchy path="chunking/domain/exceptions.py">
    ChunkingError (base)
    +-- ChunkingStrategyError
    +-- ChunkingConfigError
    +-- DocumentTooLargeError
  </chunking-hierarchy>
  <translation>
    Use chunking/infrastructure/exception_translator.py to convert between layers
  </translation>
</exceptions>

<plugin-system>
  <base-class>SemanticPlugin</base-class>
  <required-attributes>
    PLUGIN_TYPE, PLUGIN_ID, PLUGIN_VERSION
  </required-attributes>
  <required-methods>
    get_manifest() -> PluginManifest
    health_check(config) -> bool
    initialize(config) -> None
    cleanup() -> None
  </required-methods>
  <registration>
    Plugins discovered via entry points: semantik.plugins
    Built-in plugins registered at import time
    External plugins loaded by loader.load_plugins()
  </registration>
  <embedding-plugin-specific>
    BaseEmbeddingPlugin extends both SemanticPlugin and BaseEmbeddingService
    Required: get_definition(), supports_model(), embed_texts()
    Optional: list_supported_models(), get_model_config()
  </embedding-plugin-specific>
</plugin-system>

<key-patterns>
  <async-context>
    All database operations are async. Use proper session management:
    - get_async_session() for standalone operations
    - Dependency injection in FastAPI endpoints
  </async-context>
  <partition-awareness>
    The Chunk model uses composite primary key (id, collection_id, partition_key).
    Always query with collection_id to enable partition pruning.
  </partition-awareness>
  <plugin-config-flow>
    WebUI saves plugin config -> state file -> VecPipe reads on provider creation
    Factory auto-loads config via get_plugin_config(plugin_id)
  </plugin-config-flow>
</key-patterns>

<development>
  <commands>
    - Test: `poetry run pytest tests/`
    - Lint: `poetry run ruff check packages/shared/`
    - Type check: `poetry run mypy packages/shared/`
  </commands>
  <test-locations>
    - tests/unit/: Repository and utility tests
    - tests/domain/: Domain logic tests
    - chunking/domain/test_domain.py: Chunking domain tests
    - chunking/infrastructure/streaming/test_memory_pool.py: Streaming tests
  </test-locations>
</development>

<critical-patterns>
  <chunk-queries>
    # CORRECT - partition pruning enabled
    chunks = await session.execute(
        select(Chunk).where(
            Chunk.collection_id == collection_id,
            Chunk.document_id == document_id
        )
    )

    # WRONG - scans all 100 partitions
    chunks = await session.execute(
        select(Chunk).where(Chunk.document_id == document_id)
    )
  </chunk-queries>
  <embedding-mode>
    # Use QUERY mode for search, DOCUMENT mode for indexing
    query_embeddings = await provider.embed_texts(queries, mode=EmbeddingMode.QUERY)
    doc_embeddings = await provider.embed_texts(docs, mode=EmbeddingMode.DOCUMENT)
  </embedding-mode>
  <plugin-registration>
    # Plugins must be registered before use
    from shared.plugins import load_plugins
    load_plugins(["embedding", "reranker"])  # Load specific types
  </plugin-registration>
</critical-patterns>

<common-pitfalls>
  <pitfall>Querying chunks without collection_id - causes full table scan</pitfall>
  <pitfall>Not loading plugins before accessing factory - raises ValueError</pitfall>
  <pitfall>Using legacy text_processing module - prefer unified chunking</pitfall>
  <pitfall>Forgetting async session commit - changes not persisted</pitfall>
  <pitfall>Mixing EmbeddingMode in same batch - produces inconsistent embeddings</pitfall>
</common-pitfalls>
