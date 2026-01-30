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
      - BenchmarkDataset, BenchmarkQuery, BenchmarkRelevance (ground truth)
      - BenchmarkDatasetMapping (dataset-collection bindings)
      - Benchmark, BenchmarkRun, BenchmarkRunMetric, BenchmarkQueryResult
    </key-models>
    <enums>
      - CollectionStatus: PENDING -> READY -> PROCESSING -> ERROR/DEGRADED
      - OperationStatus: PENDING -> PROCESSING -> COMPLETED/FAILED/CANCELLED
      - OperationType: INDEX, APPEND, REINDEX, DELETE, REMOVE_SOURCE, PROJECTION_BUILD, BENCHMARK
      - DocumentStatus: PENDING -> PROCESSING -> COMPLETED/FAILED/DELETED
      - ProjectionRunStatus: PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED
      - BenchmarkStatus: PENDING -> RUNNING -> COMPLETED/FAILED/CANCELLED
      - BenchmarkRunStatus: PENDING -> INDEXING -> EVALUATING -> COMPLETED/FAILED
      - MappingStatus: PENDING -> RESOLVED/PARTIAL
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
      - BenchmarkRepository: Benchmark CRUD, run management, results aggregation
      - BenchmarkDatasetRepository: Dataset CRUD, query listing, mapping management
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
      - types/: Type-specific plugin bases (embedding, chunking, connector, reranker, extractor, sparse_indexer)
      - builtins/: Built-in plugins (keyword_extractor, qwen3_reranker, bm25_sparse_indexer, splade_indexer)
    </structure>
    <plugin-types>
      - EmbeddingPlugin: Model providers (local, remote, hybrid)
      - ChunkingPlugin: Text chunking strategies
      - ConnectorPlugin: Document source connectors
      - RerankerPlugin: Search result reranking
      - ExtractorPlugin: Entity/metadata extraction
      - SparseIndexerPlugin: Sparse vectors (BM25, SPLADE) for hybrid search
      - ParserPlugin: Document text extraction (text, unstructured)
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

  <module path="plugins/types/sparse_indexer.py">
    <purpose>Sparse indexer plugin base class for BM25/SPLADE hybrid search</purpose>
    <classes>
      - SparseVector: Immutable dataclass (indices, values, chunk_id, metadata)
      - SparseQueryVector: Query-only sparse vector (indices, values)
      - SparseIndexerCapabilities: Plugin capabilities (sparse_type, max_tokens, batching, IDF storage)
      - SparseIndexerPlugin: ABC base requiring encode_documents/encode_query/remove_documents
    </classes>
    <sparse-types>bm25, splade (SPARSE_TYPES constant)</sparse-types>
    <key-concepts>
      - Plugins generate sparse vectors only; vecpipe handles Qdrant persistence
      - Uses chunk_id (not document_id) for 1:1 alignment with dense vectors for RRF fusion
      - One sparse indexer per collection (BM25 OR SPLADE, not both)
    </key-concepts>
  </module>

  <module path="plugins/builtins/bm25_sparse_indexer.py">
    <purpose>BM25 sparse indexer with IDF persistence</purpose>
    <class>BM25SparseIndexerPlugin</class>
    <config>
      - k1 (default: 1.5): Term saturation parameter
      - b (default: 0.75): Length normalization (0=none, 1=full)
      - lowercase (default: true): Text normalization
      - remove_stopwords (default: true): Filter common English words
      - min_token_length (default: 2): Minimum token length
    </config>
    <idf-storage>data/sparse_indexes/{collection_name}/idf_stats.json</idf-storage>
    <performance>~1000 docs/sec (CPU), no GPU required</performance>
    <stateful>Yes - maintains corpus IDF statistics</stateful>
  </module>

  <module path="plugins/builtins/splade_indexer.py">
    <purpose>SPLADE learned sparse representations via neural models</purpose>
    <class>SPLADESparseIndexerPlugin</class>
    <default-model>naver/splade-cocondenser-ensembledistil</default-model>
    <config>
      - model_name: HuggingFace model ID
      - device: auto/cuda/cpu
      - quantization: float32/float16/int8
      - batch_size: 8-128 depending on GPU VRAM
      - max_length: 512 tokens
    </config>
    <gpu-batch-recommendations>
      4GB VRAM: batch_size=8
      8GB VRAM: batch_size=32
      24GB VRAM: batch_size=128
    </gpu-batch-recommendations>
    <performance>10-50 docs/sec (GPU), 0.3-1 docs/sec (CPU)</performance>
    <stateless>Yes - model parameters encode all knowledge</stateless>
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
      ENABLE_MEMORY_GOVERNOR, GPU_MEMORY_MAX_PERCENT, CPU_MEMORY_MAX_PERCENT
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

  <module path="llm/">
    <purpose>LLM service abstraction with provider plugins for HyDE, summarization, etc.</purpose>
    <structure>
      - base.py: BaseLLMService interface (initialize, generate, cleanup, async context manager)
      - types.py: LLMQualityTier (HIGH/LOW), LLMResponse, LLMProviderType (anthropic/openai/local)
      - exceptions.py: LLMError hierarchy (6 exception types)
      - factory.py: LLMServiceFactory - creates provider for user's configured tier
      - model_registry.py: ModelInfo dataclass, registry loader with LRU cache
      - model_registry.yaml: Curated models with memory_mb estimates for local models
      - usage_tracking.py: LLMUsageEventRepository for token usage logging
      - providers/anthropic_provider.py: HTTP client to Anthropic API
      - providers/openai_provider.py: HTTP client to OpenAI API
      - providers/local_provider.py: HTTP client to VecPipe /llm/generate endpoint
    </structure>
    <provider-pattern>
      Cloud providers (anthropic, openai) call external APIs directly.
      Local provider calls VecPipe internally - GPU memory managed by VecPipe's governor.
    </provider-pattern>
    <quality-tiers>
      - HIGH: Complex tasks (summarization, entity extraction) - best models
      - LOW: Simple tasks (HyDE, keywords) - fast/cheap models
      Users configure provider + model per tier in WebUI settings.
    </quality-tiers>
    <factory-usage>
      from shared.llm.factory import LLMServiceFactory
      from shared.llm.types import LLMQualityTier

      factory = LLMServiceFactory(session)
      provider = await factory.create_provider_for_tier(user_id, LLMQualityTier.LOW)
      async with provider:
          response = await provider.generate(prompt="...", max_tokens=256)
    </factory-usage>
    <local-provider-specifics>
      - Reuses SEARCH_API_URL + X-Internal-Api-Key (same as embedding proxies)
      - Quantization passed via kwargs: initialize(api_key="", model=model, quantization="int8")
      - No API key required for local provider
    </local-provider-specifics>
  </module>

  <module path="benchmarks/">
    <purpose>Search quality benchmarking with standard IR metrics</purpose>
    <structure>
      - __init__.py: Module exports with usage examples
      - metrics.py: Core metric functions (precision, recall, MRR, nDCG, AP)
      - evaluator.py: QueryEvaluator and ConfigurationEvaluator classes
      - types.py: TypedDicts for structured results (RetrievedChunk, RelevanceJudgment, etc.)
      - exceptions.py: BenchmarkError hierarchy
    </structure>
    <metrics>
      - collapse_chunks_to_documents: First-hit deduplication for doc-level evaluation
      - precision_at_k: Fraction of top-k slots that are relevant
      - recall_at_k: Fraction of relevant documents found in top-k
      - mean_reciprocal_rank: 1/rank of first relevant document
      - ndcg_at_k: Normalized DCG with graded relevance (0-3)
      - average_precision: Mean precision at each relevant position
      - compute_all_metrics: Convenience function for all metrics at multiple k values
    </metrics>
    <evaluators>
      - QueryEvaluator: Evaluates single query with timing and metrics
      - ConfigurationEvaluator: Evaluates all queries for a configuration
    </evaluators>
    <usage>
      from shared.benchmarks import (
          QueryEvaluator, compute_all_metrics,
          RetrievedChunk, SearchTiming
      )

      evaluator = QueryEvaluator()
      result = evaluator.evaluate(
          query_id=1,
          retrieved_chunks=chunks,
          relevance_judgments=judgments,
          k_values=[5, 10],
          timing=SearchTiming(search_time_ms=50),
      )
    </usage>
    <key-concepts>
      - Document-level evaluation: chunks collapsed to unique docs before metrics
      - Graded relevance: 0=not relevant, 1=marginally, 2=relevant, 3=highly relevant
      - Binary threshold: grade > 0 treated as relevant for precision/recall/MRR/AP
    </key-concepts>
  </module>

  <module path="pipeline/">
    <purpose>DAG-based document processing pipeline with conditional routing</purpose>
    <structure>
      - types.py: Core data structures (PipelineDAG, PipelineNode, PipelineEdge, FileReference)
      - executor.py: PipelineExecutor - orchestrates DAG execution with progress tracking
      - executor_types.py: Execution context, callbacks, result types
      - router.py: PipelineRouter - edge matching with exclusive/parallel semantics
      - predicates.py: Predicate expression evaluation (glob, negation, numeric, arrays)
      - validation.py: DAG validation (reachability, terminal nodes, catch-all requirements)
      - sniff.py: ContentSniffer - pre-routing content detection (code, scanned PDFs, structured data)
      - loader.py: Load/save pipeline configs from database/YAML
      - defaults.py: Default pipeline for backward compatibility
      - failure_tracker.py: Track consecutive failures to halt on systemic issues
      - templates/: Pre-configured pipeline templates (academic_papers, codebase, documentation, etc.)
    </structure>
    <key-types>
      - PipelineDAG: Complete graph definition with nodes and edges
      - PipelineNode: Processing step with type (PARSER, CHUNKER, EXTRACTOR, EMBEDDER), plugin_id, config
      - PipelineEdge: Connection with optional `when` predicate and `parallel` flag for fan-out
      - FileReference: Input file with namespaced metadata (source, detected, parsed)
      - NodeType: Enum of PARSER, CHUNKER, EXTRACTOR, EMBEDDER
    </key-types>
    <predicate-expressions>
      Predicates match against FileReference metadata using a mini-language:
      - Exact: {"mime_type": "application/pdf"}
      - Glob: {"mime_type": "application/*"}, {"extension": "*.md"}
      - Negation: {"mime_type": "!image/*"} (prefix with !)
      - Numeric: {"size_bytes": ">10000000"}, {"size_bytes": "<=1024"}
      - Array (OR): {"extension": [".md", ".txt", ".rst"]}
      - Nested paths: {"metadata.detected.is_code": true}
      - Boolean: {"metadata.detected.is_scanned_pdf": true}
    </predicate-expressions>
    <edge-evaluation-order>
      1. Parallel predicate edges (all matches fire for fan-out)
      2. Exclusive predicate edges (first match wins)
      3. Parallel catch-all edges (all fire)
      4. Exclusive catch-all edges (fallback)
      CRITICAL: Every DAG must have at least one non-parallel catch-all edge from _source
    </edge-evaluation-order>
    <metadata-namespacing>
      FileReference.metadata uses namespaced keys:
      - "source": Connector-provided metadata (path, filename, size_bytes, etc.)
      - "detected": Sniff results (is_code, is_scanned_pdf, detected_language, etc.)
      - "parsed": Parser-extracted metadata (page_count, has_tables, author, etc.)
    </metadata-namespacing>
    <validation-rules>
      - Must have exactly one EMBEDDER node
      - Every node must reach an embedder (or extractor for parallel paths)
      - Requires at least one non-parallel catch-all edge from _source
      - Parallel edges from same source need unique path_name values
      - Tier ordering enforced: _source -> PARSER -> CHUNKER -> EXTRACTOR/EMBEDDER -> EMBEDDER
    </validation-rules>
    <execution-modes>
      - ExecutionMode.FULL: Production - processes files, writes to database
      - ExecutionMode.DRY_RUN: Validation - runs routing/parsing without writes
    </execution-modes>
    <usage>
      from shared.pipeline import PipelineExecutor, PipelineDAG, ExecutionMode
      from shared.pipeline.loader import load_pipeline_config

      dag = load_pipeline_config(collection_id, session)
      executor = PipelineExecutor(dag, plugins, session)
      results = await executor.execute(file_refs, mode=ExecutionMode.FULL)
    </usage>
  </module>

  <module path="plugins/types/parser.py">
    <purpose>Parser plugin type for document text extraction</purpose>
    <classes>
      - ParserPlugin: Base class for parser implementations (sync methods for Celery)
      - ParsedDocument: Result with content, metadata, chunks list
      - ParserError, UnsupportedFormatError, ExtractionFailedError: Exception hierarchy
      - AgentHints: Discovery metadata for agent-driven plugin selection
    </classes>
    <key-attributes>
      - PLUGIN_TYPE = "parser"
      - SUPPORTED_EXTENSIONS: frozenset of file extensions
      - SUPPORTED_MIME_TYPES: frozenset of MIME types
      - EMITTED_FIELDS: List of metadata fields emitted for routing predicates
    </key-attributes>
    <built-in-parsers>
      - text: Lightweight text/markdown/code parser (no dependencies)
      - unstructured: Full-featured parser using unstructured library (PDF, DOCX, PPTX, HTML, email)
    </built-in-parsers>
    <sync-requirement>
      Parser methods (parse_file, parse_bytes) are SYNC, not async.
      This is required for billiard.Pool compatibility in Celery workers.
    </sync-requirement>
  </module>

  <module path="plugins/discovery.py">
    <purpose>Agent-facing plugin discovery APIs</purpose>
    <key-functions>
      - list_plugins_for_agent(plugin_type): Returns plugins with AgentHints for selection
      - find_plugins_for_input(mime_type): MIME matching with wildcard support and specificity scoring
      - get_alternative_plugins(plugin_id): Find alternatives based on input_type overlap
      - get_emitted_predicate_fields(plugin_type): Aggregate EMITTED_FIELDS for routing field discovery
    </key-functions>
    <usage>
      from shared.plugins.discovery import list_plugins_for_agent, find_plugins_for_input

      parsers = list_plugins_for_agent("parser")
      best_parser = find_plugins_for_input("application/pdf")[0]
    </usage>
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
  <llm-hierarchy path="llm/exceptions.py">
    LLMError (base)
    +-- LLMNotConfiguredError: User hasn't set up LLM settings
    +-- LLMAuthenticationError: Invalid or missing API key
    +-- LLMRateLimitError: Provider rate limit (retryable)
    +-- LLMProviderError: General provider error (network, API)
    +-- LLMTimeoutError: Request timed out
    +-- LLMContextLengthError: Input exceeds model context window
  </llm-hierarchy>
  <benchmark-hierarchy path="benchmarks/exceptions.py">
    BenchmarkError (base)
    +-- BenchmarkMetricError: Error computing metrics
    +-- BenchmarkEvaluationError: Error during query evaluation
    +-- BenchmarkValidationError: Invalid benchmark configuration
    +-- BenchmarkCancelledError: Benchmark execution cancelled
  </benchmark-hierarchy>
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
    - Test: `uv run pytest tests/shared/`
    - Lint: `uv run ruff check packages/shared/`
    - Type check: `uv run mypy packages/shared/`
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
