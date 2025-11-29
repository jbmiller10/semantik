<\!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding/removing models or repositories
     - Changing database schema or partitioning
     - Modifying chunking strategies
     - Altering exception hierarchy
     - Changing configuration patterns
     Keep this documentation in sync with the actual implementation\! -->


<component>
  <name>Shared Library</name>
  <purpose>Core models, repositories, and utilities used by all services</purpose>
  <location>packages/shared/</location>
</component>

<modules>
  <module path="database/">
    <purpose>SQLAlchemy models, repositories, and database utilities</purpose>
    <key-models>
      - User, Collection, Document, Chunk (partitioned), Operation
      - CollectionStatus enum: PENDING → READY → PROCESSING → ERROR/DEGRADED
      - OperationType enum: INDEX, APPEND, REINDEX, DELETE, REMOVE_SOURCE
    </key-models>
    <partitioning>
      Chunks table has 100 LIST partitions using abs(hashtext(collection_id)) % 100
    </partitioning>
  </module>
  
  <module path="chunking/">
    <purpose>Domain-driven chunking implementation</purpose>
    <structure>
      - domain/: Core business logic, strategies, value objects
      - application/: Use cases, DTOs
      - infrastructure/: Exception handling, repositories
    </structure>
    <strategies>CHARACTER, RECURSIVE, MARKDOWN, SEMANTIC, HIERARCHICAL, HYBRID</strategies>
  </module>
  
  <module path="text_processing/">
    <purpose>Legacy chunking (being replaced by chunking/)</purpose>
    <note>Use chunking/ module for new features</note>
  </module>
  
  <module path="embedding/">
    <purpose>Plugin-based embedding provider system</purpose>
    <pattern>Factory + Registry with auto-detection</pattern>
    <architecture>
      - plugin_base.py: BaseEmbeddingPlugin + EmbeddingProviderDefinition
      - provider_registry.py: Provider metadata registry (LRU cached)
      - factory.py: EmbeddingProviderFactory with model auto-detection
      - plugin_loader.py: Entry point discovery (semantik.embedding_providers)
      - types.py: EmbeddingMode enum (QUERY, DOCUMENT)
      - providers/: Built-in provider implementations
    </architecture>
    <built-in-providers>
      - DenseLocalEmbeddingProvider: sentence-transformers, Qwen (GPU/CPU)
      - MockEmbeddingProvider: Deterministic testing embeddings
    </built-in-providers>
    <asymmetric-mode>
      Many retrieval models need different processing for queries vs documents:
      - EmbeddingMode.QUERY: Search queries (applies prefixes/instructions)
      - EmbeddingMode.DOCUMENT: Document indexing (typically no prefix)

      ModelConfig fields: is_asymmetric, query_prefix, document_prefix, default_query_instruction
      EmbeddingProviderDefinition: supports_asymmetric flag
    </asymmetric-mode>
    <usage>
      from shared.embedding.factory import EmbeddingProviderFactory
      from shared.embedding.types import EmbeddingMode

      provider = EmbeddingProviderFactory.create_provider("model-name")

      # Query mode (default)
      embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.QUERY)

      # Document mode
      embeddings = await provider.embed_texts(texts, mode=EmbeddingMode.DOCUMENT)
    </usage>
    <api-endpoints>
      GET /embedding/providers - List providers
      GET /embedding/providers/{id} - Provider details
      GET /embedding/models - List all models
      GET /embedding/models/{name}/supported - Check support
    </api-endpoints>
  </module>
  
  <module path="managers/">
    <purpose>Qdrant vector database management</purpose>
  </module>
</modules>

<repository-pattern>
  <principle>All DB access through repositories</principle>
  <location>database/repositories/</location>
  <repositories>
    - CollectionRepository: Collection CRUD with permissions
    - DocumentRepository: Document management
    - ChunkRepository: Partition-aware chunk operations
    - OperationRepository: Async operation tracking
  </repositories>
  <critical-rule>
    Always include collection_id in chunk queries for partition pruning
  </critical-rule>
</repository-pattern>

<exceptions>
  <hierarchy>
    ApplicationError (base)
    ├── ValidationError
    ├── ResourceNotFoundError
    ├── ChunkingStrategyError
    └── DocumentTooLargeError
  </hierarchy>
  <translation>
    Use exception_translator to convert between layers
  </translation>
</exceptions>

<configuration>
  <module path="config/">
    <files>
      - base.py: Common settings
      - postgres.py: Database configuration
      - vecpipe.py: Embedding service config
      - webui.py: API service config
    </files>
    <pattern>Pydantic BaseSettings with environment variables</pattern>
  </configuration>
</configuration>

<testing>
  <requirement>Unit tests for all repositories</requirement>
  <location>tests/unit/, tests/domain/</location>
</testing>