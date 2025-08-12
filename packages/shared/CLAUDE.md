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
    <purpose>Embedding service abstraction</purpose>
    <pattern>Singleton with lazy loading</pattern>
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