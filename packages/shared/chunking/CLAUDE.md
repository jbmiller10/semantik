<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding new chunking strategies
     - Changing layer boundaries or interfaces
     - Modifying config validation rules
     - Updating factory or adapter patterns
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>Chunking System</name>
  <purpose>Domain-driven text chunking with unified strategy interface</purpose>
  <location>packages/shared/chunking/</location>
</component>

<architecture>
  Clean Architecture with 4 layers:

  ┌─────────────────────────────────────┐
  │  Unified (unified/)                 │  ← Entry point for consumers
  │  Factory, adapters for legacy APIs  │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Application (application/)         │  ← Use cases, DTOs
  │  preview_chunking, process_document │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Domain (domain/)                   │  ← Core business logic
  │  Chunk, ChunkConfig, strategies     │
  └──────────────┬──────────────────────┘
                 │
  ┌──────────────▼──────────────────────┐
  │  Infrastructure (infrastructure/)   │  ← Technical concerns
  │  Streaming, checkpoints, memory     │
  └─────────────────────────────────────┘
</architecture>

<entry-points>
  For most consumers, use the Unified layer:

  from shared.chunking.unified.factory import UnifiedChunkingFactory

  # Create strategy by name
  strategy = UnifiedChunkingFactory.create_strategy("semantic", embed_model=model)
  chunks = strategy.chunk(text, config)

  # Or with adapters for legacy text_processing interface
  from shared.chunking.unified.factory import TextProcessingStrategyAdapter
  adapter = TextProcessingStrategyAdapter(strategy, max_tokens=1000)
  results = adapter.chunk_text(text, doc_id, metadata)
</entry-points>

<strategies>
  | Strategy | Use Case | Notes |
  |----------|----------|-------|
  | CHARACTER | Fixed-size splits | Fast, predictable token counts |
  | RECURSIVE | Code and structured text | Separator-based hierarchy |
  | SEMANTIC | Natural language | Requires embed_model kwarg |
  | MARKDOWN | .md files | Preserves heading structure |
  | HIERARCHICAL | Multi-level retrieval | Parent-child relationships |
  | HYBRID | Auto-selection | Analyzes content, picks best |
</strategies>

<domain-layer>
  Entities (domain/entities/):
    - Chunk: Content with ChunkMetadata
    - ChunkCollection: Ordered collection with iteration
    - ChunkingOperation: Tracks operation lifecycle

  Value Objects (domain/value_objects/):
    - ChunkConfig: Immutable (max_tokens, min_tokens, overlap_tokens, strategy_name)
    - ChunkMetadata: Position, token count, hierarchy level, custom attributes
    - OperationStatus: State machine (PENDING → PROCESSING → COMPLETED/FAILED)

  Services (domain/services/chunking_strategies/):
    - ChunkingStrategy: ABC with chunk(), validate_content(), estimate_chunks()
    - STRATEGY_REGISTRY: Dict mapping names to strategy classes
    - register_strategy() / unregister_strategy(): Dynamic registration
</domain-layer>

<application-layer>
  Use Cases (application/use_cases/):
    - preview_chunking: Generate preview without DB persistence
    - process_document: Full chunking with operation tracking
    - compare_strategies: Side-by-side strategy comparison
    - cancel_operation: Graceful operation cancellation

  DTOs (application/dto/):
    - ChunkingRequest / ChunkingResponse
    - Validation at DTO boundary, not in domain
</application-layer>

<infrastructure-layer>
  Streaming (infrastructure/streaming/):
    - checkpoint.py: Save/restore progress for large documents
    - memory_pool.py: Reusable buffer management
    - memory_monitor.py: Track memory pressure
    - processor.py: Stream-based chunking orchestrator

  Exception Translation:
    - exception_translator.py: Domain → API-friendly error conversion
    - Use at layer boundaries to maintain abstraction
</infrastructure-layer>

<config-validation>
  ChunkConfig enforces at construction:
    - overlap_tokens < min_tokens
    - overlap_tokens < max_tokens
    - min_tokens > 0
    - max_tokens > min_tokens

  Invalid configs raise ChunkingConfigError immediately.

  from shared.chunking.domain.value_objects.chunk_config import ChunkConfig

  config = ChunkConfig(
      max_tokens=1000,
      min_tokens=100,
      overlap_tokens=50,  # Must be < min_tokens AND < max_tokens
      strategy_name="semantic",
  )
</config-validation>

<gotchas>
  <gotcha>Semantic and Hybrid strategies require embed_model kwarg</gotcha>
  <gotcha>Streaming strategies (domain/services/streaming_strategies/) are different from unified strategies</gotcha>
  <gotcha>Use exception_translator at layer boundaries for clean error conversion</gotcha>
  <gotcha>Legacy text_processing module exists but prefer unified layer</gotcha>
  <gotcha>Hierarchical strategy stores parent/child refs in custom_attributes</gotcha>
  <gotcha>overlap_tokens must be strictly less than BOTH min_tokens and max_tokens</gotcha>
</gotchas>

<adapters>
  Two adapters bridge unified strategies to legacy interfaces:

  DomainStrategyAdapter:
    Wraps unified strategy for domain layer consumers.
    Delegates all attribute access to wrapped strategy.

  TextProcessingStrategyAdapter:
    Wraps unified strategy for text_processing interface.
    Provides chunk_text(), chunk_text_async(), validate_config(), estimate_chunks().
    Handles both token-based (max_tokens) and legacy character-based (chunk_size) params.
</adapters>

<testing>
  <command>uv run pytest tests/domain/ -v</command>
  <command>uv run pytest tests/unit/chunking/ -v</command>
  <command>uv run pytest tests/streaming/ -v</command>
  <note>Domain tests in chunking/domain/test_domain.py</note>
  <note>Streaming tests in infrastructure/streaming/test_memory_pool.py</note>
</testing>
