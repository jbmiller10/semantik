<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Changing embedding models or batch sizes
     - Modifying search or reranking logic
     - Altering memory management strategies
     - Changing API endpoints or error handling
     - Updating performance targets
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>VecPipe Service</name>
  <purpose>Vector embedding generation, search, and reranking service</purpose>
  <location>packages/vecpipe/</location>
</component>

<key-modules>
  <module name="embed_chunks_unified.py">
    <purpose>Unified embedding generation for all documents</purpose>
    <models>Qwen3-Embedding-0.6B (int8 quantized)</models>
    <batch-sizes>
      - 0.6B/int8: 256 texts
      - 4B/float32: 16 texts  
      - 8B/float16: 8 texts
    </batch-sizes>
  </module>
  
  <module name="search_api.py">
    <purpose>Vector search API endpoints</purpose>
    <endpoints>
      - POST /search: Basic vector search
      - POST /hybrid_search: Vector + keyword fusion
    </endpoints>
  </module>
  
  <module name="reranker.py">
    <purpose>Cross-encoder reranking for search results</purpose>
    <models>Qwen3-Reranker models</models>
    <strategy>5x candidate multiplication, 20-200 candidate bounds</strategy>
  </module>
  
  <module name="model_manager.py">
    <purpose>Basic model lifecycle management</purpose>
    <features>
      - Lazy loading on first use
      - Auto-unload after 300s idle
      - GPU memory checking before load
    </features>
  </module>

  <module name="governed_model_manager.py">
    <purpose>Advanced model lifecycle with memory governance</purpose>
    <features>
      - Extends ModelManager with GPUMemoryGovernor
      - LRU-based eviction when memory needed
      - CPU offloading for "warm" model pool
      - Background memory pressure monitoring
    </features>
  </module>

  <module name="memory_governor.py">
    <purpose>GPU memory budget enforcement and eviction</purpose>
    <components>
      - GPUMemoryGovernor: Central memory coordinator
      - MemoryBudget: Configurable GPU/CPU limits
      - PressureLevel: LOW/MODERATE/HIGH/CRITICAL thresholds
      - TrackedModel: Model state tracking with LRU ordering
    </components>
  </module>

  <module name="cpu_offloader.py">
    <purpose>GPU⇔CPU model transfer for warm pool</purpose>
    <features>
      - Offload models to pinned CPU memory
      - Fast restore to GPU (2-5s vs 10-30s disk reload)
      - Automatic cleanup on shutdown
    </features>
  </module>
</key-modules>

<memory-management>
  <strategy>GPUMemoryGovernor with LRU eviction and CPU offloading</strategy>

  <pressure-levels>
    | Level    | Threshold | Action                                |
    |----------|-----------|---------------------------------------|
    | LOW      | &lt;60%      | No action                             |
    | MODERATE | 60-80%    | Offload models idle &gt;120s            |
    | HIGH     | 80-90%    | Aggressively offload models idle &gt;30s |
    | CRITICAL | &gt;90%      | Force unload all idle models          |
  </pressure-levels>

  <eviction-strategy>
    1. Offload LRU idle models to CPU first (preserves warm state)
    2. If CPU full or disabled, fully unload LRU models
    3. Never evict model in active use (5s grace period)
    4. Fail only if model won't fit even with empty GPU
  </eviction-strategy>

  <configuration>
    GPU limits:
      - GPU_MEMORY_RESERVE_PERCENT=0.10 (keep 10% free)
      - GPU_MEMORY_MAX_PERCENT=0.90 (never use &gt;90%)
    CPU limits (warm pool):
      - CPU_MEMORY_RESERVE_PERCENT=0.20 (keep 20% free)
      - CPU_MEMORY_MAX_PERCENT=0.50 (max 50% for warm models)
    Behavior:
      - ENABLE_CPU_OFFLOAD=true
      - EVICTION_IDLE_THRESHOLD_SECONDS=120
  </configuration>

  <vram-recommendations>
    - 24GB+ VRAM: Both embedding + reranker can coexist
    - 16GB VRAM: Default settings, occasional eviction under load
    - 8GB VRAM: More aggressive eviction, shorter idle thresholds
  </vram-recommendations>
</memory-management>

<search-flow>
  1. Resolve collection: explicit > operation_uuid lookup > default
  2. If search_type="hybrid", route to perform_hybrid_search()
  3. Embed query using same model as indexing
  4. Search Qdrant for top-k similar vectors
  5. Filter results by score_threshold (BEFORE reranking)
  6. Optional: Rerank filtered results with cross-encoder
  7. Return results with scores
</search-flow>

<collection-resolution>
  <priority-order>
    1. Explicit collection parameter in request
    2. operation_uuid database lookup (returns collection.vector_store_name)
    3. DEFAULT_COLLECTION from settings
  </priority-order>
  <error-behavior>
    If operation_uuid is provided but not found, returns HTTP 404 error.
  </error-behavior>
</collection-resolution>

<api-contracts>
  <score_threshold>
    Applies BEFORE reranking. Results with scores below threshold are
    excluded before being sent to reranker. Default is 0.0 (no filtering).
  </score_threshold>

  <search_type_hybrid>
    When search_type="hybrid", the /search endpoint internally routes
    to perform_hybrid_search() and maps results to SearchResponse format.
  </search_type_hybrid>

  <upsert_wait>
    The wait parameter is passed as a URL query parameter (?wait=true),
    not in the JSON request body, per Qdrant REST API specification.
  </upsert_wait>
</api-contracts>

<error-handling>
  <dimension-mismatch>HTTP 400 with clear error message</dimension-mismatch>
  <memory-error>HTTP 507 Insufficient Storage</memory-error>
  <timeout>HTTP 504 Gateway Timeout</timeout>
</error-handling>

<performance>
  <targets>
    - Single search: &lt;500ms p95
    - Batch embedding: &gt;100 texts/second
    - Reranking: &lt;200ms for 100 candidates
  </targets>
  <optimizations>
    - Batch processing for embeddings
    - Connection pooling for Qdrant
    - Result caching (15 min TTL)
  </optimizations>
</performance>

<testing>
  <location>tests/unit/test_embedding_*.py</location>
  <coverage>Dimension validation, OOM handling, retry logic</coverage>
  <location>tests/unit/test_memory_governor.py</location>
  <coverage>Budget calculations, LRU eviction, pressure levels, callbacks</coverage>
</testing>

<memory-api-endpoints>
  <endpoint path="/memory/stats">Memory statistics and pressure level</endpoint>
  <endpoint path="/memory/models">List tracked models with locations</endpoint>
  <endpoint path="/memory/evictions">Eviction history for debugging</endpoint>
</memory-api-endpoints>