<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Changing embedding models, reranker models, or batch sizes
     - Modifying search endpoints or adding new API routes
     - Altering memory management strategies or eviction thresholds
     - Changing model lifecycle callbacks or governor behavior
     - Updating performance targets or API contracts
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>VecPipe Service</name>
  <purpose>Vector embedding generation, semantic search, hybrid search, and reranking service with GPU memory governance</purpose>
  <location>packages/vecpipe/</location>
</component>

<architecture>
  <pattern>FastAPI service with plugin-aware provider system and memory-governed model lifecycle</pattern>
  <key-principle>Dynamic GPU memory management via LRU eviction and CPU offloading enables running large embedding + reranker models on limited VRAM</key-principle>
</architecture>

<modules>
  <module path="search_api.py">
    <purpose>Public entrypoint - thin wrapper that exports from search/ submodule</purpose>
    <note>Uses module-level __getattribute__ to forward state access to search.state</note>
  </module>

  <module path="search/app.py">
    <purpose>FastAPI application factory with router registration</purpose>
  </module>

  <module path="search/lifespan.py">
    <purpose>Application lifecycle: Qdrant connections, ModelManager initialization, plugin loading</purpose>
    <key-behavior>Uses GovernedModelManager when ENABLE_MEMORY_GOVERNOR=true, else standard ModelManager</key-behavior>
  </module>

  <module path="search/router.py">
    <purpose>HTTP endpoints for search, embed, upsert operations</purpose>
    <security>Protected endpoints require X-Internal-Api-Key header</security>
  </module>

  <module path="search/service.py">
    <purpose>Core business logic: embedding generation, search execution, reranking</purpose>
    <key-functions>
      - perform_search(): Semantic/question/code/hybrid search with optional reranking
      - resolve_collection_name(): Priority resolution (explicit > operation_uuid > default)
      - embed_texts(): Batch embedding generation
      - upsert_points(): Qdrant vector upsert
    </key-functions>
  </module>

  <module path="search/memory_api.py">
    <purpose>Memory management endpoints for monitoring and manual control</purpose>
    <endpoints>
      - GET /memory/stats: Memory usage and pressure level
      - GET /memory/models: Currently loaded models with locations
      - GET /memory/evictions: Eviction history for debugging
      - POST /memory/evict/{model_type}: Manual model eviction
      - POST /memory/preload: Pre-warm models for expected requests
      - POST /memory/defragment: Trigger CUDA cache cleanup
    </endpoints>
  </module>

  <module path="model_manager.py">
    <purpose>Base model lifecycle with lazy loading and auto-unload</purpose>
    <features>
      - Plugin-aware provider via EmbeddingProviderFactory
      - Async embedding generation
      - Reranker lifecycle with CrossEncoderReranker
      - Configurable inactivity timeout (default 300s)
    </features>
  </module>

  <module path="governed_model_manager.py">
    <purpose>Extended ModelManager with GPU memory governance</purpose>
    <extends>ModelManager</extends>
    <features>
      - Memory budget enforcement before model load
      - LRU eviction when memory needed
      - CPU offloading for warm model pool
      - Governor callback registration for embedding/reranker
      - Preload support for warming models
    </features>
    <critical-pattern>
      Must await start() after construction to begin background pressure monitor
    </critical-pattern>
  </module>

  <module path="memory_governor.py">
    <purpose>Central GPU memory coordinator with LRU eviction</purpose>
    <classes>
      - GPUMemoryGovernor: Main coordinator
      - MemoryBudget: Immutable config with GPU/CPU limits
      - TrackedModel: Model state with LRU ordering
      - PressureLevel: LOW/MODERATE/HIGH/CRITICAL thresholds
    </classes>
    <key-methods>
      - request_memory(): Check/make room before load
      - mark_loaded()/mark_unloaded(): Track model state
      - register_callbacks(): Wire up offload/unload handlers
    </key-methods>
  </module>

  <module path="cpu_offloader.py">
    <purpose>GPU-CPU model transfer for warm pool</purpose>
    <classes>
      - ModelOffloader: Singleton managing offloaded models
      - OffloadMetadata: Tracks original device and timing
      - MemoryEfficientInference: Context manager for low-memory inference
    </classes>
    <restore-time>2-5s from CPU vs 10-30s disk reload</restore-time>
  </module>

  <module path="reranker.py">
    <purpose>Cross-encoder reranking using Qwen3-Reranker models</purpose>
    <class>CrossEncoderReranker</class>
    <technique>Predicts P(yes) vs P(no) for query-document relevance</technique>
    <optimization>Uses logits_to_keep=1 to reduce VRAM for final token only</optimization>
  </module>

  <module path="sparse.py">
    <purpose>Sparse vector operations for Qdrant (BM25/SPLADE storage)</purpose>
    <key-functions>
      - ensure_sparse_collection(): Create sparse Qdrant collection if not exists
      - upsert_sparse_vectors(): Upsert sparse vectors to Qdrant
      - search_sparse_collection(): Search sparse collection with query vector
      - delete_sparse_vectors(): Delete sparse vectors by chunk_id
      - delete_sparse_collection(): Delete entire sparse collection
      - generate_sparse_collection_name(): Generate naming convention (e.g., "docs_sparse_bm25")
    </key-functions>
    <note>Plugins generate sparse vectors; this module handles storage and retrieval</note>
  </module>

  <module path="embed_chunks_unified.py">
    <purpose>CLI tool for batch embedding generation on parquet files</purpose>
    <usage>python -m vecpipe.embed_chunks_unified -i /input -o /output</usage>
    <mode>Uses EmbeddingMode.DOCUMENT for indexing</mode>
  </module>

  <module path="qwen3_search_config.py">
    <purpose>Model recommendations, batch sizes, reranker mapping</purpose>
    <key-configs>
      - QWEN3_RERANKER_MAPPING: Embedding model -> matching reranker
      - RERANK_CONFIG: Candidate multiplier, min/max bounds
      - BATCH_CONFIGS: Optimal batch sizes per model/quantization
    </key-configs>
  </module>

  <module path="memory_utils.py">
    <purpose>Memory estimation and model suggestions</purpose>
    <key-data>MODEL_MEMORY_REQUIREMENTS dict with conservative MB estimates</key-data>
  </module>
</modules>

<api-endpoints>
  <search>
    <endpoint>POST /search</endpoint>
    <endpoint>GET /search</endpoint>
    <description>Unified search with search_type routing</description>
    <search-types>semantic, question, code, hybrid</search-types>
    <reranking>Optional via use_reranker=true</reranking>
  </search>

  <batch>
    <endpoint>POST /search/batch</endpoint>
    <description>Multiple queries in one request</description>
  </batch>

  <search-modes>
    <description>search_mode parameter controls sparse/hybrid search</description>
    <modes>
      - "dense": Dense vector search only (default)
      - "sparse": Sparse vector search only (BM25/SPLADE, if enabled)
      - "hybrid": Combined dense + sparse with RRF fusion
    </modes>
    <note>Falls back to dense with warning if sparse unavailable</note>
  </search-modes>

  <embed>
    <endpoint>POST /embed</endpoint>
    <description>Batch text embedding generation</description>
  </embed>

  <upsert>
    <endpoint>POST /upsert</endpoint>
    <description>Insert/update vectors to Qdrant</description>
    <note>wait parameter passed as URL query param per Qdrant REST spec</note>
  </upsert>

  <models>
    <endpoint>GET /models</endpoint>
    <endpoint>POST /models/load</endpoint>
    <endpoint>GET /models/suggest</endpoint>
    <description>Model listing, loading, and auto-suggestion based on GPU memory</description>
  </models>
</api-endpoints>

<memory-management>
  <strategy>GPUMemoryGovernor with LRU eviction and CPU offloading</strategy>

  <pressure-levels>
    | Level    | Threshold | Action                                |
    |----------|-----------|---------------------------------------|
    | LOW      | <60%      | No action                             |
    | MODERATE | 60-80%    | Offload models idle >120s             |
    | HIGH     | 80-90%    | Aggressively offload models idle >30s |
    | CRITICAL | >90%      | Force unload all idle models (5s grace)|
  </pressure-levels>

  <eviction-strategy>
    1. Check both tracked budget AND actual GPU memory (ground truth)
    2. Offload LRU idle models to CPU first (preserves warm state)
    3. If CPU full or disabled, fully unload LRU models
    4. force=True for explicit allocation requests bypasses 5s grace period
    5. Fail only if model won't fit even with empty GPU
  </eviction-strategy>

  <configuration>
    GPU limits:
      - GPU_MEMORY_RESERVE_PERCENT=0.10 (keep 10% free)
      - GPU_MEMORY_MAX_PERCENT=0.90 (never use >90%)
    CPU limits (warm pool):
      - CPU_MEMORY_RESERVE_PERCENT=0.20 (keep 20% free)
      - CPU_MEMORY_MAX_PERCENT=0.50 (max 50% for warm models)
    Behavior:
      - ENABLE_MEMORY_GOVERNOR=true (use GovernedModelManager)
      - ENABLE_CPU_OFFLOAD=true
      - EVICTION_IDLE_THRESHOLD_SECONDS=120
      - MODEL_UNLOAD_AFTER_SECONDS=300 (base inactivity timeout)
  </configuration>

  <vram-recommendations>
    - 24GB+ VRAM: Both 8B embedding + 8B reranker in float16
    - 16GB VRAM: Both 4B models in float16, occasional eviction
    - 8GB VRAM: 8B embedding int8 + 0.6B reranker, more aggressive eviction
  </vram-recommendations>
</memory-management>

<search-flow>
  1. Resolve collection: explicit > operation_uuid lookup > DEFAULT_COLLECTION
  2. Fetch collection metadata for model/quantization/instruction defaults
  3. Handle search_mode:
     - "dense": Standard dense vector search
     - "sparse": Sparse-only search (if sparse index exists), fallback to dense with warning
     - "hybrid": Dense + sparse search with RRF fusion (rrf_k parameter)
  4. Embed query using same model that indexed the collection
  5. Search Qdrant for top-k similar vectors
  6. Filter results by score_threshold (BEFORE reranking)
  7. If use_reranker=true:
     a. Fetch 5x candidates (min=20, max=200)
     b. Ensure content is available (fetch from Qdrant if needed)
     c. Rerank with matched Qwen3-Reranker model
  8. Return results with timing metadata
</search-flow>

<collection-resolution>
  <priority-order>
    1. Explicit collection parameter in request
    2. operation_uuid database lookup (returns collection.vector_store_name)
    3. DEFAULT_COLLECTION from settings
  </priority-order>
  <error-behavior>
    If operation_uuid provided but not found, returns HTTP 404 error.
  </error-behavior>
</collection-resolution>

<reranking>
  <model-matching>get_reranker_for_embedding_model() maps embedding to reranker by size</model-matching>
  <candidate-strategy>
    - multiplier: 5x requested k
    - bounds: min=20, max=200
  </candidate-strategy>
  <technique>
    Qwen3-Reranker is causal LM that predicts P(yes) token for relevance.
    Uses softmax over yes/no logits for 0-1 score.
  </technique>
</reranking>

<error-handling>
  <dimension-mismatch>HTTP 400 with expected_dimension, actual_dimension, suggestion</dimension-mismatch>
  <insufficient-memory>HTTP 507 Insufficient Storage with suggestion for smaller model/quantization</insufficient-memory>
  <qdrant-error>HTTP 502 with extracted Qdrant error message</qdrant-error>
  <timeout>HTTP 504 Gateway Timeout</timeout>
  <embedding-error>HTTP 503 Embedding service error</embedding-error>
</error-handling>

<performance>
  <targets>
    - Single search: <500ms p95
    - Batch embedding: >100 texts/second
    - Reranking: <200ms for 100 candidates
    - CPU restore: 2-5s (vs 10-30s disk reload)
  </targets>
  <optimizations>
    - Batch processing for embeddings
    - Connection pooling for Qdrant (httpx.AsyncClient)
    - Result caching (15 min TTL via search/cache.py)
    - Collection metadata caching to reduce Qdrant calls
    - logits_to_keep=1 for reranker memory efficiency
  </optimizations>
</performance>

<testing>
  <unit-tests>
    - tests/unit/test_embedding_*.py: Dimension validation, OOM handling, retry logic
    - tests/unit/test_memory_governor.py: Budget calculations, LRU eviction, pressure levels, callbacks
    - tests/unit/test_search_*.py: Search flow, collection resolution, reranking
  </unit-tests>
  <mock-mode>
    - USE_MOCK_EMBEDDINGS=true uses MockEmbeddingProvider
    - Mock reranking generates deterministic scores from query hash
    - Useful for testing without GPU
  </mock-mode>
</testing>

<critical-patterns>
  <cuda-sync>
    Always call torch.cuda.synchronize() before torch.cuda.empty_cache().
    CUDA operations are async - without sync, memory checks may be inaccurate.
  </cuda-sync>

  <governor-callbacks>
    Governor uses callbacks for offload/unload. Must register_callbacks() before memory operations.
    Callback signatures:
    - unload_fn: async fn(model_name, quantization)
    - offload_fn: async fn(model_name, quantization, target_device)
  </governor-callbacks>

  <model-key-format>
    Format: "model_name_quantization" (e.g., "Qwen/Qwen3-Embedding-0.6B_float16")
    Parse with rsplit("_", 1) - quantization values must not contain underscores.
  </model-key-format>

  <async-shutdown>
    Use shutdown_async() for GovernedModelManager in async contexts (FastAPI lifespan).
    Sync shutdown() may cause deadlock if called from running event loop.
  </async-shutdown>
</critical-patterns>

<common-pitfalls>
  <pitfall>Forgetting to await model_mgr.start() after creating GovernedModelManager</pitfall>
  <pitfall>Not checking force=True when evicting for explicit allocation requests</pitfall>
  <pitfall>Passing wait in JSON body instead of URL query param for upsert</pitfall>
  <pitfall>Using device_map="auto" for reranker breaks CPU offloading - use device_map={"": 0}</pitfall>
  <pitfall>Assuming score_threshold applies after reranking (it's before)</pitfall>
</common-pitfalls>

<development>
  <commands>
    - Run API: uvicorn vecpipe.search_api:app --host 0.0.0.0 --port 8001
    - CLI embedding: python -m vecpipe.embed_chunks_unified -i /input -o /output
    - Tests: pytest tests/unit/test_memory_governor.py -v
  </commands>
</development>
