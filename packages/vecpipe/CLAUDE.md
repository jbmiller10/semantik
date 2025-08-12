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
    <purpose>Model lifecycle management</purpose>
    <features>
      - Lazy loading on first use
      - Auto-unload after 300s idle
      - GPU memory checking before load
    </features>
  </module>
</key-modules>

<memory-management>
  <strategy>Dynamic with auto-unloading</strategy>
  <limits>
    - Check available GPU memory before loading
    - Unload models after 5 minutes idle
    - Fall back to CPU if GPU unavailable
  </limits>
</memory-management>

<search-flow>
  1. Embed query using same model as indexing
  2. Search Qdrant for top-k similar vectors
  3. Optional: Rerank with cross-encoder
  4. Return results with scores
</search-flow>

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
</testing>