# SPLADE Implementation Plan for Semantik

## Executive Summary

This document outlines a comprehensive, phased implementation plan for integrating SPLADE (Sparse Lexical and Expansion Model) into Semantik's search architecture. The implementation leverages Qdrant's native sparse vector support (v1.7+) to provide state-of-the-art hybrid search capabilities without additional infrastructure dependencies.

## Table of Contents

1. [Technical Overview](#technical-overview)
2. [Architecture Decisions](#architecture-decisions)
3. [Implementation Phases](#implementation-phases)
4. [Technical Specifications](#technical-specifications)
5. [Performance Considerations](#performance-considerations)
6. [Risk Mitigation](#risk-mitigation)
7. [Testing Strategy](#testing-strategy)
8. [Timeline Estimates](#timeline-estimates)

---

## Technical Overview

### SPLADE Model Selection

Based on our research, we recommend:
- **Primary Model**: `naver/splade-cocondenser-ensembledistil` 
  - MRR@10: 38.3 on MS MARCO
  - R@1000: 98.3
  - Optimal balance of performance and efficiency
  
- **Alternative Models** (for comparison):
  - `naver/splade-v3`: Latest version with 40+ MRR@10
  - `prithivida/Splade_PP_en_v1`: Community optimized version

### Key Technical Requirements

1. **GPU Memory**: 8-16GB VRAM for inference (similar to current Qwen models)
2. **Storage**: ~30% increase due to sparse vectors (mitigated by sparsity)
3. **Vocabulary Size**: 30,522 dimensions (BERT tokenizer)
4. **Typical Sparsity**: 20-200 non-zero values per document

---

## Architecture Decisions

### 1. Storage Strategy: Dual Vector Approach

We will store both dense (Qwen) and sparse (SPLADE) vectors in the same Qdrant collection:

```python
{
    "vectors": {
        "dense": {
            "size": 896,  # Qwen dimension
            "distance": "Cosine"
        }
    },
    "sparse_vectors": {
        "sparse": {
            # No configuration needed, Qdrant handles dynamically
        }
    }
}
```

### 2. Fusion Strategy: Reciprocal Rank Fusion (RRF)

After extensive research, RRF is recommended for its:
- Robustness to score distribution differences
- Simplicity and proven effectiveness
- No normalization requirements

Formula: `score = Σ(1/(k + rank_i))` where k=60

### 3. Model Management Strategy

Extend existing `ModelManager` to handle dual model loading:
- Keep dense model in memory (current behavior)
- Load SPLADE on-demand for indexing
- Share tokenizer between models when possible

---

## Implementation Phases

### Phase 0: Architecture Cleanup (Pre-requisite)
**Duration**: 3-4 days

**Objective**: Resolve cross-package dependencies before SPLADE integration

**Tasks**:
1. Extract `EmbeddingService` to a shared `packages/core/` package
2. Create API endpoint for cleanup.py to query collections
3. Standardize configuration across packages
4. Update import paths and dependencies

**Deliverables**:
- Clean separation between vecpipe and webui
- Shared core package with embedding functionality
- Updated tests passing

---

### Phase 1: SPLADE Model Support
**Duration**: 5-7 days

**Objective**: Add SPLADE model loading and inference capabilities

**Tasks**:

1. **Extend Model Detection** (`embedding_service.py`):
```python
def _detect_model_type(self, model_name: str) -> ModelType:
    if "Qwen3-Embedding" in model_name:
        return ModelType.QWEN3
    elif "splade" in model_name.lower():
        return ModelType.SPLADE
    else:
        return ModelType.SENTENCE_TRANSFORMER
```

2. **Add SPLADE Model Loading**:
```python
def _load_splade_model(self, model_name: str, quantization: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if quantization == "int8":
        # Use existing quantization config
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            quantization_config=self._get_quantization_config(),
            device_map="auto"
        )
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        if quantization == "float16":
            model = model.half()
        model = model.to(self.device)
    
    return model, tokenizer
```

3. **Implement Sparse Vector Generation**:
```python
def _generate_splade_vectors(self, texts: List[str], batch_size: int) -> List[SparseVector]:
    all_sparse_vectors = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize
        inputs = self.current_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate SPLADE representations
        with torch.no_grad():
            outputs = self.current_model(**inputs)
            
            # Apply log(1 + ReLU(logits))
            logits = outputs.logits
            relu_log = torch.log(1 + torch.relu(logits))
            
            # Max pooling over sequence length
            weighted_log = relu_log * inputs.attention_mask.unsqueeze(-1)
            max_values, _ = torch.max(weighted_log, dim=1)
            
            # Convert to sparse format
            for vec in max_values:
                # Get non-zero indices and values
                nonzero_indices = torch.nonzero(vec).squeeze(-1)
                values = vec[nonzero_indices]
                
                # Filter by threshold for sparsity
                mask = values > 0.01  # Configurable threshold
                indices = nonzero_indices[mask].cpu().tolist()
                values = values[mask].cpu().tolist()
                
                all_sparse_vectors.append(SparseVector(
                    indices=indices,
                    values=values
                ))
    
    return all_sparse_vectors
```

4. **Add Model Registry Entry**:
```python
QUANTIZED_MODEL_INFO["naver/splade-cocondenser-ensembledistil"] = {
    "embedding_dim": 30522,  # BERT vocab size
    "description": "SPLADE sparse neural retrieval model",
    "max_sequence_length": 256,
    "is_sparse": True
}
```

**Deliverables**:
- SPLADE model loading with quantization support
- Sparse vector generation with configurable sparsity
- Unit tests for new functionality

---

### Phase 2: Data Structures and Storage
**Duration**: 3-4 days

**Objective**: Update data structures to support sparse vectors

**Tasks**:

1. **Define Sparse Vector Types**:
```python
from dataclasses import dataclass
from typing import List, Union

@dataclass
class SparseVector:
    indices: List[int]
    values: List[float]
    
    def to_qdrant_format(self) -> dict:
        return {
            "indices": self.indices,
            "values": self.values
        }

@dataclass
class HybridEmbedding:
    dense: np.ndarray
    sparse: SparseVector
    model_info: dict
```

2. **Update Collection Creation**:
```python
async def create_hybrid_collection(
    self,
    collection_name: str,
    dense_model: str,
    sparse_model: str,
    dense_size: int
):
    await self.qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
```

3. **Modify Point Insertion**:
```python
def create_hybrid_point(
    id: str,
    dense_vector: np.ndarray,
    sparse_vector: SparseVector,
    payload: dict
) -> models.PointStruct:
    return models.PointStruct(
        id=id,
        vector={
            "dense": dense_vector.tolist(),
            "sparse": sparse_vector.to_qdrant_format()
        },
        payload=payload
    )
```

**Deliverables**:
- Sparse vector data structures
- Updated database schema
- Migration utilities for existing collections

---

### Phase 3: Dual Embedding Pipeline
**Duration**: 4-5 days

**Objective**: Implement parallel generation of dense and sparse embeddings

**Tasks**:

1. **Create Hybrid Embedding Service**:
```python
class HybridEmbeddingService:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.dense_model = None
        self.sparse_model = None
        
    async def generate_hybrid_embeddings(
        self,
        texts: List[str],
        dense_model: str = "Qwen/Qwen3-Embedding-0.6B",
        sparse_model: str = "naver/splade-cocondenser-ensembledistil",
        quantization: str = "float16"
    ) -> List[HybridEmbedding]:
        # Generate embeddings in parallel when possible
        if self._can_load_both_models():
            dense_task = self._generate_dense_async(texts, dense_model, quantization)
            sparse_task = self._generate_sparse_async(texts, sparse_model, quantization)
            
            dense_vecs, sparse_vecs = await asyncio.gather(dense_task, sparse_task)
        else:
            # Sequential generation for memory-constrained systems
            dense_vecs = await self._generate_dense_async(texts, dense_model, quantization)
            sparse_vecs = await self._generate_sparse_async(texts, sparse_model, quantization)
        
        return [
            HybridEmbedding(dense=d, sparse=s, model_info={
                "dense_model": dense_model,
                "sparse_model": sparse_model,
                "quantization": quantization
            })
            for d, s in zip(dense_vecs, sparse_vecs)
        ]
```

2. **Update Document Processing Pipeline**:
```python
async def process_documents_hybrid(
    self,
    documents: List[Document],
    job_id: str,
    batch_size: int = 32
):
    collection_name = f"job_{job_id}"
    
    # Create hybrid collection
    await self.create_hybrid_collection(
        collection_name,
        self.dense_model,
        self.sparse_model,
        self.dense_dim
    )
    
    # Process in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.content for doc in batch]
        
        # Generate hybrid embeddings
        hybrid_embeddings = await self.hybrid_service.generate_hybrid_embeddings(
            texts,
            self.dense_model,
            self.sparse_model,
            self.quantization
        )
        
        # Create points
        points = [
            self.create_hybrid_point(
                id=doc.id,
                dense_vector=emb.dense,
                sparse_vector=emb.sparse,
                payload=doc.metadata
            )
            for doc, emb in zip(batch, hybrid_embeddings)
        ]
        
        # Insert to Qdrant
        await self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
```

**Deliverables**:
- Hybrid embedding generation pipeline
- Parallel/sequential processing based on resources
- Updated document ingestion flow

---

### Phase 4: Hybrid Search Implementation
**Duration**: 5-6 days

**Objective**: Implement RRF-based hybrid search

**Tasks**:

1. **Implement RRF Fusion**:
```python
class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        dense_results: List[ScoredPoint],
        sparse_results: List[ScoredPoint]
    ) -> List[ScoredPoint]:
        # Create rank mappings
        dense_ranks = {point.id: rank + 1 for rank, point in enumerate(dense_results)}
        sparse_ranks = {point.id: rank + 1 for rank, point in enumerate(sparse_results)}
        
        # Get all unique IDs
        all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_ranks:
                score += 1.0 / (self.k + dense_ranks[doc_id])
            if doc_id in sparse_ranks:
                score += 1.0 / (self.k + sparse_ranks[doc_id])
            rrf_scores[doc_id] = score
        
        # Sort by RRF score and return
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Reconstruct ScoredPoint objects
        id_to_point = {p.id: p for p in dense_results + sparse_results}
        return [
            ScoredPoint(
                id=doc_id,
                score=score,
                payload=id_to_point[doc_id].payload,
                vector=id_to_point[doc_id].vector
            )
            for doc_id, score in sorted_results
        ]
```

2. **Update Search API**:
```python
async def hybrid_search(
    self,
    query: str,
    collection_name: str,
    limit: int = 10,
    fusion_method: str = "rrf",
    dense_weight: float = 0.5,  # For weighted fusion
    rerank: bool = False
) -> List[SearchResult]:
    # Get collection metadata
    collection_info = await self.get_collection_info(collection_name)
    
    # Generate query embeddings
    dense_query = await self.generate_dense_embedding(
        query,
        collection_info["dense_model"]
    )
    
    sparse_query = await self.generate_sparse_embedding(
        query,
        collection_info["sparse_model"]
    )
    
    # Perform searches (in parallel)
    search_tasks = [
        self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=("dense", dense_query.tolist()),
            limit=limit * 3  # Over-fetch for fusion
        ),
        self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=("sparse", sparse_query.to_qdrant_format()),
            limit=limit * 3
        )
    ]
    
    dense_results, sparse_results = await asyncio.gather(*search_tasks)
    
    # Apply fusion
    if fusion_method == "rrf":
        fused_results = self.rrf_fusion.fuse(dense_results, sparse_results)
    else:
        # Weighted linear combination
        fused_results = self.weighted_fusion(
            dense_results,
            sparse_results,
            dense_weight
        )
    
    # Optional reranking
    if rerank:
        fused_results = await self.rerank_results(query, fused_results[:limit*2])
    
    return fused_results[:limit]
```

3. **Add Configuration Options**:
```python
class HybridSearchConfig(BaseModel):
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_threshold: float = 0.01  # Minimum sparse value to keep
    enable_reranking: bool = True
    rerank_top_k: int = 50
```

**Deliverables**:
- RRF and weighted fusion implementations
- Updated search API with hybrid support
- Configuration management
- Performance benchmarks

---

### Phase 5: API and Frontend Updates
**Duration**: 3-4 days

**Objective**: Expose hybrid search through API and update UI

**Tasks**:

1. **Update Search Endpoint**:
```python
@router.post("/search/hybrid")
async def hybrid_search_endpoint(
    request: HybridSearchRequest,
    current_user: User = Depends(get_current_user)
):
    results = await search_service.hybrid_search(
        query=request.query,
        collection_name=request.collection_name,
        limit=request.limit,
        fusion_method=request.fusion_method,
        dense_weight=request.dense_weight,
        rerank=request.enable_reranking
    )
    
    return {
        "results": results,
        "metadata": {
            "fusion_method": request.fusion_method,
            "models": {
                "dense": collection_info["dense_model"],
                "sparse": collection_info["sparse_model"]
            }
        }
    }
```

2. **Add Model Selection UI**:
```typescript
interface EmbeddingConfig {
  denseModel: string;
  sparseModel: string;
  enableHybrid: boolean;
  fusionMethod: 'rrf' | 'weighted';
  denseWeight?: number;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({ onConfigChange }) => {
  const [config, setConfig] = useState<EmbeddingConfig>({
    denseModel: 'Qwen/Qwen3-Embedding-0.6B',
    sparseModel: 'naver/splade-cocondenser-ensembledistil',
    enableHybrid: true,
    fusionMethod: 'rrf'
  });
  
  // UI implementation
};
```

3. **Update Job Creation Flow**:
   - Add model selection to job creation form
   - Store hybrid configuration in job metadata
   - Display search method in job details

**Deliverables**:
- API endpoints for hybrid search
- Updated frontend components
- User documentation

---

### Phase 6: Performance Optimization
**Duration**: 4-5 days

**Objective**: Optimize performance and resource usage

**Tasks**:

1. **Implement Sparse Vector Caching**:
```python
class SparseVectorCache:
    def __init__(self, max_size: int = 10000):
        self.cache = LRUCache(max_size)
        
    def get_or_compute(
        self,
        text: str,
        model_name: str,
        compute_fn: Callable
    ) -> SparseVector:
        cache_key = f"{model_name}:{hash(text)}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        sparse_vec = compute_fn(text)
        self.cache[cache_key] = sparse_vec
        return sparse_vec
```

2. **Optimize Batch Processing**:
   - Implement dynamic batch sizing based on text length
   - Use gradient accumulation for large batches
   - Profile and optimize GPU memory usage

3. **Add Monitoring and Metrics**:
```python
class HybridSearchMetrics:
    def __init__(self):
        self.dense_latency = Histogram("hybrid_dense_search_latency")
        self.sparse_latency = Histogram("hybrid_sparse_search_latency")
        self.fusion_latency = Histogram("hybrid_fusion_latency")
        self.total_latency = Histogram("hybrid_total_latency")
        
    async def track_search(self, search_fn):
        start = time.time()
        try:
            result = await search_fn()
            self.total_latency.observe(time.time() - start)
            return result
        except Exception as e:
            self.search_errors.inc()
            raise
```

**Deliverables**:
- Caching layer for sparse vectors
- Optimized batch processing
- Performance monitoring
- Benchmark results

---

## Technical Specifications

### Data Flow Diagram

```
User Query
    |
    v
[Search API]
    |
    ├─> [Dense Embedding Generation]
    |         |
    |         v
    |    [Qwen Model]
    |
    └─> [Sparse Embedding Generation]
              |
              v
         [SPLADE Model]
              |
    ┌─────────┴─────────┐
    |                   |
    v                   v
[Dense Search]    [Sparse Search]
    |                   |
    └─────────┬─────────┘
              |
              v
        [RRF Fusion]
              |
              v
    [Optional Reranking]
              |
              v
        [Results]
```

### Configuration Schema

```yaml
# config/hybrid_search.yaml
models:
  dense:
    default: "Qwen/Qwen3-Embedding-0.6B"
    alternatives:
      - "sentence-transformers/all-MiniLM-L6-v2"
      - "BAAI/bge-large-en-v1.5"
  
  sparse:
    default: "naver/splade-cocondenser-ensembledistil"
    alternatives:
      - "naver/splade-v3"
      - "prithivida/Splade_PP_en_v1"

fusion:
  method: "rrf"  # or "weighted"
  rrf_k: 60
  weighted_alpha: 0.7  # Dense weight

sparsity:
  threshold: 0.01
  max_terms: 200

performance:
  cache_size: 10000
  batch_size: 32
  parallel_models: false  # Load both models simultaneously
```

---

## Performance Considerations

### Expected Performance Metrics

Based on research and benchmarks:

1. **Indexing Performance**:
   - Dense only: ~1000 docs/sec
   - Hybrid (sequential): ~400 docs/sec
   - Hybrid (parallel): ~700 docs/sec

2. **Search Latency**:
   - Dense only: ~10-20ms
   - Hybrid with RRF: ~25-40ms
   - Hybrid with reranking: ~50-100ms

3. **Memory Usage**:
   - Dense model: ~2-4GB
   - SPLADE model: ~2-3GB
   - Total with caching: ~6-8GB

4. **Storage Impact**:
   - Dense vectors: 896 * 4 bytes = 3.5KB per doc
   - Sparse vectors: ~100 * 8 bytes = 0.8KB per doc
   - Total increase: ~25%

### Optimization Strategies

1. **GPU Memory Management**:
   - Use model offloading for memory-constrained systems
   - Implement gradient checkpointing for large batches
   - Dynamic batch size adjustment

2. **Latency Optimization**:
   - Cache frequent queries
   - Pre-compute sparse vectors for static content
   - Use async processing where possible

3. **Storage Optimization**:
   - Compress sparse indices using variable-length encoding
   - Prune low-value sparse entries
   - Use quantization for sparse values

---

## Risk Mitigation

### Technical Risks

1. **Memory Constraints**:
   - **Risk**: OOM when loading both models
   - **Mitigation**: Sequential loading, model offloading

2. **Performance Degradation**:
   - **Risk**: Slower search due to dual retrieval
   - **Mitigation**: Caching, parallel processing, optimization

3. **Model Compatibility**:
   - **Risk**: SPLADE models may not support all quantization levels
   - **Mitigation**: Fallback to FP16, extensive testing

### Mitigation Strategies

1. **Gradual Rollout**:
   - Feature flag for hybrid search
   - A/B testing with subset of users
   - Monitoring and rollback plan

2. **Fallback Mechanisms**:
   - Automatic fallback to dense-only search
   - Configurable timeouts
   - Error handling and recovery

3. **Resource Monitoring**:
   - Real-time GPU memory tracking
   - Alert on performance degradation
   - Automatic scaling based on load

---

## Testing Strategy

### Unit Tests

1. **Model Loading Tests**:
   - Test all SPLADE model variants
   - Verify quantization support
   - Memory usage validation

2. **Sparse Vector Generation**:
   - Verify sparsity levels
   - Test edge cases (empty text, long text)
   - Validate output format

3. **Fusion Algorithm Tests**:
   - RRF correctness
   - Weighted fusion accuracy
   - Edge cases (no results, identical results)

### Integration Tests

1. **End-to-End Pipeline**:
   - Document ingestion with hybrid embeddings
   - Search with various configurations
   - Performance benchmarks

2. **API Tests**:
   - Hybrid search endpoint
   - Error handling
   - Authentication and authorization

3. **Compatibility Tests**:
   - Existing collections still work
   - Migration tools function correctly
   - No regression in dense-only search

### Performance Tests

1. **Load Testing**:
   - Concurrent indexing operations
   - High-volume search queries
   - Memory usage under load

2. **Benchmark Suite**:
   - Compare with dense-only baseline
   - Measure latency percentiles
   - Storage efficiency metrics

---

## Timeline Estimates

### Total Duration: 6-8 weeks

| Phase | Duration | Dependencies | Team Size |
|-------|----------|--------------|-----------|
| Phase 0 | 3-4 days | None | 1-2 devs |
| Phase 1 | 5-7 days | Phase 0 | 2 devs |
| Phase 2 | 3-4 days | Phase 1 | 1-2 devs |
| Phase 3 | 4-5 days | Phase 2 | 2 devs |
| Phase 4 | 5-6 days | Phase 3 | 2-3 devs |
| Phase 5 | 3-4 days | Phase 4 | 2 devs |
| Phase 6 | 4-5 days | Phase 5 | 1-2 devs |

### Milestones

1. **Week 1-2**: Architecture cleanup and SPLADE model support
2. **Week 3-4**: Storage and embedding pipeline implementation
3. **Week 5-6**: Hybrid search and API integration
4. **Week 7-8**: Frontend updates and optimization

### Critical Path

1. Architecture cleanup (Phase 0) - Blocks all other work
2. Model support (Phase 1) - Core functionality
3. Hybrid search (Phase 4) - User-facing feature
4. Performance optimization (Phase 6) - Production readiness

---

## Success Criteria

1. **Functional Requirements**:
   - ✓ SPLADE models load and generate sparse vectors
   - ✓ Hybrid search returns relevant results
   - ✓ RRF fusion improves search quality
   - ✓ No regression in existing functionality

2. **Performance Requirements**:
   - ✓ Search latency < 50ms for 95th percentile
   - ✓ Indexing throughput > 400 docs/sec
   - ✓ Memory usage < 8GB under normal load

3. **Quality Metrics**:
   - ✓ 20%+ improvement in search relevance (A/B test)
   - ✓ Reduced "no results" queries by 30%+
   - ✓ User satisfaction score improvement

---

## Conclusion

This implementation plan provides a comprehensive roadmap for integrating SPLADE into Semantik. The phased approach ensures minimal disruption while delivering state-of-the-art hybrid search capabilities. By leveraging Qdrant's native sparse vector support and following proven architectural patterns, we can achieve this enhancement with confidence and efficiency.

The key to success will be careful attention to the architecture cleanup phase, thorough testing at each stage, and continuous performance monitoring. With proper execution, this implementation will position Semantik at the forefront of semantic search technology.