# Cross-Encoder Reranking Implementation Plan

## Executive Summary

This document outlines the comprehensive plan for implementing true cross-encoder reranking in the document-embedding-project using Qwen3-Reranker models. The implementation will significantly improve search accuracy by re-scoring top candidates from the initial vector search using more sophisticated cross-encoder models.

## Architecture Overview

### Key Design Principles
1. **Architectural Purity**: Core reranking logic resides in the `vecpipe` package
2. **Resource Efficiency**: Lazy loading and automatic unloading of models
3. **Backward Compatibility**: Reranking is opt-in via configuration flags
4. **Model Alignment**: Reranker models match embedding model size and quantization

### Component Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │     │   WebUI API     │     │   VecPipe       │
│   (React)       │────▶│   (Proxy)       │────▶│   (Core)        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       │                         │                      │
       │                         │                      ├── search_api.py
       ├── SearchInterface.tsx   ├── api/search.py     ├── reranker.py (NEW)
       ├── searchStore.ts        │                      ├── model_manager.py
       └── api.ts                │                      └── qwen3_search_config.py
```

## Detailed Implementation Plan

### Phase 1: Backend Core Implementation (vecpipe)

#### 1.1 Create Reranking Service (`packages/vecpipe/reranker.py`)

```python
"""
Cross-encoder reranking service for improved search relevance
"""

from typing import List, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Handles cross-encoder reranking using Qwen3-Reranker models"""
    
    def __init__(self, model_name: str, device: str = "cuda", quantization: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the reranker model with appropriate configuration"""
        # Implementation details...
        
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int,
        instruction: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query
        
        Returns:
            List of (index, score) tuples sorted by relevance
        """
        # Implementation details...
```

#### 1.2 Update Configuration (`packages/vecpipe/qwen3_search_config.py`)

Add the following configuration:

```python
# Qwen3 Reranker model mapping
QWEN3_RERANKER_MAPPING = {
    "Qwen/Qwen3-Embedding-0.6B": "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Embedding-4B": "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Embedding-8B": "Qwen/Qwen3-Reranker-8B",
}

# Updated reranking configuration
RERANK_CONFIG = {
    "enabled": True,
    "top_k_candidates": 50,  # Retrieve more candidates for reranking
    "final_k": 10,  # Return top-k after reranking
    "default_model": "Qwen/Qwen3-Reranker-0.6B",
    "use_hybrid_scoring": True,
    "batch_size": {
        "Qwen/Qwen3-Reranker-0.6B": {"float16": 128, "int8": 256},
        "Qwen/Qwen3-Reranker-4B": {"float16": 32, "int8": 64},
        "Qwen/Qwen3-Reranker-8B": {"float16": 16, "int8": 32},
    }
}

# Reranking instructions for different domains
RERANKING_INSTRUCTIONS = {
    "general": "Given the query and document, determine if the document is relevant.",
    "technical": "Assess if this technical document answers the technical query.",
    "code": "Determine if this code snippet is relevant to the programming query.",
    "qa": "Check if this document contains information that answers the question.",
}
```

#### 1.3 Extend Model Manager (`packages/vecpipe/model_manager.py`)

Add reranker support to the existing ModelManager:

```python
class ModelManager:
    def __init__(self, unload_after_seconds: int = 300):
        # Existing code...
        self.reranker: CrossEncoderReranker | None = None
        self.current_reranker_key: str | None = None
        
    def ensure_reranker_loaded(self, model_name: str, quantization: str) -> bool:
        """Ensure the specified reranker model is loaded"""
        # Implementation...
        
    async def rerank_async(
        self,
        query: str,
        documents: List[str],
        top_k: int,
        model_name: str,
        quantization: str,
        instruction: Optional[str] = None
    ) -> List[Tuple[int, float]]:
        """Perform async reranking with lazy model loading"""
        # Implementation...
```

#### 1.4 Update Search API (`packages/vecpipe/search_api.py`)

Modify the SearchRequest model and search endpoint:

```python
class SearchRequest(BaseModel):
    # Existing fields...
    use_reranker: bool = Field(False, description="Enable cross-encoder reranking")
    rerank_top_k: int = Field(50, description="Number of candidates to retrieve for reranking")
    reranker_model: str | None = Field(None, description="Override reranker model")

class SearchResponse(BaseModel):
    # Existing fields...
    reranking_used: bool | None = None
    reranker_model: str | None = None
    reranking_time_ms: float | None = None
```

### Phase 2: WebUI Backend Integration

#### 2.1 Update Search Proxy (`packages/webui/api/search.py`)

Add reranking parameters to the SearchRequest:

```python
class SearchRequest(BaseModel):
    # Existing fields...
    use_reranker: bool = Field(False, description="Enable cross-encoder reranking")
    rerank_top_k: int = Field(50, description="Candidates for reranking")
```

### Phase 3: Frontend Implementation

#### 3.1 Update Search Store (`apps/webui-react/src/stores/searchStore.ts`)

```typescript
export interface SearchParams {
  // Existing fields...
  useReranker: boolean;
  rerankTopK?: number;
}

// Update default state
searchParams: {
  // Existing defaults...
  useReranker: false,
  rerankTopK: 50,
}
```

#### 3.2 Update Search Interface (`apps/webui-react/src/components/SearchInterface.tsx`)

Add reranking UI controls after hybrid search options:

```tsx
{/* Reranking Options */}
<div className="mt-4 p-4 bg-gray-50 rounded-lg">
  <label className="flex items-center space-x-2">
    <input
      type="checkbox"
      checked={searchParams.useReranker}
      onChange={(e) => updateSearchParams({ useReranker: e.target.checked })}
      className="w-4 h-4 text-blue-600"
    />
    <span className="text-sm font-medium text-gray-700">
      Enable Reranking
    </span>
  </label>
  
  {searchParams.useReranker && (
    <div className="mt-2 ml-6">
      <p className="text-xs text-gray-600 mb-2">
        Reranking improves result accuracy by using a more sophisticated model 
        to re-score the top candidates. This may increase search latency.
      </p>
      <label className="block text-xs text-gray-700">
        Candidates to rerank: {searchParams.rerankTopK}
        <input
          type="range"
          min="20"
          max="100"
          step="10"
          value={searchParams.rerankTopK}
          onChange={(e) => updateSearchParams({ rerankTopK: parseInt(e.target.value) })}
          className="w-full mt-1"
        />
      </label>
    </div>
  )}
</div>
```

#### 3.3 Update API Service (`apps/webui-react/src/services/api.ts`)

Include reranking parameters in search requests:

```typescript
search: async (params: SearchParams) => {
  const response = await axiosInstance.post('/api/search', {
    // Existing parameters...
    use_reranker: params.useReranker,
    rerank_top_k: params.rerankTopK,
  });
  return response;
}
```

## Implementation Timeline

### Week 1: Backend Core
- Day 1-2: Implement CrossEncoderReranker class
- Day 3-4: Extend ModelManager with reranker support
- Day 5: Update search_api.py with reranking flow

### Week 2: Integration & Frontend
- Day 1-2: Update webui proxy layer
- Day 3-4: Implement frontend UI components
- Day 5: Integration testing

### Week 3: Testing & Optimization
- Day 1-2: Unit tests for reranking
- Day 3-4: Performance optimization
- Day 5: Documentation and deployment

## Technical Considerations

### Memory Management
1. **Model Size Requirements**:
   - Qwen3-Reranker-0.6B: ~2GB VRAM (fp16)
   - Qwen3-Reranker-4B: ~8GB VRAM (fp16)
   - Qwen3-Reranker-8B: ~16GB VRAM (fp16)

2. **Optimization Strategies**:
   - Lazy loading with automatic unloading
   - Quantization support (int8 reduces memory by ~50%)
   - Batch processing with adaptive sizing

### Performance Impact
1. **Latency Addition**:
   - 0.6B model: +100-200ms
   - 4B model: +200-400ms
   - 8B model: +400-800ms

2. **Throughput Optimization**:
   - Batch multiple queries when possible
   - Cache reranking results
   - Use smaller models for real-time applications

### Error Handling
1. **Graceful Degradation**:
   - Fall back to vector-only search if reranking fails
   - Reduce batch size on OOM errors
   - Log all errors for monitoring

2. **User Communication**:
   - Show reranking status in UI
   - Display warnings for high latency
   - Provide clear error messages

## Testing Strategy

### Unit Tests
1. Test reranker model loading/unloading
2. Test batch processing with various sizes
3. Test score normalization and ranking
4. Test error handling and fallbacks

### Integration Tests
1. End-to-end search with reranking
2. Performance benchmarks
3. Memory usage monitoring
4. Concurrent request handling

### Acceptance Criteria
1. ✅ Reranking improves search relevance by >20% on test queries
2. ✅ Latency increase is within acceptable bounds (<1s for 95th percentile)
3. ✅ Memory usage is properly managed with no leaks
4. ✅ UI clearly indicates when reranking is active
5. ✅ System gracefully handles errors without disrupting service

## Risk Mitigation

### Technical Risks
1. **Memory Overflow**: Mitigated by adaptive batch sizing and model unloading
2. **High Latency**: Mitigated by model selection and caching
3. **Model Compatibility**: Mitigated by extensive testing with different models

### Operational Risks
1. **User Confusion**: Mitigated by clear UI and documentation
2. **Resource Costs**: Mitigated by efficient resource management
3. **Backward Compatibility**: Mitigated by opt-in design

## Success Metrics

1. **Search Quality**: 20%+ improvement in relevance metrics
2. **User Satisfaction**: Positive feedback on search results
3. **System Performance**: <1s p95 latency with reranking
4. **Resource Efficiency**: <20% increase in GPU memory usage

## Future Enhancements

1. **Multi-stage Reranking**: Chain multiple rerankers for better accuracy
2. **Custom Reranker Training**: Fine-tune on domain-specific data
3. **API-based Rerankers**: Support for Cohere, OpenAI reranking APIs
4. **Progressive Results**: Show initial results while reranking continues
5. **Smart Caching**: Cache reranked results for common queries

---

This implementation plan provides a comprehensive roadmap for adding true cross-encoder reranking to the document-embedding-project. The phased approach ensures minimal disruption while delivering significant search quality improvements.