# Local LLM Guide

Run LLM inference locally without API costs or data leaving your machine.

## Overview

Local LLM inference allows you to use language models directly on your GPU, providing:

- **Privacy**: All data stays on your machine - no external API calls
- **Cost**: No per-token API charges after initial setup
- **Offline**: Works without internet once models are downloaded
- **Control**: Full control over model selection and quantization

**Trade-offs**:
- Requires NVIDIA GPU with CUDA support
- First-time model download can take several minutes
- Limited by available VRAM
- Lower throughput than cloud APIs for large workloads

## Architecture

```
┌─────────────────────┐     HTTP      ┌─────────────────────┐
│       WebUI         │ ────────────► │      VecPipe        │
│                     │               │                     │
│ LLMServiceFactory   │  /llm/generate│ LLMModelManager     │
│ creates             │  ◄────────────│ (governor callbacks)│
│ LocalLLMProvider    │               │                     │
│ (HTTP client)       │               │ HuggingFace model   │
│                     │               │ (on GPU)            │
└─────────────────────┘               └─────────────────────┘
```

**Key Components**:

| Component | Location | Role |
|-----------|----------|------|
| LocalLLMProvider | `shared/llm/providers/` | HTTP client to VecPipe |
| LLMModelManager | `vecpipe/` | GPU model lifecycle |
| GPUMemoryGovernor | `vecpipe/` | Shared memory management |
| Model Registry | `shared/llm/` | Available models and VRAM estimates |

## Requirements

### Hardware

- NVIDIA GPU with CUDA support (Ampere architecture recommended)
- Minimum 4GB VRAM (8GB+ recommended)
- SSD storage for model cache (models are 0.5-15GB each)

### VRAM Requirements

| Model | float16 | int8 | int4 | Recommended For |
|-------|---------|------|------|-----------------|
| Qwen 2.5 0.5B | 1.3GB | 0.8GB | 0.5GB | Simple tasks, low VRAM |
| Qwen 2.5 1.5B | 3.5GB | 2.0GB | 1.2GB | HyDE, query expansion |
| Qwen 2.5 3B | 7.0GB | 4.0GB | 2.5GB | Balanced quality/speed |
| Qwen 2.5 7B | 15.0GB | 8.5GB | 5.0GB | Best quality |

**Notes**:
- Memory estimates include KV cache buffer for typical request sizes
- INT4/INT8 quantization uses bitsandbytes
- Actual usage may vary based on prompt length and generation settings

### Software Dependencies

- CUDA toolkit (installed with Docker GPU support)
- bitsandbytes for INT4/INT8 quantization (included in requirements)
- transformers library (included in requirements)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_LOCAL_LLM` | `true` | Enable local LLM inference in VecPipe |
| `DEFAULT_LLM_QUANTIZATION` | `int8` | Default quantization (int4, int8, float16) |
| `LLM_UNLOAD_AFTER_SECONDS` | `300` | Idle timeout before model unload |
| `LLM_KV_CACHE_BUFFER_MB` | `1024` | KV cache memory buffer per model |
| `LLM_TRUST_REMOTE_CODE` | `false` | Allow models requiring trust_remote_code |

Set `ENABLE_LOCAL_LLM=false` if you don't have a compatible NVIDIA GPU.

### WebUI Settings

1. Navigate to **Settings** > **LLM**
2. For either quality tier (High/Low), select **Local (GPU)** as the provider
3. Choose a model from the dropdown (VRAM requirements shown)
4. Select quantization:
   - **INT8**: Better quality, moderate VRAM
   - **INT4**: Lower quality, minimal VRAM
5. Click **Save**

The first request after saving will trigger model download if not cached.

## Available Models

All local models are from the Qwen 2.5 Instruct family, optimized for chat/instruction following:

| Model | Tier | Context | Best For |
|-------|------|---------|----------|
| Qwen 2.5 0.5B | LOW | 32K | Quick responses, simple tasks |
| Qwen 2.5 1.5B | LOW | 32K | HyDE search, keyword extraction |
| Qwen 2.5 3B | HIGH | 32K | Summarization, good balance |
| Qwen 2.5 7B | HIGH | 32K | Complex reasoning, best quality |

**Tier Recommendations**:
- **LOW tier**: Use for HyDE query expansion, keyword extraction, simple reformulations
- **HIGH tier**: Use for document summarization, entity extraction, complex analysis

## Usage Patterns

### Quality Tier Selection

The LLM system uses two quality tiers to balance cost and capability:

```python
from shared.llm.factory import LLMServiceFactory
from shared.llm.types import LLMQualityTier

factory = LLMServiceFactory(session)

# LOW tier - fast, cost-effective (HyDE, keywords)
provider = await factory.create_provider_for_tier(
    user_id=user_id,
    quality_tier=LLMQualityTier.LOW,
)

# HIGH tier - best quality (summarization)
provider = await factory.create_provider_for_tier(
    user_id=user_id,
    quality_tier=LLMQualityTier.HIGH,
)
```

### Generation Example

```python
from shared.llm.factory import LLMServiceFactory
from shared.llm.types import LLMQualityTier

async def generate_hyde_passage(session, user_id: int, query: str) -> str:
    factory = LLMServiceFactory(session)
    provider = await factory.create_provider_for_tier(user_id, LLMQualityTier.LOW)

    async with provider:
        response = await provider.generate(
            prompt=f"Write a passage that would answer: {query}",
            system_prompt="You are a helpful assistant.",
            max_tokens=256,
            temperature=0.7,
        )
        return response.content
```

### Hybrid Setup (Local + Cloud)

You can configure different providers per tier:

- **HIGH tier**: Cloud provider (Anthropic/OpenAI) for complex tasks
- **LOW tier**: Local model for frequent, simple tasks

This balances cost and quality - local models handle high-volume simple tasks while cloud APIs handle complex reasoning.

## API Endpoints (Internal)

VecPipe exposes these endpoints for local LLM inference. They require `X-Internal-Api-Key` header (except health/models list).

### POST /llm/generate

Generate text using a local model.

```json
{
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "quantization": "int8",
  "prompts": ["Write about machine learning"],
  "system_prompt": "You are helpful.",
  "temperature": 0.7,
  "max_tokens": 256
}
```

**Response**:
```json
{
  "request_id": "uuid",
  "contents": ["Machine learning is..."],
  "prompt_tokens": [12],
  "completion_tokens": [150],
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct"
}
```

### GET /llm/models

List available models with VRAM requirements.

### POST /llm/models/load

Pre-warm a model into GPU memory before expected usage.

### GET /llm/health

Health check returning loaded models and governor status.

## Memory Management

Local LLMs share GPU memory with embeddings and rerankers via the GPUMemoryGovernor.

### Eviction Behavior

- **LLMs are unload-only**: Unlike embeddings, LLMs cannot be offloaded to CPU warm pool (bitsandbytes limitation)
- **LRU eviction**: Least recently used models are unloaded when memory is needed
- **Idle timeout**: Models unload after `LLM_UNLOAD_AFTER_SECONDS` of inactivity

### Memory Pressure

| Level | Threshold | LLM Behavior |
|-------|-----------|--------------|
| LOW | <60% | No action |
| MODERATE | 60-80% | Unload if idle >120s |
| HIGH | 80-90% | Unload if idle >30s |
| CRITICAL | >90% | Force unload (5s grace) |

### Shared Memory Budget

When running embeddings, rerankers, and LLMs together:

1. Each model type competes for the same GPU memory budget
2. Governor uses LRU across all types for eviction decisions
3. Consider total VRAM when selecting models

**Example**: With 8GB VRAM, running a 4GB embedding model leaves ~4GB for LLM. Choose Qwen 2.5 3B (int8: 4GB) or smaller.

## Error Handling

| HTTP Status | Error | Cause | Solution |
|-------------|-------|-------|----------|
| 507 | Insufficient Storage | Not enough GPU memory | Use smaller model or INT4 quantization |
| 503 | Service Unavailable | LLM disabled or VecPipe down | Set `ENABLE_LOCAL_LLM=true`, check VecPipe |
| 504 | Gateway Timeout | Generation too slow | Reduce max_tokens or use smaller model |

### Exception Types

| Exception | When Raised | Retryable |
|-----------|-------------|-----------|
| `LLMNotConfiguredError` | User hasn't set up LLM settings | No |
| `LLMProviderError` | VecPipe error (connection, GPU OOM) | Maybe |
| `LLMTimeoutError` | Request exceeded timeout | Yes |
| `LLMContextLengthError` | Input exceeds model context window | No |

## Troubleshooting

### Model Download Slow

**Symptom**: First request takes 5+ minutes

**Cause**: HuggingFace model download (models are 0.5-15GB)

**Solutions**:
- Use `/llm/models/load` endpoint to pre-warm models during startup
- Set `HF_HOME` to SSD-backed storage for faster cache access
- Use `HF_TOKEN` environment variable for authenticated access (faster downloads)

### Out of Memory During Generation

**Symptom**: HTTP 507 or CUDA OOM error

**Solutions**:
1. Use INT4 quantization (lowest memory)
2. Reduce `max_tokens` in generation request
3. Choose a smaller model
4. Check other models using memory (`/memory/stats` endpoint)

### High Latency

**Symptom**: Requests take 10+ seconds

**Causes and solutions**:

| Cause | Indicator | Solution |
|-------|-----------|----------|
| Model loading | First request slow | Pre-warm with `/llm/models/load` |
| GPU busy with embeddings | Concurrent embedding ops | Let embedding complete first |
| High memory pressure | Check `/memory/stats` | Reduce model sizes |
| Large max_tokens | Generation takes long | Reduce max_tokens |

### Model Not Loading

**Symptom**: RuntimeError about chat template

**Cause**: Model doesn't have chat template (not instruction-tuned)

**Solution**: Use only curated models from the registry. Custom models must be instruction-tuned with chat templates.

## Performance Expectations

| Model | Quantization | First Load | Cached Load | Generation (256 tokens) |
|-------|--------------|------------|-------------|------------------------|
| 0.5B | int8 | 3-5s | <1s | 1-2s |
| 1.5B | int8 | 5-8s | 1-2s | 2-4s |
| 3B | int8 | 8-12s | 2-3s | 4-7s |
| 7B | int8 | 15-20s | 3-5s | 8-15s |

**Notes**:
- First load includes model initialization and quantization
- Cached load uses HuggingFace disk cache (not CPU warm pool)
- Generation time varies with prompt length and GPU capability
- INT4 is ~10-20% faster than INT8 with some quality loss

## See Also

- [Configuration Guide](./CONFIGURATION.md) - Environment variable reference
- [Memory Management](./MEMORY_MANAGEMENT.md) - GPU memory governance details
- [Search System](./SEARCH_SYSTEM.md) - HyDE search using local LLMs
