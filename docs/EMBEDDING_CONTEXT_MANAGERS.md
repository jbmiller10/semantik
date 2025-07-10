# Embedding Service Context Managers

This document explains how to use context managers for proper embedding service lifecycle management in Project Semantik.

## Why Use Context Managers?

Context managers ensure that embedding services are properly cleaned up after use, preventing:
- GPU memory leaks
- Model resources not being freed
- Undefined service states after exceptions

## Available Context Managers

### 1. `embedding_service_context`

The basic context manager for the global embedding service:

```python
from shared.embedding import embedding_service_context

async with embedding_service_context() as service:
    await service.initialize("BAAI/bge-base-en-v1.5")
    embeddings = await service.embed_texts(["Hello world"])
    # Service is automatically cleaned up when exiting the context
```

### 2. `temporary_embedding_service`

Create an isolated service instance with a specific model:

```python
from shared.embedding import temporary_embedding_service

async with temporary_embedding_service("sentence-transformers/all-MiniLM-L6-v2") as service:
    # This won't affect the global embedding service
    embeddings = await service.embed_texts(["Test text"])
    # GPU memory is freed when done
```

### 3. `ManagedEmbeddingService`

A class providing more control over service lifecycle:

```python
from shared.embedding import ManagedEmbeddingService

managed = ManagedEmbeddingService(mock_mode=False)
async with managed as service:
    await service.initialize("BAAI/bge-base-en-v1.5")
    # Use the service...
```

## Usage Patterns

### Basic Usage

```python
async def generate_embeddings(texts: list[str]):
    async with embedding_service_context() as service:
        await service.initialize("BAAI/bge-base-en-v1.5")
        return await service.embed_texts(texts)
```

### Exception Handling

Context managers ensure cleanup even when exceptions occur:

```python
try:
    async with embedding_service_context() as service:
        await service.initialize("BAAI/bge-base-en-v1.5")
        embeddings = await service.embed_texts(texts)
        
        if some_condition:
            raise ValueError("Processing error")
            
except ValueError:
    # Service was still cleaned up properly
    pass
```

### Model Comparison

Load multiple models without interference:

```python
async def compare_models(text: str):
    # Load first model
    async with temporary_embedding_service("model-1") as service1:
        embedding1 = await service1.embed_single(text)
    
    # First model is cleaned up, load second model
    async with temporary_embedding_service("model-2") as service2:
        embedding2 = await service2.embed_single(text)
    
    return embedding1, embedding2
```

### FastAPI Integration

Use context managers in API endpoints:

```python
@app.post("/api/embed")
async def create_embeddings(texts: list[str], model: str = "BAAI/bge-base-en-v1.5"):
    async with temporary_embedding_service(model) as service:
        embeddings = await service.embed_texts(texts)
        return {
            "embeddings": embeddings.tolist(),
            "model": model,
            "dimension": service.get_dimension()
        }
```

### Concurrent Operations

Process with multiple models concurrently:

```python
async def process_concurrent(texts_by_model: dict[str, list[str]]):
    async def embed_with_model(model: str, texts: list[str]):
        async with temporary_embedding_service(model) as service:
            return await service.embed_texts(texts)
    
    tasks = [
        embed_with_model(model, texts)
        for model, texts in texts_by_model.items()
    ]
    
    return await asyncio.gather(*tasks)
```

## Migration Guide

### Before (Manual Cleanup)

```python
# Old way - manual cleanup required
service = await get_embedding_service()
try:
    await service.initialize("BAAI/bge-base-en-v1.5")
    embeddings = await service.embed_texts(texts)
finally:
    await service.cleanup()  # Easy to forget!
```

### After (Automatic Cleanup)

```python
# New way - automatic cleanup
async with embedding_service_context() as service:
    await service.initialize("BAAI/bge-base-en-v1.5")
    embeddings = await service.embed_texts(texts)
    # Cleanup happens automatically
```

## Best Practices

1. **Always use context managers** for temporary operations
2. **Use `temporary_embedding_service`** when you need a specific model temporarily
3. **Don't call `cleanup()` manually** - let the context manager handle it
4. **Handle exceptions properly** - context managers ensure cleanup even on errors
5. **Use concurrent contexts** for parallel processing with different models

## Performance Considerations

- Context managers add minimal overhead
- Temporary services create new instances (more memory but isolated)
- The global service context reuses the singleton instance
- Cleanup frees GPU memory immediately

## Testing

Context managers make testing easier:

```python
@pytest.mark.asyncio
async def test_embedding_generation():
    # Each test gets a fresh service instance
    async with temporary_embedding_service("test-model") as service:
        embeddings = await service.embed_texts(["test"])
        assert len(embeddings) == 1
    # Service is cleaned up after test
```

## Troubleshooting

### "Synchronous context manager not supported"

Use `async with` instead of `with`:

```python
# Wrong
with embedding_service_context() as service:
    pass

# Correct
async with embedding_service_context() as service:
    pass
```

### GPU Memory Not Freed

Ensure you're not keeping references to the service:

```python
# Wrong - service reference escapes context
service = None
async with embedding_service_context() as s:
    service = s  # Don't do this!

# Correct - service only used within context
async with embedding_service_context() as service:
    # Use service here only
    pass
```

### Model Loading Errors

The context manager will clean up even if initialization fails:

```python
try:
    async with temporary_embedding_service("invalid-model") as service:
        # This might fail
        pass
except Exception as e:
    # Service was still cleaned up
    print(f"Model loading failed: {e}")
```