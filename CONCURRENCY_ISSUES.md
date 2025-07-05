# Specific Concurrency Issues in ModelManager

## 1. Race Condition in Timestamp Updates

### Issue in `ensure_reranker_loaded()` (line 190):
```python
# Current implementation - RACE CONDITION
if self.current_reranker_key == reranker_key and self.reranker is not None:
    self.last_reranker_used = time.time()  # ❌ Outside lock!
    return True
```

**Problem**: Multiple threads could be updating `last_reranker_used` simultaneously, potentially causing:
- Lost updates (one thread overwrites another's timestamp)
- Incorrect unload timing if timestamp is partially written

### Issue in `ensure_model_loaded()` (line 90):
```python
# Current implementation - uses helper method
if self.current_model_key == model_key:
    self._update_last_used()  # This method has no lock protection
    return True
```

**Problem**: The `_update_last_used()` method modifies shared state without lock protection.

## 2. Async Task Cancellation Issues

### In `_schedule_unload()` and `_schedule_reranker_unload()`:
```python
if self.unload_task:
    self.unload_task.cancel()  # ❌ No await, task might still be running
```

**Problems**:
- Cancelled task might still be executing its finally block
- Could lead to race between old and new unload tasks
- Model might be unloaded while new request is being processed

## 3. Model Switch During Inference

### Scenario:
```
Thread 1: ensure_model_loaded("model-A", "fp16")
Thread 2: ensure_model_loaded("model-B", "fp16")  # Different model!
Thread 1: generate_embedding_async() # Which model is loaded now?
```

**Current protection**: The lock in `ensure_model_loaded()` prevents concurrent loads, but:
- No protection against model being switched after load check
- No versioning to detect if model changed between check and use

## 4. Concurrent Access Pattern Examples

### Safe Pattern (Currently Implemented):
```python
# Loading is protected
with self.lock:
    if self.embedding_service.load_model(model_name, quantization):
        self.current_model_key = model_key
        self._update_last_used()  # Still unsafe!
```

### Unsafe Pattern (Currently Implemented):
```python
# Checking without holding lock through entire operation
if not self.ensure_model_loaded(model_name, quantization):  # Lock released here
    raise RuntimeError(f"Failed to load model {model_name}")
# ... other thread could unload/switch model here ...
embedding = await loop.run_in_executor(...)  # Using potentially different model
```

## 5. ThreadPoolExecutor Task Management

### Current Implementation:
```python
self.executor = ThreadPoolExecutor(max_workers=4)
# ...
embedding = await loop.run_in_executor(
    self.executor, self.embedding_service.generate_single_embedding, ...
)
```

**Issues**:
- No tracking of in-flight tasks
- Shutdown might interrupt ongoing embeddings
- No graceful degradation under high load

## Recommended Solutions

### 1. Atomic Timestamp Updates:
```python
def ensure_reranker_loaded(self, model_name: str, quantization: str) -> bool:
    reranker_key = self._get_model_key(model_name, quantization)
    
    with self.reranker_lock:
        # Check and update timestamp atomically
        if self.current_reranker_key == reranker_key and self.reranker is not None:
            self.last_reranker_used = time.time()
            return True
    
    # Rest of loading logic...
```

### 2. Proper Task Lifecycle:
```python
async def _schedule_unload(self):
    # Cancel and await previous task
    if self.unload_task and not self.unload_task.done():
        self.unload_task.cancel()
        try:
            await self.unload_task
        except asyncio.CancelledError:
            pass
    
    # Create new task
    self.unload_task = asyncio.create_task(self._unload_after_delay())
```

### 3. Generation Tracking:
```python
class ModelManager:
    def __init__(self, ...):
        # ...
        self.model_generation = 0
        self.reranker_generation = 0
    
    async def generate_embedding_async(self, ...):
        # Capture generation at start
        generation = self.model_generation
        
        # ... do work ...
        
        # Verify model hasn't changed
        if generation != self.model_generation:
            raise RuntimeError("Model changed during operation")
```

### 4. Request Semaphore:
```python
class ModelManager:
    def __init__(self, ...):
        # ...
        self.inference_semaphore = asyncio.Semaphore(10)  # Max concurrent inferences
    
    async def generate_embedding_async(self, ...):
        async with self.inference_semaphore:
            # ... existing logic ...
```

These changes would significantly improve the robustness of the concurrent execution model.