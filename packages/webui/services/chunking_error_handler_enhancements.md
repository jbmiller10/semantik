# ChunkingErrorHandler Production Enhancements

## Overview

The ChunkingErrorHandler has been enhanced with production-ready features including correlation ID support, Redis-based state management, advanced recovery strategies, and comprehensive resource tracking.

## Key Enhancements

### 1. Correlation ID Support

All error handling methods now integrate with the correlation middleware:

- `handle_with_correlation()` - Primary method for error handling with correlation tracking
- Automatic correlation ID extraction from context
- Correlation IDs included in all logs and error reports

### 2. Advanced Recovery Strategies

#### Adaptive Batch Size Reduction
- Memory usage monitored in real-time using psutil
- Batch sizes automatically adjusted based on current memory pressure:
  - >90% usage: 4 documents per batch
  - >80% usage: 8 documents per batch  
  - >70% usage: 16 documents per batch
  - <70% usage: 32 documents per batch (normal)

#### Progressive Timeout Increases
- Timeouts increase exponentially on retry: 450s → 675s → 1012s
- Prevents premature failures for slow operations

#### Resource Pooling & Queuing
- Operations queued when resources exhausted
- Queue positions tracked in Redis
- Estimated wait times provided

### 3. State Management with Redis

#### Operation State Persistence
- Operation state saved to Redis with 24-hour TTL
- Includes context, checkpoint data, and retry counts
- Enables resumable operations after failures

#### Checkpointing Support
- Long operations can save checkpoints
- Failures can resume from last checkpoint
- Reduces re-processing overhead

#### Idempotency with Fingerprints
- Operations generate SHA256 fingerprints
- Prevents duplicate processing
- Ensures exactly-once semantics

### 4. Resource Tracking & Management

#### Real-time Resource Monitoring
- Memory usage tracked via psutil
- CPU utilization monitored
- Connection pool status checked

#### Resource Exhaustion Handling
- `handle_resource_exhaustion()` provides intelligent recovery
- Different strategies per resource type:
  - Memory: Reduce batch size or use streaming
  - CPU: Wait and retry or use simpler strategy
  - Connections: Queue operation

#### Concurrency Control
- Async locks prevent resource overload
- Per-resource-type locking
- Timeout-based lock acquisition

### 5. Enhanced Error Reporting

#### Comprehensive Error Reports
- `create_error_report()` generates detailed analytics
- Error timeline with correlation IDs
- Resource usage at time of errors
- Recovery attempt history
- Actionable recommendations

#### Error Pattern Analysis
- Errors classified and tracked over time
- Patterns identified for proactive fixes
- Recommendations based on error types

### 6. Cleanup & Recovery

#### Flexible Cleanup Strategies
- `save_partial`: Save successful results, mark failed items
- `rollback`: Full transaction rollback
- `discard`: Remove all traces

#### Resource Cleanup
- Redis keys cleaned up
- Retry counters reset
- Error history pruned
- Queue entries removed

## Usage Examples

### Basic Error Handling with Correlation

```python
result = await error_handler.handle_with_correlation(
    operation_id="op_123",
    correlation_id=correlation_id,
    error=error,
    context={
        "collection_id": "coll_456",
        "document_ids": ["doc_1", "doc_2"],
        "strategy": "semantic",
        "checkpoint": {"processed": 5, "total": 10}
    }
)

if result.recovery_action == "retry":
    await asyncio.sleep(result.retry_after)
    # Resume from checkpoint
    state = await error_handler.resume_operation("op_123")
```

### Resource Exhaustion Handling

```python
recovery = await error_handler.handle_resource_exhaustion(
    operation_id="op_123",
    resource_type=ResourceType.MEMORY,
    current_usage=14.0,  # GB
    limit=16.0  # GB
)

if recovery.action == "reduce_batch":
    # Use smaller batch size
    batch_size = recovery.new_batch_size
elif recovery.action == "queue":
    # Wait for queue position
    print(f"Queued at position {recovery.queue_position}")
```

### Error Reporting for Monitoring

```python
report = error_handler.create_error_report("op_123")

# Send alerts based on patterns
if report.error_breakdown.get("memory_error", 0) > 5:
    send_alert("Frequent memory errors detected")

# Use recommendations
for recommendation in report.recommendations:
    logger.info(f"Recommendation: {recommendation}")
```

## Integration Requirements

1. **Redis**: Required for state management features
2. **psutil**: Used for resource monitoring (already in dependencies)
3. **Correlation Middleware**: Must be enabled in FastAPI app
4. **Logging Configuration**: Should include correlation ID filter

## Best Practices

1. Always use `handle_with_correlation()` for production error handling
2. Enable Redis for resumable operations
3. Monitor error reports for patterns
4. Implement cleanup strategies for all failures
5. Use resource locks for critical operations
6. Set appropriate resource limits based on system capacity

## Thread Safety

- All methods are async-safe
- Resource locks prevent concurrent access issues
- Redis operations are atomic
- Error history limited to prevent memory leaks

## Performance Considerations

- Error history capped at 100 entries per operation
- Redis keys have 24-hour TTL
- Resource checks are lightweight (<1ms)
- Adaptive strategies prevent system overload