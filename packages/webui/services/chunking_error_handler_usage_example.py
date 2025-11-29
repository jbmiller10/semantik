"""
Example usage of the enhanced ChunkingErrorHandler with production features.

This file demonstrates how to integrate the enhanced error handler
with correlation IDs, Redis state management, and advanced recovery strategies.
"""

import asyncio
import os

from fastapi import Request
from redis.asyncio import Redis

from shared.text_processing.base_chunker import ChunkResult
from shared.chunking.exceptions import ChunkingMemoryError
from webui.middleware.correlation import get_or_generate_correlation_id
from webui.services.chunking_error_handler import ChunkingErrorHandler, ResourceType


async def example_usage() -> None:
    """Demonstrate usage of the enhanced ChunkingErrorHandler."""

    # Initialize Redis client
    redis_client = Redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        decode_responses=True,
    )

    # Initialize error handler with Redis support
    error_handler = ChunkingErrorHandler(redis_client=redis_client)

    # Example 1: Handle error with correlation ID
    operation_id = "op_12345"
    correlation_id = get_or_generate_correlation_id()

    try:
        # Simulate a memory error during chunking
        raise ChunkingMemoryError(
            detail="Out of memory while processing large document",
            correlation_id=correlation_id,
            operation_id=operation_id,
            memory_used=8_000_000_000,  # 8GB
            memory_limit=4_000_000_000,  # 4GB
        )
    except Exception as e:
        # Handle error with correlation tracking
        result = await error_handler.handle_with_correlation(
            operation_id=operation_id,
            correlation_id=correlation_id,
            error=e,
            context={
                "collection_id": "coll_123",
                "document_ids": ["doc_1", "doc_2"],
                "strategy": "semantic",
                "params": {"chunk_size": 1000},
                "checkpoint": {"processed_docs": 5, "total_docs": 10},
            },
        )

        print(f"Error handled: {result.handled}")
        print(f"Recovery action: {result.recovery_action}")
        print(f"Retry after: {result.retry_after} seconds")
        print(f"Recommendations: {result.recommendations}")

    # Example 2: Handle resource exhaustion
    try:
        # Check current memory usage
        import psutil

        memory = psutil.virtual_memory()
        current_usage_gb = (memory.total - memory.available) / (1024**3)
        limit_gb = 8.0

        if current_usage_gb > limit_gb * 0.8:  # 80% threshold
            recovery = await error_handler.handle_resource_exhaustion(
                operation_id=operation_id,
                resource_type=ResourceType.MEMORY,
                current_usage=current_usage_gb,
                limit=limit_gb,
            )

            print(f"Resource recovery action: {recovery.action}")
            if recovery.action == "queue":
                print(f"Queue position: {recovery.queue_position}")
                print(f"Estimated wait time: {recovery.wait_time} seconds")
            elif recovery.action == "reduce_batch":
                print(f"New batch size: {recovery.new_batch_size}")
                print(f"Alternative strategy: {recovery.alternative_strategy}")

    except Exception as e:
        print(f"Resource check error: {e}")

    # Example 3: Clean up failed operation
    partial_results = [
        ChunkResult(
            chunk_id="chunk_1",
            text="Sample chunk 1",
            metadata={"doc_id": "doc_1", "position": 0},
            start_offset=0,
            end_offset=100,
        ),
        ChunkResult(
            chunk_id="chunk_2",
            text="Sample chunk 2",
            metadata={"doc_id": "doc_1", "position": 1},
            start_offset=100,
            end_offset=200,
        ),
    ]

    cleanup_result = await error_handler.cleanup_failed_operation(
        operation_id=operation_id,
        partial_results=partial_results,
        cleanup_strategy="save_partial",  # or "rollback", "discard"
    )

    print(f"Cleanup successful: {cleanup_result.cleaned}")
    print(f"Partial results saved: {cleanup_result.partial_results_saved}")
    print(f"Resources freed: {cleanup_result.resources_freed}")

    # Example 4: Generate error report
    # Simulate multiple errors for reporting
    errors = [
        MemoryError("Out of memory"),
        TimeoutError("Operation timed out"),
        ValueError("Invalid chunk size"),
    ]

    report = error_handler.create_error_report(
        operation_id=operation_id,
        errors=errors,
    )

    print(f"Error report for operation {report.operation_id}")
    print(f"Total errors: {report.total_errors}")
    print(f"Error breakdown: {report.error_breakdown}")
    print(f"Resource usage: {report.resource_usage}")
    print(f"Recommendations: {report.recommendations}")

    # Close Redis connection
    await redis_client.close()


async def integration_with_fastapi(request: Request) -> None:
    """Example of integration with FastAPI endpoint."""

    # Get correlation ID from request (set by middleware)
    correlation_id = get_or_generate_correlation_id(request)

    # Initialize services
    redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    error_handler = ChunkingErrorHandler(redis_client=redis_client)

    operation_id = f"chunk_op_{correlation_id[:8]}"

    try:
        # Perform chunking operation
        # ... chunking logic here ...
        pass

    except Exception as e:
        # Handle error with correlation
        result = await error_handler.handle_with_correlation(
            operation_id=operation_id,
            correlation_id=correlation_id,
            error=e,
            context={
                "request_path": request.url.path,
                "method": request.method,
                # ... other context ...
            },
        )

        # Check if we should retry
        if result.recovery_action == "retry" and result.retry_after:
            # Schedule retry after delay
            await asyncio.sleep(result.retry_after)
            # Retry operation...

        # Clean up if failed
        if result.recovery_action == "fail":
            await error_handler.cleanup_failed_operation(
                operation_id=operation_id,
                partial_results=None,
                cleanup_strategy="rollback",
            )

        # Generate report for monitoring
        report = error_handler.create_error_report(operation_id)
        # Send to monitoring system...
        _ = report  # noqa: F841

    finally:
        await redis_client.close()


async def monitoring_and_alerting_example() -> None:
    """Example of using error handler for monitoring and alerting."""
    redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    error_handler = ChunkingErrorHandler(redis_client=redis_client)

    # Monitor error patterns across operations
    operation_ids = ["op_1", "op_2", "op_3"]

    for op_id in operation_ids:
        report = error_handler.create_error_report(op_id)

        # Check for critical patterns
        if report.total_errors > 10:
            print(f"ALERT: High error rate for operation {op_id}")

        if report.error_breakdown.get("memory_error", 0) > 5:
            print(f"ALERT: Frequent memory errors for operation {op_id}")

        # Check resource usage
        memory_percent = report.resource_usage.get("memory", {}).get("percent", 0)
        if memory_percent > 90:
            print(f"CRITICAL: Memory usage at {memory_percent}%")

    await redis_client.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
