#!/usr/bin/env python3
"""
Unit tests for chunking Celery tasks.

Tests comprehensive error handling, resource management, and task lifecycle.
"""

import asyncio
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import psutil
from celery.exceptions import SoftTimeLimitExceeded
from redis import Redis

from packages.shared.database.models import OperationStatus, OperationType
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingResourceLimitError,
    ChunkingTimeoutError,
    ResourceType,
)
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.tasks.chunking_tasks import (
    CHUNKING_MEMORY_LIMIT_GB,
    ChunkingTask,
    _calculate_batch_size,
    _check_resource_limits,
    _handle_soft_timeout,
    _monitor_resources,
    _process_chunking_operation_async,
    _send_progress_update,
    monitor_dead_letter_queue,
    process_chunking_operation,
    retry_failed_documents,
)


class TestChunkingTask:
    """Test ChunkingTask base class functionality."""

    def test_init(self):
        """Test task initialization."""
        task = ChunkingTask()
        assert task._shutdown_handler_registered is False
        assert task._graceful_shutdown is False
        assert task._redis_client is None
        assert task._error_handler is None
        assert task._circuit_breaker_state == "closed"

    @patch("packages.webui.tasks.chunking_tasks.get_redis_client")
    @patch("packages.webui.tasks.chunking_tasks.chunking_tasks_started")
    @patch("packages.webui.tasks.chunking_tasks.chunking_active_operations")
    def test_before_start(self, mock_active_ops, mock_started, mock_redis):
        """Test task setup before execution."""
        mock_redis.return_value = MagicMock(spec=Redis)
        
        task = ChunkingTask()
        task.before_start(
            task_id="test-task-123",
            args=("op-123",),
            kwargs={"correlation_id": "corr-123"},
        )
        
        assert task._redis_client is not None
        assert task._error_handler is not None
        mock_started.labels.assert_called_with(operation_type="chunking")
        mock_active_ops.inc.assert_called_once()

    @patch("packages.webui.tasks.chunking_tasks.chunking_tasks_completed")
    @patch("packages.webui.tasks.chunking_tasks.chunking_active_operations")
    def test_on_success(self, mock_active_ops, mock_completed):
        """Test successful task completion handling."""
        task = ChunkingTask()
        task._circuit_breaker_failures = 5
        task._circuit_breaker_state = "open"
        
        task.on_success(
            retval={"status": "success"},
            task_id="test-task-123",
            args=("op-123",),
            kwargs={},
        )
        
        # Circuit breaker should reset
        assert task._circuit_breaker_failures == 0
        assert task._circuit_breaker_state == "closed"
        
        mock_completed.labels.assert_called_with(
            operation_type="chunking",
            status="success",
        )
        mock_active_ops.dec.assert_called_once()

    @patch("packages.webui.tasks.chunking_tasks.chunking_tasks_failed")
    @patch("packages.webui.tasks.chunking_tasks.chunking_active_operations")
    def test_on_failure(self, mock_active_ops, mock_failed):
        """Test task failure handling."""
        task = ChunkingTask()
        task.request = Mock(retries=3)
        task.max_retries = 3
        task._redis_client = MagicMock(spec=Redis)
        
        error = ChunkingMemoryError(
            detail="Out of memory",
            correlation_id="corr-123",
            operation_id="op-123",
            memory_used=5 * 1024**3,
            memory_limit=4 * 1024**3,
        )
        
        task.on_failure(
            exc=error,
            task_id="test-task-123",
            args=("op-123",),
            kwargs={"correlation_id": "corr-123"},
            einfo=None,
        )
        
        mock_failed.labels.assert_called_with(
            operation_type="chunking",
            error_type="memory_error",
        )
        mock_active_ops.dec.assert_called_once()
        
        # Should send to DLQ since retries exceeded
        assert task._redis_client.rpush.called

    def test_circuit_breaker_logic(self):
        """Test circuit breaker state transitions."""
        task = ChunkingTask()
        
        # Initially closed
        assert task._check_circuit_breaker() is True
        
        # Open after threshold failures
        for _ in range(5):
            task._update_circuit_breaker_state()
        
        assert task._circuit_breaker_state == "open"
        assert task._check_circuit_breaker() is False
        
        # Half-open after timeout
        task._circuit_breaker_last_failure_time = 0  # Force timeout
        assert task._check_circuit_breaker() is True
        assert task._circuit_breaker_state == "half_open"

    def test_graceful_shutdown_handler(self):
        """Test graceful shutdown signal handling."""
        task = ChunkingTask()
        task._handle_shutdown(None, None)
        assert task._graceful_shutdown is True


class TestProcessChunkingOperation:
    """Test main chunking operation processing."""

    @pytest.mark.asyncio
    @patch("packages.webui.tasks.chunking_tasks.pg_connection_manager")
    @patch("packages.webui.tasks.chunking_tasks.AsyncSessionLocal")
    @patch("packages.webui.tasks.chunking_tasks.get_redis_client")
    async def test_successful_operation(
        self,
        mock_redis,
        mock_session,
        mock_pg_manager,
    ):
        """Test successful chunking operation."""
        # Setup mocks
        mock_redis.return_value = MagicMock(spec=Redis)
        mock_db = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db
        
        # Mock repositories
        mock_operation = Mock(
            id="op-123",
            status=OperationStatus.PENDING,
            collection_id="coll-123",
            type=OperationType.INDEX,
            metadata={"strategy": "recursive", "documents": []},
        )
        
        mock_collection = Mock(id="coll-123", name="Test Collection")
        
        mock_op_repo = AsyncMock()
        mock_op_repo.get_by_id.return_value = mock_operation
        mock_op_repo.update_status = AsyncMock()
        
        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_id.return_value = mock_collection
        mock_coll_repo.update_status = AsyncMock()
        
        # Mock chunking service
        mock_chunking_service = AsyncMock()
        mock_chunking_service.process_documents.return_value = []
        
        with patch("packages.webui.tasks.chunking_tasks.OperationRepository", return_value=mock_op_repo):
            with patch("packages.webui.tasks.chunking_tasks.CollectionRepository", return_value=mock_coll_repo):
                with patch("packages.webui.tasks.chunking_tasks.ChunkingService", return_value=mock_chunking_service):
                    # Create task mock
                    celery_task = Mock(spec=ChunkingTask)
                    celery_task._graceful_shutdown = False
                    
                    result = await _process_chunking_operation_async(
                        operation_id="op-123",
                        correlation_id="corr-123",
                        celery_task=celery_task,
                    )
        
        assert result["status"] == "success"
        assert result["operation_id"] == "op-123"
        mock_op_repo.update_status.assert_called()
        mock_coll_repo.update_status.assert_called()

    @pytest.mark.asyncio
    async def test_idempotency_check(self):
        """Test idempotent operation handling."""
        # Mock already completed operation
        mock_operation = Mock(
            id="op-123",
            status=OperationStatus.COMPLETED,
            metadata={"chunks_created": 100},
        )
        
        mock_op_repo = AsyncMock()
        mock_op_repo.get_by_id.return_value = mock_operation
        
        mock_db = AsyncMock()
        mock_session = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db
        
        with patch("packages.webui.tasks.chunking_tasks.AsyncSessionLocal", mock_session):
            with patch("packages.webui.tasks.chunking_tasks.OperationRepository", return_value=mock_op_repo):
                with patch("packages.webui.tasks.chunking_tasks.get_redis_client"):
                    celery_task = Mock(spec=ChunkingTask)
                    
                    result = await _process_chunking_operation_async(
                        operation_id="op-123",
                        correlation_id="corr-123",
                        celery_task=celery_task,
                    )
        
        assert result["status"] == "already_completed"
        assert result["chunks_created"] == 100

    @pytest.mark.asyncio
    @patch("packages.webui.tasks.chunking_tasks.ChunkingErrorHandler")
    async def test_partial_failure_handling(self, mock_error_handler_class):
        """Test handling of partial failures."""
        # Setup error handler mock
        mock_error_handler = AsyncMock()
        mock_error_handler.handle_partial_failure.return_value = Mock(
            recovery_operation_id="recovery-123",
            recommendations=["Retry failed documents"],
        )
        mock_error_handler_class.return_value = mock_error_handler
        
        # Create partial failure error
        error = ChunkingPartialFailureError(
            detail="Some documents failed",
            correlation_id="corr-123",
            operation_id="op-123",
            total_documents=10,
            failed_documents=["doc-1", "doc-2"],
            failure_reasons={"doc-1": "Invalid format", "doc-2": "Too large"},
            successful_chunks=8,
        )
        
        # Mock chunking service to raise partial failure
        mock_chunking_service = AsyncMock()
        mock_chunking_service.process_documents.side_effect = error
        
        # Setup other mocks
        mock_operation = Mock(
            id="op-123",
            status=OperationStatus.PROCESSING,
            collection_id="coll-123",
            type=OperationType.INDEX,
            metadata={"strategy": "recursive", "documents": [{"id": f"doc-{i}"} for i in range(10)]},
        )
        
        mock_db = AsyncMock()
        mock_session = AsyncMock()
        mock_session.return_value.__aenter__.return_value = mock_db
        
        mock_op_repo = AsyncMock()
        mock_op_repo.get_by_id.return_value = mock_operation
        
        mock_coll_repo = AsyncMock()
        mock_coll_repo.get_by_id.return_value = Mock(id="coll-123")
        
        with patch("packages.webui.tasks.chunking_tasks.AsyncSessionLocal", mock_session):
            with patch("packages.webui.tasks.chunking_tasks.OperationRepository", return_value=mock_op_repo):
                with patch("packages.webui.tasks.chunking_tasks.CollectionRepository", return_value=mock_coll_repo):
                    with patch("packages.webui.tasks.chunking_tasks.ChunkingService", return_value=mock_chunking_service):
                        with patch("packages.webui.tasks.chunking_tasks.get_redis_client"):
                            celery_task = Mock(spec=ChunkingTask)
                            celery_task._graceful_shutdown = False
                            
                            result = await _process_chunking_operation_async(
                                operation_id="op-123",
                                correlation_id="corr-123",
                                celery_task=celery_task,
                            )
        
        assert result["status"] == "partial_success"
        assert result["documents_failed"] == 2
        assert result["recovery_operation_id"] == "recovery-123"
        mock_error_handler.handle_partial_failure.assert_called_once()

    def test_soft_time_limit_handling(self):
        """Test handling of soft time limit."""
        task = ChunkingTask()
        task._check_circuit_breaker = Mock(return_value=True)
        
        with patch("packages.webui.tasks.chunking_tasks.asyncio") as mock_asyncio:
            mock_loop = Mock()
            mock_asyncio.get_event_loop.return_value = mock_loop
            mock_loop.run_until_complete.side_effect = SoftTimeLimitExceeded()
            
            with pytest.raises(ChunkingTimeoutError) as exc_info:
                process_chunking_operation(
                    task,
                    operation_id="op-123",
                    correlation_id="corr-123",
                )
            
            assert "soft time limit" in str(exc_info.value).lower()


class TestResourceManagement:
    """Test resource monitoring and limits."""

    @pytest.mark.asyncio
    async def test_check_resource_limits_memory_exhausted(self):
        """Test resource limit checking when memory is exhausted."""
        mock_error_handler = AsyncMock()
        mock_error_handler.handle_resource_exhaustion.return_value = Mock(action="fail")
        
        with patch("packages.webui.tasks.chunking_tasks.psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=95)
            
            with pytest.raises(ChunkingResourceLimitError) as exc_info:
                await _check_resource_limits(
                    error_handler=mock_error_handler,
                    operation_id="op-123",
                    correlation_id="corr-123",
                    initial_memory=1024**3,
                )
            
            assert exc_info.value.resource_type == ResourceType.MEMORY

    @pytest.mark.asyncio
    async def test_check_resource_limits_cpu_high(self):
        """Test resource checking with high CPU usage."""
        mock_error_handler = AsyncMock()
        mock_error_handler.handle_resource_exhaustion.return_value = Mock(
            action="wait_and_retry",
            wait_time=5,
        )
        
        with patch("packages.webui.tasks.chunking_tasks.psutil.virtual_memory") as mock_memory:
            with patch("packages.webui.tasks.chunking_tasks.psutil.cpu_percent") as mock_cpu:
                with patch("packages.webui.tasks.chunking_tasks.asyncio.sleep") as mock_sleep:
                    mock_memory.return_value = Mock(percent=70)
                    mock_cpu.return_value = 95
                    
                    await _check_resource_limits(
                        error_handler=mock_error_handler,
                        operation_id="op-123",
                        correlation_id="corr-123",
                        initial_memory=1024**3,
                    )
                    
                    mock_sleep.assert_called_once_with(5)

    @pytest.mark.asyncio
    async def test_monitor_resources_memory_limit(self):
        """Test monitoring exceeds memory limit."""
        mock_process = Mock(spec=psutil.Process)
        mock_process.memory_info.return_value = Mock(rss=5 * 1024**3)  # 5GB
        mock_process.cpu_times.return_value = Mock(user=100, system=50)
        
        mock_error_handler = AsyncMock()
        
        with pytest.raises(ChunkingMemoryError) as exc_info:
            await _monitor_resources(
                process=mock_process,
                operation_id="op-123",
                initial_memory=1024**3,  # 1GB initial
                initial_cpu_time=0,
                error_handler=mock_error_handler,
                correlation_id="corr-123",
            )
        
        assert exc_info.value.memory_limit == CHUNKING_MEMORY_LIMIT_GB * 1024**3

    @pytest.mark.asyncio
    async def test_calculate_batch_size(self):
        """Test adaptive batch size calculation."""
        mock_error_handler = Mock()
        mock_error_handler._calculate_adaptive_batch_size.return_value = 16
        
        with patch("packages.webui.tasks.chunking_tasks.psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = Mock(percent=75, available=4 * 1024**3)
            
            batch_size = await _calculate_batch_size(
                error_handler=mock_error_handler,
                initial_memory=1024**3,
            )
            
            assert batch_size == 16
            mock_error_handler._calculate_adaptive_batch_size.assert_called_once_with(75, 100)


class TestProgressTracking:
    """Test progress updates and monitoring."""

    @pytest.mark.asyncio
    async def test_send_progress_update(self):
        """Test sending progress updates via Redis."""
        mock_redis = AsyncMock(spec=Redis)
        
        await _send_progress_update(
            redis_client=mock_redis,
            operation_id="op-123",
            correlation_id="corr-123",
            progress=50,
            message="Processing documents",
        )
        
        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == "chunking:progress:op-123"
        assert call_args[0][1]["progress"] == 50
        assert call_args[0][1]["message"] == "Processing documents"
        
        mock_redis.expire.assert_called_once_with("chunking:progress:op-123", 3600)

    @pytest.mark.asyncio
    async def test_send_progress_update_no_redis(self):
        """Test progress update handles missing Redis gracefully."""
        # Should not raise exception
        await _send_progress_update(
            redis_client=None,
            operation_id="op-123",
            correlation_id="corr-123",
            progress=100,
            message="Complete",
        )


class TestSoftTimeout:
    """Test soft timeout handling."""

    @pytest.mark.asyncio
    async def test_handle_soft_timeout(self):
        """Test saving state on soft timeout."""
        mock_task = Mock(spec=ChunkingTask)
        mock_task.request.id = "task-123"
        
        mock_redis = MagicMock(spec=Redis)
        mock_error_handler = AsyncMock()
        
        with patch("packages.webui.tasks.chunking_tasks.get_redis_client", return_value=mock_redis):
            with patch("packages.webui.tasks.chunking_tasks.ChunkingErrorHandler", return_value=mock_error_handler):
                await _handle_soft_timeout(
                    operation_id="op-123",
                    correlation_id="corr-123",
                    celery_task=mock_task,
                )
        
        mock_error_handler._save_operation_state.assert_called_once()
        call_args = mock_error_handler._save_operation_state.call_args
        assert call_args[1]["operation_id"] == "op-123"
        assert call_args[1]["context"]["soft_timeout"] is True


class TestRetryAndMonitoring:
    """Test retry and monitoring tasks."""

    @patch("packages.webui.tasks.chunking_tasks.process_chunking_operation")
    def test_retry_failed_documents(self, mock_process):
        """Test retry task for failed documents."""
        mock_task = Mock()
        mock_process.apply_async.return_value.get.return_value = {"status": "success"}
        
        result = retry_failed_documents(
            mock_task,
            operation_id="op-123",
            failed_documents=["doc-1", "doc-2"],
            correlation_id="corr-123",
        )
        
        assert result["status"] == "success"
        mock_process.apply_async.assert_called_once()

    @patch("packages.webui.tasks.chunking_tasks.get_redis_client")
    def test_monitor_dead_letter_queue(self, mock_redis_func):
        """Test DLQ monitoring task."""
        mock_redis = MagicMock(spec=Redis)
        mock_redis_func.return_value = mock_redis
        
        # Test with items in DLQ
        mock_redis.llen.return_value = 5
        mock_redis.lrange.return_value = [
            json.dumps({
                "task_id": "task-1",
                "operation_id": "op-1",
                "error_type": "memory_error",
            })
        ]
        
        result = monitor_dead_letter_queue()
        
        assert result["dlq_size"] == 5
        assert len(result["sample_tasks"]) == 1
        assert result["alert"] is False  # Only alert if > 10
        
        # Test with many items (should alert)
        mock_redis.llen.return_value = 15
        result = monitor_dead_letter_queue()
        assert result["alert"] is True

    @patch("packages.webui.tasks.chunking_tasks.get_redis_client")
    def test_monitor_dead_letter_queue_error(self, mock_redis_func):
        """Test DLQ monitoring with Redis error."""
        mock_redis = MagicMock(spec=Redis)
        mock_redis_func.return_value = mock_redis
        mock_redis.llen.side_effect = Exception("Redis connection failed")
        
        result = monitor_dead_letter_queue()
        
        assert "error" in result
        assert result["alert"] is True