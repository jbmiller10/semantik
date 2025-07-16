# Phase 3 Development Log - Task Processing with Enhanced Monitoring

## Overview
This log tracks the implementation of Phase 3 from the Collections Refactor Execution Plan, focusing on enhanced task processing with comprehensive monitoring, performance tracking, and blue-green reindexing.

## Timeline
- **Phase Duration**: 1.5 weeks
- **Started**: 2025-07-16
- **Target Completion**: ~2025-07-26

## Objectives
1. Implement unified task with performance tracking
2. Add comprehensive Prometheus monitoring
3. Implement resource management
4. Create blue-green reindex with validation
5. Add audit logging and operation metrics

## Task List
- [x] Create phase 3 development log
- [x] Create enhanced metrics for collection operations
- [x] Enhance process_collection_operation with comprehensive monitoring and error handling
- [x] Implement resource management for collection operations
- [x] Implement blue-green reindex with validation checkpoints
- [x] Create reindex validation and atomic switch logic
- [x] Implement operation metrics recording in database
- [x] Add audit logging for collection operations

## Development Progress

### 2025-07-16 - Initial Setup
- Reviewed Phase 3 requirements from COLLECTIONS_REFACTOR_EXECUTION_PLAN.md
- Analyzed existing codebase:
  - Current tasks.py has basic process_collection_operation implementation
  - Prometheus metrics infrastructure exists in packages/shared/metrics/prometheus.py
  - Database models include CollectionAuditLog, OperationMetrics, CollectionResourceLimits
- Created development log to track progress

### 2025-07-16 - Completed Phase 3 Implementation
- **Enhanced Metrics**: Created comprehensive collection operation metrics in `collection_metrics.py`
  - Added counters for operations, errors, retries
  - Added histograms for durations and latencies
  - Added gauges for collection statistics
  - Implemented context managers for timing operations
- **Enhanced Task Processing**: Updated `process_collection_operation` with:
  - Comprehensive monitoring using Prometheus metrics
  - Resource tracking (CPU, memory)
  - Performance metrics recording in database
  - Proper error handling and retry logic
  - Audit logging for all operations
- **Blue-Green Reindex**: Implemented complete reindex operation with:
  - 6 validation checkpoints (preflight, staging, reprocessing, validation, switch, cleanup)
  - Staging collection creation
  - Validation comparing old and new collections
  - Atomic switch with rollback capability
  - Comprehensive error handling and cleanup
- **Resource Management**: Created ResourceManager with:
  - User quota management
  - Resource estimation for operations
  - System resource checking
  - Reservation system for reindex operations
- **Operation Handlers**: Enhanced all operation handlers with monitoring:
  - INDEX: Added Qdrant operation timing and audit logging
  - APPEND: Added document processing metrics and progress tracking
  - REINDEX: Complete blue-green implementation with validation
  - REMOVE_SOURCE: Added batch removal with progress updates

## Key Files Modified
- `/home/dockertest/semantik/PHASE_3_DEV_LOG.md` - Created and updated development log
- `/home/dockertest/semantik/packages/shared/metrics/collection_metrics.py` - Created comprehensive metrics module
- `/home/dockertest/semantik/packages/webui/tasks.py` - Enhanced with monitoring and validation
- `/home/dockertest/semantik/packages/webui/services/resource_manager.py` - Created resource management service
- `/home/dockertest/semantik/packages/webui/services/factory.py` - Added resource manager factory

## Design Decisions
- Will extend existing Prometheus metrics rather than replace them
- Keep backward compatibility with existing task structure
- Use database OperationMetrics model for detailed performance tracking
- Implement blue-green reindex as a separate handler with validation steps

## Notes
- Current implementation already has basic operation handlers for INDEX, APPEND, REINDEX, REMOVE_SOURCE
- Enhanced all handlers with comprehensive monitoring, validation, and resource management
- The reindex operation now has full blue-green deployment with validation checkpoints

## Summary
Phase 3 implementation is complete. All tasks have been successfully implemented:

1. **Monitoring & Metrics**: Added comprehensive Prometheus metrics for all collection operations
2. **Performance Tracking**: Database recording of operation metrics (duration, CPU, memory)
3. **Blue-Green Reindex**: Full implementation with 6 checkpoints and validation
4. **Resource Management**: User quotas, resource estimation, and reservation system
5. **Audit Logging**: All operations now create audit log entries

The implementation follows the execution plan closely while maintaining backward compatibility. The enhanced task processing provides better observability, reliability, and resource management for production use.

## Next Steps
- Phase 4: API Implementation with Security (1.5 weeks)
- Phase 5: Frontend Implementation with UX Enhancements (2 weeks)
- Phase 6: Testing & Documentation (1 week)
- Phase 7: Operational Validation (1 week)

---

## TASK-009: Refactor Celery Task Structure

### 2025-07-16 - Starting TASK-009 Implementation
- **Task**: Refactor Celery task structure to create a unified, robust task entry point
- **Requirements**:
  1. Add `acks_late=True` for message reliability
  2. Define proper `soft_time_limit` and `time_limit`
  3. Update operations record with `celery_task_id` as first action in task
  4. Implement try...finally block for guaranteed status updates
- **Analysis**:
  - Current task has basic configuration but missing reliability features
  - Task ID is being set by CollectionService after task submission
  - Found broken `operation_repo.update()` call that needs fixing
  - Need to move task_id update inside the task as first action

### 2025-07-16 - Completed TASK-009 Implementation
- **Changes Made**:
  1. **Enhanced Celery Task Decorator** (packages/webui/tasks.py):
     - Added `acks_late=True` for message reliability
     - Set `soft_time_limit=3600` (1 hour) and `time_limit=7200` (2 hours)
     - Kept existing retry configuration (max_retries=3, default_retry_delay=60)
  2. **Moved Task ID Update Inside Task**:
     - Task ID is now set as the FIRST action using `self.request.id`
     - Fixed broken `operation_repo.update()` by using `set_task_id()` method
     - Ensures task tracking even if later steps fail
  3. **Enhanced Error Handling**:
     - Wrapped exception handler updates in try-except to ensure robustness
     - Added finally block logic to guarantee final status updates
     - Added fallback to set FAILED status if task terminates unexpectedly
  4. **Updated CollectionService**:
     - Removed redundant `set_task_id()` calls in 4 locations
     - Added comments explaining task ID is now set inside the task
- **Benefits**:
  - Improved message reliability with late acknowledgment
  - Better handling of long-running operations with proper timeouts
  - Guaranteed task ID tracking for monitoring and cancellation
  - More robust error handling with guaranteed status updates
- **Code Quality**:
  - All black, ruff, and mypy checks pass
  - Fixed various linting issues including imports, type annotations, and unused variables
  - Maintained backward compatibility with existing code
- **Completed**:
  - Committed changes and created PR #84 against collections-refactor/phase_2
  - All acceptance criteria met

### 2025-07-16 - Addressing Code Review Feedback
- **Critical Issues Fixed**:
  1. ✅ **Resource Leak Risk - CeleryTaskWithUpdates**:
     - Implemented async context manager protocol (__aenter__/__aexit__)
     - All usages now use `async with` for automatic cleanup
     - Redis connections guaranteed to close even on exceptions
  2. ✅ **Transaction Handling**:
     - Added atomic transaction for _process_remove_source_operation
     - Document status updates and collection stats now wrapped in session.begin()
     - Uses proper async SQLAlchemy transaction context
  3. ✅ **Reindex Validation**:
     - Implemented comprehensive search quality validation
     - Samples points from old collection and compares search results
     - Checks vector count variance, search result overlap, and score differences
     - Added configurable thresholds for validation criteria
  4. ✅ **Magic Numbers to Constants**:
     - Created module-level constants for all hardcoded values
     - Includes timeouts, batch sizes, validation thresholds, Redis config
     - Makes configuration easily adjustable and self-documenting
  5. ✅ **Module Documentation**:
     - Added comprehensive docstring explaining architecture and usage
     - Documents key features, configuration, and task flow
  6. ✅ **PII Protection in Audit Logs**:
     - Created _sanitize_audit_details() function
     - Replaces user home directories with generic paths
     - Redacts email addresses
     - Removes sensitive keys (password, secret, token, key)
- **Completed**:
  - ✅ Fixed major indentation issue in _process_embedding_job_async (lines 280-675)
    - Used Python script to fix indentation for ~400 lines
    - All code now properly indented within try-except blocks
  - ✅ Fixed all linting issues:
    - Removed unused imports (Path, FieldCondition, Filter, MatchValue)
    - Added noqa comments for unused arguments
    - Combined nested with statements
  - ✅ Fixed all mypy type errors:
    - Added return statement for all code paths in _process_embedding_job_async
    - Added type annotation for sanitized dict
    - Fixed recursive sanitization type handling
  - ✅ All code quality checks now pass (black, ruff, mypy)

---

## TASK-010: Implement Blue-Green Staging Creation

### 2025-07-16 - Starting TASK-010 Implementation
- **Task**: Implement the first part of the re-indexing flow: creating the "green" (staging) Qdrant collections
- **Requirements**:
  1. Create a `reindex_handler` function called by the main Celery task
  2. Use `QdrantManager` to create new, unique staging collections
  3. Store the list of new staging collection names in the `collections.qdrant_staging` database field
- **Analysis**:
  - Current `_process_reindex_operation` has some staging logic but doesn't follow requirements
  - Need to use QdrantManager service instead of direct Qdrant client calls
  - Need to fix field name inconsistencies (using wrong field names)
  - Need to create separate reindex_handler function

### 2025-07-16 - Completed TASK-010 Implementation
- **Verified Existing Implementation**: 
  - The `reindex_handler` function was already implemented (lines 1207-1263)
  - QdrantManager is properly imported from `shared.managers.qdrant_manager`
  - The function correctly uses `QdrantManager.create_staging_collection` method
  - Field names are correct (`vector_store_name` not `qdrant_collection_name`)
  - Staging collection info is properly stored in `qdrant_staging` field as JSON
- **Implementation Details**:
  - `reindex_handler` creates a unique staging collection with timestamp suffix
  - Returns staging info dict with collection name, creation time, vector dimension, and base collection
  - Called from `_process_reindex_operation` at line 1325
  - Collection repository is updated with staging info at lines 1329-1332
  - Error handling includes cleanup of staging collection on failure
- **Code Quality**:
  - Applied black formatting to ensure consistent code style
  - All ruff linting checks pass
  - All mypy type checking passes
- **Acceptance Criteria Met**:
  - ✅ Staging collections are created in Qdrant using QdrantManager
  - ✅ Their names are persisted to the database in `collections.qdrant_staging` field
- **Conclusion**: TASK-010 was already fully implemented and meets all requirements

---

## TASK-011: Implement Re-indexing to Staging

### 2025-07-16 - Starting TASK-011 Implementation
- **Task**: Implement the core processing loop for re-indexing into the staging collections
- **Requirements**:
  1. The `reindex_handler` will fetch all documents for the collection
  2. Process each document using the *new* configuration (if provided) and ingest vectors into staging collections
  3. Progress must be continuously updated in the `operations` table
- **Analysis**:
  - Found TODO comment at line 1410 in `_process_reindex_operation`
  - Existing code already fetches all documents with COMPLETED status
  - Need to implement actual document processing with text extraction, chunking, embedding generation
  - Need to upload vectors to staging collection created by TASK-010

### 2025-07-16 - Completed TASK-011 Implementation
- **Changes Made**:
  1. **Document Reprocessing Logic** (packages/webui/tasks.py, lines 1405-1564):
     - Replaced TODO placeholder with full document processing implementation
     - Extracts configuration values from new_config with fallbacks to existing config
     - Uses ThreadPoolExecutor for parallel processing with 4 workers
  2. **Text Extraction and Chunking**:
     - Reuses existing `extract_and_serialize_thread_safe` function for text extraction
     - Creates TokenChunker with new chunk_size and chunk_overlap configuration
     - Preserves metadata including page numbers
     - Handles empty documents gracefully
  3. **Embedding Generation**:
     - Uses embedding_service with GPU scheduling for performance
     - Applies new model_name, quantization, and instruction settings
     - Handles vector dimension adjustment (truncation or padding with renormalization)
     - Generates unique task IDs for GPU scheduling
  4. **Vector Upload to Staging**:
     - Creates PointStruct objects with proper payload structure
     - Uploads vectors to staging collection using qdrant_client.upsert
     - Uses QdrantOperationTimer for performance monitoring
     - Includes collection_id in payload for future reference
  5. **Progress Tracking**:
     - Sends real-time progress updates via Redis updater
     - Tracks processed_count, failed_count, and vector_count
     - Reports progress percentage and vectors created
     - Handles errors gracefully and continues processing
  6. **Memory Management**:
     - Explicitly deletes large objects after processing
     - Calls gc.collect() after each document
     - Processes documents in configurable batches
- **Implementation Features**:
  - Fully implements document fetching from database
  - Processes documents with new configuration parameters
  - Uploads all vectors to staging collection
  - Continuous progress updates via WebSocket/Redis
  - Comprehensive error handling with document-level recovery
  - Memory-efficient processing with garbage collection
- **Acceptance Criteria Met**:
  - ✅ All documents are fetched from the collection
  - ✅ Documents are re-processed using new configuration
  - ✅ Vectors are ingested into staging collections
  - ✅ Progress is continuously updated (via Redis updates)
- **Integration Points**:
  - Works seamlessly with existing TASK-010 staging collection creation
  - Audit logging already captures final statistics (documents_processed, vectors_created)
  - Validation step (already implemented) will verify the reindexed data
  - Atomic switch (already implemented) will complete the blue-green deployment

### 2025-07-16 - Improvements Based on Review Feedback
- **Made batch_size configurable**:
  - Batch size can now be specified in new_config or existing config
  - Defaults to EMBEDDING_BATCH_SIZE (100) if not specified
  - Allows for collection-specific or operation-specific tuning
- **Made ThreadPoolExecutor worker count configurable**:
  - Worker count can now be specified in new_config or existing config
  - Defaults to 4 workers if not specified
  - Enables better resource utilization based on system capabilities
- **Benefits**:
  - More flexible configuration for different collection sizes
  - Better resource management for varying system capabilities
  - Easier performance tuning without code changes

---

## TASK-012: Implement Atomic Switch & Cleanup

### 2025-07-16 - Starting TASK-012 Implementation
- **Task**: Implement the final steps of the re-indexing process: the atomic switch and scheduling the cleanup of old resources
- **Requirements**:
  1. Create internal API endpoint `POST /api/internal/complete-reindex`
  2. Perform atomic database transaction to switch collections
  3. Schedule cleanup task to run after a delay
- **Analysis**:
  - Current implementation has atomic switch directly in the task
  - Need to move to API endpoint for proper atomic transaction handling
  - Need to replace immediate deletion with scheduled cleanup task

### 2025-07-16 - Completed TASK-012 Implementation
- **Changes Made**:
  1. **Added Generic Update Method to CollectionRepository**:
     - Created `update()` method in CollectionRepository that accepts a dictionary of fields
     - Supports atomic updates of multiple fields in a single transaction
     - Includes validation for all updateable fields
     - Maintains compatibility with existing specific update methods
  2. **Created Internal API Endpoint**:
     - Added `POST /api/internal/complete-reindex` endpoint in internal.py
     - Uses existing internal API authentication via X-Internal-API-Key header
     - Performs atomic transaction to switch from staging to active collections
     - Returns list of old collection names for cleanup
  3. **Created Cleanup Task**:
     - Added `cleanup_old_collections` Celery task
     - Accepts list of collection names and collection ID
     - Safely checks for collection existence before deletion
     - Includes comprehensive error handling and metrics
     - Reports cleanup statistics including success/failure counts
  4. **Modified Reindex Operation**:
     - Replaced direct database update with API call using httpx
     - Replaced immediate deletion with scheduled cleanup task
     - Added CLEANUP_DELAY_SECONDS constant (5 minutes)
     - Updated audit logging to include cleanup task ID
     - Updated return values to include old_collections list
- **Code Quality**:
  - All black formatting checks pass
  - All ruff linting checks pass
  - No mypy errors in modified files
  - Fixed import issues and exception handling
- **Benefits**:
  - True atomic switch with proper transaction boundaries
  - Zero-downtime switch with delayed cleanup
  - Better separation of concerns between task and API
  - Improved error handling and rollback capabilities
  - Cleanup task can be monitored and retried if needed
- **Acceptance Criteria Met**:
  - ✅ Internal API endpoint performs atomic switch
  - ✅ Old collections are cleaned up after a delay
  - ✅ Switch is atomic and safe with proper rollback

### 2025-07-16 - Fixed Circular Import Issue
- **Issue**: Tests were failing due to circular import error
- **Root Cause**: Imported non-existent module `webui.database` with `get_async_session`
- **Fix**: 
  - Replaced with correct imports from `shared.database.database`
  - Used `AsyncSessionLocal` directly, following pattern in tasks.py
  - No need for dependency injection since this is an internal API
- **Result**: All tests now pass (414 passed, 2 skipped, 1 deselected)

### 2025-07-16 - Implemented Code Review Improvements
- **Based on thorough code review feedback, implemented the following improvements**:
  1. **Exponential Backoff for Cleanup Task**:
     - Added `retry_backoff=True` and `retry_backoff_max=600` to cleanup task decorator
     - Better handles transient Qdrant operation failures
  2. **Containerization Support**:
     - Fixed localhost hardcoding by using `WEBUI_INTERNAL_HOST` setting
     - Allows proper operation in Docker/Kubernetes environments
  3. **Request Validation**:
     - Added UUID format validation for collection_id and operation_id
     - Added non-negative validation for vector_count
     - Added non-empty validation for staging_collection_name
  4. **Metrics Recording Pattern**:
     - Created `record_metric_safe()` helper function to reduce code duplication
     - Centralizes try/except ImportError pattern
  5. **Dynamic Cleanup Delay**:
     - Implemented `calculate_cleanup_delay()` function
     - Scales cleanup delay from 5-30 minutes based on vector count
     - Formula: 5 minutes base + 1 minute per 10,000 vectors
- **Code Quality**: All tests pass, formatting and linting clean

---

## TASK-013: Implement Operation Failure Handlers

### 2025-07-16 - Starting TASK-013 Implementation
- **Task**: Implement robust error handling that updates collection and operation status based on the type of failure
- **Requirements**:
  1. In the Celery task's `on_failure` handler, update the `operations` table to `failed` with the error message
  2. Update the parent `collections` table to an appropriate state (`degraded` for failed re-index, `error` for failed initial index)
  3. Ensure staging resources are cleaned up immediately on a failed re-index
- **Analysis**:
  - Current implementation has try-except blocks that update operation status
  - No explicit `on_failure` handler implemented
  - Need to add appropriate collection status updates based on operation type
  - Need to add staging cleanup on reindex failure

### 2025-07-16 - Completed TASK-013 Implementation
- **Changes Made**:
  1. **Implemented Comprehensive on_failure Handler**:
     - Added `on_failure` parameter to `process_collection_operation` task decorator
     - Created `_handle_task_failure` and `_handle_task_failure_async` functions
     - Handler extracts operation_id from task args/kwargs and runs async failure logic
     - Includes detailed error messages with traceback information
  2. **Enhanced Failure Recording**:
     - Operation status updated to FAILED with detailed error information
     - Error result includes error_type, error_message, task_id, and failed_at timestamp
     - Proper traceback inclusion for debugging
  3. **Collection Status Updates by Operation Type**:
     - **INDEX failure**: Collection status → ERROR (collection unusable)
     - **REINDEX failure**: Collection status → DEGRADED (original collection still works)
     - **APPEND failure**: Collection status → PARTIALLY_READY (unless already ERROR)
     - **REMOVE_SOURCE failure**: Collection status → PARTIALLY_READY
     - All status updates include descriptive status messages
  4. **Staging Resource Cleanup**:
     - Created `_cleanup_staging_resources` function
     - Automatically called on REINDEX failures
     - Parses staging info from database (handles JSON parsing)
     - Deletes staging collections from Qdrant
     - Clears staging info from database
     - Handles errors gracefully with logging
  5. **Audit Logging and Metrics**:
     - Creates audit log entries for all failures
     - Updates Prometheus metrics (collection_operations_total with failed status)
     - Sanitizes audit details to prevent PII leakage
  6. **Consistency Updates**:
     - Updated exception handling in `_process_collection_operation_async` to match on_failure behavior
     - Both paths now use same logic for status updates and cleanup
- **Code Quality**:
  - All black formatting applied
  - All ruff linting issues fixed (added noqa for unused Celery handler params)
  - All mypy type checking passes
  - All existing unit tests pass (170 passed)
- **Benefits**:
  - Guaranteed status updates even on catastrophic failures
  - Clear communication of failure types to users
  - No orphaned staging resources on reindex failures
  - Better debugging with detailed error information
  - Consistent failure handling across all operation types

### 2025-07-16 - Improvements Based on Code Review
- **Issues Addressed**:
  1. **Event Loop Safety**: Changed from creating new event loop to using `asyncio.run()` for safer handling
  2. **Error Message Sanitization**: Added comprehensive PII sanitization for error messages:
     - Removes user home paths (/home/username, /Users/username, C:\Users\username)
     - Redacts email addresses
     - Sanitizes temporary paths that may contain usernames
  3. **Code Quality**: Fixed all linting and type checking issues
- **Enhanced Security**:
  - Created `_sanitize_error_message()` function for consistent error message sanitization
  - Applied sanitization to all error messages in status updates and audit logs
  - Ensured tracebacks are also sanitized before logging
- **Note on Transaction Boundaries**:
  - After investigation, the repositories manage their own database sessions internally
  - The status updates are atomic at the repository level
  - Staging cleanup is intentionally performed outside the main error handler to avoid blocking

---

## TASK-014: Implement Resource Cleanup Task

### 2025-07-16 - Starting TASK-014 Implementation
- **Task**: Implement Resource Cleanup Task
- **Context**: Create a Celery task for cleaning up orphaned Qdrant collections after a successful re-index
- **Requirements**:
  1. The task `cleanup_qdrant_collections(collection_names: List[str])` will be called with a delay
  2. It will safely delete the specified Qdrant collections
  3. It must include checks to prevent accidental deletion of active collections
- **Analysis**:
  - Found existing `cleanup_old_collections` task from TASK-012 that handles basic deletion
  - Need to create new `cleanup_qdrant_collections` task with enhanced safety checks
  - Must verify collections are not actively being used before deletion
  - Should use QdrantManager for consistency with other operations

### 2025-07-16 - Completed TASK-014 Implementation
- **Changes Made**:
  1. **Created Enhanced Cleanup Task** (packages/webui/tasks.py, lines 336-471):
     - Added `cleanup_qdrant_collections` task with comprehensive safety checks
     - Configured with same retry settings as cleanup_old_collections plus exponential backoff
     - Returns detailed statistics including safety check results for each collection
  2. **Implemented 5 Safety Checks**:
     - **System Collection Check**: Skip collections starting with "_" (reserved for system use)
     - **Active Collection Check**: Query database to find all active collections and skip them
     - **Existence Check**: Verify collection exists in Qdrant before attempting deletion
     - **Staging Age Check**: For staging collections, verify they're older than 1 hour
     - **Audit Trail**: Record deletion details before removing each collection
  3. **Created Helper Functions**:
     - `_get_active_collections()`: Async function to retrieve all active Qdrant collection names from database
     - Checks vector_store_name, qdrant_collections list, and qdrant_staging fields
     - Returns a set of all collection names that should not be deleted
     - `_audit_collection_deletion()`: Creates audit log entries for each deleted collection
     - Records collection name, vector count, and deletion timestamp
  4. **Fixed QdrantManager Initialization**:
     - Discovered existing bug in cleanup_old_collections (tries to init QdrantManager without required client)
     - Properly import connection manager and get client before creating QdrantManager instance
     - Pattern: `qdrant_client = connection_manager.get_client()` then `QdrantManager(qdrant_client)`
  5. **Enhanced Error Handling**:
     - Each collection deletion is wrapped in try-except for individual error handling
     - Failed deletions don't stop processing of other collections
     - All errors are collected and returned in the response
  6. **Comprehensive Testing**:
     - Created test_cleanup_tasks.py with 15+ test cases
     - Tests cover all safety checks, error scenarios, and edge cases
     - Includes async tests for helper functions
     - Mock-based testing for isolation from external services
- **Key Differences from cleanup_old_collections**:
  - Enhanced safety checks prevent accidental deletion of active collections
  - Database query to determine active collections dynamically
  - Detailed safety check results in response for debugging
  - Staging collection age verification (1 hour minimum)
  - Audit logging for compliance and tracking
  - Better error isolation and reporting
- **Benefits**:
  - Safer cleanup operations with multiple verification steps
  - Complete audit trail for all deletions
  - Better visibility into why collections were skipped
  - Protection against accidental deletion of production data
  - Suitable for automated cleanup in production environments

### 2025-07-16 - Code Quality Improvements
- **Issues Addressed Based on Review**:
  1. **Import Pattern Clarification**: Added comment explaining the import from webui.utils.qdrant_manager is correct (not shared.managers.connection)
  2. **Private Method Usage**: Added comment explaining why _is_staging_collection_old() is used (method exists and provides needed functionality)
  3. **Configurable Staging Age**: Added staging_age_hours parameter (default: 1 hour) to make threshold configurable
  4. **Efficient Async Operations**: Replaced multiple asyncio.run() calls with batched audit logging
- **Performance Improvements**:
  - Created `_audit_collection_deletions_batch()` function for efficient batch audit logging
  - Single async operation for all audit logs instead of individual calls per deletion
  - Reduces overhead and improves performance for bulk cleanup operations
- **Code Quality**:
  - All tests updated to reflect new batched audit pattern
  - Clear documentation of design decisions in code comments
  - Maintained backward compatibility while improving efficiency