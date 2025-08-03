# Semantik Chunking Strategies - Enhanced Implementation Plan

## Overview
This implementation plan has been rewritten following Claude 4 prompt engineering best practices, with explicit instructions, contextual motivation, and structured formatting to maximize agent effectiveness.

**Timeline**: 4 weeks (includes buffer for unexpected issues)  
**Strategies**: 6 (character, recursive, markdown, semantic, hierarchical, hybrid)  
**Code Support**: Deferred to follow-up release (~2 weeks post-launch)

---

## Week 1: Core Foundation & Architecture

### Task 1.1: Base Architecture & Core Strategies

<task>
<metadata>
  <priority>High</priority>
  <effort>3 days</effort>
  <dependencies>llama-index, sentence-transformers</dependencies>
</metadata>

<context>
You are implementing the foundational chunking system for Semantik. This is critical because all future chunking functionality will build upon this base architecture. Poor design decisions here will compound throughout the project. The goal is to create a flexible, performant foundation that supports multiple chunking strategies while maintaining clean separation of concerns.
</context>

<instructions>
Create a robust base architecture with three core chunking strategies. Your implementation must be production-ready, secure, and performant from day one.

**Step 1: Design the BaseChunker Interface**
- Create an abstract base class that all chunking strategies will inherit from
- Include both synchronous and asynchronous methods for maximum flexibility
- Design with type safety using dataclasses for ChunkResult
- Include validation, estimation, and metadata handling capabilities
- Ensure the interface can handle streaming for large documents

**Step 2: Implement ChunkingService Layer**
- Create a service layer that separates business logic from the chunkers
- Include security validation, caching, error handling, and analytics
- Design for testability with dependency injection
- Implement preview functionality with configurable limits
- Add performance metrics collection from the start

**Step 3: Build Three Core Strategies Using LlamaIndex**
1. **Character Strategy (TokenTextSplitter)**
   - Configure for 1000 token chunks with 200 token overlap
   - Optimize for general text processing
   - Target: 1000 chunks/sec single-threaded

2. **Recursive Strategy (SentenceSplitter)**  
   - Configure for 600 token chunks with 100 token overlap
   - This will temporarily handle code files with optimized parameters
   - For code files: use 400 token chunks with 50 token overlap
   - Target: 800 chunks/sec single-threaded

3. **Markdown Strategy (MarkdownNodeParser)**
   - Respect document structure and headers
   - Preserve formatting for technical documentation
   - Target: 600 chunks/sec single-threaded
</instructions>

<available_tools>
- **context7**: Look up LlamaIndex node parsers/text splitters documentation, sentence-transformers API
- **Read/Edit/MultiEdit**: For code implementation
- **Bash**: Run tests and validate implementation
- **Grep/Glob**: Search existing codebase for patterns
- **WebSearch**: Research chunking best practices if needed
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="implementation" name="backend-api-architect">
    <purpose>Design BaseChunker interface and ChunkingService layer with proper separation of concerns</purpose>
    <focus>Architecture patterns, async support, dependency injection</focus>
  </agent>
  
  <agent type="implementation" name="database-migrations-engineer">
    <purpose>Plan data model changes for chunking metadata storage</purpose>
    <focus>Schema design, indexing strategy, performance considerations</focus>
  </agent>
  
  <agent type="review" name="backend-code-reviewer">
    <purpose>Validate architecture decisions and implementation quality</purpose>
    <focus>Security, performance, maintainability, error handling</focus>
  </agent>
  
  <agent type="testing" name="test-maestro">
    <purpose>Ensure comprehensive test coverage from the start</purpose>
    <focus>Unit tests, integration tests, performance benchmarks</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="BaseChunker Interface">
```python
# /packages/shared/text_processing/base_chunker.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChunkResult:
    """Type-safe chunk result with all necessary metadata."""
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class BaseChunker(ABC):
    """Base class for all chunking strategies with both sync and async support."""
    
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Synchronous chunking for simple use cases."""
        pass
    
    @abstractmethod
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Asynchronous chunking for I/O bound operations and better concurrency."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate strategy-specific configuration before use."""
        pass
    
    @abstractmethod
    def estimate_chunks(self, text_length: int, config: Dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning and progress tracking."""
        pass
```
</example>

<example name="ChunkingService Layer">
```python
# /packages/webui/services/chunking_service.py
class ChunkingService:
    """Service layer providing business logic, caching, and security for chunking."""
    
    async def preview_chunking(
        self,
        text: str,
        file_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        max_chunks: int = 5
    ) -> ChunkingPreviewResponse:
        """Preview chunking with comprehensive validation and caching."""
        # Security validation prevents malicious inputs
        self.security.validate_document_size(len(text))
        
        # Smart configuration based on file type
        if not config and file_type:
            config = FileTypeDetector.get_optimal_config(file_type)
        
        # Validate parameters are within safe bounds
        self.security.validate_chunk_params(config.get("params", {}))
        
        # Check cache for performance
        config_hash = self._hash_config(config)
        cached = await self._get_cached_preview(config_hash, text[:1000])
        if cached:
            return cached
        
        # Process with appropriate chunker
        chunker = ChunkingFactory.create_chunker(config)
        chunks = await chunker.chunk_text_async(text, "preview")
        
        # Build comprehensive response
        response = ChunkingPreviewResponse(
            chunks=chunks[:max_chunks],
            total_chunks=len(chunks),
            strategy_used=config["strategy"],
            is_code_file=file_type in FileTypeDetector.CODE_EXTENSIONS,
            performance_metrics=self._calculate_metrics(chunks, len(text)),
            recommendations=self._get_recommendations(chunks, file_type)
        )
        
        # Cache for next time
        await self._cache_preview(config_hash, text[:1000], response)
        
        return response
```
</example>
</implementation_examples>

<success_criteria>
- [ ] BaseChunker interface supports both sync and async operations
- [ ] ChunkingService properly separates business logic from implementation
- [ ] Security validation prevents malicious inputs (oversized docs, invalid params)
- [ ] All 3 core strategies working correctly with LlamaIndex
- [ ] Performance benchmarks meet targets (character: 1000/sec, recursive: 800/sec, markdown: 600/sec)
- [ ] Code files handled gracefully with optimized recursive parameters
- [ ] Comprehensive error handling covers all failure modes
- [ ] Test coverage > 90% for critical paths
</success_criteria>

<common_pitfalls>
- **Avoid**: Tight coupling between chunkers and business logic
- **Avoid**: Synchronous-only implementation that limits scalability
- **Avoid**: Missing security validation on user inputs
- **Avoid**: Hardcoding configuration values instead of making them configurable
- **Avoid**: Ignoring memory constraints for large documents
</common_pitfalls>
</task>

### Task 1.2: Error Handling & Recovery Framework

<task>
<metadata>
  <priority>High</priority>
  <effort>1.5 days</effort>
  <dependencies>Task 1.1 completion</dependencies>
</metadata>

<context>
Robust error handling is crucial for production reliability. Chunking operations can fail for many reasons: memory constraints, encoding issues, timeout errors, or malformed input. Without proper error handling, a single problematic document could crash the entire system or leave collections in an inconsistent state. This task ensures graceful degradation and recovery capabilities.
</context>

<instructions>
Build a comprehensive error handling framework that anticipates and gracefully handles all failure modes in the chunking system.

**Step 1: Define Error Taxonomy**
- Create an enum of all possible error types (memory, timeout, encoding, strategy, partial failure)
- Design error classes with rich context for debugging
- Include structured error codes for programmatic handling

**Step 2: Implement Recovery Strategies**
- Design retry logic with exponential backoff for transient failures
- Implement batch size reduction for memory errors
- Create timeout extension logic for slow operations
- Build fallback encoding handlers for problematic text

**Step 3: Handle Partial Failures**
- Save successful chunks even when some documents fail
- Analyze failure patterns to provide actionable recommendations
- Update collection status to reflect degraded state
- Create recovery operations for failed documents

**Step 4: Streaming Failure Recovery**
- Implement checkpoint-based recovery for large documents
- Reduce batch sizes dynamically on memory pressure
- Extend timeouts adaptively based on document size
- Mark unrecoverable documents with detailed error information
</instructions>

<available_tools>
- **context7**: Research error handling patterns in FastAPI/Celery
- **Read/Edit/MultiEdit**: Implement error handling framework
- **Bash**: Test error scenarios
- **Grep**: Search for existing error handling patterns in codebase
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="implementation" name="backend-api-architect">
    <purpose>Design robust error handling patterns that scale</purpose>
    <focus>Error taxonomy, recovery strategies, partial failure handling</focus>
  </agent>
  
  <agent type="review" name="backend-code-reviewer">
    <purpose>Ensure error handling is comprehensive and secure</purpose>
    <focus>Edge cases, security implications, error message safety</focus>
  </agent>
  
  <agent type="testing" name="qa-bug-hunter">
    <purpose>Create comprehensive error scenario tests</purpose>
    <focus>Edge cases, stress testing, recovery verification</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="Error Type Definition">
```python
# /packages/webui/services/chunking_error_handler.py
from enum import Enum
from typing import List, Dict, Any, Optional

class ChunkingErrorType(Enum):
    """Comprehensive taxonomy of chunking errors."""
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    INVALID_ENCODING = "invalid_encoding"
    STRATEGY_ERROR = "strategy_error"
    PARTIAL_FAILURE = "partial_failure"

class ChunkingErrorHandler:
    """Intelligent error handling with automatic recovery strategies."""
    
    RETRY_STRATEGIES = {
        ChunkingErrorType.MEMORY_ERROR: {
            "max_retries": 2,
            "backoff": "exponential",
            "reduce_batch_size": True,
            "initial_delay": 1.0,
            "max_delay": 10.0
        },
        ChunkingErrorType.TIMEOUT_ERROR: {
            "max_retries": 3,
            "backoff": "linear",
            "increase_timeout": True,
            "timeout_multiplier": 1.5
        },
        ChunkingErrorType.INVALID_ENCODING: {
            "max_retries": 1,
            "fallback_encoding": "utf-8",
            "lossy_decode": True,
            "strip_invalid_chars": True
        }
    }
```
</example>

<example name="Partial Failure Handling">
```python
async def handle_partial_failure(
    self,
    operation_id: str,
    processed_chunks: List[ChunkResult],
    failed_documents: List[str],
    errors: List[Exception]
) -> ChunkingOperationResult:
    """Handle partial chunking failures with grace and intelligence."""
    
    # Always save what succeeded - don't lose good work
    await self.save_partial_results(operation_id, processed_chunks)
    
    # Analyze failures to understand patterns
    failure_analysis = self.analyze_failures(errors)
    
    # Create smart recovery strategy based on failure types
    recovery_strategy = self.create_recovery_strategy(
        failure_analysis,
        failed_documents
    )
    
    # Update status with actionable information
    await self.update_collection_status(
        operation_id,
        CollectionStatus.DEGRADED,
        f"Partial failure: {len(failed_documents)} documents failed. "
        f"Primary cause: {failure_analysis.primary_cause}"
    )
    
    # Create recovery operation for retry
    recovery_op = await self.create_recovery_operation(
        operation_id,
        recovery_strategy
    )
    
    # Return comprehensive result
    return ChunkingOperationResult(
        status="partial_success",
        processed_count=len(processed_chunks),
        failed_count=len(failed_documents),
        recovery_operation_id=recovery_op.id,
        recommendations=recovery_strategy.recommendations,
        estimated_recovery_time=recovery_strategy.estimated_duration
    )
```
</example>

<example name="Streaming Recovery">
```python
async def handle_streaming_failure(
    self,
    document_id: str,
    bytes_processed: int,
    error: Exception
) -> StreamRecoveryAction:
    """Intelligently recover from streaming failures."""
    
    if isinstance(error, MemoryError):
        # Calculate new batch size based on failure point
        current_batch_size = self.get_current_batch_size()
        new_batch_size = max(
            1024,  # Minimum 1KB
            current_batch_size // 2  # Halve the batch size
        )
        
        return StreamRecoveryAction(
            action="retry_from_checkpoint",
            checkpoint=bytes_processed,
            new_batch_size=new_batch_size,
            reason="Memory pressure detected, reducing batch size"
        )
    
    elif isinstance(error, TimeoutError):
        # Adaptively extend timeout based on progress
        progress_rate = bytes_processed / self.elapsed_time
        estimated_total_time = self.document_size / progress_rate
        new_timeout = min(
            estimated_total_time * 1.5,  # 50% buffer
            self.max_timeout  # Hard limit
        )
        
        return StreamRecoveryAction(
            action="retry_with_extended_timeout",
            checkpoint=bytes_processed,
            new_timeout=new_timeout,
            reason=f"Slow processing detected, extending timeout to {new_timeout}s"
        )
    
    else:
        # Unrecoverable error - provide detailed diagnostics
        return StreamRecoveryAction(
            action="mark_failed",
            error_details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "bytes_processed": bytes_processed,
                "processing_time": self.elapsed_time,
                "suggested_action": self.suggest_manual_intervention(error)
            }
        )
```
</example>
</implementation_examples>

<success_criteria>
- [ ] All error types have defined recovery strategies
- [ ] Partial failures don't lose successfully processed data
- [ ] Memory errors trigger automatic batch size reduction
- [ ] Timeout errors adaptively extend based on document size
- [ ] Encoding errors have multiple fallback strategies
- [ ] Failed documents can be retried with adjusted parameters
- [ ] Error messages are informative but don't leak sensitive data
- [ ] Recovery operations are idempotent and resumable
</success_criteria>

<common_pitfalls>
- **Avoid**: Swallowing errors silently without logging
- **Avoid**: Exposing internal details in error messages
- **Avoid**: Infinite retry loops without backoff
- **Avoid**: Losing successful work during partial failures
- **Avoid**: Fixed retry strategies that don't adapt to failure types
</common_pitfalls>
</task>

### Task 1.3: Performance Testing Framework

<task>
<metadata>
  <priority>High</priority>
  <effort>1 day</effort>
  <dependencies>Tasks 1.1, 1.2 completion</dependencies>
</metadata>

<context>
Performance testing from Day 1 is critical to avoid nasty surprises later. Without concrete benchmarks, you might build a system that works correctly but is too slow for production use. This framework will validate performance targets continuously, catch regressions early, and guide optimization efforts. Remember: what gets measured gets managed.
</context>

<instructions>
Build a comprehensive performance testing framework that validates all chunking strategies meet their targets.

**Step 1: Define Performance Baselines**
- Hardware baseline: 4-core CPU, 8GB RAM (standard container)
- Strategy-specific targets per the requirements
- Document size profiles (1KB to 100MB)
- Memory usage limits (<100MB for 10MB document)

**Step 2: Create Performance Test Suite**
- Parameterized tests for all strategies and document sizes
- Resource monitoring during execution
- Parallel processing scalability tests
- Memory leak detection tests

**Step 3: Build Continuous Benchmarking**
- Automated benchmark runs on each commit
- Performance regression detection
- Historical trend tracking
- Automated alerts for degradation

**Step 4: Implement Load Testing**
- Concurrent document processing
- Queue saturation testing
- Resource exhaustion scenarios
- Graceful degradation verification
</instructions>

<available_tools>
- **context7**: Look up pytest-benchmark, memory_profiler documentation
- **Bash**: Run performance benchmarks
- **WebSearch**: Research chunking performance best practices
- **Read/Edit**: Implement test framework
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="implementation" name="performance-profiler">
    <purpose>Design comprehensive benchmark suite</purpose>
    <focus>Metrics collection, baseline establishment, bottleneck identification</focus>
  </agent>
  
  <agent type="review" name="backend-code-reviewer">
    <purpose>Ensure test quality and coverage</purpose>
    <focus>Test completeness, measurement accuracy, edge cases</focus>
  </agent>
  
  <agent type="analysis" name="performance-profiler">
    <purpose>Interpret initial results and guide optimization</purpose>
    <focus>Bottleneck analysis, optimization recommendations</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="Performance Test Suite">
```python
# /tests/performance/test_chunking_performance.py
import pytest
import asyncio
import time
import psutil
import memory_profiler
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """Comprehensive performance measurement results."""
    chunks_per_second: float
    peak_memory_mb: float
    avg_memory_mb: float
    total_time_seconds: float
    cpu_percent: float
    
class TestChunkingPerformance:
    """Performance tests ensuring all strategies meet their targets."""
    
    # Define concrete performance targets
    PERFORMANCE_TARGETS = {
        "character": {
            "single_thread": 1000,  # chunks/sec
            "parallel_4": 3500,
            "memory_per_mb": 50     # MB memory per MB document
        },
        "recursive": {
            "single_thread": 800,
            "parallel_4": 3000,
            "memory_per_mb": 60
        },
        "markdown": {
            "single_thread": 600,
            "parallel_4": 2200,
            "memory_per_mb": 80
        }
    }
    
    @pytest.fixture
    def performance_monitor(self):
        """Monitor resource usage during tests."""
        return PerformanceMonitor()
    
    @pytest.mark.parametrize("strategy,document_size,expected_rate", [
        ("character", "1MB", 1000),
        ("recursive", "1MB", 800),
        ("markdown", "1MB", 600),
        # Test with different sizes to ensure scalability
        ("character", "10MB", 900),   # Allow slight degradation
        ("recursive", "10MB", 700),
        ("markdown", "10MB", 500),
    ])
    async def test_single_thread_performance(
        self,
        strategy: str,
        document_size: str,
        expected_rate: int,
        performance_monitor
    ):
        """Verify single-threaded performance meets targets."""
        # Generate realistic test document
        document = self.generate_test_document(document_size)
        
        # Create chunker with production config
        config = self.get_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Start comprehensive monitoring
        performance_monitor.start()
        
        # Measure chunking performance
        start_time = time.time()
        chunks = await chunker.chunk_text_async(document, "test_doc")
        duration = time.time() - start_time
        
        # Stop monitoring and collect metrics
        metrics = performance_monitor.stop()
        
        # Calculate performance indicators
        chunks_per_second = len(chunks) / duration
        
        # Verify performance meets targets with 10% tolerance
        assert chunks_per_second >= expected_rate * 0.9, \
            f"{strategy} performance {chunks_per_second:.1f} chunks/sec " \
            f"below target {expected_rate} chunks/sec"
        
        # Verify memory usage is reasonable
        doc_size_mb = len(document) / (1024 * 1024)
        expected_memory = doc_size_mb * self.PERFORMANCE_TARGETS[strategy]["memory_per_mb"]
        assert metrics.peak_memory_mb < expected_memory * 1.2, \
            f"Memory usage {metrics.peak_memory_mb:.1f}MB exceeds " \
            f"limit {expected_memory:.1f}MB for {document_size} document"
        
        # Log results for tracking
        self.log_performance_result(
            strategy, document_size, chunks_per_second, metrics
        )
```
</example>

<example name="Parallel Processing Tests">
```python
@pytest.mark.parametrize("num_workers", [2, 4, 8])
async def test_parallel_scalability(self, num_workers: int):
    """Verify parallel processing scales efficiently."""
    documents = [
        self.generate_test_document("100KB") 
        for _ in range(100)
    ]
    
    # Baseline: single worker performance
    single_start = time.time()
    for doc in documents:
        await self.chunk_document(doc, strategy="recursive")
    single_duration = time.time() - single_start
    
    # Parallel processing
    parallel_start = time.time()
    await self.process_parallel(documents, num_workers)
    parallel_duration = time.time() - parallel_start
    
    # Calculate efficiency metrics
    speedup = single_duration / parallel_duration
    efficiency = speedup / num_workers
    
    # Should achieve at least 70% efficiency
    assert efficiency >= 0.7, \
        f"Parallel efficiency {efficiency:.2f} below 70% threshold " \
        f"for {num_workers} workers"
    
    # Log scalability results
    logger.info(
        f"Parallel scaling: {num_workers} workers, "
        f"speedup: {speedup:.2f}x, efficiency: {efficiency:.2%}"
    )
```
</example>

<example name="Memory Leak Detection">
```python
@pytest.mark.slow
async def test_memory_leak_detection(self):
    """Ensure no memory leaks during extended operation."""
    import gc
    import tracemalloc
    
    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    
    # Get baseline memory
    baseline = tracemalloc.get_traced_memory()[0]
    
    # Process many documents
    for i in range(100):
        document = self.generate_test_document("1MB")
        chunker = ChunkingFactory.create_chunker({
            "strategy": "recursive",
            "params": {"chunk_size": 600}
        })
        
        chunks = await chunker.chunk_text_async(document, f"doc_{i}")
        
        # Explicitly clean up
        del chunks
        del chunker
        
        # Check memory every 10 iterations
        if i % 10 == 0:
            gc.collect()
            current = tracemalloc.get_traced_memory()[0]
            memory_growth = (current - baseline) / (1024 * 1024)  # MB
            
            # Memory growth should be minimal
            assert memory_growth < 50, \
                f"Memory leak detected: {memory_growth:.1f}MB growth " \
                f"after {i} iterations"
    
    tracemalloc.stop()
```
</example>

<example name="Load Testing">
```python
async def test_system_under_load(self):
    """Test system behavior under heavy load."""
    # Simulate production load
    concurrent_operations = 50
    documents_per_operation = 20
    
    async def process_collection():
        """Simulate a collection processing operation."""
        docs = [
            self.generate_test_document("500KB") 
            for _ in range(documents_per_operation)
        ]
        
        service = ChunkingService(
            db_session=get_test_db(),
            collection_repo=MockCollectionRepo(),
            document_repo=MockDocumentRepo(),
            redis_client=get_test_redis(),
            security_validator=ChunkingSecurityValidator()
        )
        
        results = []
        for doc in docs:
            result = await service.process_document(doc)
            results.append(result)
        
        return results
    
    # Launch concurrent operations
    start_time = time.time()
    tasks = [
        process_collection() 
        for _ in range(concurrent_operations)
    ]
    
    # Monitor system resources during load
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    
    # Execute all operations
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    duration = time.time() - start_time
    metrics = resource_monitor.stop()
    
    # Analyze results
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    # System should handle load gracefully
    assert successful / len(results) >= 0.95, \
        f"Too many failures under load: {failed}/{len(results)}"
    
    # Resource usage should stay within bounds
    assert metrics.peak_cpu_percent < 90, \
        f"CPU usage {metrics.peak_cpu_percent}% too high under load"
    
    assert metrics.peak_memory_mb < 7000, \
        f"Memory usage {metrics.peak_memory_mb}MB exceeds limit under load"
    
    # Throughput should be reasonable
    total_docs = concurrent_operations * documents_per_operation
    docs_per_second = total_docs / duration
    
    assert docs_per_second >= 50, \
        f"Throughput {docs_per_second:.1f} docs/sec below minimum"
```
</example>
</implementation_examples>

<success_criteria>
- [ ] All strategies meet single-threaded performance targets
- [ ] Parallel processing achieves ≥70% efficiency
- [ ] Memory usage stays within defined limits
- [ ] No memory leaks detected in extended runs
- [ ] System handles concurrent load gracefully
- [ ] Performance regression detection works
- [ ] Benchmarks run automatically on commits
- [ ] Results are tracked historically
</success_criteria>

<common_pitfalls>
- **Avoid**: Testing with unrealistic documents (use production-like data)
- **Avoid**: Ignoring warmup effects in benchmarks
- **Avoid**: Testing only happy path scenarios
- **Avoid**: Neglecting resource cleanup in tests
- **Avoid**: Using mocks that hide real performance issues
</common_pitfalls>
</task>

### 🔍 Review 1.1: Foundation Validation

<review>
<metadata>
  <priority>Critical</priority>
  <effort>0.5 days</effort>
  <type>Checkpoint Review</type>
</metadata>

<context>
This is a critical checkpoint before proceeding to advanced strategies. Week 1 establishes the foundation that all future work builds upon. Any architectural flaws, performance issues, or security gaps discovered now can be fixed cheaply. Finding them later will be exponentially more expensive. This review ensures we have a solid foundation before adding complexity.
</context>

<instructions>
Conduct a comprehensive review of Week 1 deliverables before proceeding to Week 2.

**Step 1: Architecture Review**
- Validate BaseChunker interface design supports future strategies
- Ensure ChunkingService properly separates concerns
- Verify dependency injection enables testing
- Check that async/sync dual support works correctly

**Step 2: Security Validation**
- Confirm all user inputs are validated
- Verify size limits prevent DoS attacks
- Ensure error messages don't leak sensitive information
- Check parameter bounds enforcement

**Step 3: Performance Verification**
- Run all performance benchmarks
- Verify each strategy meets its targets
- Confirm memory usage stays within limits
- Validate parallel processing efficiency

**Step 4: Error Handling Assessment**
- Test all error recovery strategies
- Verify partial failures are handled gracefully
- Confirm streaming recovery works
- Validate error messages are actionable

**Step 5: Code Quality Check**
- Ensure >90% test coverage on critical paths
- Verify all code follows project standards
- Check documentation is complete
- Confirm no technical debt introduced
</instructions>

<decision_criteria>
<green_light>
- All core strategies working correctly
- Performance targets met or exceeded
- Security validation comprehensive
- Error handling covers all scenarios
- Test coverage >90%
- Code quality high
- Documentation complete
</green_light>

<yellow_light>
- Minor performance issues (within 20% of target)
- Some edge cases not fully covered
- Documentation needs minor updates
- Small refactoring needed
- **Action**: Fix issues in 1-2 days, then proceed
</yellow_light>

<red_light>
- Major architectural issues found
- Performance significantly below targets
- Security vulnerabilities discovered
- Critical functionality broken
- **Action**: Stop, reassess approach, may need major rework
</red_light>
</decision_criteria>

<review_checklist>
- [ ] BaseChunker interface supports both sync and async operations
- [ ] ChunkingService properly separates business logic
- [ ] Security validation prevents all malicious inputs
- [ ] Error handling covers all identified failure modes
- [ ] Performance tests establish accurate baselines
- [ ] All 3 core strategies working correctly
- [ ] Code files handled gracefully with recursive strategy
- [ ] Memory usage stays within defined limits
- [ ] Parallel processing achieves target efficiency
- [ ] Documentation explains all design decisions
</review_checklist>

<recommended_reviewers>
<reviewer name="backend-code-reviewer">
  <focus>Architecture quality, security, code standards</focus>
</reviewer>
<reviewer name="tech-debt-hunter">
  <focus>Identify potential future issues, maintainability</focus>
</reviewer>
<reviewer name="test-maestro">
  <focus>Test coverage analysis, edge case identification</focus>
</reviewer>
</recommended_reviewers>
</review>

---

## Week 2: Complete Strategies & Integration

### Task 2.1: Advanced Chunking Strategies

<task>
<metadata>
  <priority>High</priority>
  <effort>2 days</effort>
  <dependencies>Week 1 completion, review approval</dependencies>
</metadata>

<context>
Now that the foundation is solid, we implement the advanced strategies that differentiate Semantik. These strategies use AI and sophisticated algorithms to create superior chunks. Semantic chunking uses embeddings to find natural topic boundaries. Hierarchical chunking creates parent-child relationships for better context. Hybrid chunking intelligently selects strategies based on content. These are complex but provide significant value to users.
</context>

<instructions>
Implement the remaining three advanced chunking strategies using LlamaIndex.

**Strategy 1: Semantic Chunking**
- Use SemanticSplitterNodeParser from LlamaIndex
- Configure with appropriate embedding model (OpenAI or local)
- Set breakpoint_percentile_threshold for sensitivity
- Handle embedding API failures gracefully
- Target: 150 chunks/sec (limited by embeddings)

**Strategy 2: Hierarchical Chunking**
- Use HierarchicalNodeParser from LlamaIndex
- Configure multiple chunk size levels (2048, 512, 128)
- Maintain parent-child relationships in metadata
- Enable efficient context retrieval
- Target: 400 chunks/sec (multiple passes required)

**Strategy 3: Hybrid Chunking**
- Intelligently select strategy based on content type
- Use markdown for .md files or markdown-heavy content
- Default to recursive for general text
- Consider semantic for topic-heavy content
- Log strategy selection for analytics

**Implementation Requirements**:
- All strategies must follow BaseChunker interface
- Include comprehensive error handling
- Add performance monitoring
- Ensure streaming support for large documents
- Handle edge cases gracefully
</instructions>

<available_tools>
- **context7**: Look up sentence-transformers models, semantic chunking algorithms
- **Read/Edit/MultiEdit**: Implement advanced strategies
- **Bash**: Test with various document types
- **WebSearch**: Research semantic chunking best practices
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="implementation" name="backend-api-architect">
    <purpose>Design semantic and hierarchical chunker implementations</purpose>
    <focus>Algorithm selection, embedding integration, performance optimization</focus>
  </agent>
  
  <agent type="implementation" name="vector-search-architect">
    <purpose>Optimize embedding operations for semantic chunking</purpose>
    <focus>Embedding model selection, batch processing, caching strategies</focus>
  </agent>
  
  <agent type="review" name="backend-code-reviewer">
    <purpose>Ensure algorithm correctness and error handling</purpose>
    <focus>Edge cases, API failure handling, performance validation</focus>
  </agent>
  
  <agent type="performance" name="performance-profiler">
    <purpose>Optimize embedding operations and multi-pass algorithms</purpose>
    <focus>Bottleneck identification, memory usage, parallel processing</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="Semantic Chunker Implementation">
```python
# /packages/shared/text_processing/semantic_chunker.py
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import MockEmbedding
from llama_index.core import Document
import os
import asyncio
from typing import List, Dict, Any, Optional

class SemanticChunker(BaseChunker):
    """Semantic chunking using embeddings to find natural boundaries."""
    
    def __init__(
        self,
        embed_model=None,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1,
        max_chunk_size: int = 3000
    ):
        """Initialize with embedding model and parameters.
        
        Args:
            embed_model: Embedding model (defaults to OpenAI or Mock for testing)
            breakpoint_percentile_threshold: Sensitivity for topic boundaries (0-100)
            buffer_size: Context sentences to include around breakpoints
            max_chunk_size: Maximum tokens per chunk for safety
        """
        # Smart model selection based on environment
        if embed_model is None:
            if os.getenv("TESTING", "false").lower() == "true":
                embed_model = MockEmbedding(embed_dim=384)
                logger.info("Using MockEmbedding for testing")
            else:
                embed_model = OpenAIEmbedding(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    embed_batch_size=100  # Optimize for throughput
                )
                logger.info("Using OpenAI embeddings for semantic chunking")
        
        self.splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size
        )
        self.max_chunk_size = max_chunk_size
        self.strategy_name = "semantic"
        
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Asynchronously chunk text using semantic boundaries."""
        if not text.strip():
            return []
        
        try:
            # Create document for LlamaIndex
            doc = Document(text=text, metadata=metadata or {})
            
            # Run CPU-bound operation in executor
            loop = asyncio.get_event_loop()
            nodes = await loop.run_in_executor(
                None,
                self._chunk_with_retry,
                doc
            )
            
            # Convert nodes to ChunkResult format
            results = []
            for idx, node in enumerate(nodes):
                # Ensure chunk size safety
                chunk_text = node.text
                if len(chunk_text) > self.max_chunk_size:
                    chunk_text = chunk_text[:self.max_chunk_size]
                    logger.warning(
                        f"Truncated oversized semantic chunk: "
                        f"{len(node.text)} -> {self.max_chunk_size}"
                    )
                
                results.append(ChunkResult(
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    text=chunk_text,
                    start_offset=node.start_char_idx or 0,
                    end_offset=node.end_char_idx or len(chunk_text),
                    metadata={
                        **node.metadata,
                        "strategy": self.strategy_name,
                        "semantic_breakpoint_score": node.metadata.get("breakpoint_score", 0)
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}")
            # Fallback to recursive chunking
            logger.info("Falling back to recursive chunking")
            fallback = RecursiveChunker(chunk_size=600, chunk_overlap=100)
            return await fallback.chunk_text_async(text, doc_id, metadata)
    
    def _chunk_with_retry(self, doc: Document, max_retries: int = 3):
        """Chunk with retry logic for embedding API failures."""
        for attempt in range(max_retries):
            try:
                return self.splitter.get_nodes_from_documents([doc])
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Semantic chunking attempt {attempt + 1} failed: {e}"
                    )
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
```
</example>

<example name="Hierarchical Chunker Implementation">
```python
# /packages/shared/text_processing/hierarchical_chunker.py
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes

class HierarchicalChunker(BaseChunker):
    """Multi-level chunking for better context preservation."""
    
    def __init__(
        self,
        chunk_sizes: List[int] = None,
        chunk_overlap: int = 20
    ):
        """Initialize with multiple chunk size levels.
        
        Args:
            chunk_sizes: List of chunk sizes from largest to smallest
            chunk_overlap: Overlap between chunks at each level
        """
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        
        # Validate chunk sizes are descending
        if chunk_sizes != sorted(chunk_sizes, reverse=True):
            raise ValueError("Chunk sizes must be in descending order")
        
        self.splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap
        )
        self.strategy_name = "hierarchical"
        self.chunk_sizes = chunk_sizes
        
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Create hierarchical chunks with parent-child relationships."""
        if not text.strip():
            return []
        
        doc = Document(text=text, metadata=metadata or {})
        
        # Get hierarchical nodes
        loop = asyncio.get_event_loop()
        all_nodes = await loop.run_in_executor(
            None,
            self.splitter.get_nodes_from_documents,
            [doc]
        )
        
        # Get only leaf nodes for primary chunks
        leaf_nodes = get_leaf_nodes(all_nodes)
        
        # Build parent mapping
        parent_map = self._build_parent_map(all_nodes)
        
        # Convert to ChunkResult with hierarchy metadata
        results = []
        for idx, node in enumerate(leaf_nodes):
            parent_chain = self._get_parent_chain(node, parent_map)
            
            results.append(ChunkResult(
                chunk_id=f"{doc_id}_chunk_{idx}",
                text=node.text,
                start_offset=node.start_char_idx or 0,
                end_offset=node.end_char_idx or len(node.text),
                metadata={
                    **node.metadata,
                    "strategy": self.strategy_name,
                    "chunk_level": len(parent_chain),
                    "parent_chunks": [p.node_id for p in parent_chain],
                    "chunk_size_level": self._get_size_level(len(node.text))
                }
            ))
        
        # Also include parent chunks for context
        for parent_node in all_nodes:
            if parent_node not in leaf_nodes:
                results.append(ChunkResult(
                    chunk_id=f"{doc_id}_parent_{parent_node.node_id}",
                    text=parent_node.text,
                    start_offset=parent_node.start_char_idx or 0,
                    end_offset=parent_node.end_char_idx or len(parent_node.text),
                    metadata={
                        **parent_node.metadata,
                        "strategy": self.strategy_name,
                        "is_parent_chunk": True,
                        "child_chunks": [
                            c.node_id for c in all_nodes 
                            if self._is_child_of(c, parent_node)
                        ]
                    }
                ))
        
        return results
    
    def _get_size_level(self, text_length: int) -> int:
        """Determine which chunk size level this text belongs to."""
        for i, size in enumerate(self.chunk_sizes):
            if text_length <= size:
                return i
        return len(self.chunk_sizes) - 1
```
</example>

<example name="Hybrid Chunker Implementation">
```python
# /packages/shared/text_processing/hybrid_chunker.py
class HybridChunker(BaseChunker):
    """Intelligently selects chunking strategy based on content."""
    
    def __init__(self):
        """Initialize with strategy instances."""
        self.strategies = {
            'markdown': MarkdownChunker(),
            'semantic': SemanticChunker(
                breakpoint_percentile_threshold=90
            ),
            'recursive': RecursiveChunker(
                chunk_size=600,
                chunk_overlap=100
            )
        }
        self.strategy_name = "hybrid"
        
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Select optimal strategy based on content analysis."""
        if not text.strip():
            return []
        
        # Analyze content to select strategy
        selected_strategy = await self._select_strategy(text, metadata)
        
        logger.info(
            f"Hybrid chunker selected '{selected_strategy}' "
            f"for document {doc_id}"
        )
        
        # Use selected strategy
        chunker = self.strategies[selected_strategy]
        chunks = await chunker.chunk_text_async(text, doc_id, metadata)
        
        # Add hybrid metadata
        for chunk in chunks:
            chunk.metadata.update({
                "strategy": self.strategy_name,
                "sub_strategy": selected_strategy,
                "selection_reason": self._get_selection_reason(
                    text, metadata, selected_strategy
                )
            })
        
        return chunks
    
    async def _select_strategy(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Intelligently select the best strategy."""
        file_type = metadata.get('file_type', '') if metadata else ''
        
        # Rule 1: Use markdown for .md files
        if file_type == '.md' or self._has_markdown_structure(text):
            return 'markdown'
        
        # Rule 2: Check content characteristics
        characteristics = await self._analyze_content(text)
        
        # Use semantic for topic-heavy content
        if characteristics['topic_diversity'] > 0.7:
            return 'semantic'
        
        # Default to recursive for general text
        return 'recursive'
    
    def _has_markdown_structure(self, text: str) -> bool:
        """Check if text has significant markdown formatting."""
        import re
        
        # Count markdown indicators
        headers = len(re.findall(r'^#{1,6}\s+', text, re.MULTILINE))
        code_blocks = len(re.findall(r'```', text))
        lists = len(re.findall(r'^\s*[-*+]\s+', text, re.MULTILINE))
        
        # Threshold for markdown detection
        total_lines = len(text.split('\n'))
        markdown_density = (headers + code_blocks + lists) / max(total_lines, 1)
        
        return markdown_density > 0.1  # 10% markdown elements
    
    async def _analyze_content(self, text: str) -> Dict[str, float]:
        """Analyze content characteristics for strategy selection."""
        # This is a simplified version - could use NLP for better analysis
        sentences = text.split('.')
        words = text.split()
        
        # Calculate metrics
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Estimate topic diversity (simplified)
        unique_words = len(set(word.lower() for word in words))
        topic_diversity = unique_words / max(len(words), 1)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'topic_diversity': topic_diversity,
            'total_length': len(text)
        }
```
</example>
</implementation_examples>

<success_criteria>
- [ ] Semantic chunking uses embeddings effectively
- [ ] Hierarchical chunking maintains parent-child relationships
- [ ] Hybrid chunking selects appropriate strategies
- [ ] All strategies handle API failures gracefully
- [ ] Performance targets met (semantic: 150/sec, hierarchical: 400/sec)
- [ ] Streaming support works for large documents
- [ ] Edge cases handled (empty text, huge chunks, API errors)
- [ ] Comprehensive test coverage added
</success_criteria>

<common_pitfalls>
- **Avoid**: Calling embedding APIs without retry logic
- **Avoid**: Creating chunks larger than vector DB limits
- **Avoid**: Losing parent-child relationships in hierarchical chunking
- **Avoid**: Hard-coding strategy selection in hybrid chunker
- **Avoid**: Ignoring API rate limits for embeddings
</common_pitfalls>
</task>

### Task 2.2: Normalized Database Schema

<task>
<metadata>
  <priority>High</priority>
  <effort>1.5 days</effort>
  <dependencies>None (can be done in parallel with 2.1)</dependencies>
</metadata>

<context>
A well-designed database schema is crucial for performance at scale. The current schema is denormalized and will cause problems as data grows. By normalizing the schema, adding proper indexes, and implementing partitioning, we ensure the system can handle millions of chunks efficiently. This also enables better analytics and easier maintenance. Good schema design now saves countless hours of pain later.
</context>

<instructions>
Design and implement a normalized database schema optimized for scale.

**Step 1: Create Normalized Tables**
- Separate chunking_strategies table for strategy definitions
- Deduplicated chunking_configs with hash-based lookups
- Enhanced collections table with performance metrics
- Rich documents table with chunking metadata
- Partitioned chunks table for horizontal scaling

**Step 2: Implement Partitioning Strategy**
- Use hash partitioning on collection_id for chunks table
- Create 16 partitions initially (easily expandable)
- Ensure even distribution of data
- Test partition pruning in queries

**Step 3: Add Strategic Indexes**
- Primary key indexes on all tables
- Foreign key indexes for joins
- Composite indexes for common query patterns
- Partial indexes for filtered queries

**Step 4: Create Helper Views**
- Materialized view for collection statistics
- Regular views for common join patterns
- Functions for stats refresh
- Performance monitoring views

**Step 5: Write Migration Scripts**
- Use Alembic for version control
- Include rollback procedures
- Test on production-sized data
- Ensure zero-downtime migration
</instructions>

<available_tools>
- **context7**: PostgreSQL partitioning docs, SQLAlchemy patterns
- **Read/Edit**: Create Alembic migrations
- **Bash**: Run migrations and validate schema
- **Grep**: Search for existing schema patterns
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="implementation" name="database-migrations-engineer">
    <purpose>Design normalized schema with performance optimizations</purpose>
    <focus>Normalization, partitioning strategy, index design, migration safety</focus>
  </agent>
  
  <agent type="review" name="backend-code-reviewer">
    <purpose>Validate schema design and migration safety</purpose>
    <focus>Data integrity, performance implications, rollback procedures</focus>
  </agent>
  
  <agent type="performance" name="performance-profiler">
    <purpose>Validate query performance with new schema</purpose>
    <focus>Query plans, index usage, partition pruning</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="Core Schema Definition">
```sql
-- Normalized chunking strategies table (reference data)
CREATE TABLE chunking_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    description TEXT,
    default_params JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    performance_baseline JSONB,  -- Expected performance metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert strategies with performance baselines
INSERT INTO chunking_strategies (name, display_name, default_params, performance_baseline) VALUES
('character', 'Character-based', 
 '{"chunk_size": 1000, "chunk_overlap": 200}',
 '{"chunks_per_sec": 1000, "memory_per_mb": 50}'),
('recursive', 'Recursive Text Splitter', 
 '{"chunk_size": 600, "chunk_overlap": 100}',
 '{"chunks_per_sec": 800, "memory_per_mb": 60}'),
('markdown', 'Markdown-aware', 
 '{"preserve_headers": true, "split_by": ["#", "##", "###"]}',
 '{"chunks_per_sec": 600, "memory_per_mb": 80}'),
('semantic', 'AI-Powered Semantic', 
 '{"breakpoint_percentile_threshold": 95, "buffer_size": 1}',
 '{"chunks_per_sec": 150, "memory_per_mb": 200}'),
('hierarchical', 'Multi-level Hierarchical', 
 '{"chunk_sizes": [2048, 512, 128], "chunk_overlap": 20}',
 '{"chunks_per_sec": 400, "memory_per_mb": 150}'),
('hybrid', 'Intelligent Hybrid', 
 '{"primary_strategy": "recursive", "auto_select": true}',
 '{"chunks_per_sec": 600, "memory_per_mb": 100}');

-- Deduplicated chunking configurations
CREATE TABLE chunking_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id INTEGER NOT NULL REFERENCES chunking_strategies(id),
    params JSONB NOT NULL,
    params_hash VARCHAR(64) NOT NULL,  -- SHA256 hash for deduplication
    usage_count INTEGER DEFAULT 0,      -- Track popularity
    avg_performance JSONB,              -- Actual performance metrics
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_config_hash UNIQUE(params_hash),
    INDEX idx_config_strategy (strategy_id),
    INDEX idx_config_usage (usage_count DESC)
);

-- Function to compute config hash
CREATE OR REPLACE FUNCTION compute_config_hash(strategy_id INTEGER, params JSONB) 
RETURNS VARCHAR(64) AS $$
BEGIN
    RETURN encode(
        sha256(
            (strategy_id::TEXT || '::' || params::TEXT)::BYTEA
        ), 
        'hex'
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Enhanced collections table
CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id INTEGER NOT NULL,
    chunking_config_id UUID REFERENCES chunking_configs(id),
    
    -- Status tracking
    status collection_status NOT NULL DEFAULT 'draft',
    status_message TEXT,
    
    -- Statistics
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    code_file_count INTEGER DEFAULT 0,
    avg_chunk_size INTEGER DEFAULT 0,
    
    -- Performance tracking
    last_operation_id UUID,
    last_chunk_time_ms INTEGER,
    avg_chunk_time_ms INTEGER,
    total_processing_time_ms BIGINT DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_indexed_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes for common queries
    INDEX idx_collections_owner (owner_id),
    INDEX idx_collections_status (status),
    INDEX idx_collections_config (chunking_config_id),
    INDEX idx_collections_updated (updated_at DESC)
);

-- Create enum for collection status if not exists
DO $$ BEGIN
    CREATE TYPE collection_status AS ENUM (
        'draft', 'indexing', 'ready', 'error', 'degraded', 'archived'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;
```
</example>

<example name="Documents and Operations Tables">
```sql
-- Enhanced documents table with rich metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    
    -- File information
    file_path TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_category VARCHAR(50),  -- document, code, data, media
    mime_type VARCHAR(100),
    file_size_bytes BIGINT NOT NULL,
    file_hash VARCHAR(64),      -- SHA256 for deduplication
    
    -- Content metadata
    detected_language VARCHAR(10),
    encoding VARCHAR(50) DEFAULT 'utf-8',
    is_code_file BOOLEAN DEFAULT FALSE,
    has_images BOOLEAN DEFAULT FALSE,
    
    -- Chunking results
    chunk_count INTEGER DEFAULT 0,
    chunking_strategy_used VARCHAR(50),
    chunking_time_ms INTEGER,
    chunking_error TEXT,
    avg_chunk_size INTEGER,
    
    -- Processing status
    status document_status NOT NULL DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    last_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Indexes for performance
    INDEX idx_documents_collection_status (collection_id, status),
    INDEX idx_documents_file_type (file_type),
    INDEX idx_documents_is_code (is_code_file) WHERE is_code_file = true,
    INDEX idx_documents_hash (file_hash),
    INDEX idx_documents_processed (processed_at DESC)
);

-- Operations table for async task tracking
CREATE TABLE operations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    type operation_type NOT NULL,
    status operation_status NOT NULL DEFAULT 'pending',
    
    -- Task details
    celery_task_id VARCHAR(255),
    priority INTEGER DEFAULT 5,
    
    -- Progress tracking
    total_items INTEGER,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    progress_percent NUMERIC(5,2) DEFAULT 0,
    
    -- Performance metrics
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    total_time_ms INTEGER,
    items_per_second NUMERIC(10,2),
    
    -- Error handling
    error_summary TEXT,
    error_details JSONB,
    retry_count INTEGER DEFAULT 0,
    
    -- User info
    initiated_by INTEGER,
    cancelled_by INTEGER,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_operations_collection (collection_id),
    INDEX idx_operations_status (status),
    INDEX idx_operations_type_status (type, status),
    INDEX idx_operations_created (created_at DESC)
);
```
</example>

<example name="Partitioned Chunks Table">
```sql
-- Partitioned chunks table for horizontal scaling
CREATE TABLE chunks (
    id UUID DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL,
    document_id UUID NOT NULL,
    
    -- Chunk identity
    chunk_index INTEGER NOT NULL,
    chunk_hash VARCHAR(64),  -- For deduplication
    
    -- Content
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    
    -- Position in document
    start_offset INTEGER,
    end_offset INTEGER,
    start_page INTEGER,      -- For PDFs
    end_page INTEGER,
    
    -- Embeddings (using pgvector)
    embedding VECTOR(384),   -- Standard dimension
    embedding_model VARCHAR(50),
    
    -- Hierarchical support
    parent_chunk_id UUID,
    chunk_level INTEGER DEFAULT 0,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    strategy_used VARCHAR(50),
    
    -- Search optimization
    search_text tsvector,    -- Full-text search
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Composite primary key for partitioning
    PRIMARY KEY (id, collection_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Indexes (created on partitions)
    INDEX idx_chunks_document (document_id),
    INDEX idx_chunks_parent (parent_chunk_id) WHERE parent_chunk_id IS NOT NULL,
    INDEX idx_chunks_embedding (embedding) USING ivfflat
) PARTITION BY HASH (collection_id);

-- Create 16 partitions for chunks
DO $$
DECLARE
    i INTEGER;
BEGIN
    FOR i IN 0..15 LOOP
        EXECUTE format(
            'CREATE TABLE chunks_p%s PARTITION OF chunks 
             FOR VALUES WITH (modulus 16, remainder %s)',
            i, i
        );
        
        -- Create partition-specific indexes
        EXECUTE format(
            'CREATE INDEX idx_chunks_p%s_search ON chunks_p%s 
             USING gin(search_text)',
            i, i
        );
    END LOOP;
END $$;

-- Function to suggest partition count based on data
CREATE OR REPLACE FUNCTION suggest_partition_count(
    expected_collections INTEGER,
    avg_chunks_per_collection INTEGER
) RETURNS INTEGER AS $$
DECLARE
    total_chunks BIGINT;
    chunks_per_partition BIGINT;
    suggested_count INTEGER;
BEGIN
    total_chunks := expected_collections::BIGINT * avg_chunks_per_collection;
    chunks_per_partition := 1000000; -- Target 1M chunks per partition
    
    suggested_count := GREATEST(
        16,  -- Minimum partitions
        LEAST(
            256,  -- Maximum partitions
            POWER(2, CEIL(LOG(2, total_chunks::NUMERIC / chunks_per_partition)))
        )
    );
    
    RETURN suggested_count;
END;
$$ LANGUAGE plpgsql;
```
</example>

<example name="Analytics and Helper Tables">
```sql
-- Chunking history for auditing and rollback
CREATE TABLE chunking_history (
    id SERIAL PRIMARY KEY,
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    operation_id UUID REFERENCES operations(id),
    
    -- Configuration tracking
    previous_config_id UUID REFERENCES chunking_configs(id),
    new_config_id UUID REFERENCES chunking_configs(id),
    
    -- Impact metrics
    documents_affected INTEGER,
    chunks_before INTEGER,
    chunks_after INTEGER,
    processing_time_ms INTEGER,
    
    -- Change details
    change_type VARCHAR(50),  -- reindex, strategy_change, param_update
    change_reason TEXT,
    
    -- User tracking
    initiated_by INTEGER NOT NULL,
    approved_by INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_history_collection (collection_id),
    INDEX idx_history_created (created_at DESC)
);

-- Performance metrics table
CREATE TABLE chunking_metrics (
    id SERIAL PRIMARY KEY,
    collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    operation_id UUID REFERENCES operations(id),
    
    -- Strategy performance
    strategy VARCHAR(50) NOT NULL,
    
    -- Volume metrics
    documents_processed INTEGER NOT NULL,
    chunks_created INTEGER NOT NULL,
    total_size_bytes BIGINT NOT NULL,
    
    -- Performance metrics
    processing_time_ms INTEGER NOT NULL,
    chunks_per_second NUMERIC(10,2),
    mb_per_second NUMERIC(10,2),
    avg_chunk_size INTEGER,
    
    -- Resource usage
    peak_memory_mb INTEGER,
    cpu_seconds NUMERIC(10,2),
    
    -- Quality metrics
    empty_chunks_count INTEGER DEFAULT 0,
    oversized_chunks_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_metrics_collection_date (collection_id, created_at DESC),
    INDEX idx_metrics_strategy (strategy)
);

-- Materialized view for collection stats
CREATE MATERIALIZED VIEW collection_stats AS
SELECT 
    c.id,
    c.name,
    c.status,
    c.owner_id,
    c.document_count,
    c.chunk_count,
    c.total_size_bytes,
    c.code_file_count,
    
    -- Strategy info
    cs.name as strategy_name,
    cs.display_name as strategy_display_name,
    cc.params as chunking_params,
    
    -- Performance averages
    COALESCE(AVG(cm.chunks_per_second), 0) as avg_chunks_per_second,
    COALESCE(AVG(cm.mb_per_second), 0) as avg_mb_per_second,
    COALESCE(AVG(cm.avg_chunk_size), 0) as avg_chunk_size,
    
    -- Recent activity
    MAX(cm.created_at) as last_activity,
    COUNT(DISTINCT cm.id) as total_operations
    
FROM collections c
LEFT JOIN chunking_configs cc ON c.chunking_config_id = cc.id
LEFT JOIN chunking_strategies cs ON cc.strategy_id = cs.id
LEFT JOIN chunking_metrics cm ON c.id = cm.collection_id
GROUP BY c.id, c.name, c.status, c.owner_id, c.document_count, 
         c.chunk_count, c.total_size_bytes, c.code_file_count,
         cs.name, cs.display_name, cc.params;

-- Create indexes on materialized view
CREATE INDEX idx_collection_stats_owner ON collection_stats(owner_id);
CREATE INDEX idx_collection_stats_status ON collection_stats(status);

-- Function to refresh stats
CREATE OR REPLACE FUNCTION refresh_collection_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY collection_stats;
END;
$$ LANGUAGE plpgsql;

-- Schedule periodic refresh (requires pg_cron or similar)
-- SELECT cron.schedule('refresh-collection-stats', '*/5 * * * *', 
--                      'SELECT refresh_collection_stats()');
```
</example>

<example name="Alembic Migration">
```python
# /alembic/versions/001_create_chunking_schema.py
"""Create comprehensive chunking schema

Revision ID: 001_chunking_schema
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers
revision = '001_chunking_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create all chunking-related tables and indexes."""
    
    # Create custom types
    op.execute("""
        CREATE TYPE collection_status AS ENUM (
            'draft', 'indexing', 'ready', 'error', 'degraded', 'archived'
        );
        
        CREATE TYPE document_status AS ENUM (
            'pending', 'processing', 'completed', 'failed', 'skipped'
        );
        
        CREATE TYPE operation_type AS ENUM (
            'index', 'reindex', 'delete', 'update_strategy', 'validate'
        );
        
        CREATE TYPE operation_status AS ENUM (
            'pending', 'running', 'completed', 'failed', 'cancelled'
        );
    """)
    
    # Create pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create all tables
    # ... (include all CREATE TABLE statements from examples above)
    
    # Create helper functions
    op.execute("""
        -- Function to update updated_at timestamp
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)
    
    # Create triggers
    for table in ['collections', 'documents', 'operations', 'chunking_strategies']:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at 
            BEFORE UPDATE ON {table} 
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)
    
    # Insert default data
    op.execute("""
        -- Insert default strategies
        INSERT INTO chunking_strategies (name, display_name, default_params) 
        VALUES ... -- (from example above)
    """)

def downgrade():
    """Drop all chunking-related objects."""
    
    # Drop in reverse order due to dependencies
    op.execute("DROP MATERIALIZED VIEW IF EXISTS collection_stats CASCADE")
    op.execute("DROP TABLE IF EXISTS chunking_metrics CASCADE")
    op.execute("DROP TABLE IF EXISTS chunking_history CASCADE")
    op.execute("DROP TABLE IF EXISTS chunks CASCADE")  # Drops all partitions
    op.execute("DROP TABLE IF EXISTS operations CASCADE")
    op.execute("DROP TABLE IF EXISTS documents CASCADE")
    op.execute("DROP TABLE IF EXISTS collections CASCADE")
    op.execute("DROP TABLE IF EXISTS chunking_configs CASCADE")
    op.execute("DROP TABLE IF EXISTS chunking_strategies CASCADE")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column CASCADE")
    op.execute("DROP FUNCTION IF EXISTS compute_config_hash CASCADE")
    op.execute("DROP FUNCTION IF EXISTS suggest_partition_count CASCADE")
    op.execute("DROP FUNCTION IF EXISTS refresh_collection_stats CASCADE")
    
    # Drop types
    op.execute("DROP TYPE IF EXISTS operation_status CASCADE")
    op.execute("DROP TYPE IF EXISTS operation_type CASCADE")
    op.execute("DROP TYPE IF EXISTS document_status CASCADE")
    op.execute("DROP TYPE IF EXISTS collection_status CASCADE")
```
</example>
</implementation_examples>

<success_criteria>
- [ ] All tables properly normalized (3NF)
- [ ] Chunking configs deduplicated with hash
- [ ] Chunks table partitioned by collection_id
- [ ] All foreign keys have indexes
- [ ] Composite indexes for common queries
- [ ] Materialized view for stats performs well
- [ ] Migration scripts are idempotent
- [ ] Rollback procedures tested
- [ ] Performance validated with large datasets
</success_criteria>

<common_pitfalls>
- **Avoid**: Over-normalization that hurts query performance
- **Avoid**: Missing indexes on foreign keys
- **Avoid**: Partition key that doesn't match query patterns
- **Avoid**: Non-concurrent materialized view refreshes
- **Avoid**: Migrations that lock tables for long periods
- **Avoid**: Forgetting to update ORM models after schema changes
</common_pitfalls>
</task>

### Task 2.3: User Interface Components

<task>
<metadata>
  <priority>High</priority>
  <effort>2 days</effort>
  <dependencies>None (can be done in parallel)</dependencies>
</metadata>

<context>
The user interface is where users experience the power of chunking strategies. A well-designed UI makes complex features feel simple. The goal is to create an intuitive interface that allows users to preview chunking results, compare strategies, and make informed decisions. Good UX here will be a key differentiator for Semantik, turning a technical feature into a delightful experience.
</context>

<instructions>
Build comprehensive React components for chunking configuration and preview.

**Step 1: Create Strategy Selector Component**
- Visual cards for each strategy with icons
- Performance indicators (speed, quality metrics)
- Expandable details with strategy explanations
- Real-time parameter adjustment
- Intelligent recommendations based on file types

**Step 2: Build Chunking Preview Panel**
- Split-view showing original document and chunks
- Syntax highlighting for code files
- Visual chunk boundaries with hover effects
- Statistics sidebar (chunk count, size distribution)
- Interactive chunk navigation

**Step 3: Implement Comparison View**
- Side-by-side strategy comparison
- Synchronized scrolling between views
- Diff highlighting for chunk boundaries
- Performance metrics comparison chart
- Export comparison results

**Step 4: Create Parameter Tuning Interface**
- Visual sliders with live preview
- Preset configurations for common use cases
- Advanced mode for power users
- Validation with helpful error messages
- Save custom configurations

**Step 5: Add Analytics Dashboard**
- Strategy usage statistics
- Performance trends over time
- Chunk quality metrics
- File type distribution
- Actionable recommendations
</instructions>

<available_tools>
- **context7**: React 19 patterns, Zustand best practices, TailwindCSS
- **Read/Edit**: Create components and stores
- **Bash**: Run component tests
- **WebSearch**: UI/UX best practices for technical tools
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="design" name="ui-component-craftsman">
    <purpose>Create beautiful, accessible React components</purpose>
    <focus>Component architecture, accessibility, responsive design</focus>
  </agent>
  
  <agent type="state" name="frontend-state-architect">
    <purpose>Design Zustand stores and data flow</purpose>
    <focus>State management, React Query integration, optimistic updates</focus>
  </agent>
  
  <agent type="review" name="frontend-code-reviewer">
    <purpose>Ensure component quality and best practices</purpose>
    <focus>React patterns, performance, accessibility compliance</focus>
  </agent>
  
  <agent type="testing" name="qa-bug-hunter">
    <purpose>Create comprehensive component tests</purpose>
    <focus>Unit tests, integration tests, visual regression tests</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="Strategy Selector Component">
```tsx
// /apps/webui-react/src/components/chunking/StrategySelector.tsx
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CubeIcon, 
  DocumentTextIcon, 
  AcademicCapIcon,
  LayersIcon,
  SparklesIcon,
  BeakerIcon 
} from '@heroicons/react/24/outline';
import { useChunkingStore } from '@/stores/chunkingStore';
import { ChunkingStrategy } from '@/types/chunking';

const STRATEGY_ICONS = {
  character: CubeIcon,
  recursive: DocumentTextIcon,
  markdown: DocumentTextIcon,
  semantic: AcademicCapIcon,
  hierarchical: LayersIcon,
  hybrid: SparklesIcon,
};

const STRATEGY_DESCRIPTIONS = {
  character: "Fast, fixed-size chunks ideal for general text processing",
  recursive: "Smart splitting that respects sentence and paragraph boundaries",
  markdown: "Preserves document structure, perfect for technical documentation",
  semantic: "AI-powered chunking that groups related topics together",
  hierarchical: "Multi-level chunks for enhanced context during retrieval",
  hybrid: "Intelligently selects the best strategy based on your content",
};

export const StrategySelector: React.FC = () => {
  const { 
    strategies, 
    selectedStrategy, 
    setSelectedStrategy,
    fileType,
    recommendedStrategy 
  } = useChunkingStore();
  
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      {/* Recommendation Banner */}
      {recommendedStrategy && recommendedStrategy !== selectedStrategy && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 
                     dark:border-blue-800 rounded-lg p-4"
        >
          <div className="flex items-center space-x-2">
            <BeakerIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <p className="text-sm text-blue-800 dark:text-blue-200">
              Based on your {fileType} files, we recommend 
              <button
                onClick={() => setSelectedStrategy(recommendedStrategy)}
                className="ml-1 font-semibold underline hover:no-underline"
              >
                {strategies.find(s => s.id === recommendedStrategy)?.displayName}
              </button>
            </p>
          </div>
        </motion.div>
      )}
      
      {/* Strategy Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {strategies.map((strategy) => {
          const Icon = STRATEGY_ICONS[strategy.name];
          const isSelected = selectedStrategy === strategy.id;
          const isExpanded = expandedCard === strategy.id;
          
          return (
            <motion.div
              key={strategy.id}
              layout
              className={`
                relative rounded-xl border-2 transition-all cursor-pointer
                ${isSelected 
                  ? 'border-blue-500 dark:border-blue-400 shadow-lg' 
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                }
              `}
              onClick={() => setSelectedStrategy(strategy.id)}
            >
              {/* Recommended Badge */}
              {strategy.id === recommendedStrategy && (
                <div className="absolute -top-2 -right-2 z-10">
                  <span className="bg-blue-500 text-white text-xs px-2 py-1 
                                 rounded-full font-semibold">
                    Recommended
                  </span>
                </div>
              )}
              
              <div className="p-6">
                {/* Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <div className={`
                      p-2 rounded-lg
                      ${isSelected 
                        ? 'bg-blue-100 dark:bg-blue-900/30' 
                        : 'bg-gray-100 dark:bg-gray-800'
                      }
                    `}>
                      <Icon className={`
                        h-6 w-6
                        ${isSelected 
                          ? 'text-blue-600 dark:text-blue-400' 
                          : 'text-gray-600 dark:text-gray-400'
                        }
                      `} />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 dark:text-white">
                        {strategy.displayName}
                      </h3>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {STRATEGY_DESCRIPTIONS[strategy.name]}
                      </p>
                    </div>
                  </div>
                </div>
                
                {/* Performance Metrics */}
                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="text-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Speed</p>
                    <p className="text-sm font-semibold text-gray-900 dark:text-white">
                      {strategy.performanceBaseline?.chunksPerSec || 'N/A'}/sec
                    </p>
                  </div>
                  <div className="text-center p-2 bg-gray-50 dark:bg-gray-800 rounded">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Memory</p>
                    <p className="text-sm font-semibold text-gray-900 dark:text-white">
                      {strategy.performanceBaseline?.memoryPerMb || 'N/A'}MB
                    </p>
                  </div>
                </div>
                
                {/* Expand/Collapse Button */}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedCard(isExpanded ? null : strategy.id);
                  }}
                  className="text-sm text-blue-600 dark:text-blue-400 
                           hover:underline focus:outline-none"
                >
                  {isExpanded ? 'Show less' : 'Show more'}
                </button>
                
                {/* Expanded Details */}
                <AnimatePresence>
                  {isExpanded && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="mt-4 space-y-3"
                    >
                      {/* Best For Section */}
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 
                                     dark:text-gray-300 mb-1">
                          Best for:
                        </h4>
                        <ul className="text-sm text-gray-600 dark:text-gray-400 
                                     space-y-1">
                          {strategy.bestFor?.map((item, idx) => (
                            <li key={idx} className="flex items-center space-x-2">
                              <span className="text-green-500">✓</span>
                              <span>{item}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      {/* Parameters Preview */}
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 
                                     dark:text-gray-300 mb-1">
                          Default parameters:
                        </h4>
                        <pre className="text-xs bg-gray-100 dark:bg-gray-800 
                                      p-2 rounded overflow-x-auto">
                          {JSON.stringify(strategy.defaultParams, null, 2)}
                        </pre>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
              
              {/* Selection Indicator */}
              {isSelected && (
                <motion.div
                  layoutId="selected-indicator"
                  className="absolute inset-0 border-2 border-blue-500 
                           dark:border-blue-400 rounded-xl pointer-events-none"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};
```
</example>

<example name="Chunking Preview Component">
```tsx
// /apps/webui-react/src/components/chunking/ChunkingPreview.tsx
import React, { useMemo, useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { motion } from 'framer-motion';
import { useChunkingPreview } from '@/hooks/useChunkingPreview';
import { ChunkBoundary } from './ChunkBoundary';
import { ChunkStatistics } from './ChunkStatistics';

export const ChunkingPreview: React.FC<{
  documentId: string;
  strategyId: string;
  parameters: Record<string, any>;
}> = ({ documentId, strategyId, parameters }) => {
  const { 
    document, 
    chunks, 
    isLoading, 
    error 
  } = useChunkingPreview(documentId, strategyId, parameters);
  
  const [selectedChunk, setSelectedChunk] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'split' | 'overlay'>('split');
  
  // Generate chunk boundaries for overlay view
  const chunkBoundaries = useMemo(() => {
    if (!chunks || !document) return [];
    
    return chunks.map((chunk, index) => ({
      start: chunk.startOffset,
      end: chunk.endOffset,
      index,
      color: `hsl(${(index * 360) / chunks.length}, 70%, 50%)`,
    }));
  }, [chunks, document]);
  
  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="space-y-4 text-center">
          <div className="animate-spin rounded-full h-12 w-12 
                        border-b-2 border-blue-500 mx-auto" />
          <p className="text-gray-600 dark:text-gray-400">
            Generating preview...
          </p>
        </div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 
                    dark:border-red-800 rounded-lg p-4">
        <p className="text-red-800 dark:text-red-200">
          Error generating preview: {error.message}
        </p>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      {/* View Mode Toggle */}
      <div className="flex items-center justify-between">
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode('split')}
            className={`
              px-3 py-1 rounded-md text-sm font-medium transition-colors
              ${viewMode === 'split'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              }
            `}
          >
            Split View
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`
              px-3 py-1 rounded-md text-sm font-medium transition-colors
              ${viewMode === 'overlay'
                ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
              }
            `}
          >
            Overlay View
          </button>
        </div>
        
        {/* Statistics Summary */}
        <div className="flex items-center space-x-4 text-sm">
          <span className="text-gray-600 dark:text-gray-400">
            {chunks?.length || 0} chunks
          </span>
          <span className="text-gray-600 dark:text-gray-400">
            Avg size: {Math.round(
              (chunks?.reduce((sum, c) => sum + c.content.length, 0) || 0) / 
              (chunks?.length || 1)
            )} chars
          </span>
        </div>
      </div>
      
      {/* Main Preview Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Document/Preview Panel */}
        <div className="lg:col-span-2">
          {viewMode === 'split' ? (
            <SplitView
              document={document}
              chunks={chunks}
              selectedChunk={selectedChunk}
              onChunkSelect={setSelectedChunk}
            />
          ) : (
            <OverlayView
              document={document}
              boundaries={chunkBoundaries}
              selectedChunk={selectedChunk}
              onChunkSelect={setSelectedChunk}
            />
          )}
        </div>
        
        {/* Statistics Sidebar */}
        <div className="lg:col-span-1">
          <ChunkStatistics
            chunks={chunks}
            selectedChunk={selectedChunk}
            onChunkSelect={setSelectedChunk}
          />
        </div>
      </div>
    </div>
  );
};

// Split View Component
const SplitView: React.FC<{
  document: any;
  chunks: any[];
  selectedChunk: number | null;
  onChunkSelect: (index: number | null) => void;
}> = ({ document, chunks, selectedChunk, onChunkSelect }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Original Document */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Original Document
        </h3>
        <div className="border border-gray-200 dark:border-gray-700 
                      rounded-lg overflow-hidden">
          <pre className="p-4 text-sm overflow-x-auto max-h-96 overflow-y-auto">
            <code className="text-gray-800 dark:text-gray-200">
              {document?.content || ''}
            </code>
          </pre>
        </div>
      </div>
      
      {/* Chunks List */}
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
          Generated Chunks
        </h3>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {chunks?.map((chunk, index) => (
            <motion.div
              key={chunk.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onChunkSelect(index)}
              className={`
                p-3 rounded-lg border cursor-pointer transition-all
                ${selectedChunk === index
                  ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20'
                  : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                }
              `}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-gray-600 
                               dark:text-gray-400">
                  Chunk {index + 1}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-500">
                  {chunk.content.length} chars
                </span>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 
                         line-clamp-3">
                {chunk.content}
              </p>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};
```
</example>

<example name="Chunking Store">
```typescript
// /apps/webui-react/src/stores/chunkingStore.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { chunkingApi } from '@/api/chunking';

interface ChunkingState {
  // Strategy Management
  strategies: ChunkingStrategy[];
  selectedStrategy: string | null;
  recommendedStrategy: string | null;
  
  // Preview State
  previewDocument: Document | null;
  previewChunks: Chunk[] | null;
  previewLoading: boolean;
  previewError: Error | null;
  
  // Configuration
  currentConfig: ChunkingConfig | null;
  savedConfigs: ChunkingConfig[];
  
  // UI State
  comparisonStrategies: string[];
  showAdvancedParams: boolean;
  
  // Actions
  loadStrategies: () => Promise<void>;
  setSelectedStrategy: (strategyId: string) => void;
  updateParameters: (params: Record<string, any>) => void;
  generatePreview: (documentId: string) => Promise<void>;
  saveConfiguration: (name: string) => Promise<void>;
  compareStrategies: (strategyIds: string[]) => Promise<void>;
}

export const useChunkingStore = create<ChunkingState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial State
        strategies: [],
        selectedStrategy: null,
        recommendedStrategy: null,
        previewDocument: null,
        previewChunks: null,
        previewLoading: false,
        previewError: null,
        currentConfig: null,
        savedConfigs: [],
        comparisonStrategies: [],
        showAdvancedParams: false,
        
        // Load available strategies
        loadStrategies: async () => {
          try {
            const strategies = await chunkingApi.getStrategies();
            set({ strategies });
            
            // Set default strategy if none selected
            if (!get().selectedStrategy && strategies.length > 0) {
              set({ selectedStrategy: strategies[0].id });
            }
          } catch (error) {
            console.error('Failed to load strategies:', error);
          }
        },
        
        // Select a strategy
        setSelectedStrategy: (strategyId: string) => {
          const strategy = get().strategies.find(s => s.id === strategyId);
          if (strategy) {
            set({
              selectedStrategy: strategyId,
              currentConfig: {
                strategyId,
                params: strategy.defaultParams,
              },
            });
          }
        },
        
        // Update configuration parameters
        updateParameters: (params: Record<string, any>) => {
          const currentConfig = get().currentConfig;
          if (currentConfig) {
            set({
              currentConfig: {
                ...currentConfig,
                params: { ...currentConfig.params, ...params },
              },
            });
            
            // Debounced preview regeneration
            get().generatePreview(get().previewDocument?.id);
          }
        },
        
        // Generate chunking preview
        generatePreview: async (documentId: string) => {
          const { selectedStrategy, currentConfig } = get();
          if (!selectedStrategy || !currentConfig) return;
          
          set({ previewLoading: true, previewError: null });
          
          try {
            const result = await chunkingApi.previewChunking({
              documentId,
              strategyId: selectedStrategy,
              params: currentConfig.params,
            });
            
            set({
              previewDocument: result.document,
              previewChunks: result.chunks,
              previewLoading: false,
            });
          } catch (error) {
            set({
              previewError: error as Error,
              previewLoading: false,
            });
          }
        },
        
        // Save current configuration
        saveConfiguration: async (name: string) => {
          const { currentConfig } = get();
          if (!currentConfig) return;
          
          const savedConfig = {
            ...currentConfig,
            id: Date.now().toString(),
            name,
            createdAt: new Date().toISOString(),
          };
          
          set({
            savedConfigs: [...get().savedConfigs, savedConfig],
          });
          
          // Persist to backend
          await chunkingApi.saveConfiguration(savedConfig);
        },
        
        // Compare multiple strategies
        compareStrategies: async (strategyIds: string[]) => {
          set({ comparisonStrategies: strategyIds });
          
          // Generate previews for all strategies
          const { previewDocument } = get();
          if (!previewDocument) return;
          
          const comparisons = await Promise.all(
            strategyIds.map(strategyId =>
              chunkingApi.previewChunking({
                documentId: previewDocument.id,
                strategyId,
                params: get().strategies.find(s => s.id === strategyId)?.defaultParams || {},
              })
            )
          );
          
          // Store comparison results
          set({ comparisonResults: comparisons });
        },
      }),
      {
        name: 'chunking-config',
        partialize: (state) => ({
          selectedStrategy: state.selectedStrategy,
          savedConfigs: state.savedConfigs,
          showAdvancedParams: state.showAdvancedParams,
        }),
      }
    )
  )
);

// Hooks for specific parts of the store
export const useSelectedStrategy = () => 
  useChunkingStore(state => 
    state.strategies.find(s => s.id === state.selectedStrategy)
  );

export const useChunkingPreview = () =>
  useChunkingStore(state => ({
    document: state.previewDocument,
    chunks: state.previewChunks,
    loading: state.previewLoading,
    error: state.previewError,
  }));
```
</example>
</implementation_examples>

<success_criteria>
- [ ] Strategy selector shows all 6 strategies with clear differentiation
- [ ] Preview updates in real-time as parameters change
- [ ] Comparison view allows side-by-side strategy evaluation
- [ ] Parameter tuning provides immediate visual feedback
- [ ] Analytics dashboard shows actionable insights
- [ ] All components are fully accessible (WCAG 2.1 AA)
- [ ] Responsive design works on mobile through desktop
- [ ] Performance remains smooth with large documents
</success_criteria>

<common_pitfalls>
- **Avoid**: Overwhelming users with too many options at once
- **Avoid**: Laggy preview updates (use debouncing)
- **Avoid**: Inaccessible color-only indicators
- **Avoid**: Missing loading and error states
- **Avoid**: Hard-coded values instead of using configuration
</common_pitfalls>
</task>

### Task 2.4: Comprehensive API Implementation

<task>
<metadata>
  <priority>High</priority>
  <effort>1.5 days</effort>
  <dependencies>Tasks 2.1, 2.2, 2.3</dependencies>
</metadata>

<context>
The API is the bridge between the powerful chunking backend and the elegant UI. A well-designed API makes integration smooth for both the frontend team and future API consumers. We need RESTful endpoints that are intuitive, performant, and secure. The API should handle both synchronous preview operations and asynchronous bulk processing elegantly, with proper error handling and progress reporting.
</context>

<instructions>
Build a comprehensive FastAPI implementation for all chunking operations.

**Step 1: Strategy Management Endpoints**
- GET /api/v2/chunking/strategies - List all available strategies
- GET /api/v2/chunking/strategies/{id} - Get strategy details
- GET /api/v2/chunking/strategies/recommend - Get recommendation based on file types
- POST /api/v2/chunking/configs - Save custom configuration
- GET /api/v2/chunking/configs - List saved configurations

**Step 2: Preview Endpoints**
- POST /api/v2/chunking/preview - Generate preview with specific strategy
- POST /api/v2/chunking/compare - Compare multiple strategies
- GET /api/v2/chunking/preview/{preview_id} - Get cached preview results
- DELETE /api/v2/chunking/preview/{preview_id} - Clear preview cache

**Step 3: Collection Processing Endpoints**
- POST /api/v2/collections/{id}/chunk - Start chunking operation
- PATCH /api/v2/collections/{id}/chunking-strategy - Update strategy
- GET /api/v2/collections/{id}/chunks - Get chunks with pagination
- GET /api/v2/collections/{id}/chunking-stats - Get performance metrics

**Step 4: Analytics Endpoints**
- GET /api/v2/chunking/metrics - Global chunking metrics
- GET /api/v2/chunking/metrics/by-strategy - Metrics grouped by strategy
- GET /api/v2/chunking/quality-scores - Chunk quality analysis
- POST /api/v2/chunking/analyze - Analyze document for strategy recommendation

**Step 5: WebSocket Support**
- WS /api/v2/ws/chunking-progress - Real-time progress updates
- Support multiple concurrent operations
- Graceful disconnection handling
- Progress throttling to prevent spam
</instructions>

<available_tools>
- **context7**: FastAPI best practices, Pydantic v2 patterns
- **Read/Edit**: Implement API endpoints
- **Bash**: Test endpoints with curl/httpie
- **Grep**: Search for existing API patterns
</available_tools>

<recommended_agents>
<agent_sequence>
  <agent type="implementation" name="backend-api-architect">
    <purpose>Design RESTful API with proper separation of concerns</purpose>
    <focus>API design, request/response models, error handling</focus>
  </agent>
  
  <agent type="security" name="backend-code-reviewer">
    <purpose>Ensure API security and input validation</purpose>
    <focus>Authentication, authorization, input sanitization</focus>
  </agent>
  
  <agent type="testing" name="qa-bug-hunter">
    <purpose>Create comprehensive API tests</purpose>
    <focus>Integration tests, edge cases, performance tests</focus>
  </agent>
  
  <agent type="documentation" name="docs-scribe">
    <purpose>Generate OpenAPI documentation</purpose>
    <focus>Clear examples, response schemas, error codes</focus>
  </agent>
</agent_sequence>
</recommended_agents>

<implementation_examples>
<example name="API Router and Models">
```python
# /packages/webui/api/v2/chunking.py
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import asyncio
from uuid import UUID

from webui.dependencies import get_current_user, get_chunking_service
from webui.services.chunking_service import ChunkingService
from shared.schemas.chunking import (
    ChunkingStrategy, ChunkingConfig, ChunkingPreview,
    ChunkingMetrics, StrategyRecommendation
)

router = APIRouter(prefix="/api/v2/chunking", tags=["chunking"])

# Request/Response Models
class PreviewRequest(BaseModel):
    """Request model for chunking preview."""
    document_id: Optional[str] = None
    document_content: Optional[str] = None
    strategy_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    max_chunks: int = Field(5, ge=1, le=20)
    
    @validator('document_content')
    def validate_content_size(cls, v):
        if v and len(v) > 1_000_000:  # 1MB limit for preview
            raise ValueError("Document content too large for preview")
        return v
    
    @validator('params')
    def validate_params(cls, v, values):
        # Validate parameters based on strategy
        strategy_id = values.get('strategy_id')
        # Add strategy-specific validation here
        return v

class ComparisonRequest(BaseModel):
    """Request model for strategy comparison."""
    document_id: str
    strategy_ids: List[str] = Field(..., min_items=2, max_items=4)
    params_override: Optional[Dict[str, Dict[str, Any]]] = None

class ChunkingStartRequest(BaseModel):
    """Request model to start chunking a collection."""
    strategy_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(5, ge=1, le=10)
    force_reindex: bool = False

# Strategy Management Endpoints
@router.get("/strategies", response_model=List[ChunkingStrategy])
async def list_strategies(
    include_inactive: bool = False,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> List[ChunkingStrategy]:
    """List all available chunking strategies."""
    strategies = await service.get_strategies(include_inactive)
    
    # Add performance metrics for each strategy
    for strategy in strategies:
        metrics = await service.get_strategy_metrics(strategy.id)
        strategy.actual_performance = metrics
    
    return strategies

@router.get("/strategies/recommend")
async def recommend_strategy(
    file_types: List[str] = Query(..., description="List of file extensions"),
    avg_file_size: Optional[int] = Query(None, description="Average file size in bytes"),
    total_documents: Optional[int] = Query(None, description="Total number of documents"),
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> StrategyRecommendation:
    """Get strategy recommendation based on collection characteristics."""
    
    recommendation = await service.recommend_strategy(
        file_types=file_types,
        avg_file_size=avg_file_size,
        total_documents=total_documents
    )
    
    return StrategyRecommendation(
        recommended_strategy_id=recommendation.strategy_id,
        confidence_score=recommendation.confidence,
        reasoning=recommendation.reasoning,
        alternative_strategies=recommendation.alternatives,
        performance_estimate={
            "estimated_time_seconds": recommendation.estimated_time,
            "estimated_chunks": recommendation.estimated_chunks,
            "estimated_cost": recommendation.estimated_cost
        }
    )

# Preview Endpoints
@router.post("/preview", response_model=ChunkingPreview)
async def generate_preview(
    request: PreviewRequest,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> ChunkingPreview:
    """Generate a preview of chunking results."""
    
    # Validate user has access to document if document_id provided
    if request.document_id:
        await service.validate_document_access(
            request.document_id, 
            current_user.id
        )
    
    # Generate preview
    preview = await service.generate_preview(
        document_id=request.document_id,
        document_content=request.document_content,
        strategy_id=request.strategy_id,
        params=request.params,
        max_chunks=request.max_chunks
    )
    
    return preview

@router.post("/compare")
async def compare_strategies(
    request: ComparisonRequest,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> Dict[str, ChunkingPreview]:
    """Compare multiple chunking strategies on the same document."""
    
    # Validate access
    await service.validate_document_access(
        request.document_id,
        current_user.id
    )
    
    # Generate previews in parallel
    tasks = []
    for strategy_id in request.strategy_ids:
        params = request.params_override.get(strategy_id, {}) if request.params_override else {}
        
        tasks.append(
            service.generate_preview(
                document_id=request.document_id,
                strategy_id=strategy_id,
                params=params,
                max_chunks=10  # More chunks for comparison
            )
        )
    
    previews = await asyncio.gather(*tasks)
    
    # Return as dict keyed by strategy_id
    return {
        strategy_id: preview 
        for strategy_id, preview in zip(request.strategy_ids, previews)
    }
```
</example>

<example name="Collection Processing Endpoints">
```python
# Collection chunking endpoints
@router.post("/collections/{collection_id}/chunk")
async def start_chunking(
    collection_id: UUID,
    request: ChunkingStartRequest,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> Dict[str, Any]:
    """Start asynchronous chunking operation for a collection."""
    
    # Validate collection ownership
    collection = await service.validate_collection_access(
        collection_id,
        current_user.id
    )
    
    # Check if already processing
    if collection.status == "indexing":
        raise HTTPException(
            status_code=409,
            detail="Collection is already being processed"
        )
    
    # Start chunking operation
    operation = await service.start_chunking(
        collection_id=collection_id,
        strategy_id=request.strategy_id,
        params=request.params,
        priority=request.priority,
        force_reindex=request.force_reindex,
        user_id=current_user.id
    )
    
    return {
        "operation_id": str(operation.id),
        "status": operation.status,
        "estimated_time_seconds": operation.estimated_time,
        "message": "Chunking operation started successfully"
    }

@router.get("/collections/{collection_id}/chunks")
async def get_collection_chunks(
    collection_id: UUID,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    document_id: Optional[UUID] = None,
    search: Optional[str] = None,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> Dict[str, Any]:
    """Get chunks for a collection with pagination."""
    
    # Validate access
    await service.validate_collection_access(
        collection_id,
        current_user.id
    )
    
    # Get chunks
    chunks, total = await service.get_chunks(
        collection_id=collection_id,
        document_id=document_id,
        search=search,
        offset=offset,
        limit=limit
    )
    
    return {
        "chunks": chunks,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total
    }

@router.get("/collections/{collection_id}/chunking-stats")
async def get_chunking_stats(
    collection_id: UUID,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> Dict[str, Any]:
    """Get detailed chunking statistics for a collection."""
    
    # Validate access
    await service.validate_collection_access(
        collection_id,
        current_user.id
    )
    
    stats = await service.get_collection_chunking_stats(collection_id)
    
    return {
        "collection_id": str(collection_id),
        "strategy_used": stats.strategy_name,
        "total_documents": stats.total_documents,
        "total_chunks": stats.total_chunks,
        "avg_chunk_size": stats.avg_chunk_size,
        "chunk_size_distribution": stats.size_distribution,
        "processing_time_ms": stats.processing_time_ms,
        "chunks_per_second": stats.chunks_per_second,
        "memory_used_mb": stats.memory_used_mb,
        "quality_metrics": {
            "empty_chunks": stats.empty_chunks,
            "oversized_chunks": stats.oversized_chunks,
            "avg_overlap_ratio": stats.avg_overlap_ratio
        },
        "last_updated": stats.last_updated
    }
```
</example>

<example name="WebSocket Progress Updates">
```python
# WebSocket connection manager
class ChunkingProgressManager:
    """Manages WebSocket connections for chunking progress updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.operation_status: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, operation_id: str):
        await websocket.accept()
        if operation_id not in self.active_connections:
            self.active_connections[operation_id] = []
        self.active_connections[operation_id].append(websocket)
        
        # Send current status if available
        if operation_id in self.operation_status:
            await websocket.send_json(self.operation_status[operation_id])
    
    def disconnect(self, websocket: WebSocket, operation_id: str):
        if operation_id in self.active_connections:
            self.active_connections[operation_id].remove(websocket)
            if not self.active_connections[operation_id]:
                del self.active_connections[operation_id]
    
    async def broadcast_progress(self, operation_id: str, progress: Dict):
        """Broadcast progress to all connected clients."""
        self.operation_status[operation_id] = progress
        
        if operation_id in self.active_connections:
            disconnected = []
            for websocket in self.active_connections[operation_id]:
                try:
                    await websocket.send_json(progress)
                except:
                    disconnected.append(websocket)
            
            # Clean up disconnected clients
            for websocket in disconnected:
                self.disconnect(websocket, operation_id)

# Global instance
progress_manager = ChunkingProgressManager()

@router.websocket("/ws/chunking-progress/{operation_id}")
async def chunking_progress_websocket(
    websocket: WebSocket,
    operation_id: str,
    token: str = Query(...)  # Auth token in query params for WebSocket
):
    """WebSocket endpoint for real-time chunking progress updates."""
    
    # Validate token
    try:
        user = await validate_websocket_token(token)
    except:
        await websocket.close(code=4001, reason="Invalid authentication")
        return
    
    # Validate operation access
    try:
        operation = await get_operation_with_access_check(operation_id, user.id)
    except:
        await websocket.close(code=4003, reason="Access denied")
        return
    
    # Connect client
    await progress_manager.connect(websocket, operation_id)
    
    try:
        # Keep connection alive
        while True:
            # Wait for any message (ping/pong)
            data = await websocket.receive_text()
            
            # Echo back for keep-alive
            if data == "ping":
                await websocket.send_text("pong")
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        progress_manager.disconnect(websocket, operation_id)

# Analytics Endpoints
@router.get("/metrics")
async def get_global_metrics(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d)$"),
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> ChunkingMetrics:
    """Get global chunking metrics across all collections."""
    
    # Only admins can see global metrics
    if not current_user.is_admin:
        # Return user-specific metrics instead
        return await service.get_user_metrics(current_user.id, time_range)
    
    return await service.get_global_metrics(time_range)

@router.get("/metrics/by-strategy")
async def get_metrics_by_strategy(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d)$"),
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> Dict[str, ChunkingMetrics]:
    """Get chunking metrics grouped by strategy."""
    
    strategies = await service.get_strategies()
    metrics = {}
    
    for strategy in strategies:
        if current_user.is_admin:
            metrics[strategy.name] = await service.get_strategy_metrics(
                strategy.id, 
                time_range
            )
        else:
            metrics[strategy.name] = await service.get_user_strategy_metrics(
                current_user.id,
                strategy.id,
                time_range
            )
    
    return metrics

@router.post("/analyze")
async def analyze_document(
    document_id: str,
    current_user = Depends(get_current_user),
    service: ChunkingService = Depends(get_chunking_service)
) -> Dict[str, Any]:
    """Analyze a document to provide chunking insights."""
    
    # Validate access
    await service.validate_document_access(document_id, current_user.id)
    
    analysis = await service.analyze_document(document_id)
    
    return {
        "document_id": document_id,
        "characteristics": {
            "file_type": analysis.file_type,
            "content_type": analysis.content_type,
            "total_length": analysis.total_length,
            "has_structure": analysis.has_structure,
            "detected_patterns": analysis.patterns
        },
        "recommendations": {
            "primary": {
                "strategy": analysis.recommended_strategy,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning
            },
            "alternatives": analysis.alternative_strategies
        },
        "estimated_results": {
            strategy.name: {
                "chunk_count": estimate.chunk_count,
                "avg_chunk_size": estimate.avg_size,
                "processing_time_ms": estimate.time_ms
            }
            for strategy, estimate in analysis.estimates.items()
        }
    }
```
</example>

<example name="Service Layer Implementation">
```python
# /packages/webui/services/chunking_service.py
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from redis.asyncio import Redis

from shared.database.repositories import ChunkingRepository
from shared.text_processing import ChunkingFactory
from webui.tasks import process_collection_chunking

class ChunkingService:
    """Service layer for all chunking operations."""
    
    def __init__(
        self,
        db: AsyncSession,
        redis: Redis,
        chunking_repo: ChunkingRepository
    ):
        self.db = db
        self.redis = redis
        self.chunking_repo = chunking_repo
        self.factory = ChunkingFactory()
    
    async def generate_preview(
        self,
        document_id: Optional[str],
        document_content: Optional[str],
        strategy_id: str,
        params: Dict[str, Any],
        max_chunks: int = 5
    ) -> ChunkingPreview:
        """Generate a preview of chunking results."""
        
        # Get content
        if document_id:
            document = await self.chunking_repo.get_document(document_id)
            content = await self._load_document_content(document)
            metadata = {
                "file_type": document.file_type,
                "file_name": document.file_name
            }
        else:
            content = document_content
            metadata = {}
        
        # Create chunker
        chunker = self.factory.create_chunker({
            "strategy": strategy_id,
            "params": params
        })
        
        # Generate chunks
        start_time = datetime.utcnow()
        chunks = await chunker.chunk_text_async(
            content,
            "preview",
            metadata
        )
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Limit chunks for preview
        preview_chunks = chunks[:max_chunks]
        
        return ChunkingPreview(
            chunks=preview_chunks,
            total_chunks=len(chunks),
            strategy_used=strategy_id,
            parameters_used=params,
            processing_time_ms=processing_time,
            avg_chunk_size=sum(len(c.text) for c in chunks) // len(chunks) if chunks else 0,
            preview_truncated=len(chunks) > max_chunks
        )
    
    async def start_chunking(
        self,
        collection_id: UUID,
        strategy_id: str,
        params: Dict[str, Any],
        priority: int,
        force_reindex: bool,
        user_id: int
    ) -> Operation:
        """Start an asynchronous chunking operation."""
        
        # Create operation record
        operation = await self.chunking_repo.create_operation(
            collection_id=collection_id,
            type="index" if not force_reindex else "reindex",
            initiated_by=user_id,
            priority=priority
        )
        
        # Create or get chunking config
        config = await self.chunking_repo.get_or_create_config(
            strategy_id=strategy_id,
            params=params
        )
        
        # Update collection
        await self.chunking_repo.update_collection(
            collection_id,
            chunking_config_id=config.id,
            status="indexing"
        )
        
        # Queue task with priority
        task = process_collection_chunking.apply_async(
            args=[str(collection_id), str(operation.id)],
            priority=priority,
            task_id=str(operation.id)
        )
        
        # Update operation with task ID
        await self.chunking_repo.update_operation(
            operation.id,
            celery_task_id=task.id
        )
        
        return operation
    
    async def get_collection_chunking_stats(
        self,
        collection_id: UUID
    ) -> CollectionStats:
        """Get detailed chunking statistics for a collection."""
        
        # Use materialized view for performance
        stats = await self.db.execute(
            select(collection_stats).where(
                collection_stats.c.id == collection_id
            )
        )
        
        row = stats.first()
        if not row:
            raise ValueError(f"Collection {collection_id} not found")
        
        # Get additional metrics
        metrics = await self.chunking_repo.get_latest_metrics(collection_id)
        
        # Get chunk size distribution
        distribution = await self.db.execute(
            select(
                func.width_bucket(
                    chunks.c.content_length,
                    0, 5000, 10
                ).label('bucket'),
                func.count().label('count')
            )
            .where(chunks.c.collection_id == collection_id)
            .group_by('bucket')
        )
        
        return CollectionStats(
            **row._asdict(),
            size_distribution=[
                {"range": f"{i*500}-{(i+1)*500}", "count": count}
                for i, count in distribution
            ],
            latest_metrics=metrics
        )
```
</example>
</implementation_examples>

<success_criteria>
- [ ] All endpoints follow RESTful conventions
- [ ] Proper authentication and authorization on all endpoints
- [ ] Input validation with helpful error messages
- [ ] Async operations return immediately with operation ID
- [ ] WebSocket provides real-time progress updates
- [ ] Preview operations complete in <2 seconds
- [ ] Comprehensive error handling with proper HTTP codes
- [ ] OpenAPI documentation auto-generated
</success_criteria>

<common_pitfalls>
- **Avoid**: Synchronous operations for large collections
- **Avoid**: Missing rate limiting on preview endpoints
- **Avoid**: Leaking internal errors to API responses
- **Avoid**: N+1 queries when fetching related data
- **Avoid**: WebSocket connections without proper cleanup
</common_pitfalls>
</task>

---