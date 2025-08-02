# Semantik Chunking Strategies - Implementation Plan (No Legacy, No Agentic, No Code)

## Executive Summary

This implementation plan reflects the decision to defer code-specific chunking to a follow-up release, with improvements based on architectural review feedback. We're implementing 6 core strategies in 4 weeks (with buffer), with code files handled by optimized recursive chunking until dedicated code support arrives ~2 weeks post-launch.

**Timeline**: 4 weeks (was 3.5 - added buffer)  
**Strategies**: 6 (character, recursive, markdown, semantic, hierarchical, hybrid)  
**Code Support**: Deferred to follow-up release

---

## Week 1: Core Foundation & Architecture

### Task 1.1: Base Architecture & Core Strategies

**Priority**: High  
**Effort**: 3 days  
**Dependencies**: llama-index, sentence-transformers

**Available Tools & Resources**:
- **context7**: Look up LlamaIndex node parsers/text splitters documentation, sentence-transformers API
- **Read/Edit/MultiEdit**: For code implementation
- **Bash**: Run tests and validate implementation
- **Grep/Glob**: Search existing codebase for patterns

**Recommended Subagents**:
- **Implementation**: `backend-api-architect` for designing BaseChunker interface and ChunkingService layer
- **Implementation**: `database-migrations-engineer` for planning data model changes
- **Review**: `backend-code-reviewer` for validating architecture and implementation
- **Testing**: `test-maestro` for ensuring comprehensive test coverage

#### Requirements
Start with architecture and 3 core strategies to ensure solid foundation:

1. **BaseChunker interface** with async support
2. **ChunkingService layer** for business logic separation
3. **Core strategies** using LlamaIndex:
   - **character** - Simple fixed-size splitting (TokenTextSplitter)
   - **recursive** - Smart general-purpose splitting (SentenceSplitter - handles code files temporarily)
   - **markdown** - For technical documentation (MarkdownNodeParser)

#### Implementation

**Enhanced Base Chunker Interface:**
```python
# /packages/shared/text_processing/base_chunker.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ChunkResult:
    """Type-safe chunk result."""
    chunk_id: str
    text: str
    start_offset: int
    end_offset: int
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class BaseChunker(ABC):
    """Base class for all chunking strategies."""
    
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Synchronous chunking."""
        pass
    
    @abstractmethod
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Asynchronous chunking for I/O bound operations."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate strategy-specific configuration."""
        pass
    
    @abstractmethod
    def estimate_chunks(self, text_length: int, config: Dict[str, Any]) -> int:
        """Estimate number of chunks for capacity planning."""
        pass
```

**ChunkingFactory Implementation:**
```python
# /packages/shared/text_processing/chunking_factory.py
from typing import Dict, Any
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.embeddings import MockEmbedding
import os

class ChunkingFactory:
    """Factory for creating chunking strategies using LlamaIndex."""
    
    @staticmethod
    def create_chunker(config: Dict[str, Any]) -> BaseChunker:
        """Create appropriate chunker based on configuration."""
        strategy = config["strategy"]
        params = config.get("params", {})
        
        if strategy == "character":
            return CharacterChunker(**params)
        elif strategy == "recursive":
            return RecursiveChunker(**params)
        elif strategy == "markdown":
            return MarkdownChunker(**params)
        elif strategy == "semantic":
            # Handle embedding model
            if "embed_model" not in params:
                # Use mock for testing or real model for production
                if os.getenv("TESTING", "false").lower() == "true":
                    params["embed_model"] = MockEmbedding(embed_dim=384)
                else:
                    params["embed_model"] = OpenAIEmbedding()
            return SemanticChunker(**params)
        elif strategy == "hierarchical":
            return HierarchicalChunker(**params)
        elif strategy == "hybrid":
            return HybridChunker(**params)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
```

**ChunkingService Layer:**
```python
# /packages/webui/services/chunking_service.py
from typing import List, Dict, Any, Optional
import hashlib
import json

class ChunkingService:
    """Service layer for chunking operations."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        redis_client: Redis,
        security_validator: ChunkingSecurityValidator
    ):
        self.db = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.redis = redis_client
        self.security = security_validator
        
    async def preview_chunking(
        self,
        text: str,
        file_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        max_chunks: int = 5
    ) -> ChunkingPreviewResponse:
        """Preview chunking with validation and caching."""
        # Validate input size
        self.security.validate_document_size(len(text))
        
        # Get or validate config
        if not config and file_type:
            config = FileTypeDetector.get_optimal_config(file_type)
        
        # Validate config security
        self.security.validate_chunk_params(config.get("params", {}))
        
        # Check cache
        config_hash = self._hash_config(config)
        cached = await self._get_cached_preview(config_hash, text[:1000])
        if cached:
            return cached
        
        # Create chunker and process
        chunker = ChunkingFactory.create_chunker(config)
        chunks = await chunker.chunk_text_async(text, "preview")
        
        # Build response
        response = ChunkingPreviewResponse(
            chunks=chunks[:max_chunks],
            total_chunks=len(chunks),
            strategy_used=config["strategy"],
            is_code_file=file_type in FileTypeDetector.CODE_EXTENSIONS,
            performance_metrics=self._calculate_metrics(chunks, len(text)),
            recommendations=self._get_recommendations(chunks, file_type)
        )
        
        # Cache result
        await self._cache_preview(config_hash, text[:1000], response)
        
        return response
```

**Security Validator:**
```python
# /packages/webui/services/chunking_security.py
class ChunkingSecurityValidator:
    """Validate chunking requests for security."""
    
    # Configurable limits
    MAX_CHUNK_SIZE = 10000
    MIN_CHUNK_SIZE = 50
    MAX_DOCUMENT_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CHUNKS_PER_DOCUMENT = 50000
    MAX_PREVIEW_SIZE = 1 * 1024 * 1024  # 1MB for preview
    
    @staticmethod
    def validate_chunk_params(params: Dict[str, Any]) -> None:
        """Validate chunking parameters are within safe bounds."""
        chunk_size = params.get("chunk_size", 0)
        if not (ChunkingSecurityValidator.MIN_CHUNK_SIZE <= 
                chunk_size <= 
                ChunkingSecurityValidator.MAX_CHUNK_SIZE):
            raise ValidationError(
                f"Chunk size must be between {ChunkingSecurityValidator.MIN_CHUNK_SIZE} "
                f"and {ChunkingSecurityValidator.MAX_CHUNK_SIZE}"
            )
        
        # Validate overlap
        overlap = params.get("chunk_overlap", 0)
        if overlap >= chunk_size:
            raise ValidationError("Chunk overlap must be less than chunk size")
            
    @staticmethod
    def validate_document_size(size: int, is_preview: bool = False) -> None:
        """Prevent processing of oversized documents."""
        max_size = (ChunkingSecurityValidator.MAX_PREVIEW_SIZE if is_preview 
                   else ChunkingSecurityValidator.MAX_DOCUMENT_SIZE)
        if size > max_size:
            raise ValidationError(f"Document too large: {size} > {max_size}")
```

#### Performance Benchmarks

**Concrete benchmark specifications:**
```python
# /tests/performance/chunking_benchmarks.py
class ChunkingBenchmarks:
    """Performance benchmarks for chunking strategies."""
    
    HARDWARE_BASELINE = {
        "cpu": "4 cores",
        "memory": "8GB",
        "description": "Standard container specs"
    }
    
    PERFORMANCE_TARGETS = {
        "character": {  # TokenTextSplitter
            "single_thread": 1000,  # chunks/sec
            "parallel_4": 3500,     # chunks/sec with 4 workers
            "memory_per_mb": 50     # MB memory per MB document
        },
        "recursive": {  # SentenceSplitter
            "single_thread": 800,
            "parallel_4": 3000,
            "memory_per_mb": 60
        },
        "markdown": {   # MarkdownNodeParser
            "single_thread": 600,
            "parallel_4": 2200,
            "memory_per_mb": 80
        },
        "semantic": {   # SemanticSplitterNodeParser
            "single_thread": 150,   # Lower due to embeddings
            "parallel_4": 400,      # Limited by embedding model
            "memory_per_mb": 200
        },
        "hierarchical": {  # HierarchicalNodeParser
            "single_thread": 400,   # Multiple passes
            "parallel_4": 1500,
            "memory_per_mb": 150
        }
    }
    
    DOCUMENT_PROFILES = [
        {"name": "small", "size": "1KB", "chunks_expected": 2},
        {"name": "medium", "size": "100KB", "chunks_expected": 100},
        {"name": "large", "size": "10MB", "chunks_expected": 10000},
        {"name": "xlarge", "size": "100MB", "chunks_expected": 100000}
    ]
```

---

### Task 1.2: Error Handling & Recovery Framework

**Priority**: High  
**Effort**: 1.5 days  
**Dependencies**: Task 1.1

**Available Tools & Resources**:
- **context7**: Research error handling patterns in FastAPI/Celery
- **Read/Edit/MultiEdit**: Implement error handling framework
- **Bash**: Test error scenarios

**Recommended Subagents**:
- **Implementation**: `backend-api-architect` for designing robust error handling patterns
- **Review**: `backend-code-reviewer` for security and completeness review
- **Testing**: `qa-bug-hunter` for creating edge case tests

#### Requirements
Comprehensive error handling for production reliability:

```python
# /packages/webui/services/chunking_error_handler.py
from enum import Enum
from typing import List, Dict, Any, Optional

class ChunkingErrorType(Enum):
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    INVALID_ENCODING = "invalid_encoding"
    STRATEGY_ERROR = "strategy_error"
    PARTIAL_FAILURE = "partial_failure"

class ChunkingErrorHandler:
    """Handle errors during chunking operations."""
    
    RETRY_STRATEGIES = {
        ChunkingErrorType.MEMORY_ERROR: {
            "max_retries": 2,
            "backoff": "exponential",
            "reduce_batch_size": True
        },
        ChunkingErrorType.TIMEOUT_ERROR: {
            "max_retries": 3,
            "backoff": "linear",
            "increase_timeout": True
        },
        ChunkingErrorType.INVALID_ENCODING: {
            "max_retries": 1,
            "fallback_encoding": "utf-8",
            "lossy_decode": True
        }
    }
    
    async def handle_partial_failure(
        self,
        operation_id: str,
        processed_chunks: List[ChunkResult],
        failed_documents: List[str],
        errors: List[Exception]
    ) -> ChunkingOperationResult:
        """Handle partial chunking failures gracefully."""
        
        # Save successful chunks
        await self.save_partial_results(operation_id, processed_chunks)
        
        # Analyze failure patterns
        failure_analysis = self.analyze_failures(errors)
        
        # Create recovery strategy
        recovery_strategy = self.create_recovery_strategy(
            failure_analysis,
            failed_documents
        )
        
        # Update collection status
        await self.update_collection_status(
            operation_id,
            CollectionStatus.DEGRADED,
            f"Partial failure: {len(failed_documents)} documents failed"
        )
        
        # Create recovery operation
        recovery_op = await self.create_recovery_operation(
            operation_id,
            recovery_strategy
        )
        
        return ChunkingOperationResult(
            status="partial_success",
            processed_count=len(processed_chunks),
            failed_count=len(failed_documents),
            recovery_operation_id=recovery_op.id,
            recommendations=recovery_strategy.recommendations
        )
    
    async def handle_streaming_failure(
        self,
        document_id: str,
        bytes_processed: int,
        error: Exception
    ) -> StreamRecoveryAction:
        """Handle failures during streaming processing."""
        if isinstance(error, MemoryError):
            # Reduce batch size and retry from checkpoint
            return StreamRecoveryAction(
                action="retry_from_checkpoint",
                checkpoint=bytes_processed,
                new_batch_size=self.calculate_reduced_batch_size(error)
            )
        elif isinstance(error, TimeoutError):
            # Extend timeout and retry
            return StreamRecoveryAction(
                action="retry_with_extended_timeout",
                checkpoint=bytes_processed,
                new_timeout=self.calculate_extended_timeout()
            )
        else:
            # Unrecoverable - mark document
            return StreamRecoveryAction(
                action="mark_failed",
                error_details=str(error)
            )
```

---

### Task 1.3: Performance Testing Framework

**Priority**: High  
**Effort**: 1 day  
**Dependencies**: Tasks 1.1, 1.2

**Available Tools & Resources**:
- **context7**: Look up pytest-benchmark, memory_profiler documentation
- **Bash**: Run performance benchmarks
- **WebSearch**: Research chunking performance best practices

**Recommended Subagents**:
- **Implementation**: `performance-profiler` for designing benchmark suite
- **Review**: `backend-code-reviewer` for test quality
- **Analysis**: `performance-profiler` for interpreting results

#### Requirements
Build performance testing early to validate targets:

```python
# /tests/performance/test_chunking_performance.py
import pytest
import asyncio
import time
import psutil
from typing import Dict, List

class TestChunkingPerformance:
    """Performance tests for chunking strategies."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Monitor resource usage during tests."""
        return PerformanceMonitor()
    
    @pytest.mark.parametrize("strategy,document_size,expected_rate", [
        ("character", "1MB", 1000),
        ("recursive", "1MB", 800),
        ("markdown", "1MB", 600),
    ])
    async def test_single_thread_performance(
        self,
        strategy: str,
        document_size: str,
        expected_rate: int,
        performance_monitor
    ):
        """Test single-threaded chunking performance."""
        # Generate test document
        document = self.generate_test_document(document_size)
        
        # Create chunker
        config = self.get_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Start monitoring
        performance_monitor.start()
        
        # Measure chunking
        start_time = time.time()
        chunks = await chunker.chunk_text_async(document, "test_doc")
        duration = time.time() - start_time
        
        # Stop monitoring
        metrics = performance_monitor.stop()
        
        # Calculate rate
        chunks_per_second = len(chunks) / duration
        
        # Assertions
        assert chunks_per_second >= expected_rate * 0.9, \
            f"{strategy} performance {chunks_per_second:.1f} below target {expected_rate}"
        
        assert metrics.peak_memory_mb < 100, \
            f"Memory usage {metrics.peak_memory_mb}MB exceeds limit"
        
        # Log results for tracking
        self.log_performance_result(strategy, document_size, chunks_per_second, metrics)
    
    @pytest.mark.parametrize("num_workers", [2, 4, 8])
    async def test_parallel_performance(self, num_workers: int):
        """Test parallel chunking scalability."""
        documents = [self.generate_test_document("100KB") for _ in range(100)]
        
        # Single worker baseline
        single_start = time.time()
        for doc in documents:
            await self.chunk_document(doc)
        single_duration = time.time() - single_start
        
        # Parallel processing
        parallel_start = time.time()
        await self.process_parallel(documents, num_workers)
        parallel_duration = time.time() - parallel_start
        
        # Calculate speedup
        speedup = single_duration / parallel_duration
        efficiency = speedup / num_workers
        
        # Should achieve at least 70% efficiency
        assert efficiency >= 0.7, \
            f"Parallel efficiency {efficiency:.2f} below threshold"
```

---

### 🔍 Review 1.1: Foundation Validation

**Priority**: Critical  
**Effort**: 0.5 days

**Available Tools & Resources**:
- **Read**: Review all implemented code
- **Bash**: Run test suite and benchmarks
- **TodoWrite**: Track review findings

**Recommended Subagents**:
- **Review**: `backend-code-reviewer` for architecture validation
- **Review**: `tech-debt-hunter` for identifying potential issues
- **Testing**: `test-maestro` for test coverage analysis  

#### Checklist
- [ ] BaseChunker interface supports sync and async operations
- [ ] ChunkingService properly separates concerns
- [ ] Security validation prevents malicious inputs
- [ ] Error handling covers all failure modes
- [ ] Performance tests establish baselines
- [ ] All 3 core strategies working correctly
- [ ] Code files handled gracefully with recursive

#### Decision Gate
- **Green**: Continue to remaining strategies
- **Yellow**: Fix issues, may need 1-2 days adjustment
- **Red**: Major architecture issues, reassess approach

---

## Week 2: Complete Strategies & Integration

### Task 2.1: Advanced Chunking Strategies

**Priority**: High  
**Effort**: 2 days  
**Dependencies**: Week 1 completion

**Available Tools & Resources**:
- **context7**: Look up sentence-transformers models, semantic chunking algorithms
- **Read/Edit/MultiEdit**: Implement advanced strategies
- **Bash**: Test with various document types
- **WebSearch**: Research semantic chunking best practices

**Recommended Subagents**:
- **Implementation**: `backend-api-architect` for semantic/hierarchical chunker design
- **Implementation**: `vector-search-architect` for embedding integration
- **Review**: `backend-code-reviewer` for algorithm correctness
- **Performance**: `performance-profiler` for optimizing embedding operations

#### Requirements
Implement remaining 3 strategies:

1. **semantic** - AI-powered semantic boundaries (using LlamaIndex SemanticSplitterNodeParser)
2. **hierarchical** - Multi-level chunking (using LlamaIndex HierarchicalNodeParser)
3. **hybrid** - Mixed content handling (combination of strategies)

#### Implementation Focus

**Semantic Chunker with LlamaIndex:**
```python
# /packages/shared/text_processing/semantic_chunker.py
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document
from typing import List, Dict, Any
import asyncio

class SemanticChunker(BaseChunker):
    """Semantic chunking using LlamaIndex native implementation."""
    
    def __init__(
        self,
        embed_model=None,
        breakpoint_percentile_threshold: int = 95,
        buffer_size: int = 1
    ):
        self.splitter = SemanticSplitterNodeParser(
            embed_model=embed_model or OpenAIEmbedding(),
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            buffer_size=buffer_size
        )
        self.strategy_name = "semantic"
        
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Async semantic chunking using LlamaIndex."""
        if not text.strip():
            return []
        
        # Create a temporary document
        doc = Document(text=text, metadata=metadata or {})
        
        # Get nodes using semantic splitter
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None,
            self.splitter.get_nodes_from_documents,
            [doc]
        )
        
        # Convert to ChunkResult
        results = []
        for idx, node in enumerate(nodes):
            results.append(ChunkResult(
                chunk_id=f"{doc_id}_chunk_{idx}",
                text=node.text,
                start_offset=node.start_char_idx or 0,
                end_offset=node.end_char_idx or len(node.text),
                metadata={**node.metadata, "strategy": self.strategy_name}
            ))
        
        return results


**Other Core Chunkers:**
```python
# Character/Token-based Chunker
from llama_index.core.node_parser import TokenTextSplitter

class CharacterChunker(BaseChunker):
    """Character/token-based chunking using LlamaIndex."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
        self.strategy_name = "character"

# Recursive/Sentence-based Chunker
from llama_index.core.node_parser import SentenceSplitter

class RecursiveChunker(BaseChunker):
    """Recursive sentence-based chunking using LlamaIndex."""
    
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100):
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.strategy_name = "recursive"
    
    # All chunkers follow the same async pattern
    async def chunk_text_async(self, text: str, doc_id: str, 
                             metadata: Dict[str, Any] = None) -> List[ChunkResult]:
        """Same implementation pattern as CharacterChunker"""
        doc = Document(text=text, metadata=metadata or {})
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None, self.splitter.get_nodes_from_documents, [doc]
        )
        return self._convert_nodes_to_chunks(nodes, doc_id)

# Markdown Chunker
from llama_index.core.node_parser import MarkdownNodeParser

class MarkdownChunker(BaseChunker):
    """Markdown-aware chunking using LlamaIndex."""
    
    def __init__(self):
        self.splitter = MarkdownNodeParser()
        self.strategy_name = "markdown"
```

**Advanced Strategies Implementation:**
```python
# Hierarchical Chunker
from llama_index.core.node_parser import HierarchicalNodeParser

class HierarchicalChunker(BaseChunker):
    """Hierarchical multi-level chunking using LlamaIndex."""
    
    def __init__(self, chunk_sizes: List[int] = None):
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        self.splitter = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes
        )
        self.strategy_name = "hierarchical"
    
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Creates parent-child hierarchical chunks."""
        doc = Document(text=text, metadata=metadata or {})
        
        # Get hierarchical nodes
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None,
            self.splitter.get_nodes_from_documents,
            [doc]
        )
        
        # Convert to ChunkResult with parent-child relationships
        results = []
        for idx, node in enumerate(nodes):
            parent_id = None
            if hasattr(node, 'parent_node') and node.parent_node:
                parent_id = node.parent_node.node_id
                
            results.append(ChunkResult(
                chunk_id=f"{doc_id}_chunk_{idx}",
                text=node.text,
                start_offset=node.start_char_idx or 0,
                end_offset=node.end_char_idx or len(node.text),
                metadata={
                    **node.metadata,
                    "strategy": self.strategy_name,
                    "parent_chunk_id": parent_id,
                    "chunk_level": getattr(node, 'level', 0)
                }
            ))
        
        return results

# Hybrid Chunker
class HybridChunker(BaseChunker):
    """Hybrid chunking that selects strategy based on content."""
    
    def __init__(self):
        self.strategies = {
            'markdown': MarkdownNodeParser(),
            'default': SentenceSplitter(chunk_size=600, chunk_overlap=100)
        }
        self.strategy_name = "hybrid"
    
    async def chunk_text_async(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[ChunkResult]:
        """Selects appropriate strategy based on content type."""
        # Detect content type
        file_type = metadata.get('file_type', '') if metadata else ''
        
        # Select strategy
        if file_type == '.md' or self._has_markdown_headers(text):
            splitter = self.strategies['markdown']
            used_strategy = 'markdown'
        else:
            splitter = self.strategies['default']
            used_strategy = 'recursive'
        
        # Process with selected strategy
        doc = Document(text=text, metadata=metadata or {})
        
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None,
            splitter.get_nodes_from_documents,
            [doc]
        )
        
        # Convert to ChunkResult
        results = []
        for idx, node in enumerate(nodes):
            results.append(ChunkResult(
                chunk_id=f"{doc_id}_chunk_{idx}",
                text=node.text,
                start_offset=node.start_char_idx or 0,
                end_offset=node.end_char_idx or len(node.text),
                metadata={
                    **node.metadata,
                    "strategy": self.strategy_name,
                    "sub_strategy": used_strategy
                }
            ))
        
        return results
    
    def _has_markdown_headers(self, text: str) -> bool:
        """Check if text contains markdown headers."""
        import re
        return bool(re.search(r'^#{1,6}\s+', text, re.MULTILINE))
```

---

### Task 2.2: Normalized Database Schema

**Priority**: High  
**Effort**: 1.5 days  
**Dependencies**: None

**Available Tools & Resources**:
- **context7**: PostgreSQL partitioning docs, SQLAlchemy patterns
- **Read/Edit**: Create Alembic migrations
- **Bash**: Run migrations and validate schema

**Recommended Subagents**:
- **Implementation**: `database-migrations-engineer` for schema design and migrations
- **Review**: `backend-code-reviewer` for data model validation
- **Performance**: `performance-profiler` for index optimization

#### Requirements
Implement normalized schema with performance optimizations:

```sql
-- Normalized chunking strategies table
CREATE TABLE chunking_strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    description TEXT,
    default_params JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default strategies
INSERT INTO chunking_strategies (name, display_name, default_params) VALUES
('character', 'Character-based', '{"chunk_size": 1000, "chunk_overlap": 200}'),
('recursive', 'Recursive', '{"chunk_size": 600, "chunk_overlap": 100}'),
('markdown', 'Markdown', '{"split_by_headers": true}'),
('semantic', 'Semantic', '{"breakpoint_percentile_threshold": 95, "buffer_size": 1}'),
('hierarchical', 'Hierarchical', '{"chunk_sizes": [2048, 512, 128]}'),
('hybrid', 'Hybrid', '{"primary_strategy": "recursive"}');

-- Chunking configurations (deduped)
CREATE TABLE chunking_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id INTEGER REFERENCES chunking_strategies(id),
    params JSONB NOT NULL,
    params_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(params_hash)
);

-- Collections with normalized config
CREATE TABLE collections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id INTEGER NOT NULL,
    chunking_config_id UUID REFERENCES chunking_configs(id),
    
    -- Status and stats
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    code_file_count INTEGER DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    
    -- Performance tracking
    last_chunk_time_ms INTEGER,
    avg_chunk_time_ms INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_collections_owner (owner_id),
    INDEX idx_collections_config (chunking_config_id),
    INDEX idx_collections_status (status)
);

-- Documents with enhanced tracking
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50),
    file_category VARCHAR(50),
    is_code_file BOOLEAN DEFAULT FALSE,
    file_size_bytes BIGINT,
    
    -- Chunking results
    chunk_count INTEGER DEFAULT 0,
    chunking_time_ms INTEGER,
    chunking_error TEXT,
    
    -- Status tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    processed_at TIMESTAMP WITH TIME ZONE,
    retry_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_documents_collection_status (collection_id, status),
    INDEX idx_documents_category (file_category),
    INDEX idx_documents_code (is_code_file) WHERE is_code_file = true
);

-- Chunking history for auditing
CREATE TABLE chunking_history (
    id SERIAL PRIMARY KEY,
    collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    operation_id UUID REFERENCES operations(id),
    previous_config_id UUID REFERENCES chunking_configs(id),
    new_config_id UUID REFERENCES chunking_configs(id),
    documents_affected INTEGER,
    chunks_before INTEGER,
    chunks_after INTEGER,
    processing_time_ms INTEGER,
    created_by INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_chunking_history_collection (collection_id)
);

-- Performance metrics table
CREATE TABLE chunking_metrics (
    id SERIAL PRIMARY KEY,
    collection_id UUID REFERENCES collections(id) ON DELETE CASCADE,
    strategy VARCHAR(50),
    chunks_created INTEGER,
    documents_processed INTEGER,
    total_size_bytes BIGINT,
    processing_time_ms INTEGER,
    chunks_per_second NUMERIC(10,2),
    avg_chunk_size INTEGER,
    peak_memory_mb INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_metrics_collection_date (collection_id, created_at DESC),
    INDEX idx_metrics_strategy (strategy)
);

-- Partitioned chunks table for scale
CREATE TABLE chunks (
    id UUID DEFAULT gen_random_uuid(),
    collection_id UUID NOT NULL,
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    embedding VECTOR(384),  -- Using pgvector
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    PRIMARY KEY (id, collection_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
) PARTITION BY HASH (collection_id);

-- Create 16 partitions for chunks
DO $$
BEGIN
    FOR i IN 0..15 LOOP
        EXECUTE format('CREATE TABLE chunks_p%s PARTITION OF chunks FOR VALUES WITH (modulus 16, remainder %s)', i, i);
    END LOOP;
END $$;

-- Materialized view for collection stats
CREATE MATERIALIZED VIEW collection_stats AS
SELECT 
    c.id,
    c.name,
    c.status,
    c.document_count,
    c.chunk_count,
    c.code_file_count,
    c.total_size_bytes,
    cs.name as strategy_name,
    cc.params as chunking_params,
    COALESCE(AVG(cm.chunks_per_second), 0) as avg_chunks_per_second,
    COALESCE(AVG(cm.avg_chunk_size), 0) as avg_chunk_size
FROM collections c
LEFT JOIN chunking_configs cc ON c.chunking_config_id = cc.id
LEFT JOIN chunking_strategies cs ON cc.strategy_id = cs.id
LEFT JOIN chunking_metrics cm ON c.id = cm.collection_id
GROUP BY c.id, c.name, c.status, c.document_count, c.chunk_count, 
         c.code_file_count, c.total_size_bytes, cs.name, cc.params;

-- Refresh stats periodically
CREATE OR REPLACE FUNCTION refresh_collection_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY collection_stats;
END;
$$ LANGUAGE plpgsql;
```

---

### Task 2.3: Enhanced API with Security

**Priority**: High  
**Effort**: 2 days  
**Dependencies**: Tasks 2.1, 2.2

**Available Tools & Resources**:
- **context7**: FastAPI security docs, rate limiting libraries
- **Read/Edit/MultiEdit**: Implement API endpoints
- **Bash**: Test API with curl/httpie
- **playwright**: Test WebSocket endpoints

**Recommended Subagents**:
- **Implementation**: `backend-api-architect` for API design and security
- **Review**: `backend-code-reviewer` for security audit
- **Testing**: `qa-bug-hunter` for API testing
- **Testing**: `test-maestro` for comprehensive endpoint coverage

#### Requirements
Build comprehensive API with all missing endpoints:

```python
# /packages/webui/api/collections.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi_limiter.depends import RateLimiter
from pydantic import BaseModel, validator, Field
from typing import Dict, Any, List, Optional, Literal
import asyncio

router = APIRouter(prefix="/collections", tags=["collections"])

# Request/Response Models
class ChunkingConfigRequest(BaseModel):
    """Validated chunking configuration."""
    strategy: Literal["character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]
    params: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("params")
    def validate_params(cls, v, values):
        """Validate params based on strategy."""
        strategy = values.get("strategy")
        # Strategy-specific validation
        if strategy == "character":
            chunk_size = v.get("chunk_size", 1000)
            if not 50 <= chunk_size <= 10000:
                raise ValueError("chunk_size must be between 50 and 10000")
        # ... more validations
        return v

class ChunkingPreviewRequest(BaseModel):
    """Preview request with size limits."""
    text_sample: str = Field(..., max_length=1_000_000)  # 1MB max
    file_type: Optional[str] = None
    strategy: Optional[ChunkingConfigRequest] = None
    max_chunks: int = Field(default=5, ge=1, le=20)

# Endpoints
@router.get("/chunking/strategies")
async def list_chunking_strategies(
    service: ChunkingService = Depends(get_chunking_service)
) -> List[ChunkingStrategyInfo]:
    """List available chunking strategies with details."""
    return await service.get_available_strategies()

@router.post("/chunking/preview")
@limiter.limit("10/minute")  # Rate limiting
async def preview_chunking(
    request: ChunkingPreviewRequest,
    background_tasks: BackgroundTasks,
    service: ChunkingService = Depends(get_chunking_service)
) -> ChunkingPreviewResponse:
    """Preview chunking with rate limiting."""
    result = await service.preview_chunking(
        text=request.text_sample,
        file_type=request.file_type,
        config=request.strategy.dict() if request.strategy else None,
        max_chunks=request.max_chunks
    )
    
    # Track usage analytics in background
    background_tasks.add_task(
        service.track_preview_usage,
        strategy=result.strategy_used,
        file_type=request.file_type
    )
    
    return result

@router.post("/chunking/recommend")
@limiter.limit("20/minute")
async def recommend_chunking_strategy(
    file_paths: List[str] = Field(..., max_items=100),
    service: ChunkingService = Depends(get_chunking_service)
) -> ChunkingRecommendation:
    """Recommend optimal chunking strategy."""
    return await service.recommend_strategy(file_paths)

@router.get("/{collection_id}/chunking/stats")
async def get_chunking_statistics(
    collection_id: str,
    days: int = Field(default=30, ge=1, le=365),
    service: ChunkingService = Depends(get_chunking_service),
    current_user: User = Depends(get_current_user)
) -> ChunkingStatistics:
    """Get detailed chunking statistics."""
    # Verify user has access
    await service.verify_collection_access(collection_id, current_user.id)
    
    return await service.get_chunking_statistics(collection_id, days)

@router.post("/{collection_id}/chunking/validate")
async def validate_chunking_config(
    collection_id: str,
    config: ChunkingConfigRequest,
    sample_size: int = Field(default=5, ge=1, le=20),
    service: ChunkingService = Depends(get_chunking_service),
    current_user: User = Depends(get_current_user)
) -> ChunkingValidationResult:
    """Validate chunking config against collection documents."""
    await service.verify_collection_access(collection_id, current_user.id)
    
    return await service.validate_config_for_collection(
        collection_id,
        config.dict(),
        sample_size
    )

# WebSocket for chunking progress
@router.websocket("/{collection_id}/chunking/progress")
async def chunking_progress(
    websocket: WebSocket,
    collection_id: str,
    service: ChunkingService = Depends(get_chunking_service)
):
    """WebSocket endpoint for real-time chunking progress."""
    await websocket.accept()
    
    try:
        # Subscribe to progress updates
        async for update in service.get_chunking_progress(collection_id):
            await websocket.send_json(update.dict())
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))
```

---

### Task 2.4: Async Processing with Priority Queues

**Priority**: High  
**Effort**: 1.5 days  
**Dependencies**: Previous tasks

**Available Tools & Resources**:
- **context7**: Celery priority queues, Redis patterns
- **Read/Edit**: Implement Celery tasks
- **Bash**: Test async processing
- **LS/Glob**: Find existing Celery patterns

**Recommended Subagents**:
- **Implementation**: `backend-api-architect` for async task design
- **Infrastructure**: `devops-sentinel` for Celery/Redis configuration
- **Review**: `backend-code-reviewer` for concurrency safety
- **Testing**: `qa-bug-hunter` for async edge cases

#### Requirements
Implement scalable async processing:

```python
# /packages/webui/tasks/chunking_tasks.py
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
import asyncio
from typing import Dict, Any, List

# Configure priority queues
CELERY_TASK_ROUTES = {
    'chunking.preview': {
        'queue': 'chunking_preview',
        'routing_key': 'preview',
        'priority': 9  # Highest priority
    },
    'chunking.process_small': {
        'queue': 'chunking_small',
        'routing_key': 'small',
        'priority': 5
    },
    'chunking.process_large': {
        'queue': 'chunking_large', 
        'routing_key': 'large',
        'priority': 3
    },
    'chunking.semantic': {
        'queue': 'chunking_ml',
        'routing_key': 'ml',
        'priority': 1  # Lowest priority, resource intensive
    }
}

class IdempotentChunkingTask(Task):
    """Base class for idempotent chunking tasks."""
    
    def before_start(self, task_id, args, kwargs):
        """Check if task already completed."""
        operation_id = kwargs.get('operation_id')
        if redis_client.get(f"chunking_complete:{operation_id}"):
            # Already completed, skip
            return {"status": "already_completed"}
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        operation_id = kwargs.get('operation_id')
        # Update operation status
        asyncio.run(self.update_operation_failed(operation_id, str(exc)))
        
        # Determine if retryable
        if self.is_retryable(exc):
            # Exponential backoff
            retry_delay = min(300, 2 ** self.request.retries * 10)
            raise self.retry(exc=exc, countdown=retry_delay)

@celery_app.task(
    base=IdempotentChunkingTask,
    bind=True,
    max_retries=3,
    soft_time_limit=300,  # 5 minutes
    time_limit=360        # 6 minutes hard limit
)
def process_chunking_operation(
    self,
    operation_id: str,
    collection_id: str,
    document_ids: List[str],
    config: Dict[str, Any]
):
    """Process chunking operation with idempotency."""
    
    # Acquire distributed lock
    lock_key = f"chunking_lock:{operation_id}"
    lock_timeout = 3600  # 1 hour
    
    with redis_client.lock(lock_key, timeout=lock_timeout):
        try:
            # Initialize progress tracking
            progress_tracker = ChunkingProgressTracker(
                operation_id,
                total_documents=len(document_ids)
            )
            
            # Process documents in batches
            batch_size = self.determine_batch_size(config["strategy"])
            
            for batch_start in range(0, len(document_ids), batch_size):
                batch = document_ids[batch_start:batch_start + batch_size]
                
                # Process batch
                results = asyncio.run(
                    self.process_batch(batch, config, progress_tracker)
                )
                
                # Update progress
                progress_tracker.update(len(results))
                
                # Check for cancellation
                if redis_client.get(f"chunking_cancel:{operation_id}"):
                    raise TaskCancelled("Operation cancelled by user")
        
        except SoftTimeLimitExceeded:
            # Gracefully handle timeout
            asyncio.run(self.handle_timeout(operation_id, progress_tracker))
            raise
        
        except Exception as e:
            # Log and handle error
            logger.error(f"Chunking error: {e}")
            raise
        
        finally:
            # Mark as complete
            redis_client.setex(
                f"chunking_complete:{operation_id}",
                86400,  # 24 hour TTL
                "1"
            )
            
            # Clean up lock
            redis_client.delete(lock_key)

    async def process_batch(
        self,
        document_ids: List[str],
        config: Dict[str, Any],
        progress_tracker: ChunkingProgressTracker
    ) -> List[ChunkingResult]:
        """Process a batch of documents."""
        async with get_db_session() as db:
            service = ChunkingService(db)
            
            # Process documents concurrently
            tasks = []
            for doc_id in document_ids:
                task = service.chunk_document(doc_id, config)
                tasks.append(task)
            
            # Gather results with error handling
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle individual failures
            successful = []
            failed = []
            
            for doc_id, result in zip(document_ids, results):
                if isinstance(result, Exception):
                    failed.append((doc_id, result))
                    await self.handle_document_failure(doc_id, result)
                else:
                    successful.append(result)
            
            # Update metrics
            await service.record_batch_metrics(
                successful_count=len(successful),
                failed_count=len(failed),
                config=config
            )
            
            return successful
```

---

### 🔍 Review 2.1: Integration Validation

**Priority**: Critical  
**Effort**: 0.5 days

**Available Tools & Resources**:
- **Read**: Review all integration code
- **Bash**: Run integration tests
- **WebFetch**: Check external API documentation

**Recommended Subagents**:
- **Review**: `backend-code-reviewer` for integration patterns
- **Review**: `tech-debt-hunter` for code quality issues
- **Performance**: `performance-profiler` for bottleneck analysis  

#### Checklist
- [ ] All 6 strategies implemented and tested
- [ ] Database schema normalized and indexed
- [ ] API endpoints secure with rate limiting
- [ ] Async processing with priority queues working
- [ ] Error recovery handles partial failures
- [ ] Performance meets benchmarks
- [ ] Code files handled appropriately

---

## Week 3: Testing, Performance & Polish

### Task 3.1: Comprehensive Test Suite

**Priority**: High  
**Effort**: 2 days  
**Dependencies**: All core features complete

**Available Tools & Resources**:
- **context7**: pytest patterns, test fixture best practices, LlamaIndex testing utilities
- **Read/Edit/MultiEdit**: Write comprehensive tests
- **Bash**: Run test suite with coverage
- **playwright**: E2E testing for UI components

**Recommended Subagents**:
- **Implementation**: `qa-bug-hunter` for test strategy and implementation
- **Implementation**: `test-maestro` for coverage analysis
- **Review**: `backend-code-reviewer` for test quality
- **UI Testing**: `frontend-code-reviewer` if UI tests needed

#### Requirements
Build test suite covering all aspects with LlamaIndex mock utilities:

```python
# /tests/unit/test_all_chunking_strategies.py
from llama_index.core.embeddings import MockEmbedding
from llama_index.core import Document

class TestChunkingStrategies:
    """Comprehensive tests for all strategies."""
    
    @pytest.fixture
    def mock_embed_model(self):
        """Mock embedding model for semantic chunking tests."""
        return MockEmbedding(embed_dim=384)
    
    # Test data fixtures
    EDGE_CASES = {
        "empty": "",
        "single_char": "A",
        "unicode": "Hello 世界! 🌍 → €£¥",
        "very_long_line": "a" * 50000,
        "null_bytes": "Hello\x00World",
        "mixed_encoding": b"Hello\xff\xfeWorld".decode('utf-8', errors='ignore'),
        "only_whitespace": "   \n\n\t\t  ",
        "html_injection": "<script>alert('xss')</script>",
        "sql_injection": "'; DROP TABLE chunks; --"
    }
    
    PERFORMANCE_DOCS = {
        "small": generate_doc(1 * KB),
        "medium": generate_doc(100 * KB),
        "large": generate_doc(10 * MB),
        "xlarge": generate_doc(100 * MB)
    }
    
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @pytest.mark.parametrize("edge_case_name,text", EDGE_CASES.items())
    async def test_edge_cases(self, strategy: str, edge_case_name: str, text: str):
        """Test all strategies handle edge cases gracefully."""
        config = get_default_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Should not raise exception
        try:
            chunks = await chunker.chunk_text_async(text, "test")
            
            # Validate chunks
            if text.strip():  # Non-empty input
                assert isinstance(chunks, list)
                for chunk in chunks:
                    assert isinstance(chunk, ChunkResult)
                    assert chunk.text  # Non-empty chunk
            else:  # Empty input
                assert chunks == []
                
        except Exception as e:
            pytest.fail(f"{strategy} failed on {edge_case_name}: {e}")
    
    @pytest.mark.integration
    async def test_code_file_optimization(self):
        """Test code files get optimized parameters."""
        code_file = """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
        
    def multiply(self, a, b):
        return a * b
"""
        
        # Test with .py file type
        config = FileTypeDetector.get_optimal_config("test.py")
        assert config["strategy"] == "recursive"
        assert config["params"]["chunk_size"] == 400  # Optimized for code
        assert config["params"]["chunk_overlap"] == 50
        
        # Test chunking preserves code structure reasonably
        chunker = ChunkingFactory.create_chunker(config)
        chunks = await chunker.chunk_text_async(code_file, "test.py")
        
        # Verify chunks
        assert len(chunks) >= 2
        assert any("fibonacci" in chunk.text for chunk in chunks)
        assert any("Calculator" in chunk.text for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_semantic_chunking_with_mock(self, mock_embed_model):
        """Test semantic chunking with mock embeddings."""
        config = {
            "strategy": "semantic",
            "params": {
                "embed_model": mock_embed_model,
                "breakpoint_percentile_threshold": 95
            }
        }
        
        chunker = ChunkingFactory.create_chunker(config)
        text = "This is the first topic. It talks about AI. " \
               "This is the second topic. It discusses databases."
        
        chunks = await chunker.chunk_text_async(text, "test_doc")
        
        # Verify semantic chunking worked
        assert len(chunks) > 0
        assert all(chunk.metadata["strategy"] == "semantic" for chunk in chunks)
```

**Integration Tests:**
```python
# /tests/integration/test_chunking_e2e.py
class TestChunkingE2E:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_collection_lifecycle(self, test_client, test_db):
        """Test complete collection lifecycle with chunking."""
        # Create collection
        response = await test_client.post(
            "/collections",
            json={
                "name": "Test Collection",
                "description": "E2E test",
                "chunking": {
                    "strategy": "recursive",
                    "params": {"chunk_size": 500}
                }
            }
        )
        assert response.status_code == 201
        collection = response.json()
        
        # Add documents including code files
        files = [
            ("file", ("doc.txt", "Test document content", "text/plain")),
            ("file", ("code.py", "def test(): pass", "text/x-python")),
            ("file", ("readme.md", "# Test\n\nContent", "text/markdown"))
        ]
        
        response = await test_client.post(
            f"/collections/{collection['id']}/documents",
            files=files
        )
        assert response.status_code == 202
        operation = response.json()
        
        # Wait for processing
        await wait_for_operation(operation['id'])
        
        # Get stats
        response = await test_client.get(
            f"/collections/{collection['id']}/stats"
        )
        stats = response.json()
        
        assert stats['document_count'] == 3
        assert stats['code_file_count'] == 1
        assert stats['chunk_count'] > 0
        
        # Update strategy
        response = await test_client.patch(
            f"/collections/{collection['id']}/chunking",
            json={
                "chunking": {
                    "strategy": "semantic",
                    "params": {"breakpoint_threshold": 90}
                },
                "reindex": True
            }
        )
        assert response.status_code == 202
        
        # Verify reindexing started
        operation = response.json()
        assert operation['type'] == 'reindex'
```

---

### Task 3.2: Performance Optimization

**Priority**: High  
**Effort**: 1.5 days  
**Dependencies**: Test suite

**Available Tools & Resources**:
- **context7**: Python profiling tools, async optimization
- **Bash**: Run profilers and benchmarks
- **Read/Edit**: Implement optimizations

**Recommended Subagents**:
- **Implementation**: `performance-profiler` for identifying bottlenecks
- **Implementation**: `backend-api-architect` for optimization strategies
- **Review**: `backend-code-reviewer` for optimization correctness

#### Requirements
Optimize based on benchmark results:

```python
# /packages/shared/text_processing/optimizations.py
import asyncio
from functools import lru_cache
from typing import List, Dict, Any
import numpy as np

class ChunkingOptimizations:
    """Performance optimizations for chunking."""
    
    @staticmethod
    def enable_streaming_for_large_docs(
        chunker: BaseChunker,
        threshold_mb: int = 10
    ) -> StreamingChunker:
        """Wrap chunker with streaming support for large documents."""
        return StreamingChunker(chunker, threshold_mb)
        
    @staticmethod
    @lru_cache(maxsize=1000)
    def cache_chunking_config(config_hash: str) -> Dict[str, Any]:
        """Cache parsed configurations."""
        # Retrieve and parse config
        pass
    
    @staticmethod
    async def batch_chunk_documents(
        documents: List[Document],
        config: Dict[str, Any],
        max_concurrent: int = 10
    ) -> List[ChunkResult]:
        """Process multiple documents concurrently with limits."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def chunk_with_limit(doc):
            async with semaphore:
                return await chunk_document(doc, config)
        
        tasks = [chunk_with_limit(doc) for doc in documents]
        return await asyncio.gather(*tasks)

class StreamingChunker:
    """Stream large documents to avoid memory issues."""
    
    def __init__(self, base_chunker: BaseChunker, threshold_mb: int):
        self.base_chunker = base_chunker
        self.threshold_bytes = threshold_mb * 1024 * 1024
        
    async def chunk_document_stream(
        self,
        file_path: str,
        doc_id: str,
        config: Dict[str, Any]
    ) -> AsyncIterator[ChunkResult]:
        """Stream chunks for large documents."""
        file_size = os.path.getsize(file_path)
        
        if file_size < self.threshold_bytes:
            # Small file, process normally
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            chunks = await self.base_chunker.chunk_text_async(content, doc_id)
            for chunk in chunks:
                yield chunk
        else:
            # Large file, stream processing
            buffer = ""
            chunk_index = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    # Read in chunks
                    new_content = f.read(1024 * 1024)  # 1MB at a time
                    if not new_content:
                        break
                    
                    buffer += new_content
                    
                    # Process when buffer is large enough
                    if len(buffer) >= config.get('chunk_size', 1000) * 2:
                        # Find good break point
                        break_point = self.find_break_point(buffer)
                        
                        # Process this portion
                        to_process = buffer[:break_point]
                        chunks = await self.base_chunker.chunk_text_async(
                            to_process,
                            doc_id
                        )
                        
                        # Yield chunks with adjusted indices
                        for chunk in chunks:
                            chunk.chunk_index += chunk_index
                            yield chunk
                        
                        chunk_index += len(chunks)
                        
                        # Keep remainder in buffer
                        buffer = buffer[break_point:]
            
            # Process remaining buffer
            if buffer:
                chunks = await self.base_chunker.chunk_text_async(buffer, doc_id)
                for chunk in chunks:
                    chunk.chunk_index += chunk_index
                    yield chunk
```

---

### Task 3.3: Monitoring & Analytics

**Priority**: Medium  
**Effort**: 1 day  
**Dependencies**: Previous tasks

**Available Tools & Resources**:
- **context7**: Prometheus Python client, monitoring patterns
- **Read/Edit**: Implement metrics collection
- **Bash**: Test metric endpoints

**Recommended Subagents**:
- **Implementation**: `backend-api-architect` for metrics design
- **Infrastructure**: `devops-sentinel` for monitoring setup
- **Review**: `backend-code-reviewer` for completeness

#### Requirements
Add comprehensive monitoring:

```python
# /packages/webui/services/chunking_analytics.py
from dataclasses import dataclass
from typing import Dict, List, Any
import prometheus_client
from datetime import datetime, timedelta

# Prometheus metrics
chunking_duration = prometheus_client.Histogram(
    'chunking_duration_seconds',
    'Time spent chunking documents',
    ['strategy', 'file_type']
)

chunking_rate = prometheus_client.Counter(
    'chunks_created_total',
    'Total number of chunks created',
    ['strategy']
)

chunking_errors = prometheus_client.Counter(
    'chunking_errors_total',
    'Total chunking errors',
    ['strategy', 'error_type']
)

class ChunkingAnalyticsService:
    """Analytics and monitoring for chunking."""
    
    async def record_chunking_operation(
        self,
        operation_id: str,
        collection_id: str,
        strategy: str,
        documents_processed: int,
        chunks_created: int,
        processing_time_ms: int,
        errors: List[Dict[str, Any]] = None
    ):
        """Record comprehensive metrics for operation."""
        # Prometheus metrics
        chunking_duration.labels(
            strategy=strategy,
            file_type='mixed'
        ).observe(processing_time_ms / 1000)
        
        chunking_rate.labels(strategy=strategy).inc(chunks_created)
        
        if errors:
            for error in errors:
                chunking_errors.labels(
                    strategy=strategy,
                    error_type=error.get('type', 'unknown')
                ).inc()
        
        # Database metrics
        metric = ChunkingMetric(
            collection_id=collection_id,
            operation_id=operation_id,
            strategy=strategy,
            documents_processed=documents_processed,
            chunks_created=chunks_created,
            processing_time_ms=processing_time_ms,
            chunks_per_second=chunks_created / (processing_time_ms / 1000),
            errors_count=len(errors) if errors else 0,
            success_rate=1 - (len(errors) / documents_processed if errors else 0)
        )
        
        async with self.db:
            self.db.add(metric)
            await self.db.commit()
    
    async def get_strategy_performance_comparison(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Compare performance across strategies."""
        since = datetime.utcnow() - timedelta(days=days)
        
        query = """
        SELECT 
            strategy,
            COUNT(DISTINCT operation_id) as operations,
            SUM(documents_processed) as total_documents,
            SUM(chunks_created) as total_chunks,
            AVG(chunks_per_second) as avg_speed,
            AVG(success_rate) as avg_success_rate,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY chunks_per_second) as median_speed,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY chunks_per_second) as p95_speed
        FROM chunking_metrics
        WHERE created_at > :since
        GROUP BY strategy
        ORDER BY avg_speed DESC
        """
        
        results = await self.db.execute(query, {"since": since})
        
        return {
            "strategies": [dict(row) for row in results],
            "period_days": days,
            "generated_at": datetime.utcnow()
        }
    
    async def get_code_file_insights(self) -> Dict[str, Any]:
        """Analyze code file processing to inform future development."""
        query = """
        SELECT 
            d.file_type,
            COUNT(*) as file_count,
            AVG(d.chunk_count) as avg_chunks,
            AVG(d.chunking_time_ms) as avg_time_ms,
            AVG(d.file_size_bytes) as avg_size
        FROM documents d
        WHERE d.is_code_file = true
        GROUP BY d.file_type
        ORDER BY file_count DESC
        """
        
        results = await self.db.execute(query)
        
        return {
            "code_file_stats": [dict(row) for row in results],
            "total_code_files": sum(row.file_count for row in results),
            "recommendation": "High code file usage - prioritize dedicated code chunking"
        }
```

---

### Task 3.4: Documentation & Communication

**Priority**: High  
**Effort**: 1.5 days  
**Dependencies**: All features complete

**Available Tools & Resources**:
- **Read/Write**: Create documentation files
- **WebSearch**: Research documentation best practices
- **context7**: API documentation tools

**Recommended Subagents**:
- **Implementation**: `docs-scribe` for comprehensive documentation
- **Review**: `plan-feedback-specialist` for documentation quality
- **Frontend**: `frontend-state-architect` for UI integration docs

#### Requirements
Create comprehensive documentation:

1. **API Documentation** - Update with all new endpoints
2. **Strategy Guide** - When to use each strategy
3. **Code File Handling** - Clear explanation of current approach and roadmap
4. **Performance Tuning** - Guide for optimizing chunking
5. **Migration Guide** - For when code support arrives

---

### 🔍 Review 3.1: Pre-Launch Validation

**Priority**: Critical  
**Effort**: 0.5 days

**Available Tools & Resources**:
- **Read**: Review all code and documentation
- **Bash**: Run full test suite
- **TodoWrite**: Track final issues

**Recommended Subagents**:
- **Review**: `backend-code-reviewer` for final code review
- **Review**: `tech-debt-hunter` for technical debt assessment
- **Testing**: `test-maestro` for test completeness
- **Docs**: `docs-scribe` for documentation accuracy  

#### Checklist
- [ ] All tests passing with >90% coverage
- [ ] Performance meets all benchmarks
- [ ] Security validation prevents attacks
- [ ] Error handling covers all scenarios
- [ ] Monitoring provides visibility
- [ ] Documentation is complete
- [ ] Code file messaging is clear

---

## Week 4: Final Polish & Buffer

### Task 4.1: Performance Validation

**Priority**: High  
**Effort**: 1 day  
**Dependencies**: All features complete

**Available Tools & Resources**:
- **Bash**: Run load tests and benchmarks
- **Read**: Analyze performance results
- **context7**: Load testing tools documentation

**Recommended Subagents**:
- **Implementation**: `performance-profiler` for comprehensive validation
- **Infrastructure**: `devops-sentinel` for resource monitoring
- **Review**: `backend-code-reviewer` for performance findings

Run comprehensive performance tests at scale.

### Task 4.2: Security Audit

**Priority**: High  
**Effort**: 1 day  
**Dependencies**: All features complete

**Available Tools & Resources**:
- **Read**: Review security implementations
- **Bash**: Run security scanners
- **WebSearch**: Security best practices

**Recommended Subagents**:
- **Review**: `backend-code-reviewer` for security audit
- **Testing**: `qa-bug-hunter` for security test cases
- **Infrastructure**: `devops-sentinel` for container security

Final security review and penetration testing.

### Task 4.3: Documentation Review

**Priority**: Medium  
**Effort**: 0.5 days  
**Dependencies**: Documentation complete

**Available Tools & Resources**:
- **Read**: Review all documentation
- **Edit**: Fix any issues found

**Recommended Subagents**:
- **Review**: `docs-scribe` for documentation completeness
- **Review**: `plan-feedback-specialist` for clarity and structure

Review all documentation for accuracy and completeness.

### Task 4.4: Deployment Preparation

**Priority**: High  
**Effort**: 1 day  
**Dependencies**: All tasks complete

**Available Tools & Resources**:
- **Read/Edit**: Create deployment scripts
- **Bash**: Test deployment process
- **context7**: Docker/Kubernetes best practices

**Recommended Subagents**:
- **Implementation**: `devops-sentinel` for deployment scripts
- **Review**: `backend-code-reviewer` for deployment safety
- **Documentation**: `docs-scribe` for deployment guides

Prepare deployment scripts, monitoring, and rollback plans.

### 🔍 Final Review: Launch Readiness

**Priority**: Critical  
**Effort**: 0.5 days

**Available Tools & Resources**:
- **Read**: Final review of all components
- **Bash**: Run full validation suite
- **TodoWrite**: Track any final issues

**Recommended Subagents**:
- **Review**: `task-orchestrator` for overall readiness assessment
- **Review**: `backend-code-reviewer` for final sign-off
- **Review**: `tech-debt-hunter` for post-launch improvements  

#### Launch Checklist
- [ ] All features working correctly
- [ ] Performance validated at scale
- [ ] Security audit passed
- [ ] Documentation complete and accurate
- [ ] Monitoring and alerts configured
- [ ] Rollback plan tested
- [ ] Team trained on support

#### Go/No-Go Decision
- **Ship**: Ready for production
- **Delay**: Critical issues need resolution

---

## Summary

### What We're Building (4 weeks)
- 6 chunking strategies (code support deferred)
- Normalized database schema with performance optimizations
- Comprehensive API with security and rate limiting
- Scalable async processing with priority queues
- Robust error handling and recovery
- Performance monitoring and analytics
- Clear communication about code support roadmap

### Key Improvements from Review
- **Week 1**: Reduced scope to 3 strategies + architecture
- **Added**: Dedicated error handling framework
- **Added**: Concrete performance benchmarks
- **Added**: Security validation throughout
- **Added**: ChunkingService layer for separation of concerns
- **Added**: Streaming support for large documents
- **Added**: Comprehensive test specifications
- **Added**: 0.5 week buffer for unexpected issues

### Risk Mitigation
- Early performance validation in Week 1
- Phased strategy implementation
- Comprehensive error handling
- Security validation at every layer
- Buffer time for issues

This revised plan addresses all major concerns raised in the reviews while maintaining the 4-week timeline for delivering a robust, scalable chunking system.