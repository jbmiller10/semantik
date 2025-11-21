# Chunking API Documentation

## Overview

The Chunking API provides a comprehensive set of endpoints for managing document chunking strategies within the Semantik platform. This API enables intelligent document splitting, strategy comparison, and real-time processing of large document collections with WebSocket support for progress tracking.

## Architecture and Design Decisions

### Core Design Principles

1. **Strategy-Based Architecture**: The API implements multiple chunking strategies (Fixed Size, Semantic, Recursive, Sliding Window, Document Structure, and Hybrid), each optimized for different content types and use cases.

2. **Asynchronous Processing**: Long-running chunking operations are handled asynchronously with real-time progress updates via WebSockets, ensuring responsive user experience.

3. **Caching and Performance**: Preview operations are cached for 15 minutes to improve performance during strategy comparison and testing.

4. **Rate Limiting**: Implemented per-endpoint rate limiting to prevent abuse and ensure fair resource usage across users.

5. **Quality Metrics**: Built-in quality analysis and scoring system to help users select optimal chunking strategies for their content.

### Key Components

- **Strategy Management**: Endpoints for listing, recommending, and configuring chunking strategies
- **Preview Operations**: Real-time preview generation with caching and comparison capabilities
- **Collection Processing**: Bulk processing of document collections with progress tracking
- **Analytics**: Comprehensive metrics and quality analysis for chunking operations
- **Configuration Management**: Save and reuse custom chunking configurations

## Authentication

All chunking API endpoints require JWT authentication. Include the bearer token in the Authorization header:

```http
Authorization: Bearer {jwt_token}
```

## Base URL

```
http://localhost:8080/api/v2/chunking
```

## Endpoint Reference

### Strategy Management Endpoints

#### List Available Strategies

```http
GET /api/v2/chunking/strategies
```

Returns a comprehensive list of all available chunking strategies with their characteristics and default configurations.

**Response (200):**
```json
[
  {
    "id": "semantic",
    "name": "Semantic Chunking",
    "description": "Groups content based on semantic similarity",
    "best_for": ["pdf", "docx", "md", "html"],
    "pros": [
      "Preserves semantic coherence",
      "Better context retention",
      "Improved search quality"
    ],
    "cons": [
      "Slower processing",
      "Higher memory usage",
      "Requires embedding model",
      "Variable chunk sizes"
    ],
    "default_config": {
      "strategy": "semantic",
      "chunk_size": 512,
      "chunk_overlap": 50,
      "preserve_sentences": true
    },
    "performance_characteristics": {
      "speed": "slow",
      "memory_usage": "high",
      "quality": "high"
    }
  }
]
```

#### Get Strategy Details

```http
GET /api/v2/chunking/strategies/{strategy_id}
```

Retrieves detailed information about a specific chunking strategy.

**Path Parameters:**
- `strategy_id`: Strategy identifier (e.g., "semantic", "fixed_size", "recursive")

**Response (200):** Same structure as single strategy in list response

**Error Responses:**
- `404 Not Found`: Strategy not found

#### Recommend Strategy

```http
POST /api/v2/chunking/strategies/recommend
```

Get an AI-powered strategy recommendation based on file types and content characteristics.

**Query Parameters:**
- `file_types` (required): List of file extensions to analyze (e.g., ["pdf", "md", "txt"])

**Response (200):**
```json
{
  "recommended_strategy": "semantic",
  "confidence": 0.85,
  "reasoning": "Based on the presence of PDF and Markdown files, semantic chunking will best preserve document structure and meaning",
  "alternative_strategies": ["recursive", "document_structure"],
  "suggested_config": {
    "strategy": "semantic",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "preserve_sentences": true
  }
}
```

## Plugin Author Guide

Third-party chunking strategies register themselves at runtime through the strategy registry and factory. Each plugin must provide:

1. A strategy implementation class that subclasses `packages.shared.chunking.domain.services.chunking_strategies.base.ChunkingStrategy` and implements `chunk(text, config)`.
2. A call to `ChunkingStrategyFactory.register_strategy(<internal_name>, <class>)` so the orchestrator can instantiate it.
3. A call to `register_strategy_definition(...)` so the strategy surfaces in `/strategies` responses with defaults and metadata.

Minimal example (register at import time in your plugin module):

```python
from packages.shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy
from webui.services.chunking.strategy_registry import register_strategy_definition
from webui.services.chunking_strategy_factory import ChunkingStrategyFactory


class DemoPluginStrategy(ChunkingStrategy):
    INTERNAL_NAME = "demo_plugin"
    API_ID = "demo_plugin"

    def chunk(self, text: str, config):  # config is a ChunkConfig
        return [text]  # trivial passthrough


register_strategy_definition(
    api_id="demo_plugin",
    internal_id="demo_plugin",
    display_name="Demo Plugin",
    description="Example plugin strategy",
    manager_defaults={"chunk_size": 256, "chunk_overlap": 0},
    builder_defaults={"chunk_size": 128, "chunk_overlap": 0},
    visual_example={"url": "https://example.com/demo.png"},
    is_plugin=True,
)
ChunkingStrategyFactory.register_strategy("demo_plugin", DemoPluginStrategy)
```

After registration the orchestrator accepts `strategy_name="demo_plugin"` along with optional `strategy_config` and treats it like any built-in strategy.

## Configuration Persistence

User-defined chunking presets are persisted in the database table `chunking_config_profiles` (Alembic migration `202511201200`). The legacy `data/chunking_configs.json` file is no longer used. The FastAPI endpoints `/configs` (save/list) now store and retrieve rows via the orchestrator's configuration manager, enabling multi-replica safe storage and auditability.

### Preview Operations

#### Generate Preview

```http
POST /api/v2/chunking/preview
```

Generate a preview of how content would be chunked using a specific strategy. Results are cached for 15 minutes.

**Rate Limit:** 10 requests per minute per user

**Request Body:**
```json
{
  "document_id": "doc_123",  // Optional if content provided
  "content": "Document text...",  // Optional if document_id provided
  "strategy": "semantic",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "preserve_sentences": true
  },
  "max_chunks": 10,
  "include_metrics": true
}
```

**Response (200):**
```json
{
  "preview_id": "prev_abc123",
  "strategy": "semantic",
  "config": {
    "strategy": "semantic",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "preserve_sentences": true
  },
  "chunks": [
    {
      "index": 0,
      "content": "This is the first chunk of text...",
      "token_count": 125,
      "char_count": 512,
      "metadata": {
        "start_pos": 0,
        "end_pos": 512
      },
      "quality_score": 0.92,
      "overlap_info": {
        "next_overlap": 50
      }
    }
  ],
  "total_chunks": 25,
  "metrics": {
    "avg_chunk_size": 480,
    "size_variance": 0.15,
    "quality_score": 0.88
  },
  "processing_time_ms": 245,
  "cached": false,
  "expires_at": "2024-01-15T10:15:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid configuration or missing required fields
- `413 Payload Too Large`: Content exceeds 10MB limit
- `429 Too Many Requests`: Rate limit exceeded
- `504 Gateway Timeout`: Processing timeout

#### Compare Strategies

```http
POST /api/v2/chunking/compare
```

Compare multiple chunking strategies on the same content for side-by-side analysis.

**Rate Limit:** 5 requests per minute per user

**Request Body:**
```json
{
  "document_id": "doc_123",  // Optional if content provided
  "content": "Document text...",  // Optional if document_id provided
  "strategies": ["semantic", "fixed_size", "recursive"],
  "configs": {
    "semantic": {
      "chunk_size": 512,
      "chunk_overlap": 50,
      "preserve_sentences": true
    }
  },
  "max_chunks_per_strategy": 5
}
```

**Response (200):**
```json
{
  "comparison_id": "comp_xyz789",
  "comparisons": [
    {
      "strategy": "semantic",
      "config": {...},
      "sample_chunks": [...],
      "total_chunks": 25,
      "avg_chunk_size": 480,
      "size_variance": 0.15,
      "quality_score": 0.88,
      "processing_time_ms": 245,
      "pros": ["Preserves semantic coherence"],
      "cons": ["Slower processing"]
    }
  ],
  "recommendation": {
    "recommended_strategy": "semantic",
    "confidence": 0.85,
    "reasoning": "Highest quality score for your content type",
    "alternative_strategies": ["recursive"],
    "suggested_config": {...}
  },
  "processing_time_ms": 850
}
```

#### Get Cached Preview

```http
GET /api/v2/chunking/preview/{preview_id}
```

Retrieve cached preview results by ID.

**Path Parameters:**
- `preview_id`: Preview identifier returned from preview generation

**Response (200):** Same structure as preview generation response

**Error Responses:**
- `404 Not Found`: Preview not found or expired

#### Clear Preview Cache

```http
DELETE /api/v2/chunking/preview/{preview_id}
```

Clear cached preview results.

**Response (204):** No content

### Collection Processing

#### Start Chunking Operation

```http
POST /api/v2/chunking/collections/{collection_id}/chunk
```

Start an asynchronous chunking operation on a collection. Returns immediately with operation ID for tracking.

**Request Body:**
```json
{
  "strategy": "semantic",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "preserve_sentences": true
  },
  "document_ids": ["doc_1", "doc_2"],  // Optional, processes all if not specified
  "priority": 5,  // 1-10, higher is more priority
  "notify_on_completion": true
}
```

**Response (202 Accepted):**
```json
{
  "operation_id": "op_abc123",
  "collection_id": "coll_xyz789",
  "status": "pending",
  "strategy": "semantic",
  "estimated_time_seconds": 300,
  "queued_position": 2,
  "websocket_channel": "chunking:coll_xyz789:op_abc123"
}
```

**WebSocket Events:** Progress updates are sent to the returned `websocket_channel`

#### Update Collection Chunking Strategy

```http
PATCH /api/v2/chunking/collections/{collection_id}/chunking-strategy
```

Update the default chunking strategy for a collection.

**Request Body:**
```json
{
  "strategy": "semantic",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "preserve_sentences": true
  },
  "reprocess_existing": true  // Rechunk existing documents
}
```

**Response (200/202):**
- If `reprocess_existing` is false: Returns 200 with immediate update
- If `reprocess_existing` is true: Returns 202 with operation details

#### Get Collection Chunks

```http
GET /api/v2/chunking/collections/{collection_id}/chunks
```

Retrieve paginated list of chunks for a collection.

**Query Parameters:**
- `page` (optional): Page number (default: 1)
- `page_size` (optional): Items per page (1-100, default: 20)
- `document_id` (optional): Filter by document

**Response (200):**
```json
{
  "chunks": [
    {
      "id": "chunk_123",
      "document_id": "doc_456",
      "content": "Chunk content...",
      "position": 0,
      "token_count": 125,
      "metadata": {}
    }
  ],
  "total": 500,
  "page": 1,
  "page_size": 20,
  "has_next": true
}
```

#### Get Chunking Statistics

```http
GET /api/v2/chunking/collections/{collection_id}/chunking-stats
```

Get detailed statistics about chunking for a collection.

**Response (200):**
```json
{
  "total_chunks": 1250,
  "total_documents": 50,
  "avg_chunk_size": 485,
  "min_chunk_size": 150,
  "max_chunk_size": 512,
  "size_variance": 0.12,
  "strategy_used": "semantic",
  "last_updated": "2024-01-15T10:00:00Z",
  "processing_time_seconds": 45.5,
  "quality_metrics": {
    "overall_quality": 0.88,
    "coherence_score": 0.92,
    "completeness_score": 0.85
  }
}
```

### Analytics Endpoints

#### Get Global Metrics

```http
GET /api/v2/chunking/metrics
```

Retrieve global chunking metrics across all collections.

**Query Parameters:**
- `period_days` (optional): Analysis period in days (1-365, default: 30)

**Response (200):**
```json
{
  "total_collections_processed": 125,
  "total_chunks_created": 45000,
  "total_documents_processed": 1500,
  "avg_chunks_per_document": 30.0,
  "most_used_strategy": "semantic",
  "avg_processing_time": 25.5,
  "success_rate": 0.98,
  "period_start": "2024-01-01T00:00:00Z",
  "period_end": "2024-01-31T23:59:59Z"
}
```

#### Get Metrics by Strategy

```http
GET /api/v2/chunking/metrics/by-strategy
```

Get performance metrics grouped by chunking strategy.

**Query Parameters:**
- `period_days` (optional): Analysis period in days (1-365, default: 30)

**Response (200):**
```json
[
  {
    "strategy": "semantic",
    "usage_count": 450,
    "avg_chunk_size": 485,
    "avg_processing_time": 2.5,
    "success_rate": 0.97,
    "avg_quality_score": 0.88,
    "best_for_types": ["pdf", "docx", "md"]
  }
]
```

#### Analyze Document

```http
POST /api/v2/chunking/analyze
```

Analyze a document to recommend optimal chunking strategy.

**Request Body:**
```json
{
  "document_id": "doc_123",  // Optional if content provided
  "content": "Document text...",  // Optional if document_id provided
  "file_type": "pdf",
  "deep_analysis": true
}
```

**Response (200):**
```json
{
  "document_type": "pdf",
  "content_structure": {
    "sections": 5,
    "paragraphs": 20,
    "sentences": 100,
    "words": 1500
  },
  "recommended_strategy": {
    "recommended_strategy": "semantic",
    "confidence": 0.92,
    "reasoning": "Document has clear semantic boundaries",
    "alternative_strategies": ["document_structure"],
    "suggested_config": {...}
  },
  "estimated_chunks": {
    "fixed_size": 10,
    "semantic": 8,
    "recursive": 12
  },
  "complexity_score": 0.65,
  "special_considerations": [
    "Document contains tables",
    "Mixed language content detected"
  ]
}
```

#### Get Quality Analysis

```http
GET /api/v2/chunking/quality-scores
```

Analyze chunk quality across collections.

**Query Parameters:**
- `collection_id` (optional): Specific collection to analyze

**Response (200):**
```json
{
  "overall_quality": "good",
  "quality_score": 0.78,
  "coherence_score": 0.82,
  "completeness_score": 0.75,
  "size_consistency": 0.77,
  "recommendations": [
    "Consider using semantic chunking for better coherence",
    "Adjust chunk size for more consistent results"
  ],
  "issues_detected": [
    "Some chunks are too small",
    "Overlapping content detected"
  ]
}
```

### Configuration Management

#### Save Configuration

```http
POST /api/v2/chunking/configs
```

Save a custom chunking configuration for reuse.

**Request Body:**
```json
{
  "name": "My Custom Config",
  "description": "Optimized for technical documentation",
  "strategy": "semantic",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "preserve_sentences": true
  },
  "is_default": false,
  "tags": ["technical", "documentation"]
}
```

**Response (201):**
```json
{
  "id": "conf_abc123",
  "name": "My Custom Config",
  "description": "Optimized for technical documentation",
  "strategy": "semantic",
  "config": {...},
  "created_by": 1,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-15T10:00:00Z",
  "usage_count": 0,
  "is_default": false,
  "tags": ["technical", "documentation"]
}
```

#### List Configurations

```http
GET /api/v2/chunking/configs
```

List saved chunking configurations.

**Query Parameters:**
- `strategy` (optional): Filter by strategy type
- `is_default` (optional): Filter default configurations

**Response (200):**
```json
[
  {
    "id": "conf_abc123",
    "name": "My Custom Config",
    "description": "Optimized for technical documentation",
    "strategy": "semantic",
    "config": {...},
    "created_by": 1,
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": "2024-01-15T10:00:00Z",
    "usage_count": 5,
    "is_default": false,
    "tags": ["technical", "documentation"]
  }
]
```

### Progress Tracking

#### Get Operation Progress

```http
GET /api/v2/chunking/operations/{operation_id}/progress
```

Get current progress of a chunking operation.

**Response (200):**
```json
{
  "operation_id": "op_abc123",
  "status": "in_progress",
  "progress_percentage": 45.5,
  "documents_processed": 23,
  "total_documents": 50,
  "chunks_created": 690,
  "current_document": "document_24.pdf",
  "estimated_time_remaining": 120,
  "errors": []
}
```

## WebSocket Usage

### Connecting to Chunking Progress Channel

The chunking API provides real-time progress updates via WebSocket connections. When starting a chunking operation, the response includes a `websocket_channel` that clients can connect to for live updates.

#### Connection Example

```javascript
// Connect to the WebSocket channel returned from chunking operation
const wsUrl = `ws://localhost:8080/ws/channel/${websocketChannel}?token=${jwtToken}`;
const ws = new WebSocket(wsUrl);

ws.onopen = () => {
  console.log('Connected to chunking progress channel');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  handleChunkingUpdate(message);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
};
```

#### Message Types

##### Chunking Started
```json
{
  "type": "chunking_started",
  "operation_id": "op_abc123",
  "collection_id": "coll_xyz789",
  "strategy": "semantic",
  "timestamp": "2024-01-15T10:00:00Z"
}
```

##### Progress Update
```json
{
  "type": "chunking_progress",
  "data": {
    "progress_percentage": 45.5,
    "documents_processed": 23,
    "total_documents": 50,
    "chunks_created": 690,
    "current_document": "document_24.pdf"
  }
}
```

##### Document Processing
```json
{
  "type": "chunking_document_start",
  "data": {
    "document_id": "doc_123",
    "document_name": "technical_guide.pdf",
    "document_size": 1024000
  }
}
```

```json
{
  "type": "chunking_document_complete",
  "data": {
    "document_id": "doc_123",
    "chunks_created": 25,
    "processing_time_ms": 1250
  }
}
```

##### Operation Complete
```json
{
  "type": "chunking_completed",
  "data": {
    "operation_id": "op_abc123",
    "total_documents": 50,
    "total_chunks": 1500,
    "processing_time_seconds": 125.5,
    "success_rate": 0.98
  }
}
```

##### Error Notification
```json
{
  "type": "chunking_failed",
  "data": {
    "operation_id": "op_abc123",
    "error": "Processing failed",
    "error_code": "CHUNK_PROCESSING_ERROR",
    "document_id": "doc_456",
    "timestamp": "2024-01-15T10:05:00Z"
  }
}
```

### WebSocket Throttling

Progress updates are throttled to prevent overwhelming clients:
- Minimum 500ms between progress updates per operation
- Document-level events are not throttled
- Error and completion events are always sent immediately

## Error Handling

### Error Response Format

All error responses follow the standard Semantik error format:

```json
{
  "error": "Error message",
  "error_type": "ERROR_TYPE_CODE",
  "correlation_id": "corr_abc123",
  "details": {
    "field": "Additional context"
  },
  "suggestions": [
    "Try reducing chunk size",
    "Use a different strategy"
  ],
  "timestamp": "2024-01-15T10:00:00Z"
}
```

### Common Error Types

#### ChunkingValidationError (400)
Occurs when request parameters are invalid:
```json
{
  "error": "Invalid configuration: chunk_overlap must be less than chunk_size",
  "error_type": "CHUNKING_VALIDATION_ERROR",
  "correlation_id": "corr_abc123",
  "details": {
    "chunk_size": 100,
    "chunk_overlap": 150
  },
  "suggestions": ["Reduce chunk_overlap to less than 100"]
}
```

#### ChunkingMemoryError (413)
Occurs when content exceeds size limits:
```json
{
  "error": "Content too large for preview (max 10MB)",
  "error_type": "CHUNKING_MEMORY_ERROR",
  "correlation_id": "corr_abc123"
}
```

#### ChunkingTimeoutError (504)
Occurs when processing exceeds time limits:
```json
{
  "error": "Chunking operation timed out after 30 seconds",
  "error_type": "CHUNKING_TIMEOUT_ERROR",
  "correlation_id": "corr_abc123"
}
```

#### ChunkingConfigurationError (400)
Occurs when strategy configuration is invalid:
```json
{
  "error": "Invalid configuration for semantic strategy",
  "error_type": "CHUNKING_CONFIGURATION_ERROR",
  "correlation_id": "corr_abc123",
  "details": {
    "missing_field": "embedding_model"
  }
}
```

## Security Considerations

### Authentication and Authorization
- All endpoints require valid JWT authentication
- Users can only access their own collections and configurations
- Operation tracking is user-scoped

### Input Validation
- Content size limited to 10MB for preview operations
- Null bytes and potentially malicious patterns are rejected
- File type validation for strategy recommendations

### Rate Limiting
Rate limits are enforced per user to prevent abuse:

| Endpoint | Limit | Window |
|----------|-------|--------|
| Preview Generation | 10 requests | 1 minute |
| Strategy Comparison | 5 requests | 1 minute |
| Collection Chunking | 5 operations | 5 minutes |
| Configuration Save | 10 requests | 1 minute |

### Data Privacy
- Chunking previews are cached per-user and expire after 15 minutes
- WebSocket channels are user and operation specific
- No cross-user data leakage in multi-tenant environments

## Performance Guidelines

### Optimal Configuration

#### Chunk Size Selection
- **Small chunks (100-300 tokens)**: Better for Q&A systems, higher precision
- **Medium chunks (300-600 tokens)**: Balanced for most use cases
- **Large chunks (600-1000 tokens)**: Better for summarization, maintaining context

#### Overlap Configuration
- **No overlap**: Fastest processing, risk of losing context at boundaries
- **Small overlap (10-20%)**: Good balance of performance and context preservation
- **Large overlap (30-50%)**: Better context but increased storage and processing

### Strategy Selection Guide

| Content Type | Recommended Strategy | Chunk Size | Overlap |
|-------------|---------------------|------------|---------|
| Technical Docs | Semantic | 512 | 50 |
| Code Files | Recursive | 400 | 100 |
| PDFs | Document Structure | 600 | 75 |
| Logs | Fixed Size | 300 | 0 |
| Mixed Content | Hybrid | 500 | 50 |

### Performance Optimization

1. **Batch Processing**: Process multiple documents in a single operation
2. **Priority Queuing**: Use priority parameter for time-sensitive operations
3. **Preview Caching**: Leverage 15-minute cache for iterative testing
4. **WebSocket Efficiency**: Use throttled progress updates for large operations

## Best Practices

### Strategy Testing
1. Always preview with a representative sample before full processing
2. Compare multiple strategies using the comparison endpoint
3. Monitor quality metrics to ensure optimal results

### Configuration Management
1. Save successful configurations for reuse
2. Use tags to organize configurations by use case
3. Set organization-wide defaults for consistency

### Error Recovery
1. Implement exponential backoff for rate limit errors
2. Use correlation IDs for debugging failed operations
3. Monitor WebSocket disconnections and implement reconnection logic

### Monitoring
1. Track operation success rates per strategy
2. Monitor average processing times for capacity planning
3. Analyze quality metrics to identify improvement opportunities

## Migration Guide

For users migrating from the legacy chunking system:

### API Changes
- Endpoint path changed from `/api/chunks` to `/api/v2/chunking`
- Strategy names standardized (e.g., "fixed" → "fixed_size")
- Configuration structure unified across all strategies

### Feature Enhancements
- Real-time progress via WebSocket (previously polling-based)
- Strategy comparison capability (new feature)
- Quality metrics and analysis (new feature)
- Configuration management system (new feature)

### Deprecated Features
- Legacy `/api/chunks/process` endpoint (use collection chunking)
- Synchronous chunking (all operations now async)
- Strategy-specific endpoints (unified under strategy pattern)
