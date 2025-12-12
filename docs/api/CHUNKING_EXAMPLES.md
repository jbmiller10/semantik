# Chunking API Examples

Practical examples for curl, Python, JavaScript/TypeScript, WebSocket.

## Auth

```bash
curl -X POST /api/auth/login -H "Content-Type: application/json" \
  -d '{"username": "...", "password": "..."}'
# Returns access_token
```

## Quick Reference

### Available Chunking Strategies

| Strategy | Value | Description |
|----------|-------|-------------|
| Fixed Size | `fixed_size` | Splits content into fixed-size chunks |
| Semantic | `semantic` | Uses semantic similarity to group related content |
| Recursive | `recursive` | Recursively splits using hierarchical separators |
| Markdown | `markdown` | Optimized for markdown document structure |
| Hierarchical | `hierarchical` | Preserves document hierarchy (headings, sections) |
| Hybrid | `hybrid` | Combines multiple strategies |
| Sliding Window | `sliding_window` | Legacy - uses overlapping windows |
| Document Structure | `document_structure` | Legacy - structure-aware chunking |

### Operation Status Values

| Status | Description |
|--------|-------------|
| `pending` | Operation queued, not yet started |
| `in_progress` | Operation is currently running |
| `completed` | Operation finished successfully |
| `failed` | Operation encountered an error |
| `cancelled` | Operation was cancelled by user |
| `partial` | Operation completed with some errors |

### ChunkingOperationResponse Schema

When starting a chunking operation, the API returns:

```json
{
  "operation_id": "uuid-string",
  "collection_id": "collection-uuid",
  "status": "pending",
  "strategy": "semantic",
  "estimated_time_seconds": 120,
  "queued_position": 1,
  "websocket_channel": "chunking:collection-uuid:operation-uuid"
}
```

**Important:** When connecting to WebSocket for progress updates, use the `operation_id` field directly in the URL path, NOT the `websocket_channel` field:

```
ws://localhost:8080/ws/operations/{operation_id}?token={jwt_token}
```

### WebSocket Message Structure

Messages received from the WebSocket have this structure:

```json
{
  "type": "operation_progress",
  "data": {
    "status": "in_progress",
    "progress": 45.5,
    "message": "Processing document 5 of 10",
    "documents_processed": 5,
    "total_documents": 10,
    "chunks_created": 150
  }
}
```

**Message Types:**
- `operation_progress` - Progress update
- `operation_completed` - Operation finished successfully
- `operation_failed` - Operation encountered an error

**Data Fields:**
- `status` - Current operation status (see status values above)
- `progress` - Percentage complete (0-100)
- `message` - Human-readable status message
- `error_message` - Error details (when status is `failed`)
- `documents_processed` - Number of documents processed
- `total_documents` - Total documents to process
- `chunks_created` - Number of chunks created so far

## Common Use Cases

### Use Case 1: Finding the Best Strategy for Your Documents

**Scenario:** You have a collection of technical documentation in various formats and want to determine the optimal chunking strategy.

```python
import requests
import json

# Configuration
base_url = "http://localhost:8080/api/v2/chunking"
token = "your_jwt_token"
headers = {"Authorization": f"Bearer {token}"}

# Step 1: Get strategy recommendation based on file types
file_types = ["pdf", "md", "txt"]
response = requests.post(
    f"{base_url}/strategies/recommend",
    headers=headers,
    params={"file_types": file_types}
)
recommendation = response.json()
print(f"Recommended strategy: {recommendation['recommended_strategy']}")
print(f"Reasoning: {recommendation['reasoning']}")

# Step 2: Test the recommendation with a sample document
sample_content = """
# Technical Documentation

## Introduction
This document describes the architecture of our system...

## Components
The system consists of three main components...
"""

preview_response = requests.post(
    f"{base_url}/preview",
    headers=headers,
    json={
        "content": sample_content,
        "strategy": recommendation["recommended_strategy"],
        "config": recommendation["suggested_config"],
        "max_chunks": 5,
        "include_metrics": True
    }
)
preview = preview_response.json()
print(f"Preview generated {preview['total_chunks']} chunks")
print(f"Quality score: {preview['metrics']['quality_score']}")
```

### Use Case 2: Comparing Multiple Strategies

**Scenario:** You want to compare how different strategies handle your specific content.

```python
# Compare three strategies on the same content
comparison_response = requests.post(
    f"{base_url}/compare",
    headers=headers,
    json={
        "content": sample_content,
        "strategies": ["semantic", "fixed_size", "recursive"],
        "max_chunks_per_strategy": 3
    }
)

comparison = comparison_response.json()
for comp in comparison["comparisons"]:
    print(f"\nStrategy: {comp['strategy']}")
    print(f"  Total chunks: {comp['total_chunks']}")
    print(f"  Avg chunk size: {comp['avg_chunk_size']}")
    print(f"  Quality score: {comp['quality_score']}")
    print(f"  Processing time: {comp['processing_time_ms']}ms")

print(f"\nBest strategy: {comparison['recommendation']['recommended_strategy']}")
```

### Use Case 3: Processing an Entire Collection

**Scenario:** You need to chunk all documents in a collection with real-time progress tracking.

```python
import asyncio
import websockets
import json

async def process_collection_with_progress(collection_id, strategy, config):
    # Start the chunking operation
    response = requests.post(
        f"{base_url}/collections/{collection_id}/chunk",
        headers=headers,
        json={
            "strategy": strategy,
            "config": config,
            "priority": 7,
            "notify_on_completion": True
        }
    )
    
    operation = response.json()
    operation_id = operation["operation_id"]
    print(f"Operation started: {operation_id}")
    
    # Connect to operation WebSocket for progress updates
    ws_url = f"ws://localhost:8080/ws/operations/{operation_id}?token={token}"
    
    async with websockets.connect(ws_url) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            progress = data.get("data", {}).get("progress")
            if progress is not None:
                print(f"Progress: {progress:.1f}%")

            status = data.get("data", {}).get("status")
            if data.get("type") == "operation_completed" or status == "completed":
                print("Chunking completed!")
                break
            if data.get("type") == "operation_failed" or status == "failed":
                print(f"Chunking failed: {data.get('data', {}).get('error_message')}")
                break

# Run the async function
asyncio.run(process_collection_with_progress(
    "coll_123",
    "semantic",
    {"chunk_size": 512, "chunk_overlap": 50, "preserve_sentences": True}
))
```

## curl Examples

### List Available Strategies

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/strategies" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Generate Preview

```bash
curl -X POST "http://localhost:8080/api/v2/chunking/preview" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a sample document that needs to be chunked for semantic search...",
    "strategy": "semantic",
    "config": {
      "chunk_size": 512,
      "chunk_overlap": 50,
      "preserve_sentences": true
    },
    "max_chunks": 10,
    "include_metrics": true
  }'
```

### Start Collection Chunking

```bash
curl -X POST "http://localhost:8080/api/v2/chunking/collections/${COLLECTION_ID}/chunk" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "semantic",
    "config": {
      "chunk_size": 512,
      "chunk_overlap": 50,
      "preserve_sentences": true
    },
    "priority": 5
  }'
```

### Get Operation Progress

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/operations/${OPERATION_ID}/progress" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Compare Strategies

```bash
curl -X POST "http://localhost:8080/api/v2/chunking/compare" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Sample document content for comparison...",
    "strategies": ["semantic", "fixed_size", "recursive"],
    "max_chunks_per_strategy": 5
  }'
```

### Save Custom Configuration

```bash
curl -X POST "http://localhost:8080/api/v2/chunking/configs" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Technical Docs Config",
    "description": "Optimized for technical documentation",
    "strategy": "semantic",
    "config": {
      "chunk_size": 600,
      "chunk_overlap": 75,
      "preserve_sentences": true
    },
    "tags": ["technical", "documentation"]
  }'
```

### List Saved Configurations

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/configs" \
  -H "Authorization: Bearer ${TOKEN}"

# Filter by strategy
curl -X GET "http://localhost:8080/api/v2/chunking/configs?strategy=semantic" \
  -H "Authorization: Bearer ${TOKEN}"

# Filter for default configs only
curl -X GET "http://localhost:8080/api/v2/chunking/configs?is_default=true" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Get Strategy Details

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/strategies/semantic" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Get Cached Preview

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/preview/${PREVIEW_ID}" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Clear Preview Cache

```bash
curl -X DELETE "http://localhost:8080/api/v2/chunking/preview/${PREVIEW_ID}" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Update Collection Chunking Strategy

```bash
curl -X PATCH "http://localhost:8080/api/v2/chunking/collections/${COLLECTION_ID}/chunking-strategy" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "hierarchical",
    "config": {
      "chunk_size": 512,
      "chunk_overlap": 50,
      "preserve_sentences": true
    },
    "reprocess_existing": true
  }'
```

### Get Collection Chunks (Paginated)

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/collections/${COLLECTION_ID}/chunks?page=1&page_size=20" \
  -H "Authorization: Bearer ${TOKEN}"

# Filter by document
curl -X GET "http://localhost:8080/api/v2/chunking/collections/${COLLECTION_ID}/chunks?document_id=${DOC_ID}" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Get Global Metrics

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/metrics?period_days=30" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Get Metrics by Strategy

```bash
curl -X GET "http://localhost:8080/api/v2/chunking/metrics/by-strategy?period_days=30" \
  -H "Authorization: Bearer ${TOKEN}"
```

### Get Quality Scores

```bash
# Global quality scores
curl -X GET "http://localhost:8080/api/v2/chunking/quality-scores" \
  -H "Authorization: Bearer ${TOKEN}"

# Quality scores for specific collection
curl -X GET "http://localhost:8080/api/v2/chunking/quality-scores?collection_id=${COLLECTION_ID}" \
  -H "Authorization: Bearer ${TOKEN}"
```

## HTTPie Examples

HTTPie provides a more user-friendly command-line interface:

### List Strategies

```bash
http GET localhost:8080/api/v2/chunking/strategies \
  "Authorization: Bearer ${TOKEN}"
```

### Generate Preview

```bash
http POST localhost:8080/api/v2/chunking/preview \
  "Authorization: Bearer ${TOKEN}" \
  content="Document content..." \
  strategy=semantic \
  config:='{"chunk_size": 512, "chunk_overlap": 50}' \
  max_chunks=10 \
  include_metrics=true
```

### Analyze Document

```bash
http POST localhost:8080/api/v2/chunking/analyze \
  "Authorization: Bearer ${TOKEN}" \
  content="Document to analyze..." \
  file_type=pdf \
  deep_analysis=true
```

## Python Examples

### Complete Python Client Class

```python
import requests
import asyncio
import websockets
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"
    # Legacy/aliases (backward compatibility)
    SLIDING_WINDOW = "sliding_window"
    DOCUMENT_STRUCTURE = "document_structure"

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_sentences: bool = True
    metadata: Optional[Dict] = None

class ChunkingAPIClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = f"{base_url}/api/v2/chunking"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def list_strategies(self) -> List[Dict]:
        """List all available chunking strategies."""
        response = self.session.get(f"{self.base_url}/strategies")
        response.raise_for_status()
        return response.json()
    
    def recommend_strategy(self, file_types: List[str]) -> Dict:
        """Get strategy recommendation based on file types."""
        response = self.session.post(
            f"{self.base_url}/strategies/recommend",
            params={"file_types": file_types}
        )
        response.raise_for_status()
        return response.json()
    
    def generate_preview(
        self,
        content: str = None,
        document_id: str = None,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
        config: ChunkingConfig = None,
        max_chunks: int = 10
    ) -> Dict:
        """Generate a preview of chunking results."""
        if not content and not document_id:
            raise ValueError("Either content or document_id must be provided")
        
        payload = {
            "strategy": strategy.value,
            "max_chunks": max_chunks,
            "include_metrics": True
        }
        
        if content:
            payload["content"] = content
        if document_id:
            payload["document_id"] = document_id
        if config:
            payload["config"] = {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "preserve_sentences": config.preserve_sentences
            }
        
        response = self.session.post(f"{self.base_url}/preview", json=payload)
        response.raise_for_status()
        return response.json()
    
    def compare_strategies(
        self,
        content: str,
        strategies: List[ChunkingStrategy],
        configs: Optional[Dict[str, ChunkingConfig]] = None
    ) -> Dict:
        """Compare multiple chunking strategies."""
        payload = {
            "content": content,
            "strategies": [s.value for s in strategies],
            "max_chunks_per_strategy": 5
        }
        
        if configs:
            payload["configs"] = {
                strategy: {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "preserve_sentences": config.preserve_sentences
                }
                for strategy, config in configs.items()
            }
        
        response = self.session.post(f"{self.base_url}/compare", json=payload)
        response.raise_for_status()
        return response.json()
    
    def start_collection_chunking(
        self,
        collection_id: str,
        strategy: ChunkingStrategy,
        config: ChunkingConfig = None,
        document_ids: Optional[List[str]] = None,
        priority: int = 5
    ) -> Dict:
        """Start chunking operation on a collection."""
        payload = {
            "strategy": strategy.value,
            "priority": priority,
            "notify_on_completion": True
        }
        
        if config:
            payload["config"] = {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "preserve_sentences": config.preserve_sentences
            }
        
        if document_ids:
            payload["document_ids"] = document_ids
        
        response = self.session.post(
            f"{self.base_url}/collections/{collection_id}/chunk",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_operation_progress(self, operation_id: str) -> Dict:
        """Get current progress of a chunking operation."""
        response = self.session.get(
            f"{self.base_url}/operations/{operation_id}/progress"
        )
        response.raise_for_status()
        return response.json()
    
    def get_collection_stats(self, collection_id: str) -> Dict:
        """Get chunking statistics for a collection."""
        response = self.session.get(
            f"{self.base_url}/collections/{collection_id}/chunking-stats"
        )
        response.raise_for_status()
        return response.json()
    
    async def monitor_operation(self, operation_id: str):
        """Monitor chunking operation progress via WebSocket."""
        token = self.headers["Authorization"].replace("Bearer ", "")
        ws_url = f"ws://localhost:8080/ws/operations/{operation_id}?token={token}"
        
        async with websockets.connect(ws_url) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                print(f"[{data.get('type')}] ", end="")

                progress = data.get("data", {}).get("progress")
                message = data.get("data", {}).get("message", "")
                status = data.get("data", {}).get("status")

                if progress is not None:
                    print(f"{progress:.1f}% - {message}")

                if data.get("type") in ("operation_completed", "operation_failed") or status in ("completed", "failed"):
                    print(status or data.get("type"))
                    break

# Example usage
def main():
    client = ChunkingAPIClient("http://localhost:8080", "your_token")
    
    # List strategies
    strategies = client.list_strategies()
    for strategy in strategies:
        print(f"- {strategy['name']}: {strategy['description']}")
    
    # Get recommendation
    recommendation = client.recommend_strategy(["pdf", "md"])
    print(f"Recommended: {recommendation['recommended_strategy']}")
    
    # Generate preview
    preview = client.generate_preview(
        content="Your document content here...",
        strategy=ChunkingStrategy.SEMANTIC,
        config=ChunkingConfig(chunk_size=512, chunk_overlap=50)
    )
    print(f"Preview generated {preview['total_chunks']} chunks")
    
    # Compare strategies
    comparison = client.compare_strategies(
        content="Document to compare...",
        strategies=[ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE]
    )
    print(f"Best strategy: {comparison['recommendation']['recommended_strategy']}")

if __name__ == "__main__":
    main()
```

## JavaScript/TypeScript Examples

### TypeScript Client Implementation

```typescript
// types.ts
export enum ChunkingStrategy {
  FIXED_SIZE = 'fixed_size',
  SEMANTIC = 'semantic',
  RECURSIVE = 'recursive',
  MARKDOWN = 'markdown',
  HIERARCHICAL = 'hierarchical',
  HYBRID = 'hybrid',
  // Legacy/aliases (backward compatibility)
  SLIDING_WINDOW = 'sliding_window',
  DOCUMENT_STRUCTURE = 'document_structure'
}

export interface ChunkingConfig {
  chunk_size: number;
  chunk_overlap: number;
  preserve_sentences: boolean;
  metadata?: Record<string, any>;
}

export interface PreviewRequest {
  document_id?: string;
  content?: string;
  strategy: ChunkingStrategy;
  config?: ChunkingConfig;
  max_chunks?: number;
  include_metrics?: boolean;
}

export interface ChunkPreview {
  index: number;
  content: string;
  token_count: number;
  char_count: number;
  metadata: Record<string, any>;
  quality_score: number;
}

export interface PreviewResponse {
  preview_id: string;
  strategy: ChunkingStrategy;
  config: ChunkingConfig;
  chunks: ChunkPreview[];
  total_chunks: number;
  metrics?: Record<string, any>;
  processing_time_ms: number;
  cached: boolean;
  expires_at: string;
}

// chunking-client.ts
export class ChunkingAPIClient {
  private baseUrl: string;
  private headers: HeadersInit;

  constructor(baseUrl: string, token: string) {
    this.baseUrl = `${baseUrl}/api/v2/chunking`;
    this.headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }

  async listStrategies(): Promise<any[]> {
    const response = await fetch(`${this.baseUrl}/strategies`, {
      headers: this.headers
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  }

  async generatePreview(request: PreviewRequest): Promise<PreviewResponse> {
    const response = await fetch(`${this.baseUrl}/preview`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(request)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `API error: ${response.status}`);
    }
    
    return response.json();
  }

  async compareStrategies(
    content: string,
    strategies: ChunkingStrategy[]
  ): Promise<any> {
    const response = await fetch(`${this.baseUrl}/compare`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        content,
        strategies,
        max_chunks_per_strategy: 5
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  }

  async startCollectionChunking(
    collectionId: string,
    strategy: ChunkingStrategy,
    config?: ChunkingConfig
  ): Promise<any> {
    const response = await fetch(
      `${this.baseUrl}/collections/${collectionId}/chunk`,
      {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify({
          strategy,
          config,
          priority: 5,
          notify_on_completion: true
        })
      }
    );
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return response.json();
  }

  connectToProgress(
    operationId: string,
    token: string,
    onMessage: (data: any) => void
  ): WebSocket {
    const wsUrl = `ws://localhost:8080/ws/operations/${operationId}?token=${token}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    return ws;
  }
}

// usage.ts
async function demonstrateChunkingAPI() {
  const client = new ChunkingAPIClient('http://localhost:8080', 'your_token');
  
  try {
    // List strategies
    const strategies = await client.listStrategies();
    console.log('Available strategies:', strategies);
    
    // Generate preview
    const preview = await client.generatePreview({
      content: 'Sample document content...',
      strategy: ChunkingStrategy.SEMANTIC,
      config: {
        chunk_size: 512,
        chunk_overlap: 50,
        preserve_sentences: true
      },
      max_chunks: 10,
      include_metrics: true
    });
    
    console.log(`Generated ${preview.total_chunks} chunks`);
    console.log('First chunk:', preview.chunks[0]);
    
    // Compare strategies
    const comparison = await client.compareStrategies(
      'Document to analyze...',
      [ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE]
    );
    
    console.log('Comparison results:', comparison);
    
    // Start collection chunking with progress monitoring
    const operation = await client.startCollectionChunking(
      'coll_123',
      ChunkingStrategy.SEMANTIC,
      {
        chunk_size: 512,
        chunk_overlap: 50,
        preserve_sentences: true
      }
    );
    
    console.log('Operation started:', operation.operation_id);
    
    // Connect to WebSocket for progress using operation_id (NOT websocket_channel)
    // The WebSocket URL path uses operation_id directly: /ws/operations/{operation_id}
    const ws = client.connectToProgress(
      operation.operation_id,
      'your_token',
      (data) => {
        // Message structure: { type: string, data: { status, progress, message, ... } }
        const status = data.data?.status;
        const progress = data.data?.progress;

        if (progress != null) {
          console.log(`Progress: ${progress.toFixed(1)}%`);
        }

        if (data.type === 'operation_completed' || status === 'completed') {
          console.log('Chunking completed!');
          ws.close();
        }

        if (data.type === 'operation_failed' || status === 'failed') {
          console.error('Chunking failed:', data.data?.error_message);
          ws.close();
        }
      }
    );
    
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the demonstration
demonstrateChunkingAPI();
```

### React Hook Example

```typescript
import { useState, useEffect } from 'react';

interface UseChunkingProgressOptions {
  operationId: string;  // Use operation_id from the response, NOT websocket_channel
  token: string;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

// Status values from ChunkingStatus enum
type OperationStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled' | 'partial';

export function useChunkingProgress({ operationId, token, onComplete, onError }: UseChunkingProgressOptions) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<OperationStatus>('pending');

  useEffect(() => {
    // WebSocket URL uses operation_id directly in the path
    const wsUrl = `ws://localhost:8080/ws/operations/${operationId}?token=${token}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      // Message structure: { type: string, data: { status, progress, message, error_message, ... } }
      const msgStatus = msg.data?.status;
      const msgProgress = msg.data?.progress;

      if (msgProgress != null) setProgress(msgProgress);
      if (msgStatus) setStatus(msgStatus);

      if (msg.type === 'operation_completed' || msgStatus === 'completed') {
        setStatus('completed');
        setProgress(100);
        onComplete?.();
      }

      if (msg.type === 'operation_failed' || msgStatus === 'failed') {
        setStatus('failed');
        onError?.(msg.data?.error_message || 'Operation failed');
      }
    };

    ws.onerror = () => {
      setStatus('failed');
      onError?.('WebSocket connection error');
    };

    return () => ws.close();
  }, [operationId, token, onComplete, onError]);

  return { progress, status };
}

function ChunkingProgressDisplay({ operationId, token }: any) {
  const { progress, status } = useChunkingProgress({
    operationId,
    token,
    onComplete: () => console.log('Chunking completed!'),
    onError: (error) => console.error('Chunking error:', error),
  });

  return (
    <div>
      <h3>Chunking Progress</h3>
      <div>Status: {status}</div>
      <div>Progress: {progress.toFixed(1)}%</div>
      <progress value={progress} max={100} />
    </div>
  );
}
```

## WebSocket Examples

### Node.js WebSocket Client

```javascript
const WebSocket = require('ws');

class ChunkingProgressMonitor {
  constructor(operationId, token) {
    // IMPORTANT: Use operation_id from the API response, not websocket_channel
    // The WebSocket URL uses operation_id directly in the path
    this.operationId = operationId;
    this.token = token;
    this.ws = null;
  }

  connect() {
    // WebSocket URL format: /ws/operations/{operation_id}?token={jwt_token}
    const wsUrl = `ws://localhost:8080/ws/operations/${this.operationId}?token=${this.token}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.on('open', () => {
      console.log('Connected to operation progress channel');
    });

    this.ws.on('message', (data) => {
      const message = JSON.parse(data);
      this.handleMessage(message);
    });

    this.ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    this.ws.on('close', () => {
      console.log('WebSocket connection closed');
    });
  }

  handleMessage(message) {
    // Message structure: { type: string, data: { status, progress, message, error_message, ... } }
    // Status values: 'pending', 'in_progress', 'completed', 'failed', 'cancelled', 'partial'
    const status = message.data?.status;
    const progress = message.data?.progress;
    const text = message.data?.message;

    if (progress != null) {
      console.log(`Progress: ${progress.toFixed(1)}% ${text || ''}`);
    }

    if (message.type === 'operation_completed' || status === 'completed') {
      console.log('Chunking completed!');
      this.close();
    }

    if (message.type === 'operation_failed' || status === 'failed') {
      console.error(`Chunking failed: ${message.data?.error_message || 'Operation failed'}`);
      this.close();
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage - pass the operation_id from the chunking operation response
const monitor = new ChunkingProgressMonitor('op_456', 'your_token');
monitor.connect();
```

## Error Handling Examples

### Python Error Handling

```python
import requests
from typing import Optional

class ChunkingAPIError(Exception):
    """Base exception for Chunking API errors."""
    def __init__(self, message: str, error_type: str = None, correlation_id: str = None):
        super().__init__(message)
        self.error_type = error_type
        self.correlation_id = correlation_id

class ChunkingValidationError(ChunkingAPIError):
    """Raised when request validation fails."""
    pass

class ChunkingMemoryError(ChunkingAPIError):
    """Raised when content exceeds size limits."""
    pass

class ChunkingTimeoutError(ChunkingAPIError):
    """Raised when operation times out."""
    pass

class RateLimitError(ChunkingAPIError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after

def handle_api_response(response: requests.Response) -> dict:
    """Handle API response and raise appropriate exceptions."""
    if response.status_code == 200 or response.status_code == 201:
        return response.json()
    
    if response.status_code == 202:
        return response.json()
    
    if response.status_code == 204:
        return None
    
    # Handle error responses
    try:
        error_data = response.json()
        error_message = error_data.get('error', 'Unknown error')
        error_type = error_data.get('error_type')
        correlation_id = error_data.get('correlation_id')
    except:
        error_message = f"API error: {response.status_code}"
        error_type = None
        correlation_id = None
    
    if response.status_code == 400:
        if error_type == 'CHUNKING_VALIDATION_ERROR':
            raise ChunkingValidationError(error_message, error_type, correlation_id)
        elif error_type == 'CHUNKING_CONFIGURATION_ERROR':
            raise ChunkingAPIError(error_message, error_type, correlation_id)
    
    elif response.status_code == 413:
        raise ChunkingMemoryError(error_message, error_type, correlation_id)
    
    elif response.status_code == 429:
        retry_after = response.headers.get('X-RateLimit-Reset-After', 60)
        raise RateLimitError(error_message, int(retry_after))
    
    elif response.status_code == 504:
        raise ChunkingTimeoutError(error_message, error_type, correlation_id)
    
    else:
        raise ChunkingAPIError(error_message, error_type, correlation_id)

# Usage with retry logic
import time
from typing import Any

def make_api_request_with_retry(
    url: str,
    method: str = 'GET',
    headers: dict = None,
    json_data: dict = None,
    max_retries: int = 3
) -> Optional[dict]:
    """Make API request with retry logic for rate limits and timeouts."""
    retries = 0
    
    while retries < max_retries:
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=json_data)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return handle_api_response(response)
            
        except RateLimitError as e:
            print(f"Rate limit exceeded. Waiting {e.retry_after} seconds...")
            time.sleep(e.retry_after)
            retries += 1
            
        except ChunkingTimeoutError as e:
            print(f"Request timed out. Retry {retries + 1}/{max_retries}")
            retries += 1
            if retries < max_retries:
                time.sleep(2 ** retries)  # Exponential backoff
                
        except ChunkingValidationError as e:
            # Don't retry validation errors
            print(f"Validation error: {e}")
            if e.correlation_id:
                print(f"Correlation ID: {e.correlation_id}")
            raise
            
        except ChunkingMemoryError as e:
            # Don't retry memory errors
            print(f"Content too large: {e}")
            raise
            
        except ChunkingAPIError as e:
            print(f"API error: {e}")
            if e.correlation_id:
                print(f"Correlation ID for debugging: {e.correlation_id}")
            raise
    
    raise ChunkingAPIError(f"Max retries ({max_retries}) exceeded")

# Example usage
try:
    result = make_api_request_with_retry(
        url="http://localhost:8080/api/v2/chunking/preview",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
        json_data={
            "content": "Document content...",
            "strategy": "semantic"
        }
    )
    print("Success:", result)
    
except ChunkingValidationError as e:
    print(f"Invalid request: {e}")
    
except ChunkingMemoryError as e:
    print(f"Content too large, consider splitting: {e}")
    
except ChunkingAPIError as e:
    print(f"API error occurred: {e}")
```

### JavaScript Error Handling

```javascript
class ChunkingAPIError extends Error {
  constructor(message, errorType = null, correlationId = null) {
    super(message);
    this.name = 'ChunkingAPIError';
    this.errorType = errorType;
    this.correlationId = correlationId;
  }
}

class ChunkingValidationError extends ChunkingAPIError {
  constructor(message, errorType, correlationId) {
    super(message, errorType, correlationId);
    this.name = 'ChunkingValidationError';
  }
}

class RateLimitError extends ChunkingAPIError {
  constructor(message, retryAfter) {
    super(message);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

async function handleAPIResponse(response) {
  if (response.ok) {
    if (response.status === 204) {
      return null;
    }
    return await response.json();
  }

  let errorData;
  try {
    errorData = await response.json();
  } catch {
    errorData = {
      error: `API error: ${response.status}`,
      error_type: 'UNKNOWN'
    };
  }

  const errorMessage = errorData.error || 'Unknown error';
  const errorType = errorData.error_type;
  const correlationId = errorData.correlation_id;

  switch (response.status) {
    case 400:
      if (errorType === 'CHUNKING_VALIDATION_ERROR') {
        throw new ChunkingValidationError(errorMessage, errorType, correlationId);
      }
      throw new ChunkingAPIError(errorMessage, errorType, correlationId);

    case 413:
      throw new ChunkingAPIError('Content too large', 'CHUNKING_MEMORY_ERROR', correlationId);

    case 429:
      const retryAfter = response.headers.get('X-RateLimit-Reset-After') || 60;
      throw new RateLimitError(errorMessage, parseInt(retryAfter));

    case 504:
      throw new ChunkingAPIError('Operation timed out', 'CHUNKING_TIMEOUT_ERROR', correlationId);

    default:
      throw new ChunkingAPIError(errorMessage, errorType, correlationId);
  }
}

// Retry logic with exponential backoff
async function makeAPIRequestWithRetry(url, options = {}, maxRetries = 3) {
  let lastError;
  
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(url, options);
      return await handleAPIResponse(response);
      
    } catch (error) {
      lastError = error;
      
      if (error instanceof RateLimitError) {
        console.log(`Rate limited. Waiting ${error.retryAfter} seconds...`);
        await new Promise(resolve => setTimeout(resolve, error.retryAfter * 1000));
        
      } else if (error instanceof ChunkingValidationError) {
        // Don't retry validation errors
        throw error;
        
      } else if (error.name === 'ChunkingAPIError' && 
                 error.errorType === 'CHUNKING_TIMEOUT_ERROR') {
        // Exponential backoff for timeouts
        const delay = Math.pow(2, attempt) * 1000;
        console.log(`Timeout. Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        
      } else {
        // Don't retry other errors
        throw error;
      }
    }
  }
  
  throw lastError;
}

// Usage example
async function exampleWithErrorHandling() {
  try {
    const result = await makeAPIRequestWithRetry(
      'http://localhost:8080/api/v2/chunking/preview',
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          content: 'Document content...',
          strategy: 'semantic'
        })
      }
    );
    
    console.log('Success:', result);
    
  } catch (error) {
    if (error instanceof ChunkingValidationError) {
      console.error('Validation failed:', error.message);
      if (error.correlationId) {
        console.error('Correlation ID:', error.correlationId);
      }
      
    } else if (error instanceof RateLimitError) {
      console.error('Rate limit exceeded. Try again later.');
      
    } else if (error instanceof ChunkingAPIError) {
      console.error('API error:', error.message);
      console.error('Type:', error.errorType);
      
    } else {
      console.error('Unexpected error:', error);
    }
  }
}
```

## Testing Best Practices

### Integration Test Example

```python
import pytest
import requests
from typing import Generator

@pytest.fixture
def api_client() -> Generator:
    """Create an authenticated API client for testing."""
    # Login and get token
    response = requests.post(
        "http://localhost:8080/api/auth/login",
        json={"username": "test_user", "password": "test_password"}
    )
    token = response.json()["access_token"]
    
    client = ChunkingAPIClient("http://localhost:8080", token)
    yield client
    
    # Cleanup if needed

def test_chunking_workflow(api_client):
    """Test complete chunking workflow."""
    
    # Step 1: Get strategy recommendation
    recommendation = api_client.recommend_strategy(["pdf", "md"])
    assert recommendation["recommended_strategy"] in ["semantic", "document_structure"]
    
    # Step 2: Generate preview with recommended strategy
    preview = api_client.generate_preview(
        content="# Test Document\n\nThis is test content for chunking.",
        strategy=ChunkingStrategy(recommendation["recommended_strategy"]),
        max_chunks=5
    )
    assert preview["total_chunks"] > 0
    assert len(preview["chunks"]) <= 5
    
    # Step 3: Compare strategies
    comparison = api_client.compare_strategies(
        content="Test content for comparison",
        strategies=[ChunkingStrategy.SEMANTIC, ChunkingStrategy.FIXED_SIZE]
    )
    assert len(comparison["comparisons"]) == 2
    assert "recommendation" in comparison
    
    # Step 4: Verify caching works
    cached_preview = api_client.session.get(
        f"{api_client.base_url}/preview/{preview['preview_id']}"
    ).json()
    assert cached_preview["cached"] == True

def test_error_handling(api_client):
    """Test error handling scenarios."""
    
    # Test validation error
    with pytest.raises(ChunkingValidationError):
        api_client.generate_preview(
            content="Test",
            strategy=ChunkingStrategy.SEMANTIC,
            config=ChunkingConfig(chunk_size=50, chunk_overlap=100)  # Invalid
        )
    
    # Test content size limit
    large_content = "x" * (10 * 1024 * 1024 + 1)  # > 10MB
    with pytest.raises(ChunkingMemoryError):
        api_client.generate_preview(
            content=large_content,
            strategy=ChunkingStrategy.FIXED_SIZE
        )
```

## Performance Tips

1. **Use Preview Caching**: Preview results are cached for 15 minutes. Reuse `preview_id` when possible.

2. **Batch Operations**: Process multiple documents in a single collection operation rather than individual requests.

3. **Optimize Chunk Sizes**: Larger chunks reduce total count but may lose granularity. Find the right balance.

4. **Monitor WebSocket Throttling**: Progress updates are throttled to prevent overwhelming clients.

5. **Use Priority Queuing**: Set higher priority (1-10) for time-sensitive operations.

6. **Strategy Selection**: Use the recommendation endpoint for optimal strategy selection based on content type.

## Troubleshooting Common Issues

### Rate Limiting
- Implement exponential backoff when hitting rate limits
- Check `X-RateLimit-*` headers to track usage
- Consider caching preview results

### WebSocket Disconnections
- Implement automatic reconnection logic
- Store operation ID to resume monitoring
- Check for network issues or token expiration

### Memory Errors
- Split large documents before processing
- Use streaming for very large files
- Monitor server memory usage

### Timeout Errors
- Reduce chunk size for faster processing
- Use simpler strategies for large documents
- Consider breaking into smaller operations
