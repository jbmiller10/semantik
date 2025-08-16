# Chunking API Examples

This document provides practical examples for using the Semantik Chunking API with various tools and programming languages.

## Table of Contents

- [Authentication](#authentication)
- [Common Use Cases](#common-use-cases)
- [curl Examples](#curl-examples)
- [HTTPie Examples](#httpie-examples)
- [Python Examples](#python-examples)
- [JavaScript/TypeScript Examples](#javascripttypescript-examples)
- [WebSocket Examples](#websocket-examples)
- [Error Handling Examples](#error-handling-examples)

## Authentication

All API requests require a JWT token. First, obtain a token:

```bash
# Using curl
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "...",
  "token_type": "bearer"
}
```

Store the `access_token` for use in subsequent requests.

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
    print(f"Operation started: {operation['operation_id']}")
    print(f"WebSocket channel: {operation['websocket_channel']}")
    
    # Connect to WebSocket for progress updates
    ws_url = f"ws://localhost:8080/ws/channel/{operation['websocket_channel']}?token={token}"
    
    async with websockets.connect(ws_url) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "chunking_progress":
                progress = data["data"]
                print(f"Progress: {progress['progress_percentage']:.1f}%")
                print(f"  Documents: {progress['documents_processed']}/{progress['total_documents']}")
                print(f"  Chunks created: {progress['chunks_created']}")
                
            elif data["type"] == "chunking_completed":
                print("Chunking completed!")
                print(f"  Total chunks: {data['data']['total_chunks']}")
                print(f"  Processing time: {data['data']['processing_time_seconds']}s")
                break
                
            elif data["type"] == "chunking_failed":
                print(f"Chunking failed: {data['data']['error']}")
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
    SLIDING_WINDOW = "sliding_window"
    DOCUMENT_STRUCTURE = "document_structure"
    HYBRID = "hybrid"

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
    
    async def monitor_operation(self, operation_id: str, websocket_channel: str):
        """Monitor chunking operation progress via WebSocket."""
        token = self.headers["Authorization"].replace("Bearer ", "")
        ws_url = f"ws://localhost:8080/ws/channel/{websocket_channel}?token={token}"
        
        async with websockets.connect(ws_url) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                print(f"[{data['type']}] ", end="")
                
                if data["type"] == "chunking_progress":
                    progress = data["data"]
                    print(f"{progress['progress_percentage']:.1f}% - "
                          f"{progress['documents_processed']}/{progress['total_documents']} docs")
                
                elif data["type"] == "chunking_completed":
                    print(f"Completed! {data['data']['total_chunks']} chunks created")
                    break
                
                elif data["type"] == "chunking_failed":
                    print(f"Failed: {data['data']['error']}")
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
  SLIDING_WINDOW = 'sliding_window',
  DOCUMENT_STRUCTURE = 'document_structure',
  HYBRID = 'hybrid'
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
    websocketChannel: string,
    token: string,
    onMessage: (data: any) => void
  ): WebSocket {
    const wsUrl = `ws://localhost:8080/ws/channel/${websocketChannel}?token=${token}`;
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
    
    // Connect to WebSocket for progress
    const ws = client.connectToProgress(
      operation.websocket_channel,
      'your_token',
      (data) => {
        switch(data.type) {
          case 'chunking_progress':
            console.log(`Progress: ${data.data.progress_percentage}%`);
            break;
          case 'chunking_completed':
            console.log('Chunking completed!');
            ws.close();
            break;
          case 'chunking_failed':
            console.error('Chunking failed:', data.data.error);
            ws.close();
            break;
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
import { useState, useEffect, useCallback } from 'react';

interface UseChunkingProgressOptions {
  operationId: string;
  websocketChannel: string;
  token: string;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

export function useChunkingProgress({
  operationId,
  websocketChannel,
  token,
  onComplete,
  onError
}: UseChunkingProgressOptions) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'pending' | 'in_progress' | 'completed' | 'failed'>('pending');
  const [documentsProcessed, setDocumentsProcessed] = useState(0);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [chunksCreated, setChunksCreated] = useState(0);
  const [currentDocument, setCurrentDocument] = useState<string | null>(null);

  useEffect(() => {
    const wsUrl = `ws://localhost:8080/ws/channel/${websocketChannel}?token=${token}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch(data.type) {
        case 'chunking_started':
          setStatus('in_progress');
          break;
          
        case 'chunking_progress':
          const progressData = data.data;
          setProgress(progressData.progress_percentage);
          setDocumentsProcessed(progressData.documents_processed);
          setTotalDocuments(progressData.total_documents);
          setChunksCreated(progressData.chunks_created);
          setCurrentDocument(progressData.current_document);
          break;
          
        case 'chunking_completed':
          setStatus('completed');
          setProgress(100);
          onComplete?.();
          break;
          
        case 'chunking_failed':
          setStatus('failed');
          onError?.(data.data.error);
          break;
      }
    };

    ws.onerror = () => {
      setStatus('failed');
      onError?.('WebSocket connection error');
    };

    return () => {
      ws.close();
    };
  }, [operationId, websocketChannel, token, onComplete, onError]);

  return {
    progress,
    status,
    documentsProcessed,
    totalDocuments,
    chunksCreated,
    currentDocument
  };
}

// Component usage
function ChunkingProgressDisplay({ operationId, websocketChannel, token }: any) {
  const {
    progress,
    status,
    documentsProcessed,
    totalDocuments,
    chunksCreated,
    currentDocument
  } = useChunkingProgress({
    operationId,
    websocketChannel,
    token,
    onComplete: () => console.log('Chunking completed!'),
    onError: (error) => console.error('Chunking error:', error)
  });

  return (
    <div>
      <h3>Chunking Progress</h3>
      <div>Status: {status}</div>
      <div>Progress: {progress.toFixed(1)}%</div>
      <div>Documents: {documentsProcessed}/{totalDocuments}</div>
      <div>Chunks Created: {chunksCreated}</div>
      {currentDocument && <div>Processing: {currentDocument}</div>}
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
  constructor(websocketChannel, token) {
    this.websocketChannel = websocketChannel;
    this.token = token;
    this.ws = null;
  }

  connect() {
    const wsUrl = `ws://localhost:8080/ws/channel/${this.websocketChannel}?token=${this.token}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.on('open', () => {
      console.log('Connected to chunking progress channel');
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
    switch(message.type) {
      case 'chunking_started':
        console.log(`Chunking started for collection ${message.collection_id}`);
        break;

      case 'chunking_progress':
        const progress = message.data;
        console.log(`Progress: ${progress.progress_percentage.toFixed(1)}%`);
        console.log(`  Documents: ${progress.documents_processed}/${progress.total_documents}`);
        console.log(`  Chunks: ${progress.chunks_created}`);
        if (progress.current_document) {
          console.log(`  Current: ${progress.current_document}`);
        }
        break;

      case 'chunking_document_start':
        console.log(`Starting document: ${message.data.document_name}`);
        break;

      case 'chunking_document_complete':
        console.log(`Completed document: ${message.data.document_id}`);
        console.log(`  Chunks created: ${message.data.chunks_created}`);
        break;

      case 'chunking_completed':
        console.log('Chunking operation completed!');
        console.log(`  Total chunks: ${message.data.total_chunks}`);
        console.log(`  Processing time: ${message.data.processing_time_seconds}s`);
        this.close();
        break;

      case 'chunking_failed':
        console.error('Chunking operation failed!');
        console.error(`  Error: ${message.data.error}`);
        console.error(`  Error code: ${message.data.error_code}`);
        this.close();
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage
const monitor = new ChunkingProgressMonitor('chunking:coll_123:op_456', 'your_token');
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