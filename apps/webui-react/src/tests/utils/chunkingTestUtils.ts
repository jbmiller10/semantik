import { vi } from 'vitest';
import type { 
  ChunkingStrategyType, 
  ChunkingConfiguration,
  ChunkingPreviewResponse,
  ChunkingComparisonResult,
  ChunkingAnalytics,
  ChunkingPreset,
  ChunkPreview
} from '../../types/chunking';
import type { WebSocketMessage } from '../../services/websocket';

/**
 * Mock WebSocket class for testing
 */
export class MockChunkingWebSocket {
  url: string;
  readyState: number = WebSocket.CONNECTING;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  addEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | AddEventListenerOptions) => void;
  removeEventListener: (type: string, listener: EventListenerOrEventListenerObject, options?: boolean | EventListenerOptions) => void;
  dispatchEvent: (event: Event) => boolean;
  
  private messageQueue: WebSocketMessage[] = [];
  private isAuthenticated = false;
  
  constructor(url: string) {
    this.url = url;
    // Add event listener methods for compatibility
    this.addEventListener = vi.fn((event: string, handler: EventListenerOrEventListenerObject | ((event: Event) => void)) => {
      if (event === 'open' && this.onopen === null) {
        this.onopen = handler as (event: Event) => void;
      } else if (event === 'close' && this.onclose === null) {
        this.onclose = handler as (event: CloseEvent) => void;
      } else if (event === 'error' && this.onerror === null) {
        this.onerror = handler as (event: Event) => void;
      } else if (event === 'message' && this.onmessage === null) {
        this.onmessage = handler as (event: MessageEvent) => void;
      }
    });
    this.removeEventListener = vi.fn();
    this.dispatchEvent = vi.fn();
    
    // Simulate connection opening
    setTimeout(() => this.simulateOpen(), 10);
  }
  
  simulateOpen() {
    this.readyState = WebSocket.OPEN;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }
  
  simulateMessage(message: WebSocketMessage) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', {
        data: JSON.stringify(message)
      }));
    }
  }
  
  simulateError() {
    if (this.onerror) {
      this.onerror(new Event('error'));
    }
  }
  
  simulateClose(code = 1000, reason = 'Normal closure') {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code, reason }));
    }
  }
  
  send(data: string) {
    const message = JSON.parse(data);
    
    // Simulate authentication flow
    if (message.type === 'auth_request') {
      setTimeout(() => {
        this.isAuthenticated = true;
        this.simulateMessage({
          type: 'auth_success',
          data: { userId: 'test-user', sessionId: 'test-session' },
          timestamp: Date.now()
        });
      }, 10);
    }
    
    // Process any queued messages if authenticated
    if (this.isAuthenticated && this.messageQueue.length > 0) {
      this.messageQueue.forEach(msg => this.simulateMessage(msg));
      this.messageQueue = [];
    }
  }
  
  close() {
    this.simulateClose();
  }
  
  // Helper method to queue messages for testing
  queueMessage(message: WebSocketMessage) {
    if (this.isAuthenticated) {
      this.simulateMessage(message);
    } else {
      this.messageQueue.push(message);
    }
  }
}

/**
 * Factory to create mock WebSocket instances
 */
export function createMockWebSocket() {
  const MockWebSocketConstructor = vi.fn((url: string) => new MockChunkingWebSocket(url));
  return MockWebSocketConstructor as unknown as typeof WebSocket;
}

/**
 * Sample chunking preview response for testing
 */
export const mockChunkingPreviewResponse: ChunkingPreviewResponse = {
  chunks: [
    {
      id: 'chunk-1',
      content: 'This is the first chunk of content with some meaningful text.',
      startIndex: 0,
      endIndex: 60,
      metadata: { position: 0, size: 60 },
      tokens: 10,
      overlapWithPrevious: 0,
      overlapWithNext: 10
    },
    {
      id: 'chunk-2', 
      content: 'Some overlapping text. This is the second chunk with more content.',
      startIndex: 50,
      endIndex: 116,
      metadata: { position: 50, size: 66 },
      tokens: 12,
      overlapWithPrevious: 10,
      overlapWithNext: 0
    }
  ],
  statistics: {
    totalChunks: 2,
    avgChunkSize: 63,
    minChunkSize: 60,
    maxChunkSize: 66,
    totalSize: 126,
    totalTokens: 22,
    avgTokensPerChunk: 11,
    overlapPercentage: 7.9,
    sizeDistribution: [
      { range: '0-100', count: 2, percentage: 100 }
    ]
  },
  performance: {
    processingTimeMs: 45,
    chunksPerSecond: 44.4,
    memoryUsageMB: 2.5,
    estimatedFullProcessingTimeMs: 450
  },
  warnings: []
};

/**
 * Sample comparison results for testing
 */
export const mockComparisonResults: ChunkingComparisonResult[] = [
  {
    strategy: 'recursive',
    configuration: {
      strategy: 'recursive',
      parameters: {
        chunk_size: 600,
        chunk_overlap: 100
      }
    },
    preview: mockChunkingPreviewResponse,
    score: {
      overall: 85,
      quality: 88,
      performance: 82
    }
  },
  {
    strategy: 'character',
    configuration: {
      strategy: 'character',
      parameters: {
        chunk_size: 500,
        chunk_overlap: 0
      }
    },
    preview: {
      ...mockChunkingPreviewResponse,
      statistics: {
        ...mockChunkingPreviewResponse.statistics,
        avgChunkSize: 500,
        overlapPercentage: 0
      }
    },
    score: {
      overall: 75,
      quality: 70,
      performance: 95
    }
  }
];

/**
 * Sample analytics data for testing
 */
export const mockChunkingAnalytics: ChunkingAnalytics = {
  totalDocumentsProcessed: 150,
  totalChunksCreated: 3500,
  averageChunkSize: 512,
  averageProcessingTime: 234,
  strategyUsage: {
    recursive: 45,
    character: 20,
    semantic: 25,
    markdown: 5,
    hierarchical: 3,
    hybrid: 2
  },
  performanceTrends: [
    { date: '2024-01-01', avgProcessingTime: 220, documentsProcessed: 20 },
    { date: '2024-01-02', avgProcessingTime: 245, documentsProcessed: 25 },
    { date: '2024-01-03', avgProcessingTime: 230, documentsProcessed: 30 }
  ],
  topConfigurations: [
    {
      strategy: 'recursive',
      parameters: { chunk_size: 600, chunk_overlap: 100 },
      usageCount: 45,
      avgScore: 85
    },
    {
      strategy: 'semantic',
      parameters: { chunk_size: 800, chunk_overlap: 50 },
      usageCount: 25,
      avgScore: 92
    }
  ]
};

/**
 * Sample presets for testing
 */
export const mockChunkingPresets: ChunkingPreset[] = [
  {
    id: 'default-recursive',
    name: 'Default Recursive',
    description: 'Standard recursive splitting',
    strategy: 'recursive',
    configuration: {
      strategy: 'recursive',
      parameters: {
        chunk_size: 600,
        chunk_overlap: 100
      }
    },
    isSystem: true,
    isRecommended: true
  },
  {
    id: 'custom-semantic',
    name: 'Custom Semantic',
    description: 'Custom semantic configuration',
    strategy: 'semantic',
    configuration: {
      strategy: 'semantic',
      parameters: {
        chunk_size: 800,
        chunk_overlap: 50,
        similarity_threshold: 0.75
      }
    },
    isSystem: false,
    isRecommended: false
  }
];

/**
 * Helper to create mock API responses
 */
export function createMockApiResponse<T>(data: T, delay = 10) {
  return new Promise<{ data: T }>((resolve) => {
    setTimeout(() => resolve({ data }), delay);
  });
}

/**
 * Helper to simulate WebSocket chunking messages
 */
export function simulateChunkingProgress(
  ws: MockChunkingWebSocket,
  chunks: ChunkPreview[],
  requestId?: string
) {
  // Send preview start
  ws.queueMessage({
    type: 'preview_start',
    data: { totalChunks: chunks.length },
    timestamp: Date.now(),
    requestId
  });
  
  // Send each chunk with progress
  chunks.forEach((chunk, index) => {
    ws.queueMessage({
      type: 'preview_progress',
      data: {
        percentage: ((index + 1) / chunks.length) * 100,
        currentChunk: index + 1,
        totalChunks: chunks.length
      },
      timestamp: Date.now(),
      requestId
    });
    
    ws.queueMessage({
      type: 'preview_chunk',
      data: {
        chunk,
        index,
        total: chunks.length
      },
      timestamp: Date.now(),
      requestId
    });
  });
  
  // Send completion
  ws.queueMessage({
    type: 'preview_complete',
    data: {
      statistics: mockChunkingPreviewResponse.statistics,
      performance: mockChunkingPreviewResponse.performance
    },
    timestamp: Date.now(),
    requestId
  });
}

/**
 * Helper to create test document
 */
export function createTestDocument(overrides = {}) {
  return {
    id: 'test-doc-1',
    name: 'test-document.txt',
    content: 'This is a test document with enough content to be chunked into multiple pieces. '.repeat(10),
    size: 800,
    type: 'text/plain',
    ...overrides
  };
}

/**
 * Helper to create chunking configuration
 */
export function createChunkingConfig(
  strategy: ChunkingStrategyType = 'recursive',
  overrides = {}
): ChunkingConfiguration {
  return {
    strategy,
    parameters: {
      chunk_size: 600,
      chunk_overlap: 100,
      ...overrides
    }
  };
}