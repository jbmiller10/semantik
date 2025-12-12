import type {
  ChunkingStrategyType,
  ChunkingConfiguration,
  ChunkingPreviewResponse,
  ChunkingComparisonResult,
  ChunkingAnalytics,
  ChunkingPreset
} from '../../types/chunking';

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
