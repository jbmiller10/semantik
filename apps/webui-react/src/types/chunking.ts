export type ChunkingStrategyType = 
  | 'character'
  | 'recursive'
  | 'markdown'
  | 'semantic'
  | 'hierarchical'
  | 'hybrid';

export interface ChunkingStrategy {
  type: ChunkingStrategyType;
  name: string;
  description: string;
  icon: string; // lucide-react icon name
  performance: {
    speed: 'fast' | 'medium' | 'slow';
    quality: 'basic' | 'good' | 'excellent';
    memoryUsage: 'low' | 'medium' | 'high';
  };
  supportedFileTypes: string[];
  parameters: ChunkingParameter[];
  isRecommended?: boolean;
  recommendedFor?: string[];
}

export interface ChunkingParameter {
  name: string;
  key: string;
  type: 'number' | 'boolean' | 'select';
  min?: number;
  max?: number;
  step?: number;
  options?: { value: string | number; label: string }[];
  defaultValue: number | boolean | string;
  description: string;
  unit?: string;
  advanced?: boolean;
}

export interface ChunkingConfiguration {
  strategy: ChunkingStrategyType;
  parameters: Record<string, number | boolean | string>;
}

export interface ChunkingPreset {
  id: string;
  name: string;
  description: string;
  strategy: ChunkingStrategyType;
  configuration: ChunkingConfiguration;
  fileTypes?: string[];
}

export interface ChunkPreview {
  id: string;
  content: string;
  startIndex: number;
  endIndex: number;
  metadata?: Record<string, unknown>;
  tokens?: number;
  overlapWithPrevious?: number;
  overlapWithNext?: number;
}

export interface ChunkingPreviewRequest {
  documentId?: string;
  content?: string;
  strategy: ChunkingStrategyType;
  configuration: Record<string, number | boolean | string>;
  maxPreviewChunks?: number;
}

export interface ChunkingPreviewResponse {
  chunks: ChunkPreview[];
  statistics: ChunkingStatistics;
  performance: {
    processingTimeMs: number;
    estimatedFullProcessingTimeMs: number;
  };
}

export interface ChunkingStatistics {
  totalChunks: number;
  avgChunkSize: number;
  minChunkSize: number;
  maxChunkSize: number;
  totalTokens?: number;
  avgTokensPerChunk?: number;
  overlapPercentage?: number;
  sizeDistribution: {
    range: string;
    count: number;
    percentage: number;
  }[];
}

export interface ChunkingComparisonResult {
  strategy: ChunkingStrategyType;
  configuration: ChunkingConfiguration;
  preview: ChunkingPreviewResponse;
  score?: {
    quality: number;
    performance: number;
    overall: number;
  };
}

export interface ChunkingAnalytics {
  strategyUsage: {
    strategy: ChunkingStrategyType;
    count: number;
    percentage: number;
    trend: 'up' | 'down' | 'stable';
  }[];
  performanceMetrics: {
    strategy: ChunkingStrategyType;
    avgProcessingTimeMs: number;
    avgChunksPerDocument: number;
    successRate: number;
  }[];
  fileTypeDistribution: {
    fileType: string;
    count: number;
    preferredStrategy: ChunkingStrategyType;
  }[];
  recommendations: ChunkingRecommendation[];
}

export interface ChunkingRecommendation {
  id: string;
  type: 'strategy' | 'parameter' | 'general';
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  action?: {
    label: string;
    configuration?: ChunkingConfiguration;
  };
}

// Strategy definitions with metadata
export const CHUNKING_STRATEGIES: Record<ChunkingStrategyType, Omit<ChunkingStrategy, 'type'>> = {
  character: {
    name: 'Character-based',
    description: 'Simple fixed-size splitting for basic needs. Fast and predictable.',
    icon: 'Type',
    performance: {
      speed: 'fast',
      quality: 'basic',
      memoryUsage: 'low'
    },
    supportedFileTypes: ['*'],
    parameters: [
      {
        name: 'Chunk Size',
        key: 'chunk_size',
        type: 'number',
        min: 100,
        max: 2000,
        step: 100,
        defaultValue: 600,
        description: 'Number of characters per chunk',
        unit: 'characters'
      },
      {
        name: 'Chunk Overlap',
        key: 'chunk_overlap',
        type: 'number',
        min: 0,
        max: 500,
        step: 50,
        defaultValue: 100,
        description: 'Number of overlapping characters between chunks',
        unit: 'characters'
      }
    ]
  },
  recursive: {
    name: 'Recursive',
    description: 'Intelligent splitting that respects sentence and paragraph boundaries. Recommended for most documents.',
    icon: 'GitBranch',
    performance: {
      speed: 'fast',
      quality: 'good',
      memoryUsage: 'low'
    },
    supportedFileTypes: ['*'],
    parameters: [
      {
        name: 'Chunk Size',
        key: 'chunk_size',
        type: 'number',
        min: 100,
        max: 2000,
        step: 100,
        defaultValue: 600,
        description: 'Target size for each chunk',
        unit: 'characters'
      },
      {
        name: 'Chunk Overlap',
        key: 'chunk_overlap',
        type: 'number',
        min: 0,
        max: 500,
        step: 50,
        defaultValue: 100,
        description: 'Overlap between consecutive chunks',
        unit: 'characters'
      },
      {
        name: 'Preserve Sentences',
        key: 'preserve_sentences',
        type: 'boolean',
        defaultValue: true,
        description: 'Avoid splitting in the middle of sentences',
        advanced: true
      }
    ],
    isRecommended: true,
    recommendedFor: ['General documents', 'Articles', 'Reports']
  },
  markdown: {
    name: 'Markdown-aware',
    description: 'Respects Markdown structure including headers, lists, and code blocks.',
    icon: 'FileText',
    performance: {
      speed: 'medium',
      quality: 'excellent',
      memoryUsage: 'medium'
    },
    supportedFileTypes: ['md', 'mdx', 'markdown'],
    parameters: [
      {
        name: 'Max Header Level',
        key: 'max_header_level',
        type: 'number',
        min: 1,
        max: 6,
        step: 1,
        defaultValue: 3,
        description: 'Maximum header level to use as chunk boundary'
      },
      {
        name: 'Include Headers',
        key: 'include_headers',
        type: 'boolean',
        defaultValue: true,
        description: 'Include parent headers in each chunk for context'
      },
      {
        name: 'Min Chunk Size',
        key: 'min_chunk_size',
        type: 'number',
        min: 100,
        max: 1000,
        step: 100,
        defaultValue: 300,
        description: 'Minimum size before forcing a chunk split',
        unit: 'characters',
        advanced: true
      }
    ],
    recommendedFor: ['Documentation', 'Technical docs', 'README files']
  },
  semantic: {
    name: 'Semantic',
    description: 'Uses AI embeddings to find natural topic boundaries. Best quality but slower.',
    icon: 'Brain',
    performance: {
      speed: 'slow',
      quality: 'excellent',
      memoryUsage: 'high'
    },
    supportedFileTypes: ['*'],
    parameters: [
      {
        name: 'Breakpoint Threshold',
        key: 'breakpoint_percentile_threshold',
        type: 'number',
        min: 70,
        max: 99,
        step: 5,
        defaultValue: 90,
        description: 'Similarity threshold for identifying topic changes',
        unit: 'percentile'
      },
      {
        name: 'Max Chunk Size',
        key: 'max_chunk_size',
        type: 'number',
        min: 500,
        max: 3000,
        step: 100,
        defaultValue: 1000,
        description: 'Maximum allowed chunk size',
        unit: 'characters'
      },
      {
        name: 'Buffer Size',
        key: 'buffer_size',
        type: 'number',
        min: 1,
        max: 5,
        step: 1,
        defaultValue: 2,
        description: 'Number of sentences to consider for semantic similarity',
        advanced: true
      }
    ],
    recommendedFor: ['Research papers', 'Books', 'Complex documents']
  },
  hierarchical: {
    name: 'Hierarchical',
    description: 'Creates parent-child relationships between chunks for better context preservation.',
    icon: 'Network',
    performance: {
      speed: 'medium',
      quality: 'excellent',
      memoryUsage: 'medium'
    },
    supportedFileTypes: ['*'],
    parameters: [
      {
        name: 'Chunk Sizes',
        key: 'chunk_sizes',
        type: 'select',
        options: [
          { value: '512,1024,2048', label: 'Small (512, 1024, 2048)' },
          { value: '1024,2048,4096', label: 'Medium (1024, 2048, 4096)' },
          { value: '2048,4096,8192', label: 'Large (2048, 4096, 8192)' }
        ],
        defaultValue: '1024,2048,4096',
        description: 'Size hierarchy for parent and child chunks'
      },
      {
        name: 'Overlap Ratio',
        key: 'overlap_ratio',
        type: 'number',
        min: 0,
        max: 0.5,
        step: 0.1,
        defaultValue: 0.2,
        description: 'Overlap ratio between hierarchical levels',
        advanced: true
      }
    ],
    recommendedFor: ['Long documents', 'Technical manuals', 'Legal documents']
  },
  hybrid: {
    name: 'Hybrid Auto-Select',
    description: 'Automatically selects the best strategy based on content type and structure.',
    icon: 'Sparkles',
    performance: {
      speed: 'medium',
      quality: 'excellent',
      memoryUsage: 'medium'
    },
    supportedFileTypes: ['*'],
    parameters: [
      {
        name: 'Quality Preference',
        key: 'quality_preference',
        type: 'select',
        options: [
          { value: 'speed', label: 'Optimize for Speed' },
          { value: 'balanced', label: 'Balanced' },
          { value: 'quality', label: 'Optimize for Quality' }
        ],
        defaultValue: 'balanced',
        description: 'Balance between processing speed and chunk quality'
      },
      {
        name: 'Auto-detect Code',
        key: 'auto_detect_code',
        type: 'boolean',
        defaultValue: true,
        description: 'Automatically optimize for code files when detected'
      }
    ],
    isRecommended: true,
    recommendedFor: ['Mixed content', 'Unknown file types', 'First-time users']
  }
};

// Preset configurations
export const CHUNKING_PRESETS: ChunkingPreset[] = [
  {
    id: 'default-documents',
    name: 'General Documents',
    description: 'Optimized for PDFs, Word docs, and general text',
    strategy: 'recursive',
    configuration: {
      strategy: 'recursive',
      parameters: {
        chunk_size: 600,
        chunk_overlap: 100,
        preserve_sentences: true
      }
    },
    fileTypes: ['pdf', 'doc', 'docx', 'txt']
  },
  {
    id: 'technical-docs',
    name: 'Technical Documentation',
    description: 'Perfect for README files and API documentation',
    strategy: 'markdown',
    configuration: {
      strategy: 'markdown',
      parameters: {
        max_header_level: 3,
        include_headers: true,
        min_chunk_size: 300
      }
    },
    fileTypes: ['md', 'mdx', 'rst']
  },
  {
    id: 'code-files',
    name: 'Source Code',
    description: 'Optimized for code files with smaller chunks',
    strategy: 'recursive',
    configuration: {
      strategy: 'recursive',
      parameters: {
        chunk_size: 400,
        chunk_overlap: 50,
        preserve_sentences: false
      }
    },
    fileTypes: ['py', 'js', 'ts', 'java', 'cpp', 'go', 'rust']
  },
  {
    id: 'research-papers',
    name: 'Academic Papers',
    description: 'Semantic chunking for research and academic content',
    strategy: 'semantic',
    configuration: {
      strategy: 'semantic',
      parameters: {
        breakpoint_percentile_threshold: 90,
        max_chunk_size: 1000,
        buffer_size: 2
      }
    },
    fileTypes: ['pdf', 'tex']
  },
  {
    id: 'large-documents',
    name: 'Large Documents',
    description: 'Hierarchical chunking for books and manuals',
    strategy: 'hierarchical',
    configuration: {
      strategy: 'hierarchical',
      parameters: {
        chunk_sizes: '1024,2048,4096',
        overlap_ratio: 0.2
      }
    },
    fileTypes: ['pdf', 'epub']
  }
];