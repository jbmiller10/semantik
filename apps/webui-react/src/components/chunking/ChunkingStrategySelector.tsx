import { useState } from 'react';
import { 
  Type, 
  GitBranch, 
  FileText, 
  Brain, 
  Network, 
  Sparkles,
  ChevronDown,
  ChevronUp,
  Info,
  Zap,
  Target,
  Database,
  Settings
} from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';
import { CHUNKING_STRATEGIES } from '../../types/chunking';
import { ChunkingParameterTuner } from './ChunkingParameterTuner';
import type { ChunkingStrategyType } from '../../types/chunking';

interface ChunkingStrategySelectorProps {
  onStrategyChange?: (strategy: ChunkingStrategyType) => void;
  disabled?: boolean;
  fileType?: string;
}

const strategyIcons: Record<ChunkingStrategyType, React.ElementType> = {
  character: Type,
  recursive: GitBranch,
  markdown: FileText,
  semantic: Brain,
  hierarchical: Network,
  hybrid: Sparkles
};

const performanceColors = {
  speed: {
    fast: 'text-green-600',
    medium: 'text-yellow-600',
    slow: 'text-red-600'
  },
  quality: {
    basic: 'text-gray-600',
    good: 'text-blue-600',
    excellent: 'text-purple-600'
  },
  memoryUsage: {
    low: 'text-green-600',
    medium: 'text-yellow-600',
    high: 'text-red-600'
  }
};

const performanceLabels = {
  speed: {
    fast: 'Fast',
    medium: 'Medium',
    slow: 'Slow'
  },
  quality: {
    basic: 'Basic',
    good: 'Good',
    excellent: 'Excellent'
  },
  memoryUsage: {
    low: 'Low',
    medium: 'Medium',
    high: 'High'
  }
};

export function ChunkingStrategySelector({ 
  onStrategyChange, 
  disabled = false,
  fileType 
}: ChunkingStrategySelectorProps) {
  const { 
    selectedStrategy, 
    setStrategy, 
    getRecommendedStrategy 
  } = useChunkingStore();
  
  const [expandedStrategy, setExpandedStrategy] = useState<ChunkingStrategyType | null>(null);
  const recommendedStrategy = fileType ? getRecommendedStrategy(fileType) : null;

  const handleStrategySelect = (strategyType: ChunkingStrategyType) => {
    if (!disabled) {
      setStrategy(strategyType);
      onStrategyChange?.(strategyType);
    }
  };

  const toggleExpanded = (strategyType: ChunkingStrategyType) => {
    setExpandedStrategy(expandedStrategy === strategyType ? null : strategyType);
  };

  const [showParameterTuner, setShowParameterTuner] = useState(false);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900">Chunking Strategy</h3>
        {fileType && (
          <span className="text-sm text-gray-500">
            Detected file type: <span className="font-medium">{fileType}</span>
          </span>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {(Object.entries(CHUNKING_STRATEGIES) as [ChunkingStrategyType, typeof CHUNKING_STRATEGIES[ChunkingStrategyType]][]).map(
          ([strategyType, strategy]) => {
            const Icon = strategyIcons[strategyType];
            const isSelected = selectedStrategy === strategyType;
            const isRecommended = strategy.isRecommended || strategyType === recommendedStrategy;
            const isExpanded = expandedStrategy === strategyType;

            return (
              <div
                key={strategyType}
                className={`relative rounded-lg border-2 transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 bg-white'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              >
                {/* Recommended Badge */}
                {isRecommended && (
                  <div className="absolute -top-3 left-4 px-2 py-0.5 bg-green-100 text-green-800 text-xs font-medium rounded-full">
                    Recommended
                  </div>
                )}

                {/* Main Card Content */}
                <div
                  className="p-4"
                  onClick={() => handleStrategySelect(strategyType)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleStrategySelect(strategyType);
                    }
                  }}
                  aria-label={`Select ${strategy.name} strategy`}
                  aria-pressed={isSelected}
                >
                  <div className="flex items-start space-x-3">
                    <div className={`p-2 rounded-lg ${
                      isSelected ? 'bg-blue-100' : 'bg-gray-100'
                    }`}>
                      <Icon className={`h-5 w-5 ${
                        isSelected ? 'text-blue-600' : 'text-gray-600'
                      }`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-gray-900 truncate">
                        {strategy.name}
                      </h4>
                      <p className="mt-1 text-xs text-gray-500 line-clamp-2">
                        {strategy.description}
                      </p>
                    </div>
                  </div>

                  {/* Performance Indicators */}
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                    <div className="flex items-center space-x-1">
                      <Zap className={`h-3 w-3 ${performanceColors.speed[strategy.performance.speed]}`} />
                      <span className={performanceColors.speed[strategy.performance.speed]}>
                        {performanceLabels.speed[strategy.performance.speed]}
                      </span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Target className={`h-3 w-3 ${performanceColors.quality[strategy.performance.quality]}`} />
                      <span className={performanceColors.quality[strategy.performance.quality]}>
                        {performanceLabels.quality[strategy.performance.quality]}
                      </span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Database className={`h-3 w-3 ${performanceColors.memoryUsage[strategy.performance.memoryUsage]}`} />
                      <span className={performanceColors.memoryUsage[strategy.performance.memoryUsage]}>
                        {performanceLabels.memoryUsage[strategy.performance.memoryUsage]}
                      </span>
                    </div>
                  </div>

                  {/* Recommended For */}
                  {strategy.recommendedFor && strategy.recommendedFor.length > 0 && (
                    <div className="mt-2 flex items-start space-x-1">
                      <Info className="h-3 w-3 text-gray-400 mt-0.5 flex-shrink-0" />
                      <p className="text-xs text-gray-500">
                        Best for: {strategy.recommendedFor.join(', ')}
                      </p>
                    </div>
                  )}
                </div>

                {/* Expand/Collapse Button */}
                <button
                  className="w-full px-4 pb-3 pt-1 flex items-center justify-center text-xs text-gray-500 hover:text-gray-700 transition-colors"
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleExpanded(strategyType);
                  }}
                  aria-expanded={isExpanded}
                  aria-label={`${isExpanded ? 'Hide' : 'Show'} details for ${strategy.name}`}
                >
                  {isExpanded ? (
                    <>
                      <ChevronUp className="h-3 w-3 mr-1" />
                      Hide Details
                    </>
                  ) : (
                    <>
                      <ChevronDown className="h-3 w-3 mr-1" />
                      Show Details
                    </>
                  )}
                </button>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-4 pb-4 border-t border-gray-100">
                    <div className="mt-3 space-y-3">
                      {/* Parameters */}
                      <div>
                        <h5 className="text-xs font-medium text-gray-700 mb-1">Parameters</h5>
                        <ul className="space-y-1 text-xs text-gray-600">
                          {strategy.parameters.filter(p => !p.advanced).map(param => (
                            <li key={param.key} className="flex justify-between">
                              <span>{param.name}:</span>
                              <span className="font-mono text-gray-500">
                                {param.defaultValue}{param.unit ? ` ${param.unit}` : ''}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      {/* Supported File Types */}
                      <div>
                        <h5 className="text-xs font-medium text-gray-700 mb-1">Supported Files</h5>
                        <p className="text-xs text-gray-600">
                          {strategy.supportedFileTypes.includes('*') 
                            ? 'All file types'
                            : strategy.supportedFileTypes.join(', ')
                          }
                        </p>
                      </div>

                      {/* Technical Details */}
                      <div>
                        <h5 className="text-xs font-medium text-gray-700 mb-1">How it works</h5>
                        <p className="text-xs text-gray-600 leading-relaxed">
                          {getStrategyDetails(strategyType)}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          }
        )}
      </div>

      {/* Parameter Tuner Section */}
      <div className="mt-6 border-t pt-6">
        <button
          type="button"
          onClick={() => setShowParameterTuner(!showParameterTuner)}
          className="flex items-center justify-between w-full text-left focus:outline-none"
        >
          <h4 className="text-lg font-medium text-gray-900 flex items-center">
            <Settings className="h-5 w-5 mr-2" />
            Configure Parameters
          </h4>
          {showParameterTuner ? (
            <ChevronUp className="h-5 w-5 text-gray-400" />
          ) : (
            <ChevronDown className="h-5 w-5 text-gray-400" />
          )}
        </button>
        
        {showParameterTuner && (
          <div className="mt-4">
            <ChunkingParameterTuner 
              disabled={disabled}
              showPreview={false}
              onParameterChange={() => onStrategyChange?.(selectedStrategy)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

// Helper function to provide technical details for each strategy
function getStrategyDetails(strategy: ChunkingStrategyType): string {
  const details: Record<ChunkingStrategyType, string> = {
    character: 'Splits text at fixed character intervals. Simple and fast but may break words or sentences. Best for uniform content where context preservation is less critical.',
    recursive: 'Intelligently splits at natural boundaries (sentences, paragraphs) while respecting size limits. Tries multiple separators in order of preference. Ideal for most text documents.',
    markdown: 'Parses Markdown structure and splits at logical boundaries like headers and sections. Preserves document hierarchy and includes parent headers for context.',
    semantic: 'Uses AI embeddings to identify topic changes and natural breakpoints. Computes semantic similarity between sentences to find optimal split points. Produces the highest quality chunks but requires more processing.',
    hierarchical: 'Creates multiple levels of chunks with parent-child relationships. Larger parent chunks provide context while smaller child chunks enable precise retrieval. Excellent for long, structured documents.',
    hybrid: 'Automatically analyzes content and selects the best strategy. Uses heuristics to detect document type, structure, and content patterns. Perfect when you\'re unsure which strategy to use.'
  };
  
  return details[strategy];
}