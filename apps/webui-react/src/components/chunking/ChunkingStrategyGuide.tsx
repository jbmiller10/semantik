import { useState } from 'react';
import { X, CheckCircle, Info } from 'lucide-react';
import { CHUNKING_STRATEGIES } from '../../types/chunking';
import type { ChunkingStrategyType } from '../../types/chunking';

interface ChunkingStrategyGuideProps {
  onClose: () => void;
  currentStrategy?: ChunkingStrategyType;
  fileType?: string;
}

const performanceLabels = {
  speed: { fast: 'Fast', medium: 'Medium', slow: 'Slow' },
  quality: { basic: 'Basic', good: 'Good', excellent: 'Excellent' },
  memoryUsage: { low: 'Low', medium: 'Medium', high: 'High' }
};

const performanceColors = {
  speed: { fast: 'text-green-600', medium: 'text-yellow-600', slow: 'text-red-600' },
  quality: { basic: 'text-gray-600', good: 'text-blue-600', excellent: 'text-purple-600' },
  memoryUsage: { low: 'text-green-600', medium: 'text-yellow-600', high: 'text-red-600' }
};

export function ChunkingStrategyGuide({ 
  onClose, 
  currentStrategy, 
  fileType 
}: ChunkingStrategyGuideProps) {
  const [selectedTab, setSelectedTab] = useState<'comparison' | 'examples'>('comparison');

  // Prevent body scroll when modal is open
  useState(() => {
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  });

  const getFileTypeRecommendation = (strategy: ChunkingStrategyType): boolean => {
    if (!fileType) return false;
    const strategyInfo = CHUNKING_STRATEGIES[strategy];
    return strategyInfo.supportedFileTypes.includes('*') || 
           strategyInfo.supportedFileTypes.includes(fileType);
  };

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-5xl w-full max-h-[90vh] overflow-hidden shadow-xl">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">Chunking Strategy Guide</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex">
            <button
              onClick={() => setSelectedTab('comparison')}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                selectedTab === 'comparison'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Strategy Comparison
            </button>
            <button
              onClick={() => setSelectedTab('examples')}
              className={`px-6 py-3 text-sm font-medium border-b-2 transition-colors ${
                selectedTab === 'examples'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              Visual Examples
            </button>
          </nav>
        </div>

        {/* Content */}
        <div className="overflow-y-auto" style={{ maxHeight: 'calc(90vh - 140px)' }}>
          {selectedTab === 'comparison' ? (
            <div className="p-6">
              {/* Quick Recommendation */}
              <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h3 className="font-medium text-blue-900 mb-2 flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  Quick Recommendation
                </h3>
                <p className="text-sm text-blue-700">
                  <strong>Not sure which to choose?</strong> Use <span className="font-semibold">Hybrid Auto-Select</span> — 
                  it automatically picks the best strategy based on your content type. Perfect for most use cases!
                </p>
              </div>

              {/* Comparison Table */}
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <th className="text-left py-3 px-4 font-medium text-gray-900">Strategy</th>
                      <th className="text-center py-3 px-4 font-medium text-gray-900">Speed</th>
                      <th className="text-center py-3 px-4 font-medium text-gray-900">Quality</th>
                      <th className="text-center py-3 px-4 font-medium text-gray-900">Memory</th>
                      <th className="text-left py-3 px-4 font-medium text-gray-900">Best For</th>
                      <th className="text-center py-3 px-4 font-medium text-gray-900">Recommended</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(Object.entries(CHUNKING_STRATEGIES) as [ChunkingStrategyType, typeof CHUNKING_STRATEGIES[ChunkingStrategyType]][]).map(
                      ([strategyType, strategy], index) => {
                        const isCurrentStrategy = strategyType === currentStrategy;
                        const isRecommendedForFile = getFileTypeRecommendation(strategyType);
                        const isRecommended = strategy.isRecommended || strategyType === 'hybrid';

                        return (
                          <tr 
                            key={strategyType} 
                            className={`border-b border-gray-100 ${
                              isCurrentStrategy ? 'bg-blue-50' : index % 2 === 0 ? 'bg-gray-50' : 'bg-white'
                            }`}
                          >
                            <td className="py-4 px-4">
                              <div>
                                <p className="font-medium text-gray-900">{strategy.name}</p>
                                <p className="text-xs text-gray-500 mt-1">{strategy.description}</p>
                              </div>
                            </td>
                            <td className="py-4 px-4 text-center">
                              <span className={`text-sm font-medium ${performanceColors.speed[strategy.performance.speed]}`}>
                                {performanceLabels.speed[strategy.performance.speed]}
                              </span>
                            </td>
                            <td className="py-4 px-4 text-center">
                              <span className={`text-sm font-medium ${performanceColors.quality[strategy.performance.quality]}`}>
                                {performanceLabels.quality[strategy.performance.quality]}
                              </span>
                            </td>
                            <td className="py-4 px-4 text-center">
                              <span className={`text-sm font-medium ${performanceColors.memoryUsage[strategy.performance.memoryUsage]}`}>
                                {performanceLabels.memoryUsage[strategy.performance.memoryUsage]}
                              </span>
                            </td>
                            <td className="py-4 px-4">
                              <p className="text-sm text-gray-600">
                                {strategy.recommendedFor?.join(', ') || 'General use'}
                              </p>
                            </td>
                            <td className="py-4 px-4 text-center">
                              {(isRecommended || isRecommendedForFile) && (
                                <CheckCircle className="h-5 w-5 text-green-500 mx-auto" />
                              )}
                            </td>
                          </tr>
                        );
                      }
                    )}
                  </tbody>
                </table>
              </div>

              {/* File Type Specific Recommendations */}
              {fileType && (
                <div className="mt-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <h3 className="font-medium text-gray-900 mb-2">
                    Recommendations for {fileType.toUpperCase()} files
                  </h3>
                  <div className="space-y-2 text-sm text-gray-600">
                    {fileType === 'md' && (
                      <p>• <strong>Markdown-aware</strong> is ideal for preserving document structure</p>
                    )}
                    {fileType === 'pdf' && (
                      <p>• <strong>Semantic</strong> works best for complex PDFs with mixed content</p>
                    )}
                    {['py', 'js', 'ts', 'jsx', 'tsx'].includes(fileType) && (
                      <p>• <strong>Markdown-aware</strong> handles code files effectively</p>
                    )}
                    <p>• <strong>Hybrid Auto-Select</strong> will automatically choose the best strategy</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">How Each Strategy Chunks Text</h3>
              
              {/* Visual Examples */}
              <div className="space-y-6">
                {/* Character-based Example */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">Character-based Chunking</h4>
                  <div className="bg-gray-50 p-4 rounded text-xs space-y-3">
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 1 (50 chars)</span>
                      <div className="bg-blue-100 p-2 rounded border border-blue-200">
                        <span className="font-mono">Machine learning algorithms analyze patterns in dat</span>
                      </div>
                    </div>
                    <div className="border-t-2 border-dashed border-gray-300 -my-1"></div>
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 2 (50 chars)</span>
                      <div className="bg-green-100 p-2 rounded border border-green-200">
                        <span className="font-mono">a to make predictions. Deep neural networks have r</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    Splits at exactly 50 characters, breaking words mid-sentence.
                  </p>
                </div>

                {/* Recursive Example */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">Recursive Chunking</h4>
                  <div className="bg-gray-50 p-4 rounded text-xs space-y-3">
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 1</span>
                      <div className="bg-blue-100 p-2 rounded border border-blue-200">
                        <span className="font-mono">Machine learning algorithms analyze patterns in data to make predictions.</span>
                      </div>
                    </div>
                    <div className="border-t-2 border-dashed border-gray-300 -my-1"></div>
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 2</span>
                      <div className="bg-green-100 p-2 rounded border border-green-200">
                        <span className="font-mono">Deep neural networks have revolutionized computer vision and natural language processing.</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    Splits at sentence boundaries, preserving complete thoughts.
                  </p>
                </div>

                {/* Semantic Example */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">Semantic Chunking</h4>
                  <div className="bg-gray-50 p-4 rounded text-xs space-y-3">
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 1: Machine Learning</span>
                      <div className="bg-blue-100 p-2 rounded border border-blue-200">
                        <span className="font-mono">Machine learning algorithms analyze patterns in data to make predictions. Deep neural networks have revolutionized computer vision. Training models requires large datasets and computational resources.</span>
                      </div>
                    </div>
                    <div className="border-t-2 border-dashed border-gray-300 -my-1"></div>
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 2: Data Privacy</span>
                      <div className="bg-green-100 p-2 rounded border border-green-200">
                        <span className="font-mono">Data privacy is crucial when handling user information. GDPR regulations require explicit consent for data processing. Companies must implement strong security measures to protect sensitive data.</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    Groups sentences by topic, keeping related concepts together.
                  </p>
                </div>

                {/* Markdown Example */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">Markdown-aware Chunking</h4>
                  <div className="bg-gray-50 p-4 rounded text-xs space-y-3">
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 1</span>
                      <div className="bg-blue-100 p-2 rounded border border-blue-200 font-mono">
                        <div className="font-bold">## Getting Started</div>
                        <div className="mt-1">1. Install dependencies with `npm install`</div>
                        <div>2. Set up your environment variables</div>
                        <div>3. Run `npm run dev` to start the server</div>
                      </div>
                    </div>
                    <div className="border-t-2 border-dashed border-gray-300 -my-1"></div>
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 2</span>
                      <div className="bg-green-100 p-2 rounded border border-green-200 font-mono">
                        <div className="font-bold">## Configuration</div>
                        <div className="mt-1">The app uses environment variables for configuration.</div>
                        <div>Create a `.env` file based on `.env.example`</div>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    Keeps headers with their content, respecting document structure.
                  </p>
                </div>

                {/* Hierarchical Example */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">Hierarchical Chunking</h4>
                  <div className="bg-gray-50 p-4 rounded text-xs space-y-3">
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Parent Chunk</span>
                      <div className="bg-purple-100 p-2 rounded border border-purple-200">
                        <span className="font-mono">Chapter 3: Advanced Techniques. This chapter covers optimization strategies...</span>
                        <div className="mt-2 ml-4 space-y-2">
                          <div className="bg-purple-50 p-1 rounded border-l-2 border-purple-300 pl-2">
                            <span className="text-xs text-gray-600">Child 1:</span> Performance optimization
                          </div>
                          <div className="bg-purple-50 p-1 rounded border-l-2 border-purple-300 pl-2">
                            <span className="text-xs text-gray-600">Child 2:</span> Memory management
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    Creates parent-child relationships for better context retention.
                  </p>
                </div>

                {/* Hybrid Example with Overlap */}
                <div className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3">Chunk Overlap Visualization</h4>
                  <div className="bg-gray-50 p-4 rounded text-xs">
                    <div className="relative">
                      <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 1</span>
                      <div className="bg-blue-100 p-2 rounded-t border border-blue-200 border-b-0">
                        <span className="font-mono">Neural networks consist of layers of interconnected nodes. Each node processes input signals and produces an output.</span>
                      </div>
                      <div className="bg-gradient-to-b from-blue-100 to-green-100 p-2 border-x border-blue-200">
                        <span className="font-mono text-gray-700">The output is passed through an activation function.</span>
                        <span className="text-xs text-gray-500 ml-2">[Overlap: 100 chars]</span>
                      </div>
                      <div className="relative">
                        <span className="absolute -top-2 left-2 text-xs font-medium text-gray-500 bg-gray-50 px-1">Chunk 2</span>
                        <div className="bg-green-100 p-2 rounded-b border border-green-200 border-t-0">
                          <span className="font-mono">Common activation functions include ReLU, sigmoid, and tanh.</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 mt-3">
                    Overlap ensures context continuity between chunks.
                  </p>
                </div>
              </div>

              {/* Tips */}
              <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <h4 className="font-medium text-yellow-900 mb-2 flex items-center gap-2">
                  <Info className="h-5 w-5" />
                  Pro Tips
                </h4>
                <ul className="space-y-1 text-sm text-yellow-700">
                  <li>• Larger chunks preserve more context but may reduce search precision</li>
                  <li>• Smaller chunks improve search accuracy but may lose context</li>
                  <li>• Overlap helps maintain continuity between chunks</li>
                  <li>• Test different strategies with your specific content for best results</li>
                </ul>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}