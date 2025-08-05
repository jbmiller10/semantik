import { useState, useEffect } from 'react';
import { 
  GitCompare,
  Plus,
  X,
  Download,
  BarChart3,
  Target,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle
} from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';
import { CHUNKING_STRATEGIES } from '../../types/chunking';
import type { ChunkingStrategyType } from '../../types/chunking';

interface ChunkingComparisonViewProps {
  document?: { id?: string; content?: string; name?: string };
  maxStrategies?: number;
}

export function ChunkingComparisonView({ 
  document,
  maxStrategies = 3 
}: ChunkingComparisonViewProps) {
  const {
    comparisonStrategies,
    comparisonResults,
    comparisonLoading,
    comparisonError,
    addComparisonStrategy,
    removeComparisonStrategy,
    compareStrategies,
    selectedStrategy
  } = useChunkingStore();

  const [showAddStrategy, setShowAddStrategy] = useState(false);
  const [syncScroll, setSyncScroll] = useState(true);
  const [exportFormat, setExportFormat] = useState<'json' | 'csv'>('json');

  // Auto-compare when strategies change
  useEffect(() => {
    if (document && comparisonStrategies.length > 0) {
      compareStrategies();
    }
  }, [comparisonStrategies, document, compareStrategies]);

  const availableStrategies = Object.keys(CHUNKING_STRATEGIES).filter(
    strategy => !comparisonStrategies.includes(strategy as ChunkingStrategyType)
  ) as ChunkingStrategyType[];

  const handleAddStrategy = (strategy: ChunkingStrategyType) => {
    addComparisonStrategy(strategy);
    setShowAddStrategy(false);
  };

  const handleExport = () => {
    const data = Array.from(comparisonResults.entries()).map(([strategy, result]) => ({
      strategy,
      ...result.preview.statistics,
      processingTime: result.preview.performance.processingTimeMs,
      score: result.score
    }));

    if (exportFormat === 'json') {
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = `chunking-comparison-${new Date().toISOString()}.json`;
      a.click();
    } else {
      // CSV export
      const headers = ['Strategy', 'Total Chunks', 'Avg Size', 'Min Size', 'Max Size', 'Processing Time (ms)', 'Quality Score', 'Performance Score'];
      const rows = data.map(d => [
        d.strategy,
        d.totalChunks,
        d.avgChunkSize,
        d.minChunkSize,
        d.maxChunkSize,
        d.processingTime,
        d.score?.quality || 'N/A',
        d.score?.performance || 'N/A'
      ]);
      
      const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = `chunking-comparison-${new Date().toISOString()}.csv`;
      a.click();
    }
  };

  const handleSyncScroll = (e: React.UIEvent<HTMLDivElement>, strategyIndex: number) => {
    if (!syncScroll) return;

    const scrollTop = e.currentTarget.scrollTop;
    const scrollContainers = window.document.querySelectorAll('.comparison-scroll-container');
    scrollContainers.forEach((container: Element, index: number) => {
      if (index !== strategyIndex && container instanceof HTMLElement) {
        container.scrollTop = scrollTop;
      }
    });
  };

  const getScoreTrend = (score: number): React.ReactElement => {
    if (score >= 80) return <TrendingUp className="h-4 w-4 text-green-600" />;
    if (score >= 60) return <Minus className="h-4 w-4 text-yellow-600" />;
    return <TrendingDown className="h-4 w-4 text-red-600" />;
  };

  if (!document) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center">
          <GitCompare className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-sm text-gray-600">Select a document to compare chunking strategies</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900 flex items-center">
          <GitCompare className="h-5 w-5 mr-2" />
          Strategy Comparison
        </h3>
        
        <div className="flex items-center space-x-2">
          {/* Sync Scroll Toggle */}
          <button
            onClick={() => setSyncScroll(!syncScroll)}
            className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
              syncScroll 
                ? 'bg-blue-100 text-blue-700' 
                : 'bg-gray-100 text-gray-700'
            }`}
          >
            Sync Scroll: {syncScroll ? 'ON' : 'OFF'}
          </button>

          {/* Export Button */}
          {comparisonResults.size > 0 && (
            <div className="flex items-center space-x-1">
              <select
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value as 'json' | 'csv')}
                className="text-sm border-gray-300 rounded-md"
              >
                <option value="json">JSON</option>
                <option value="csv">CSV</option>
              </select>
              <button
                onClick={handleExport}
                className="px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-700 flex items-center"
              >
                <Download className="h-4 w-4 mr-1" />
                Export
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Strategy Selection */}
      <div className="flex items-center space-x-2 flex-wrap">
        {comparisonStrategies.map(strategy => (
          <div
            key={strategy}
            className="flex items-center space-x-2 bg-blue-50 border border-blue-200 rounded-lg px-3 py-1"
          >
            <span className="text-sm font-medium text-blue-700">
              {CHUNKING_STRATEGIES[strategy].name}
            </span>
            <button
              onClick={() => removeComparisonStrategy(strategy)}
              className="text-blue-600 hover:text-blue-800"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ))}
        
        {comparisonStrategies.length < maxStrategies && (
          <div className="relative">
            {showAddStrategy ? (
              <div className="absolute z-10 mt-1 bg-white rounded-md shadow-lg border border-gray-200">
                <div className="py-1">
                  {availableStrategies.map(strategy => (
                    <button
                      key={strategy}
                      onClick={() => handleAddStrategy(strategy)}
                      className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    >
                      {CHUNKING_STRATEGIES[strategy].name}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <button
                onClick={() => setShowAddStrategy(true)}
                className="flex items-center space-x-1 px-3 py-1 text-sm font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
              >
                <Plus className="h-4 w-4" />
                <span>Add Strategy</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* Comparison Results */}
      {comparisonLoading && (
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
          <div className="text-center">
            <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="text-sm text-gray-600">Comparing strategies...</p>
          </div>
        </div>
      )}

      {comparisonError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-700">{comparisonError}</p>
        </div>
      )}

      {comparisonResults.size > 0 && !comparisonLoading && (
        <div className="space-y-6">
          {/* Metrics Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Array.from(comparisonResults.entries()).map(([strategy, result]) => {
              const isCurrentStrategy = strategy === selectedStrategy;
              const stats = result.preview.statistics;
              const perf = result.preview.performance;
              const score = result.score;

              return (
                <div
                  key={strategy}
                  className={`bg-white rounded-lg border-2 p-4 ${
                    isCurrentStrategy ? 'border-blue-500' : 'border-gray-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium text-gray-900">
                      {CHUNKING_STRATEGIES[strategy].name}
                    </h4>
                    {isCurrentStrategy && (
                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
                        Current
                      </span>
                    )}
                  </div>

                  {/* Key Metrics */}
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center">
                        <BarChart3 className="h-4 w-4 mr-1" />
                        Chunks
                      </span>
                      <span className="font-medium">{stats.totalChunks}</span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center">
                        <Target className="h-4 w-4 mr-1" />
                        Avg Size
                      </span>
                      <span className="font-medium">{stats.avgChunkSize} chars</span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-gray-600 flex items-center">
                        <Clock className="h-4 w-4 mr-1" />
                        Time
                      </span>
                      <span className="font-medium">{perf.processingTimeMs}ms</span>
                    </div>

                    {score && (
                      <>
                        <div className="border-t pt-2 mt-2">
                          <div className="flex items-center justify-between">
                            <span className="text-gray-600">Quality</span>
                            <div className="flex items-center space-x-1">
                              <span className="font-medium">{score.quality.toFixed(0)}%</span>
                              {getScoreTrend(score.quality)}
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between mt-1">
                            <span className="text-gray-600">Performance</span>
                            <div className="flex items-center space-x-1">
                              <span className="font-medium">{score.performance.toFixed(0)}%</span>
                              {getScoreTrend(score.performance)}
                            </div>
                          </div>
                        </div>
                      </>
                    )}
                  </div>

                  {/* Winner Badge */}
                  {score && score.overall >= 85 && (
                    <div className="mt-3 flex items-center justify-center text-green-600">
                      <CheckCircle className="h-5 w-5 mr-1" />
                      <span className="text-sm font-medium">Recommended</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Detailed Comparison Table */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h4 className="text-sm font-medium text-gray-900">Detailed Metrics</h4>
            </div>
            
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                      Metric
                    </th>
                    {Array.from(comparisonResults.keys()).map(strategy => (
                      <th key={strategy} className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        {CHUNKING_STRATEGIES[strategy].name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  <tr>
                    <td className="px-4 py-2 text-sm text-gray-600">Total Chunks</td>
                    {Array.from(comparisonResults.values()).map((result, i) => (
                      <td key={i} className="px-4 py-2 text-sm font-medium">
                        {result.preview.statistics.totalChunks}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="px-4 py-2 text-sm text-gray-600">Average Size</td>
                    {Array.from(comparisonResults.values()).map((result, i) => (
                      <td key={i} className="px-4 py-2 text-sm font-medium">
                        {result.preview.statistics.avgChunkSize} chars
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="px-4 py-2 text-sm text-gray-600">Size Range</td>
                    {Array.from(comparisonResults.values()).map((result, i) => (
                      <td key={i} className="px-4 py-2 text-sm font-medium">
                        {result.preview.statistics.minChunkSize} - {result.preview.statistics.maxChunkSize}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="px-4 py-2 text-sm text-gray-600">Processing Time</td>
                    {Array.from(comparisonResults.values()).map((result, i) => (
                      <td key={i} className="px-4 py-2 text-sm font-medium">
                        {result.preview.performance.processingTimeMs}ms
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="px-4 py-2 text-sm text-gray-600">Est. Full Time</td>
                    {Array.from(comparisonResults.values()).map((result, i) => (
                      <td key={i} className="px-4 py-2 text-sm font-medium">
                        {(result.preview.performance.estimatedFullProcessingTimeMs / 1000).toFixed(1)}s
                      </td>
                    ))}
                  </tr>
                  {comparisonResults.values().next().value?.preview.statistics.overlapPercentage !== undefined && (
                    <tr>
                      <td className="px-4 py-2 text-sm text-gray-600">Overlap %</td>
                      {Array.from(comparisonResults.values()).map((result, i) => (
                        <td key={i} className="px-4 py-2 text-sm font-medium">
                          {result.preview.statistics.overlapPercentage || 0}%
                        </td>
                      ))}
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Chunk Preview Comparison */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h4 className="text-sm font-medium text-gray-900">Chunk Preview Comparison</h4>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 divide-x divide-gray-200">
              {Array.from(comparisonResults.entries()).map(([strategy, result], index) => (
                <div key={strategy} className="p-4">
                  <h5 className="text-sm font-medium text-gray-700 mb-3">
                    {CHUNKING_STRATEGIES[strategy].name}
                  </h5>
                  <div 
                    className="comparison-scroll-container h-64 overflow-y-auto space-y-2"
                    onScroll={(e) => handleSyncScroll(e, index)}
                  >
                    {result.preview.chunks.slice(0, 5).map((chunk, i) => (
                      <div key={chunk.id} className="p-3 bg-gray-50 rounded-md">
                        <div className="text-xs text-gray-500 mb-1">Chunk {i + 1}</div>
                        <p className="text-sm text-gray-700 line-clamp-3">
                          {chunk.content}
                        </p>
                      </div>
                    ))}
                    {result.preview.chunks.length > 5 && (
                      <p className="text-sm text-gray-500 text-center py-2">
                        ... and {result.preview.chunks.length - 5} more chunks
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}