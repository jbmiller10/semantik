import { useEffect, useState } from 'react';
import { 
  BarChart3,
  TrendingUp,
  TrendingDown,
  Minus,
  FileText,
  Zap,
  Target,
  AlertCircle,
  ChevronRight,
  RefreshCw,
  Download
} from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';
import { CHUNKING_STRATEGIES } from '../../types/chunking';
import type { ChunkingRecommendation } from '../../types/chunking';

interface ChunkingAnalyticsDashboardProps {
  onApplyRecommendation?: (recommendation: ChunkingRecommendation) => void;
}

export function ChunkingAnalyticsDashboard({ 
  onApplyRecommendation 
}: ChunkingAnalyticsDashboardProps) {
  const { 
    analyticsData, 
    analyticsLoading, 
    loadAnalytics,
    setStrategy,
    updateConfiguration
  } = useChunkingStore();
  
  const [selectedTimeRange, setSelectedTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [showDetails, setShowDetails] = useState<Record<string, boolean>>({});

  useEffect(() => {
    loadAnalytics();
  }, [loadAnalytics]);

  const handleRefresh = () => {
    loadAnalytics();
  };

  const handleExportAnalytics = () => {
    if (!analyticsData) return;

    const exportData = {
      exportDate: new Date().toISOString(),
      timeRange: selectedTimeRange,
      ...analyticsData
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chunking-analytics-${new Date().toISOString()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const applyRecommendation = (recommendation: ChunkingRecommendation) => {
    if (recommendation.action?.configuration) {
      setStrategy(recommendation.action.configuration.strategy);
      updateConfiguration(recommendation.action.configuration.parameters);
    }
    onApplyRecommendation?.(recommendation);
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'down':
        return <TrendingDown className="h-4 w-4 text-red-600" />;
      case 'stable':
        return <Minus className="h-4 w-4 text-gray-600" />;
    }
  };

  const getPriorityColor = (priority: 'high' | 'medium' | 'low') => {
    switch (priority) {
      case 'high':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'low':
        return 'text-blue-600 bg-blue-50 border-blue-200';
    }
  };

  if (analyticsLoading && !analyticsData) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
        <div className="text-center">
          <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-sm text-gray-600">Loading analytics...</p>
        </div>
      </div>
    );
  }

  if (!analyticsData) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
        <div className="text-center">
          <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-sm text-gray-600">No analytics data available</p>
          <button
            onClick={handleRefresh}
            className="mt-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            Load Analytics
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-gray-900 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2" />
          Chunking Analytics
        </h3>
        
        <div className="flex items-center space-x-2">
          {/* Time Range Selector */}
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value as '7d' | '30d' | '90d')}
            className="text-sm border-gray-300 rounded-md"
          >
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>

          {/* Action Buttons */}
          <button
            onClick={handleRefresh}
            disabled={analyticsLoading}
            className="p-2 text-gray-600 hover:text-gray-800 disabled:opacity-50"
            title="Refresh analytics"
          >
            <RefreshCw className={`h-4 w-4 ${analyticsLoading ? 'animate-spin' : ''}`} />
          </button>
          
          <button
            onClick={handleExportAnalytics}
            className="p-2 text-gray-600 hover:text-gray-800"
            title="Export analytics"
          >
            <Download className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Strategy Usage */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-200">
          <h4 className="text-sm font-medium text-gray-900">Strategy Usage</h4>
        </div>
        <div className="p-4">
          <div className="space-y-3">
            {analyticsData.strategyUsage.map(usage => {
              const strategy = CHUNKING_STRATEGIES[usage.strategy];
              const maxCount = Math.max(...analyticsData.strategyUsage.map(u => u.count));
              const widthPercent = (usage.count / maxCount) * 100;

              return (
                <div key={usage.strategy} className="space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-gray-700">
                      {strategy.name}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="text-gray-600">
                        {usage.count} ({usage.percentage}%)
                      </span>
                      {getTrendIcon(usage.trend)}
                    </div>
                  </div>
                  <div className="relative bg-gray-200 rounded-full h-2">
                    <div
                      className="absolute top-0 left-0 h-full bg-blue-600 rounded-full transition-all duration-300"
                      style={{ width: `${widthPercent}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-200">
          <h4 className="text-sm font-medium text-gray-900">Performance Metrics</h4>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Strategy
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  <div className="flex items-center">
                    <Zap className="h-3 w-3 mr-1" />
                    Avg Time
                  </div>
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  <div className="flex items-center">
                    <Target className="h-3 w-3 mr-1" />
                    Avg Chunks
                  </div>
                </th>
                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Success Rate
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {analyticsData.performanceMetrics.map(metric => (
                <tr key={metric.strategy}>
                  <td className="px-4 py-2 text-sm font-medium text-gray-900">
                    {CHUNKING_STRATEGIES[metric.strategy].name}
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-600">
                    {metric.avgProcessingTimeMs}ms
                  </td>
                  <td className="px-4 py-2 text-sm text-gray-600">
                    {metric.avgChunksPerDocument}
                  </td>
                  <td className="px-4 py-2 text-sm">
                    <span className={`font-medium ${
                      metric.successRate >= 95 ? 'text-green-600' : 
                      metric.successRate >= 90 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {metric.successRate.toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* File Type Distribution */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-4 py-3 border-b border-gray-200">
          <h4 className="text-sm font-medium text-gray-900">File Type Distribution</h4>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {analyticsData.fileTypeDistribution.slice(0, 8).map(fileType => (
              <div key={fileType.fileType} className="text-center">
                <div className="flex items-center justify-center h-12 w-12 mx-auto mb-2 bg-gray-100 rounded-lg">
                  <FileText className="h-6 w-6 text-gray-600" />
                </div>
                <p className="text-sm font-medium text-gray-900">.{fileType.fileType}</p>
                <p className="text-xs text-gray-600">{fileType.count} files</p>
                <p className="text-xs text-blue-600 mt-1">
                  {CHUNKING_STRATEGIES[fileType.preferredStrategy].name}
                </p>
              </div>
            ))}
          </div>
          {analyticsData.fileTypeDistribution.length > 8 && (
            <p className="text-sm text-gray-500 text-center mt-4">
              And {analyticsData.fileTypeDistribution.length - 8} more file types...
            </p>
          )}
        </div>
      </div>

      {/* Recommendations */}
      {analyticsData.recommendations.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-gray-900 flex items-center">
            <AlertCircle className="h-4 w-4 mr-2" />
            Recommendations
          </h4>
          
          {analyticsData.recommendations.map(recommendation => (
            <div
              key={recommendation.id}
              className={`rounded-lg border p-4 ${getPriorityColor(recommendation.priority)}`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <h5 className="font-medium mb-1">{recommendation.title}</h5>
                  <p className="text-sm opacity-90">{recommendation.description}</p>
                  
                  {showDetails[recommendation.id] && recommendation.action && (
                    <div className="mt-3 p-3 bg-white bg-opacity-50 rounded-md">
                      <p className="text-sm font-medium mb-1">Suggested Configuration:</p>
                      <pre className="text-xs font-mono">
                        {JSON.stringify(recommendation.action.configuration, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
                
                <div className="flex items-center space-x-2 ml-4">
                  {recommendation.action && (
                    <>
                      <button
                        onClick={() => setShowDetails({
                          ...showDetails,
                          [recommendation.id]: !showDetails[recommendation.id]
                        })}
                        className="text-sm font-medium hover:underline"
                      >
                        {showDetails[recommendation.id] ? 'Hide' : 'Details'}
                      </button>
                      <button
                        onClick={() => applyRecommendation(recommendation)}
                        className="flex items-center px-3 py-1 text-sm font-medium bg-white bg-opacity-70 rounded-md hover:bg-opacity-100 transition-colors"
                      >
                        Apply
                        <ChevronRight className="h-3 w-3 ml-1" />
                      </button>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <p className="text-sm text-blue-600 font-medium">Total Documents Processed</p>
          <p className="text-2xl font-bold text-blue-900 mt-1">
            {analyticsData.strategyUsage.reduce((sum, u) => sum + u.count, 0).toLocaleString()}
          </p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <p className="text-sm text-green-600 font-medium">Most Used Strategy</p>
          <p className="text-2xl font-bold text-green-900 mt-1">
            {CHUNKING_STRATEGIES[analyticsData.strategyUsage[0]?.strategy || 'recursive'].name}
          </p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <p className="text-sm text-purple-600 font-medium">Average Success Rate</p>
          <p className="text-2xl font-bold text-purple-900 mt-1">
            {(
              analyticsData.performanceMetrics.reduce((sum, m) => sum + m.successRate, 0) / 
              analyticsData.performanceMetrics.length
            ).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
}