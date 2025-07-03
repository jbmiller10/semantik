import { useState, useEffect } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useJobsStore } from '../stores/jobsStore';
import api from '../services/api';

interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  gpu_percent?: number;
  gpu_memory_used_gb?: number;
  gpu_memory_total_gb?: number;
}

interface MetricsHistory {
  timestamp: number;
  metrics: SystemMetrics;
}

function JobMetricsModal() {
  const { showJobMetricsModal, setShowJobMetricsModal } = useUIStore();
  const jobs = useJobsStore((state) => state.jobs);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [metricsHistory, setMetricsHistory] = useState<MetricsHistory[]>([]);
  const [averageWindow, setAverageWindow] = useState<5 | 10 | 30>(5);
  const [currentMetrics, setCurrentMetrics] = useState<SystemMetrics | null>(null);

  const job = jobs.find((j) => j.id === showJobMetricsModal);

  // Parse Prometheus metrics format
  const parsePrometheusMetrics = (prometheusText: string): SystemMetrics | null => {
    try {
      const lines = prometheusText.split('\n');
      const metrics: SystemMetrics = {
        cpu_percent: 0,
        memory_percent: 0,
        memory_used_gb: 0,
        memory_total_gb: 0,
      };

      for (const line of lines) {
        if (line.startsWith('#') || !line.trim()) continue;
        
        if (line.startsWith('embedding_cpu_utilization_percent')) {
          metrics.cpu_percent = parseFloat(line.split(' ')[1]) || 0;
        } else if (line.startsWith('embedding_memory_utilization_percent')) {
          metrics.memory_percent = parseFloat(line.split(' ')[1]) || 0;
        } else if (line.startsWith('embedding_memory_used_bytes')) {
          const bytes = parseFloat(line.split(' ')[1]) || 0;
          metrics.memory_used_gb = bytes / (1024 * 1024 * 1024);
        } else if (line.startsWith('embedding_memory_total_bytes')) {
          const bytes = parseFloat(line.split(' ')[1]) || 0;
          metrics.memory_total_gb = bytes / (1024 * 1024 * 1024);
        } else if (line.includes('embedding_gpu_utilization_percent')) {
          metrics.gpu_percent = parseFloat(line.split(' ')[1]) || 0;
        } else if (line.includes('embedding_gpu_memory_used_bytes')) {
          const bytes = parseFloat(line.split(' ')[1]) || 0;
          metrics.gpu_memory_used_gb = bytes / (1024 * 1024 * 1024);
        } else if (line.includes('embedding_gpu_memory_total_bytes')) {
          const bytes = parseFloat(line.split(' ')[1]) || 0;
          metrics.gpu_memory_total_gb = bytes / (1024 * 1024 * 1024);
        }
      }

      // If memory_used_gb wasn't set from bytes, calculate from percentage
      if (metrics.memory_used_gb === 0 && metrics.memory_percent > 0 && metrics.memory_total_gb > 0) {
        metrics.memory_used_gb = (metrics.memory_percent / 100) * metrics.memory_total_gb;
      }

      // If memory_total_gb is still 0, use a fallback
      if (metrics.memory_total_gb === 0) {
        metrics.memory_total_gb = 16; // Fallback to 16GB
        metrics.memory_used_gb = (metrics.memory_percent / 100) * metrics.memory_total_gb;
      }

      return metrics;
    } catch (error) {
      console.error('Error parsing Prometheus metrics:', error);
      return null;
    }
  };

  // Calculate rolling average
  const calculateAverage = (): SystemMetrics | null => {
    if (metricsHistory.length === 0) return currentMetrics;
    
    const cutoffTime = Date.now() - (averageWindow * 1000);
    const relevantMetrics = metricsHistory.filter(h => h.timestamp >= cutoffTime);
    
    if (relevantMetrics.length === 0) return currentMetrics;

    const avg: SystemMetrics = {
      cpu_percent: 0,
      memory_percent: 0,
      memory_used_gb: 0,
      memory_total_gb: relevantMetrics[0]?.metrics.memory_total_gb || 0,
    };

    relevantMetrics.forEach(h => {
      avg.cpu_percent += h.metrics.cpu_percent;
      avg.memory_percent += h.metrics.memory_percent;
      avg.memory_used_gb += h.metrics.memory_used_gb;
      if (h.metrics.gpu_percent !== undefined) {
        avg.gpu_percent = (avg.gpu_percent || 0) + h.metrics.gpu_percent;
      }
      if (h.metrics.gpu_memory_used_gb !== undefined) {
        avg.gpu_memory_used_gb = (avg.gpu_memory_used_gb || 0) + h.metrics.gpu_memory_used_gb;
      }
      if (h.metrics.gpu_memory_total_gb !== undefined) {
        avg.gpu_memory_total_gb = h.metrics.gpu_memory_total_gb;
      }
    });

    const count = relevantMetrics.length;
    avg.cpu_percent /= count;
    avg.memory_percent /= count;
    avg.memory_used_gb /= count;
    if (avg.gpu_percent !== undefined) avg.gpu_percent /= count;
    if (avg.gpu_memory_used_gb !== undefined) avg.gpu_memory_used_gb /= count;

    return avg;
  };

  // Fetch metrics
  useEffect(() => {
    if (!showJobMetricsModal || !autoRefresh) return;

    const fetchMetrics = async () => {
      try {
        const response = await api.get('/api/metrics');
        if (response.data.data) {
          const metrics = parsePrometheusMetrics(response.data.data);
          if (metrics) {
            setCurrentMetrics(metrics);
            setMetricsHistory(prev => {
              const newHistory = [...prev, { timestamp: Date.now(), metrics }];
              // Keep only last 30 seconds of history
              const cutoff = Date.now() - 30000;
              return newHistory.filter(h => h.timestamp >= cutoff);
            });
          }
        }
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 1000);
    return () => clearInterval(interval);
  }, [showJobMetricsModal, autoRefresh]);

  if (!showJobMetricsModal || !job) return null;

  const displayMetrics = calculateAverage() || currentMetrics;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 overflow-y-auto h-full w-full z-50">
      <div className="relative top-10 mx-auto p-4 w-full max-w-4xl">
        <div className="bg-white rounded-lg shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gray-100 px-6 py-4 flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">
              Job Monitor: {job.collection_name}
            </h2>
            <div className="flex items-center space-x-4">
              <label className="flex items-center text-sm text-gray-600">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="mr-2"
                />
                Auto-refresh
              </label>
              <button
                onClick={() => setShowJobMetricsModal(null)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Real-time Progress (Blue) */}
                <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                  <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
                    <svg className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Real-time Progress
                  </h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-blue-700">Files Progress</span>
                      <span className="text-sm font-semibold text-blue-900">
                        {job.processed_documents || job.processed_files || 0} / {job.total_documents || job.total_files || 0}
                      </span>
                    </div>
                    <div className="w-full bg-blue-200 rounded-full h-3">
                      <div 
                        className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${job.progress || 0}%` }}
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-blue-600">Status</p>
                        <p className="font-semibold text-blue-900">
                          {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                        </p>
                      </div>
                      <div>
                        <p className="text-blue-600">Progress</p>
                        <p className="font-semibold text-blue-900">
                          {job.progress || 0}%
                        </p>
                      </div>
                    </div>
                    {job.current_file && (
                      <div className="mt-2 p-2 bg-blue-100 rounded">
                        <p className="text-xs text-blue-700 truncate">
                          Current: {job.current_file.split('/').pop()}
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Job Information (Green) */}
                <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                  <h3 className="text-lg font-semibold text-green-900 mb-3 flex items-center">
                    <svg className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Job Information
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-green-700">Collection</span>
                      <span className="font-semibold text-green-900 truncate ml-2" title={job.collection_name}>
                        {job.collection_name}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-700">Model</span>
                      <span className="font-semibold text-green-900">
                        {job.model_name || 'Default'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-700">Created</span>
                      <span className="font-semibold text-green-900">
                        {new Date(job.created_at).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-green-700">Directory</span>
                      <span className="font-semibold text-green-900 truncate ml-2" title={job.directory}>
                        .../{job.directory?.split('/').pop() || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Resource Usage (Purple) */}
                <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                  <h3 className="text-lg font-semibold text-purple-900 mb-3 flex items-center justify-between">
                    <div className="flex items-center">
                      <svg className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                      </svg>
                      Resource Usage
                    </div>
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => setAverageWindow(5)}
                        className={`px-2 py-0.5 text-xs rounded ${averageWindow === 5 ? 'bg-purple-600 text-white' : 'bg-purple-200 text-purple-700 hover:bg-purple-300'}`}
                      >
                        5s
                      </button>
                      <button
                        onClick={() => setAverageWindow(10)}
                        className={`px-2 py-0.5 text-xs rounded ${averageWindow === 10 ? 'bg-purple-600 text-white' : 'bg-purple-200 text-purple-700 hover:bg-purple-300'}`}
                      >
                        10s
                      </button>
                      <button
                        onClick={() => setAverageWindow(30)}
                        className={`px-2 py-0.5 text-xs rounded ${averageWindow === 30 ? 'bg-purple-600 text-white' : 'bg-purple-200 text-purple-700 hover:bg-purple-300'}`}
                      >
                        30s
                      </button>
                    </div>
                  </h3>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-purple-700">CPU Usage</span>
                        <span className="text-sm font-semibold text-purple-900">
                          {displayMetrics?.cpu_percent?.toFixed(1) || '0'}%
                        </span>
                      </div>
                      <div className="w-full bg-purple-200 rounded-full h-2">
                        <div 
                          className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${displayMetrics?.cpu_percent || 0}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-purple-700">System Memory</span>
                        <span className="text-sm font-semibold text-purple-900">
                          {displayMetrics?.memory_used_gb?.toFixed(1) || '0'} / {displayMetrics?.memory_total_gb?.toFixed(0) || '0'} GB ({displayMetrics?.memory_percent?.toFixed(1) || '0'}%)
                        </span>
                      </div>
                      <div className="w-full bg-purple-200 rounded-full h-2">
                        <div 
                          className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${displayMetrics?.memory_percent || 0}%` }}
                        />
                      </div>
                    </div>
                    {displayMetrics?.gpu_percent !== undefined && (
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm text-purple-700">GPU Usage</span>
                          <span className="text-sm font-semibold text-purple-900">
                            {displayMetrics.gpu_percent.toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-purple-200 rounded-full h-2">
                          <div 
                            className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${displayMetrics.gpu_percent}%` }}
                          />
                        </div>
                      </div>
                    )}
                    {displayMetrics?.gpu_memory_used_gb !== undefined && displayMetrics?.gpu_memory_total_gb !== undefined && (
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm text-purple-700">VRAM Usage</span>
                          <span className="text-sm font-semibold text-purple-900">
                            {displayMetrics.gpu_memory_used_gb.toFixed(1)} / {displayMetrics.gpu_memory_total_gb.toFixed(0)} GB
                          </span>
                        </div>
                        <div className="w-full bg-purple-200 rounded-full h-2">
                          <div 
                            className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${(displayMetrics.gpu_memory_used_gb / displayMetrics.gpu_memory_total_gb) * 100}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Error Tracking (Red) */}
                <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                  <h3 className="text-lg font-semibold text-red-900 mb-3 flex items-center">
                    <svg className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Error Tracking
                  </h3>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-red-700">Failed Files</span>
                      <span className={`text-sm font-semibold ${job.failed_files > 0 ? 'text-red-900' : 'text-green-700'}`}>
                        {job.failed_files || 0}
                      </span>
                    </div>
                    {job.failed_files > 0 && (
                      <div className="mt-2 p-2 bg-red-100 rounded">
                        <p className="text-xs text-red-800">
                          Check logs for details about failed files
                        </p>
                      </div>
                    )}
                    {job.error && (
                      <div className="mt-2 p-2 bg-red-100 rounded">
                        <p className="text-xs text-red-800">{job.error}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

            {/* Job Configuration Details */}
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Job Configuration</h3>
              <div className="grid grid-cols-3 gap-4 text-xs">
                <div>
                  <span className="text-gray-500">Model:</span>
                  <span className="ml-1 text-gray-700">{job.model_name || 'Default'}</span>
                </div>
                <div>
                  <span className="text-gray-500">Chunk Size:</span>
                  <span className="ml-1 text-gray-700">{job.chunk_size || 600}</span>
                </div>
                <div>
                  <span className="text-gray-500">Batch Size:</span>
                  <span className="ml-1 text-gray-700">{job.batch_size || 96}</span>
                </div>
                <div>
                  <span className="text-gray-500">Quantization:</span>
                  <span className="ml-1 text-gray-700">{job.quantization || 'float32'}</span>
                </div>
                <div>
                  <span className="text-gray-500">Vector Dim:</span>
                  <span className="ml-1 text-gray-700">{job.vector_dim || 'Auto'}</span>
                </div>
                <div>
                  <span className="text-gray-500">Directory:</span>
                  <span className="ml-1 text-gray-700 truncate" title={job.directory}>
                    {job.directory}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default JobMetricsModal;