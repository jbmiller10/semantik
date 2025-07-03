import { useState } from 'react';
import type { Job } from '../stores/jobsStore';
import { useUIStore } from '../stores/uiStore';
import { useJobProgress } from '../hooks/useJobProgress';
import { jobsApi } from '../services/api';

interface JobCardProps {
  job: Job;
  onDelete: () => void;
}

function JobCard({ job, onDelete }: JobCardProps) {
  const { addToast, setShowJobMetricsModal } = useUIStore();
  const [isCancelling, setIsCancelling] = useState(false);
  
  // Use WebSocket hook for active jobs
  const isActive = job.status === 'processing' || job.status === 'waiting';
  useJobProgress(job.id, isActive);

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this job?')) {
      return;
    }

    try {
      await jobsApi.delete(job.id);
      addToast({ type: 'success', message: 'Job deleted successfully' });
      onDelete();
    } catch (error: any) {
      addToast({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to delete job',
      });
    }
  };

  const handleCancel = async () => {
    if (!confirm(`Are you sure you want to cancel "${job.collection_name}"?\n\nThis action cannot be undone.`)) {
      return;
    }

    setIsCancelling(true);
    try {
      const response = await jobsApi.cancel(job.id);
      addToast({ type: 'warning', message: response.data.message || 'Job cancellation requested' });
      // The WebSocket connection will update the job status
    } catch (error: any) {
      addToast({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to cancel job',
      });
    } finally {
      setIsCancelling(false);
    }
  };

  const handleSearch = () => {
    // Navigate to search tab with this collection selected
    useUIStore.getState().setActiveTab('search');
    // TODO: Set the collection in search params
  };

  const isRunning = job.status === 'processing' || job.status === 'waiting';

  return (
    <div className={`border rounded-lg p-4 transition-all duration-300 ${
      isRunning 
        ? 'border-blue-300 shadow-lg job-card-running' 
        : 'border-gray-200 shadow hover:shadow-md'
    }`}>
      {/* Header Section */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-bold text-gray-900">{job.collection_name}</h3>
          <p className="text-sm text-gray-600 mt-0.5">{job.directory_path || job.directory}</p>
        </div>
        <div className="flex items-center space-x-2">
          {/* Status Indicator Dot */}
          <div className="relative">
            <span className={`h-3 w-3 rounded-full inline-block ${
              job.status === 'processing' ? 'bg-blue-500' : 
              job.status === 'waiting' ? 'bg-yellow-500' :
              job.status === 'completed' ? 'bg-green-500' :
              job.status === 'failed' ? 'bg-red-500' :
              'bg-gray-500'
            }`}>
              {(job.status === 'processing' || job.status === 'waiting') && (
                <span className="absolute inset-0 rounded-full bg-current opacity-75 animate-ping"></span>
              )}
            </span>
          </div>
          {/* Status Badge */}
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
            job.status === 'processing' ? 'bg-blue-100 text-blue-800 animate-pulse' : 
            job.status === 'waiting' ? 'bg-yellow-100 text-yellow-800 animate-pulse' :
            job.status === 'completed' ? 'bg-green-100 text-green-800' :
            job.status === 'failed' ? 'bg-red-100 text-red-800' :
            job.status === 'cancelled' ? 'bg-gray-100 text-gray-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
          </span>
        </div>
      </div>

      {/* Statistics Grid */}
      <div className="grid grid-cols-3 gap-4 mb-3">
        <div>
          <p className="text-xs text-gray-500">Total Files</p>
          <p className="text-sm font-semibold text-gray-900">{job.total_files || 0}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Processed</p>
          <p className={`text-sm font-semibold ${isRunning ? 'text-blue-600' : 'text-gray-900'}`}>
            {job.processed_documents || job.processed_files || 0}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Failed</p>
          <p className={`text-sm font-semibold ${job.failed_files > 0 ? 'text-red-600' : 'text-gray-900'}`}>
            {job.failed_files || 0}
          </p>
        </div>
      </div>

      {/* Progress Section (for running jobs) */}
      {job.status === 'processing' && (
        <div className="mb-3">
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm text-gray-600">Progress</span>
            <span className="text-sm font-medium text-gray-900">{job.progress || 0}%</span>
          </div>
          <div className="relative w-full bg-gray-200 rounded-full h-2 overflow-hidden">
            <div 
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${job.progress || 0}%` }}
            >
              <div className="absolute inset-0 progress-shimmer"></div>
            </div>
          </div>
          {job.current_file && (
            <div className="mt-2 flex items-center text-sm text-gray-600">
              <svg className="animate-spin h-4 w-4 mr-2 text-blue-500" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span className="truncate">Processing: {job.current_file.split('/').pop()}</span>
            </div>
          )}
        </div>
      )}

      {/* Completion Indicator (for completed jobs) */}
      {job.status === 'completed' && (
        <div className="mb-3">
          <div className="w-full bg-green-500 rounded-full h-2"></div>
        </div>
      )}

      {/* Error Section */}
      {job.status === 'failed' && job.error && (
        <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
          {job.error}
        </div>
      )}

      {/* Metadata Grid */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs mb-3">
        <div>
          <span className="text-gray-500">Model:</span>
          <span className="ml-1 text-gray-700">{job.model_name || 'N/A'}</span>
        </div>
        <div>
          <span className="text-gray-500">Created:</span>
          <span className="ml-1 text-gray-700">{new Date(job.created_at).toLocaleDateString()}</span>
        </div>
        <div>
          <span className="text-gray-500">Vector Size:</span>
          <span className="ml-1 text-gray-700">{job.vector_dim || 'Auto'}</span>
        </div>
        <div>
          <span className="text-gray-500">Quantization:</span>
          <span className="ml-1 text-gray-700">{job.quantization || 'float32'}</span>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-end space-x-2 pt-3 border-t border-gray-200">
        {/* Monitor Button (for processing jobs) */}
        {job.status === 'processing' && (
          <button
            onClick={() => setShowJobMetricsModal(job.id)}
            className="px-3 py-1.5 text-sm font-medium text-blue-700 bg-blue-50 border border-blue-300 rounded-md hover:bg-blue-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            <svg className="inline-block h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Monitor
          </button>
        )}

        {/* Cancel Button (for processing/waiting jobs) */}
        {(job.status === 'processing' || job.status === 'waiting') && (
          <button
            onClick={handleCancel}
            disabled={isCancelling}
            className="px-3 py-1.5 text-sm font-medium text-red-700 bg-red-50 border border-red-300 rounded-md hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <svg className="inline-block h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
            Cancel
          </button>
        )}

        {/* Search Button (for completed jobs) */}
        {job.status === 'completed' && (
          <button
            onClick={handleSearch}
            className="px-3 py-1.5 text-sm font-medium text-green-700 bg-green-50 border border-green-300 rounded-md hover:bg-green-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
          >
            <svg className="inline-block h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Search
          </button>
        )}

        {/* Delete Button (always shown) */}
        <button
          onClick={handleDelete}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
        >
          <svg className="inline-block h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
          Delete
        </button>
      </div>
    </div>
  );
}

export default JobCard;