import { useOperationProgress } from '../hooks/useOperationProgress';
import type { Operation } from '../types/collection';

// Helper to safely get source_path from config
function getSourcePath(config: Record<string, unknown> | undefined): string | null {
  if (!config || !('source_path' in config)) return null;
  const sourcePath = config.source_path;
  return typeof sourcePath === 'string' ? sourcePath : null;
}

interface OperationProgressProps {
  operation: Operation;
  className?: string;
  showDetails?: boolean;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

function OperationProgress({
  operation,
  className = '',
  showDetails = true,
  onComplete,
  onError,
}: OperationProgressProps) {
  // Connect to WebSocket for real-time updates
  const { isConnected } = useOperationProgress(operation.id, {
    onComplete,
    onError,
    showToasts: false, // Parent component can handle toasts
  });
  
  // Format operation type for display
  const getOperationTypeLabel = (type: string) => {
    switch (type) {
      case 'index':
        return 'Initial Indexing';
      case 'append':
        return 'Adding Data';
      case 'reindex':
        return 'Re-indexing';
      case 'remove_source':
        return 'Removing Source';
      case 'delete':
        return 'Deleting Collection';
      default:
        return type.charAt(0).toUpperCase() + type.slice(1);
    }
  };
  
  // Format status for display
  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'pending':
        return 'Queued';
      case 'processing':
        return 'Processing';
      case 'completed':
        return 'Completed';
      case 'failed':
        return 'Failed';
      case 'cancelled':
        return 'Cancelled';
      default:
        return status.charAt(0).toUpperCase() + status.slice(1);
    }
  };
  
  // Get status color classes
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'text-gray-600 bg-gray-100';
      case 'processing':
        return 'text-blue-600 bg-blue-100';
      case 'completed':
        return 'text-green-600 bg-green-100';
      case 'failed':
        return 'text-red-600 bg-red-100';
      case 'cancelled':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };
  
  // Format ETA
  const formatETA = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
  };
  
  return (
    <div className={`space-y-3 ${className}`}>
      {/* Operation Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <h4 className="text-sm font-medium text-gray-900">
            {getOperationTypeLabel(operation.type)}
          </h4>
          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(operation.status)}`}>
            {getStatusLabel(operation.status)}
          </span>
          {operation.status === 'processing' && isConnected && (
            <span className="inline-flex items-center text-xs text-gray-500">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-1"></div>
              Live
            </span>
          )}
        </div>
        
        {operation.eta && operation.status === 'processing' && (
          <span className="text-sm text-gray-500">
            ETA: {formatETA(operation.eta)}
          </span>
        )}
      </div>
      
      {/* Progress Bar (for processing operations) */}
      {operation.status === 'processing' && (
        <div>
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-600">Progress</span>
            <span className="text-xs font-medium text-gray-900">
              {operation.progress || 0}%
            </span>
          </div>
          <div className="relative w-full bg-gray-200 rounded-full h-2 overflow-hidden">
            <div 
              className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${operation.progress || 0}%` }}
            >
              <div className="absolute inset-0 progress-shimmer"></div>
            </div>
          </div>
        </div>
      )}
      
      {/* Operation Details */}
      {showDetails && (
        <div className="text-xs text-gray-600 space-y-1">
          {/* Source path from config */}
          {(() => {
            const sourcePath = getSourcePath(operation.config);
            return sourcePath ? (
              <div className="flex items-start">
                <span className="text-gray-500 mr-2">Source:</span>
                <span className="font-mono break-all">{sourcePath}</span>
              </div>
            ) : null;
          })()}
          
          {/* Error message for failed operations */}
          {operation.status === 'failed' && operation.error_message && (
            <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-red-700">
              <span className="font-medium">Error:</span> {operation.error_message}
            </div>
          )}
          
          {/* Timing information */}
          {operation.started_at && (
            <div className="flex items-center text-gray-500">
              <span>Started: {new Date(operation.started_at).toLocaleTimeString()}</span>
            </div>
          )}
          
          {operation.completed_at && (
            <div className="flex items-center text-gray-500">
              <span>Completed: {new Date(operation.completed_at).toLocaleTimeString()}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default OperationProgress;