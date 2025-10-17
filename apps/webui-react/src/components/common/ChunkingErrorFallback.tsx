import { 
  AlertTriangle, 
  RefreshCw, 
  Settings, 
  FileText, 
  Wifi, 
  WifiOff,
  Database,
  Zap
} from 'lucide-react';
import type { ReactNode } from 'react';

interface ChunkingErrorFallbackProps {
  error: Error;
  resetError: () => void;
  variant?: 'preview' | 'comparison' | 'analytics' | 'configuration';
  onResetConfiguration?: () => void;
  children?: ReactNode;
}

export function ChunkingErrorFallback({ 
  error, 
  resetError,
  variant = 'preview',
  onResetConfiguration
}: ChunkingErrorFallbackProps) {
  const isNetworkError = error.message.toLowerCase().includes('network') || 
                        error.message.toLowerCase().includes('websocket') ||
                        error.message.toLowerCase().includes('connection');
  
  const isConfigError = error.message.toLowerCase().includes('configuration') ||
                       error.message.toLowerCase().includes('invalid') ||
                       error.message.toLowerCase().includes('parameter');
  
  const isDataError = error.message.toLowerCase().includes('data') ||
                     error.message.toLowerCase().includes('parse') ||
                     error.message.toLowerCase().includes('format');

  const getIcon = () => {
    if (isNetworkError) return <WifiOff className="h-6 w-6 text-red-500" />;
    if (isConfigError) return <Settings className="h-6 w-6 text-orange-500" />;
    if (isDataError) return <Database className="h-6 w-6 text-yellow-500" />;
    return <AlertTriangle className="h-6 w-6 text-red-500" />;
  };

  const getTitle = () => {
    switch (variant) {
      case 'preview':
        return 'Preview Generation Failed';
      case 'comparison':
        return 'Strategy Comparison Failed';
      case 'analytics':
        return 'Analytics Loading Failed';
      case 'configuration':
        return 'Configuration Error';
      default:
        return 'Chunking Error';
    }
  };

  const getSuggestion = () => {
    if (isNetworkError) {
      return 'Check your connection and try again. The WebSocket connection may have been interrupted.';
    }
    if (isConfigError) {
      return 'The current configuration may be invalid. Try resetting to default settings.';
    }
    if (isDataError) {
      return 'The document format may not be supported or the data is corrupted.';
    }
    
    switch (variant) {
      case 'preview':
        return 'Unable to generate preview. Try selecting a different document or adjusting parameters.';
      case 'comparison':
        return 'Could not compare strategies. Ensure at least one strategy is selected.';
      case 'analytics':
        return 'Failed to load analytics data. This may be a temporary issue.';
      case 'configuration':
        return 'Invalid configuration detected. Reset to defaults or adjust parameters.';
      default:
        return 'An error occurred while processing. Please try again.';
    }
  };

  return (
    <div className="w-full p-6 bg-gradient-to-br from-red-50 to-orange-50 rounded-lg border border-red-200">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-start space-x-3">
          {getIcon()}
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900">
              {getTitle()}
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              {getSuggestion()}
            </p>
          </div>
        </div>

        {/* Error Message */}
        <div className="bg-white/80 backdrop-blur rounded-md p-3 border border-red-100">
          <div className="flex items-start space-x-2">
            <Zap className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
            <p className="text-sm text-red-700 font-mono flex-1">
              {error.message}
            </p>
          </div>
        </div>

        {/* Contextual Help */}
        {isNetworkError && (
          <div className="bg-blue-50 rounded-md p-3 border border-blue-200">
            <div className="flex items-start space-x-2">
              <Wifi className="h-4 w-4 text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-800">
                <p className="font-medium">Connection Tips:</p>
                <ul className="mt-1 ml-4 list-disc">
                  <li>Check if the backend services are running</li>
                  <li>Verify your network connection is stable</li>
                  <li>Try refreshing the page if the issue persists</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {isConfigError && (
          <div className="bg-yellow-50 rounded-md p-3 border border-yellow-200">
            <div className="flex items-start space-x-2">
              <Settings className="h-4 w-4 text-yellow-600 mt-0.5" />
              <div className="text-sm text-yellow-800">
                <p className="font-medium">Configuration Tips:</p>
                <ul className="mt-1 ml-4 list-disc">
                  <li>Chunk size should be between 100-10000 characters</li>
                  <li>Overlap should not exceed chunk size</li>
                  <li>Some strategies have specific requirements</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {variant === 'preview' && (
          <div className="bg-green-50 rounded-md p-3 border border-green-200">
            <div className="flex items-start space-x-2">
              <FileText className="h-4 w-4 text-green-600 mt-0.5" />
              <div className="text-sm text-green-800">
                <p className="font-medium">Alternative Actions:</p>
                <ul className="mt-1 ml-4 list-disc">
                  <li>Try a different chunking strategy</li>
                  <li>Select a smaller document for preview</li>
                  <li>Adjust the chunk size parameters</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex space-x-3">
          <button
            onClick={resetError}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Try Again</span>
          </button>
          
          {(isConfigError || variant === 'configuration') && onResetConfiguration && (
            <button
              onClick={onResetConfiguration}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
            >
              <Settings className="h-4 w-4" />
              <span>Reset to Defaults</span>
            </button>
          )}
        </div>

        {/* Technical Details (Collapsed by default) */}
        <details className="text-sm text-gray-500">
          <summary className="cursor-pointer hover:text-gray-700 font-medium">
            Technical details for developers
          </summary>
          <div className="mt-3 bg-gray-100 p-3 rounded text-xs">
            <strong>Error Type:</strong> {error.name}<br />
            <strong>Variant:</strong> {variant}<br />
            <strong>Timestamp:</strong> {new Date().toISOString()}<br />
            <strong>Stack Trace:</strong>
            <pre className="mt-2 overflow-auto max-h-32 text-gray-700">
              {error.stack}
            </pre>
          </div>
        </details>
      </div>
    </div>
  );
}

// Specialized variants for specific use cases
export function PreviewErrorFallback({ error, resetError }: { error: Error; resetError: () => void }) {
  return (
    <ChunkingErrorFallback 
      error={error} 
      resetError={resetError} 
      variant="preview" 
    />
  );
}

export function ComparisonErrorFallback({ error, resetError }: { error: Error; resetError: () => void }) {
  return (
    <ChunkingErrorFallback 
      error={error} 
      resetError={resetError} 
      variant="comparison" 
    />
  );
}

export function AnalyticsErrorFallback({ error, resetError }: { error: Error; resetError: () => void }) {
  return (
    <ChunkingErrorFallback 
      error={error} 
      resetError={resetError} 
      variant="analytics" 
    />
  );
}

export function ConfigurationErrorFallback({ 
  error, 
  resetError,
  onResetConfiguration 
}: { 
  error: Error; 
  resetError: () => void;
  onResetConfiguration?: () => void;
}) {
  return (
    <ChunkingErrorFallback 
      error={error} 
      resetError={resetError} 
      variant="configuration"
      onResetConfiguration={onResetConfiguration}
    />
  );
}