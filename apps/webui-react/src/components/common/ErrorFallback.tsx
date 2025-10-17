import { AlertTriangle, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';

interface ErrorFallbackProps {
  error: Error;
  resetError: () => void;
  componentName?: string;
  suggestion?: string;
  showDetails?: boolean;
}

export function ErrorFallback({ 
  error, 
  resetError, 
  componentName,
  suggestion,
  showDetails = true
}: ErrorFallbackProps) {
  const [detailsExpanded, setDetailsExpanded] = useState(false);

  return (
    <div className="w-full p-6 bg-red-50 rounded-lg border border-red-200">
      <div className="flex items-start space-x-3">
        <AlertTriangle className="h-6 w-6 text-red-500 mt-0.5 flex-shrink-0" />
        <div className="flex-1 space-y-3">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              {componentName ? `${componentName} Error` : 'Component Error'}
            </h3>
            <p className="text-sm text-gray-600 mt-1">
              {suggestion || 'An unexpected error occurred. Please try again.'}
            </p>
          </div>

          <div className="bg-white rounded-md p-3 border border-red-100">
            <p className="text-sm text-red-700 font-mono">
              {error.message}
            </p>
          </div>

          {showDetails && (
            <button
              onClick={() => setDetailsExpanded(!detailsExpanded)}
              className="flex items-center space-x-1 text-sm text-gray-600 hover:text-gray-800"
            >
              {detailsExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
              <span>{detailsExpanded ? 'Hide' : 'Show'} details</span>
            </button>
          )}

          {showDetails && detailsExpanded && (
            <div className="bg-gray-100 p-3 rounded text-xs">
              <pre className="overflow-auto max-h-48 text-gray-700">
                {error.stack}
              </pre>
            </div>
          )}

          <button
            onClick={resetError}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Try Again</span>
          </button>
        </div>
      </div>
    </div>
  );
}