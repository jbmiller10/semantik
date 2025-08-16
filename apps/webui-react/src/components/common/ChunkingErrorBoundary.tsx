import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Settings, FileText } from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';

interface Props {
  children?: ReactNode;
  componentName?: string;
  onReset?: () => void;
  preserveConfiguration?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  savedConfiguration?: Record<string, unknown>;
}

class ChunkingErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    savedConfiguration: undefined
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { 
      hasError: true, 
      error
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`[ChunkingErrorBoundary] Error in ${this.props.componentName || 'chunking component'}:`, {
      error,
      errorInfo,
      timestamp: new Date().toISOString()
    });

    // Save current configuration if needed
    if (this.props.preserveConfiguration) {
      // Note: We can't use hooks in class components, so we'll need to pass this via props
      // or use a different pattern for saving configuration
      this.setState({ 
        errorInfo,
        savedConfiguration: { 
          // This would be populated from props or context
          timestamp: Date.now() 
        }
      });
    } else {
      this.setState({ errorInfo });
    }
  }

  private handleReset = () => {
    // Call custom reset handler if provided
    if (this.props.onReset) {
      this.props.onReset();
    }

    // Reset error state
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  private handleResetConfiguration = () => {
    // This would reset the chunking configuration to defaults
    // We'd need to pass this functionality via props
    console.log('Resetting chunking configuration to defaults');
    this.handleReset();
  };

  public render() {
    const { hasError, error, errorInfo } = this.state;
    const { children, componentName } = this.props;

    if (hasError && error) {
      return (
        <div className="w-full p-6 bg-red-50 rounded-lg border border-red-200">
          <div className="space-y-4">
            <div className="flex items-start space-x-3">
              <AlertTriangle className="h-6 w-6 text-red-500 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900">
                  Chunking Component Error
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  {componentName 
                    ? `The ${componentName} encountered an error.`
                    : 'An error occurred in the chunking component.'}
                </p>
              </div>
            </div>

            <div className="bg-white rounded-md p-3 border border-red-100">
              <p className="text-sm text-red-700 font-mono">
                {error.message}
              </p>
            </div>

            {error.message.includes('WebSocket') && (
              <div className="bg-blue-50 rounded-md p-3 border border-blue-200">
                <div className="flex items-start space-x-2">
                  <FileText className="h-4 w-4 text-blue-600 mt-0.5" />
                  <div className="text-sm text-blue-800">
                    <p className="font-medium">Connection Issue Detected</p>
                    <p className="mt-1">
                      This might be a temporary network issue. Try refreshing the preview or checking your connection.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {error.message.includes('configuration') && (
              <div className="bg-yellow-50 rounded-md p-3 border border-yellow-200">
                <div className="flex items-start space-x-2">
                  <Settings className="h-4 w-4 text-yellow-600 mt-0.5" />
                  <div className="text-sm text-yellow-800">
                    <p className="font-medium">Configuration Issue</p>
                    <p className="mt-1">
                      The current chunking configuration may be invalid. Try resetting to default settings.
                    </p>
                  </div>
                </div>
              </div>
            )}

            <details className="text-sm text-gray-500">
              <summary className="cursor-pointer hover:text-gray-700 font-medium">
                Show technical details
              </summary>
              <div className="mt-3 space-y-2">
                <div className="bg-gray-100 p-3 rounded text-xs">
                  <strong>Component:</strong> {componentName || 'Unknown'}
                </div>
                {errorInfo && (
                  <div className="bg-gray-100 p-3 rounded text-xs">
                    <strong>Component Stack:</strong>
                    <pre className="mt-1 overflow-auto max-h-32 text-gray-700">
                      {errorInfo.componentStack}
                    </pre>
                  </div>
                )}
                <div className="bg-gray-100 p-3 rounded text-xs">
                  <strong>Error Stack:</strong>
                  <pre className="mt-1 overflow-auto max-h-32 text-gray-700">
                    {error.stack}
                  </pre>
                </div>
              </div>
            </details>

            <div className="flex space-x-3">
              <button
                onClick={this.handleReset}
                className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Try Again</span>
              </button>
              
              {error.message.includes('configuration') && (
                <button
                  onClick={this.handleResetConfiguration}
                  className="flex items-center space-x-2 px-4 py-2 bg-gray-500 text-white rounded-md hover:bg-gray-600 transition-colors"
                >
                  <Settings className="h-4 w-4" />
                  <span>Reset Settings</span>
                </button>
              )}
            </div>
          </div>
        </div>
      );
    }

    return children;
  }
}

// Functional component wrapper to use with hooks
export function ChunkingErrorBoundaryWrapper({ 
  children, 
  componentName,
  preserveConfiguration = false 
}: { 
  children: ReactNode;
  componentName?: string;
  preserveConfiguration?: boolean;
}) {
  const resetChunkingState = useChunkingStore((state) => state.resetToDefaults);

  const handleReset = () => {
    if (preserveConfiguration) {
      // Preserve current configuration while resetting errors
      console.log('Preserving chunking configuration during reset');
    } else {
      // Reset to defaults
      resetChunkingState();
    }
  };

  return (
    <ChunkingErrorBoundary
      componentName={componentName}
      onReset={handleReset}
      preserveConfiguration={preserveConfiguration}
    >
      {children}
    </ChunkingErrorBoundary>
  );
}

export default ChunkingErrorBoundary;