import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children?: ReactNode;
  /** Name of the section for error messages */
  sectionName: string;
  /** Callback when retry is clicked */
  onRetry?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * Error boundary specialized for settings sections.
 * Provides a compact error UI that fits within collapsible sections.
 */
class SectionErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`[SectionErrorBoundary] Error in ${this.props.sectionName}:`, {
      error,
      errorInfo,
      timestamp: new Date().toISOString(),
    });
    this.setState({ errorInfo });
  }

  private handleReset = () => {
    this.props.onRetry?.();
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  public render() {
    const { hasError, error } = this.state;
    const { children, sectionName } = this.props;

    if (hasError && error) {
      return (
        <div className="p-4 bg-red-50 rounded-lg border border-red-200">
          <div className="flex items-start space-x-3">
            <AlertTriangle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div className="flex-1 space-y-3">
              <div>
                <h4 className="text-sm font-medium text-gray-900">
                  Failed to load {sectionName}
                </h4>
                <p className="text-sm text-gray-600 mt-1">
                  {error.message || 'An unexpected error occurred'}
                </p>
              </div>

              <details className="text-xs text-gray-500">
                <summary className="cursor-pointer hover:text-gray-700">
                  Technical details
                </summary>
                <pre className="mt-2 p-2 bg-gray-100 rounded overflow-auto max-h-24 text-xs">
                  {error.stack}
                </pre>
              </details>

              <button
                onClick={this.handleReset}
                className="flex items-center space-x-1 px-3 py-1.5 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>Try Again</span>
              </button>
            </div>
          </div>
        </div>
      );
    }

    return children;
  }
}

export default SectionErrorBoundary;
