import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children?: ReactNode;
  fallback?: (error: Error, resetError: () => void) => ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  resetKeys?: Array<string | number>;
  resetOnPropsChange?: boolean;
  isolate?: boolean;
  level?: 'page' | 'section' | 'component';
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

class ErrorBoundary extends Component<Props, State> {
  private resetTimeoutId: NodeJS.Timeout | null = null;
  private previousResetKeys: Array<string | number> = [];

  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    errorId: ''
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    const errorId = `error-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
    return { 
      hasError: true, 
      error,
      errorId
    };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console with context
    console.error('[ErrorBoundary] Component error caught:', {
      error,
      errorInfo,
      errorId: this.state.errorId,
      level: this.props.level || 'component',
      timestamp: new Date().toISOString()
    });

    // Store error info in state
    this.setState({ errorInfo });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Future: Send to monitoring service
    // this.logErrorToService(error, errorInfo);
  }

  public componentDidUpdate(prevProps: Props) {
    const { resetKeys, resetOnPropsChange } = this.props;
    const { hasError } = this.state;
    
    // Reset error boundary when resetKeys change
    if (hasError && prevProps.resetKeys !== resetKeys) {
      if (resetKeys?.some((key, idx) => key !== this.previousResetKeys[idx])) {
        this.resetErrorBoundary();
      }
    }
    
    // Reset on any props change if specified
    if (hasError && resetOnPropsChange && prevProps.children !== this.props.children) {
      this.resetErrorBoundary();
    }
    
    this.previousResetKeys = resetKeys || [];
  }

  private resetErrorBoundary = () => {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
    }
    
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    });
  };

  private handleReset = () => {
    this.resetErrorBoundary();
  };

  private renderDefaultFallback() {
    const { error, errorInfo } = this.state;
    const { level = 'component', isolate = false } = this.props;
    
    const containerClass = level === 'page' 
      ? 'min-h-screen flex items-center justify-center bg-red-50'
      : isolate 
        ? 'flex items-center justify-center p-8 bg-red-50 rounded-lg'
        : 'w-full p-8 bg-red-50 rounded-lg';

    return (
      <div className={containerClass}>
        <div className="max-w-md w-full space-y-4">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="h-8 w-8 text-red-500" />
            <h2 className="text-xl font-semibold text-gray-900">
              {level === 'page' ? 'Page Error' : 'Component Error'}
            </h2>
          </div>
          
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <p className="text-sm text-gray-600 mb-2">
              An unexpected error occurred in this {level}.
            </p>
            <p className="text-xs text-red-600 font-mono bg-red-50 p-2 rounded">
              {error?.message || 'Unknown error'}
            </p>
          </div>

          <details className="text-sm text-gray-500">
            <summary className="cursor-pointer hover:text-gray-700">
              Technical details
            </summary>
            <div className="mt-2 space-y-2">
              <div className="bg-gray-100 p-2 rounded text-xs">
                <strong>Error ID:</strong> {this.state.errorId}
              </div>
              {errorInfo && (
                <div className="bg-gray-100 p-2 rounded text-xs">
                  <strong>Component Stack:</strong>
                  <pre className="mt-1 overflow-auto max-h-32">
                    {errorInfo.componentStack}
                  </pre>
                </div>
              )}
              {error?.stack && (
                <div className="bg-gray-100 p-2 rounded text-xs">
                  <strong>Error Stack:</strong>
                  <pre className="mt-1 overflow-auto max-h-32">
                    {error.stack}
                  </pre>
                </div>
              )}
            </div>
          </details>

          <div className="flex space-x-3">
            <button
              onClick={this.handleReset}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Try Again</span>
            </button>
            
            {level === 'page' && (
              <button
                onClick={() => window.location.href = '/'}
                className="flex items-center space-x-2 px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600 transition-colors"
              >
                <Home className="h-4 w-4" />
                <span>Go Home</span>
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  public render() {
    const { hasError, error } = this.state;
    const { children, fallback } = this.props;

    if (hasError && error) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback(error, this.resetErrorBoundary);
      }
      
      // Otherwise use default fallback
      return this.renderDefaultFallback();
    }

    return children;
  }
}

export default ErrorBoundary;