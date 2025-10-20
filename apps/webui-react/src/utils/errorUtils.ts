import { AxiosError } from 'axios';

/**
 * Type guard to check if an error is an AxiosError
 */
export function isAxiosError(error: unknown): error is AxiosError {
  return error instanceof AxiosError;
}

/**
 * Type guard to check if an error is a standard Error
 */
export function isError(error: unknown): error is Error {
  return error instanceof Error;
}

/**
 * Extract error message from various error types
 */
export function getErrorMessage(error: unknown): string {
  if (isAxiosError(error)) {
    const data = error.response?.data as { detail?: unknown };
    const detail = data?.detail;
    
    // Handle structured error details
    if (typeof detail === 'object' && detail !== null && !Array.isArray(detail)) {
      if ('message' in detail && typeof (detail as { message: unknown }).message === 'string') {
        return (detail as { message: string }).message;
      }
      if ('error' in detail && typeof (detail as { error: unknown }).error === 'string') {
        return (detail as { error: string }).error;
      }
    }

    // Handle array-based validation errors (e.g., Pydantic validation response)
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => {
          if (typeof item === 'string') {
            return item;
          }
          if (item && typeof item === 'object' && 'msg' in item) {
            const message = (item as { msg?: unknown }).msg;
            if (typeof message === 'string') {
              const loc = (item as { loc?: unknown }).loc;
              if (Array.isArray(loc) && loc.length > 1 && typeof loc[1] === 'string') {
                return `${loc[1]}: ${message}`;
              }
              return message;
            }
          }
          return null;
        })
        .filter((msg): msg is string => Boolean(msg));

      if (messages.length > 0) {
        return messages.join('; ');
      }
    }

    // Handle string error details
    if (typeof detail === 'string') {
      return detail;
    }

    // Fallback to status text or generic message
    return error.response?.statusText || error.message || 'Network error occurred';
  }
  
  if (isError(error)) {
    return error.message;
  }
  
  if (typeof error === 'string') {
    return error;
  }
  
  return 'An unexpected error occurred';
}

/**
 * Extract structured error information for specific error types
 */
export interface StructuredError {
  error: string;
  message?: string;
  suggestion?: string;
}

export function isStructuredError(detail: unknown): detail is StructuredError {
  return (
    typeof detail === 'object' &&
    detail !== null &&
    'error' in detail &&
    typeof (detail as { error: unknown }).error === 'string'
  );
}

/**
 * Handle insufficient memory errors specifically
 */
export function isInsufficientMemoryError(error: unknown): boolean {
  if (!isAxiosError(error)) return false;
  
  const data = error.response?.data as { detail?: unknown };
  const detail = data?.detail;
  return (
    error.response?.status === 507 &&
    isStructuredError(detail) &&
    detail.error === 'insufficient_memory'
  );
}

/**
 * Get structured error details for insufficient memory errors
 */
export function getInsufficientMemoryErrorDetails(error: unknown): {
  message: string;
  suggestion: string;
} | null {
  if (!isInsufficientMemoryError(error)) return null;
  
  const data = (error as AxiosError).response?.data as { detail?: unknown };
  const detail = data?.detail as StructuredError;
  return {
    message: detail.message || 'Insufficient GPU memory for reranking',
    suggestion: detail.suggestion || 'Try using a smaller model or different quantization',
  };
}
