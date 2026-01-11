import axios, { type AxiosError } from 'axios';
import { getErrorMessage, isAxiosError } from '../utils/errorUtils';

/**
 * Error categories for consistent API error classification.
 * Maps HTTP status codes and error types to semantic categories.
 */
export type ApiErrorCategory =
  | 'auth'
  | 'validation'
  | 'not_found'
  | 'conflict'
  | 'server'
  | 'network'
  | 'cancelled'
  | 'insufficient_resources'
  | 'unknown';

export const ErrorCategories = {
  /** Authentication/authorization errors (401, 403) */
  AUTH: 'auth' as const,
  /** Validation errors (400, 422) */
  VALIDATION: 'validation' as const,
  /** Resource not found (404) */
  NOT_FOUND: 'not_found' as const,
  /** Resource conflict (409) */
  CONFLICT: 'conflict' as const,
  /** Server errors (500+) */
  SERVER: 'server' as const,
  /** Network/connection errors (no response) */
  NETWORK: 'network' as const,
  /** Request cancelled by user */
  CANCELLED: 'cancelled' as const,
  /** Insufficient resources (507) */
  INSUFFICIENT_RESOURCES: 'insufficient_resources' as const,
  /** Unknown error type */
  UNKNOWN: 'unknown' as const,
};

/**
 * Typed API error with category classification.
 * Provides structured error information for consistent handling across the app.
 */
export class ApiError extends Error {
  readonly category: ApiErrorCategory;
  readonly statusCode: number | undefined;
  readonly details: Record<string, unknown> | undefined;

  constructor(
    message: string,
    category: ApiErrorCategory,
    statusCode?: number,
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
    this.category = category;
    this.statusCode = statusCode;
    this.details = details;
    // Ensure proper prototype chain for instanceof checks
    Object.setPrototypeOf(this, ApiError.prototype);
  }

  /**
   * Check if this is an authentication error (user needs to log in)
   */
  isAuthError(): boolean {
    return this.category === ErrorCategories.AUTH;
  }

  /**
   * Check if this error is retryable (network, server, rate limit)
   */
  isRetryable(): boolean {
    return (
      this.category === ErrorCategories.NETWORK ||
      this.category === ErrorCategories.SERVER ||
      this.statusCode === 429 ||
      this.statusCode === 503
    );
  }
}

/**
 * Centralized API error handler.
 * Provides consistent error classification and message extraction.
 */
export class ApiErrorHandler {
  /**
   * Convert any error to a typed ApiError with category classification.
   * Use this when you need the full error object with category info.
   */
  static handle(error: unknown): ApiError {
    // Log in development for debugging
    if (import.meta.env.DEV) {
      console.error('[ApiError]', error);
    }

    // Handle request cancellation
    if (axios.isCancel(error)) {
      return new ApiError('Request was cancelled', ErrorCategories.CANCELLED);
    }

    // Handle Axios errors (HTTP responses)
    if (isAxiosError(error)) {
      const axiosError = error as AxiosError<{ detail?: unknown }>;
      const status = axiosError.response?.status;
      const message = getErrorMessage(error);

      // Map status codes to categories
      const category = this.categorizeByStatus(status, axiosError);

      return new ApiError(
        message,
        category,
        status,
        axiosError.response?.data as Record<string, unknown> | undefined
      );
    }

    // Handle network errors (no response)
    if (error instanceof Error && 'code' in error) {
      const networkError = error as Error & { code?: string };
      if (
        networkError.code === 'ECONNABORTED' ||
        networkError.code === 'ENOTFOUND' ||
        networkError.code === 'ERR_NETWORK'
      ) {
        return new ApiError(
          'Network error. Please check your connection.',
          ErrorCategories.NETWORK
        );
      }
    }

    // Fallback for other errors
    const message = getErrorMessage(error);
    return new ApiError(message, ErrorCategories.UNKNOWN);
  }

  /**
   * Categorize error by HTTP status code.
   */
  private static categorizeByStatus(
    status: number | undefined,
    error: AxiosError
  ): ApiErrorCategory {
    if (!status) {
      // No response received - network error
      if (!error.response) {
        return ErrorCategories.NETWORK;
      }
      return ErrorCategories.UNKNOWN;
    }

    if (status === 401 || status === 403) {
      return ErrorCategories.AUTH;
    }
    if (status === 400 || status === 422) {
      return ErrorCategories.VALIDATION;
    }
    if (status === 404) {
      return ErrorCategories.NOT_FOUND;
    }
    if (status === 409) {
      return ErrorCategories.CONFLICT;
    }
    if (status === 507) {
      return ErrorCategories.INSUFFICIENT_RESOURCES;
    }
    if (status >= 500) {
      return ErrorCategories.SERVER;
    }

    return ErrorCategories.UNKNOWN;
  }

  /**
   * Extract just the error message string.
   * Use this for backward compatibility with existing error handlers.
   */
  static getMessage(error: unknown): string {
    // Handle cancellation
    if (axios.isCancel(error)) {
      return 'Request was cancelled';
    }
    return getErrorMessage(error);
  }

  /**
   * Check if error is an authentication error.
   */
  static isAuthError(error: unknown): boolean {
    if (isAxiosError(error)) {
      const status = (error as AxiosError).response?.status;
      return status === 401 || status === 403;
    }
    return false;
  }

  /**
   * Check if error is retryable (network, server, rate limit).
   */
  static isRetryable(error: unknown): boolean {
    if (axios.isCancel(error)) {
      return false;
    }

    if (isAxiosError(error)) {
      const axiosError = error as AxiosError;
      const status = axiosError.response?.status;

      // No response = network error = retryable
      if (!axiosError.response) {
        return true;
      }

      // Specific retryable status codes
      return status === 429 || status === 503 || status === 504 || (status !== undefined && status >= 500);
    }

    return false;
  }
}
