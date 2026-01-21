import { describe, it, expect, vi, beforeEach, afterAll } from 'vitest';
import axios, { AxiosError } from 'axios';
import { ApiError, ApiErrorHandler, ErrorCategories } from '../api-error-handler';

// Mock console.error for development logging tests
const originalConsoleError = console.error;
beforeEach(() => {
  console.error = vi.fn();
});

describe('ApiError', () => {
  it('should create error with all properties', () => {
    const error = new ApiError('Test message', 'auth', 401, { key: 'value' });

    expect(error.message).toBe('Test message');
    expect(error.category).toBe('auth');
    expect(error.statusCode).toBe(401);
    expect(error.details).toEqual({ key: 'value' });
    expect(error.name).toBe('ApiError');
  });

  it('should work with instanceof', () => {
    const error = new ApiError('Test', 'unknown');
    expect(error instanceof ApiError).toBe(true);
    expect(error instanceof Error).toBe(true);
  });

  it('should identify auth errors', () => {
    const authError = new ApiError('Unauthorized', 'auth', 401);
    const otherError = new ApiError('Not found', 'not_found', 404);

    expect(authError.isAuthError()).toBe(true);
    expect(otherError.isAuthError()).toBe(false);
  });

  it('should identify retryable errors', () => {
    const networkError = new ApiError('Network error', 'network');
    const serverError = new ApiError('Server error', 'server', 500);
    const rateLimitError = new ApiError('Rate limited', 'unknown', 429);
    const validationError = new ApiError('Invalid', 'validation', 400);

    expect(networkError.isRetryable()).toBe(true);
    expect(serverError.isRetryable()).toBe(true);
    expect(rateLimitError.isRetryable()).toBe(true);
    expect(validationError.isRetryable()).toBe(false);
  });
});

describe('ErrorCategories', () => {
  it('should have all expected categories', () => {
    expect(ErrorCategories.AUTH).toBe('auth');
    expect(ErrorCategories.VALIDATION).toBe('validation');
    expect(ErrorCategories.NOT_FOUND).toBe('not_found');
    expect(ErrorCategories.CONFLICT).toBe('conflict');
    expect(ErrorCategories.SERVER).toBe('server');
    expect(ErrorCategories.NETWORK).toBe('network');
    expect(ErrorCategories.CANCELLED).toBe('cancelled');
    expect(ErrorCategories.INSUFFICIENT_RESOURCES).toBe('insufficient_resources');
    expect(ErrorCategories.UNKNOWN).toBe('unknown');
  });
});

describe('ApiErrorHandler.handle', () => {
  it('should handle cancelled requests', () => {
    const cancelError = new axios.Cancel('cancelled');
    const result = ApiErrorHandler.handle(cancelError);

    expect(result.category).toBe('cancelled');
    expect(result.message).toBe('Request was cancelled');
  });

  it('should categorize 401 as auth error', () => {
    const error = new AxiosError('Unauthorized', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 401,
      statusText: 'Unauthorized',
      data: { detail: 'Invalid token' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('auth');
    expect(result.statusCode).toBe(401);
    expect(result.message).toBe('Invalid token');
  });

  it('should categorize 403 as auth error', () => {
    const error = new AxiosError('Forbidden', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 403,
      statusText: 'Forbidden',
      data: { detail: 'Access denied' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('auth');
    expect(result.statusCode).toBe(403);
  });

  it('should categorize 400 as validation error', () => {
    const error = new AxiosError('Bad Request', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 400,
      statusText: 'Bad Request',
      data: { detail: 'Invalid input' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('validation');
    expect(result.statusCode).toBe(400);
  });

  it('should categorize 422 as validation error', () => {
    const error = new AxiosError('Unprocessable Entity', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 422,
      statusText: 'Unprocessable Entity',
      data: { detail: 'Validation failed' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('validation');
    expect(result.statusCode).toBe(422);
  });

  it('should categorize 404 as not_found error', () => {
    const error = new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 404,
      statusText: 'Not Found',
      data: { detail: 'Resource not found' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('not_found');
    expect(result.statusCode).toBe(404);
  });

  it('should categorize 409 as conflict error', () => {
    const error = new AxiosError('Conflict', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 409,
      statusText: 'Conflict',
      data: { detail: 'Resource already exists' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('conflict');
    expect(result.statusCode).toBe(409);
  });

  it('should categorize 500+ as server error', () => {
    const error = new AxiosError('Server Error', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 500,
      statusText: 'Internal Server Error',
      data: { detail: 'Something went wrong' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('server');
    expect(result.statusCode).toBe(500);
  });

  it('should categorize 507 as insufficient_resources error', () => {
    const error = new AxiosError('Insufficient Storage', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 507,
      statusText: 'Insufficient Storage',
      data: { detail: 'Insufficient GPU memory' },
      headers: {},
      config: {} as never,
    });

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('insufficient_resources');
    expect(result.statusCode).toBe(507);
  });

  it('should handle axios errors without response as network errors', () => {
    const error = new AxiosError('Network Error', 'ERR_NETWORK');

    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('network');
  });

  it('should handle unknown errors', () => {
    const result = ApiErrorHandler.handle('some string error');

    expect(result.category).toBe('unknown');
    expect(result.message).toBe('some string error');
  });

  it('should handle Error objects', () => {
    const error = new Error('Something went wrong');
    const result = ApiErrorHandler.handle(error);

    expect(result.category).toBe('unknown');
    expect(result.message).toBe('Something went wrong');
  });
});

describe('ApiErrorHandler.getMessage', () => {
  it('should extract message from axios error', () => {
    const error = new AxiosError('Bad Request', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 400,
      statusText: 'Bad Request',
      data: { detail: 'Invalid configuration' },
      headers: {},
      config: {} as never,
    });

    const message = ApiErrorHandler.getMessage(error);

    expect(message).toBe('Invalid configuration');
  });

  it('should handle cancelled requests', () => {
    const cancelError = new axios.Cancel('cancelled');
    const message = ApiErrorHandler.getMessage(cancelError);

    expect(message).toBe('Request was cancelled');
  });

  it('should handle string errors', () => {
    const message = ApiErrorHandler.getMessage('String error');
    expect(message).toBe('String error');
  });

  it('should handle unknown errors', () => {
    const message = ApiErrorHandler.getMessage(undefined);
    expect(message).toBe('An unexpected error occurred');
  });
});

describe('ApiErrorHandler.isAuthError', () => {
  it('should return true for 401', () => {
    const error = new AxiosError('Unauthorized', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 401,
      statusText: 'Unauthorized',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isAuthError(error)).toBe(true);
  });

  it('should return true for 403', () => {
    const error = new AxiosError('Forbidden', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 403,
      statusText: 'Forbidden',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isAuthError(error)).toBe(true);
  });

  it('should return false for other errors', () => {
    const error = new AxiosError('Not Found', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 404,
      statusText: 'Not Found',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isAuthError(error)).toBe(false);
  });

  it('should return false for non-axios errors', () => {
    expect(ApiErrorHandler.isAuthError(new Error('test'))).toBe(false);
  });
});

describe('ApiErrorHandler.isRetryable', () => {
  it('should return true for network errors', () => {
    const error = new AxiosError('Network Error', 'ERR_NETWORK');
    expect(ApiErrorHandler.isRetryable(error)).toBe(true);
  });

  it('should return true for 429 rate limit', () => {
    const error = new AxiosError('Too Many Requests', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 429,
      statusText: 'Too Many Requests',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isRetryable(error)).toBe(true);
  });

  it('should return true for 503 service unavailable', () => {
    const error = new AxiosError('Service Unavailable', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 503,
      statusText: 'Service Unavailable',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isRetryable(error)).toBe(true);
  });

  it('should return true for 500 server errors', () => {
    const error = new AxiosError('Server Error', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 500,
      statusText: 'Internal Server Error',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isRetryable(error)).toBe(true);
  });

  it('should return false for cancelled requests', () => {
    const cancelError = new axios.Cancel('cancelled');
    expect(ApiErrorHandler.isRetryable(cancelError)).toBe(false);
  });

  it('should return false for 400 validation errors', () => {
    const error = new AxiosError('Bad Request', 'ERR_BAD_REQUEST', undefined, undefined, {
      status: 400,
      statusText: 'Bad Request',
      data: {},
      headers: {},
      config: {} as never,
    });

    expect(ApiErrorHandler.isRetryable(error)).toBe(false);
  });
});

// Restore console.error after tests
afterAll(() => {
  console.error = originalConsoleError;
});
