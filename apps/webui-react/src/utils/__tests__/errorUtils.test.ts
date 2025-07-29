import { AxiosError, InternalAxiosRequestConfig } from 'axios';
import {
  isAxiosError,
  isError,
  getErrorMessage,
  isStructuredError,
  isInsufficientMemoryError,
  getInsufficientMemoryErrorDetails,
} from '../errorUtils';

describe('errorUtils', () => {
  describe('isAxiosError', () => {
    it('should return true for AxiosError instances', () => {
      const error = new AxiosError('Network error');
      expect(isAxiosError(error)).toBe(true);
    });

    it('should return false for non-AxiosError instances', () => {
      expect(isAxiosError(new Error('Regular error'))).toBe(false);
      expect(isAxiosError('string error')).toBe(false);
      expect(isAxiosError(null)).toBe(false);
      expect(isAxiosError(undefined)).toBe(false);
    });
  });

  describe('isError', () => {
    it('should return true for Error instances', () => {
      expect(isError(new Error('Test error'))).toBe(true);
      expect(isError(new TypeError('Type error'))).toBe(true);
    });

    it('should return false for non-Error instances', () => {
      expect(isError('string error')).toBe(false);
      expect(isError(null)).toBe(false);
      expect(isError(undefined)).toBe(false);
      expect(isError({ message: 'not an error' })).toBe(false);
    });
  });

  describe('getErrorMessage', () => {
    it('should extract message from AxiosError with string detail', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: 'Custom error message' },
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(getErrorMessage(error)).toBe('Custom error message');
    });

    it('should extract message from AxiosError with object detail', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: { message: 'Structured error message' } },
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(getErrorMessage(error)).toBe('Structured error message');
    });

    it('should extract error from AxiosError with error field', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: { error: 'Error field message' } },
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(getErrorMessage(error)).toBe('Error field message');
    });

    it('should fallback to statusText for AxiosError', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: {},
        status: 404,
        statusText: 'Not Found',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(getErrorMessage(error)).toBe('Not Found');
    });

    it('should extract message from regular Error', () => {
      const error = new Error('Regular error message');
      expect(getErrorMessage(error)).toBe('Regular error message');
    });

    it('should handle string errors', () => {
      expect(getErrorMessage('String error')).toBe('String error');
    });

    it('should handle unknown error types', () => {
      expect(getErrorMessage(null)).toBe('An unexpected error occurred');
      expect(getErrorMessage(undefined)).toBe('An unexpected error occurred');
      expect(getErrorMessage({ someField: 'value' })).toBe('An unexpected error occurred');
    });
  });

  describe('isStructuredError', () => {
    it('should return true for valid structured errors', () => {
      expect(isStructuredError({ error: 'test_error' })).toBe(true);
      expect(isStructuredError({ error: 'test_error', message: 'Test message' })).toBe(true);
      expect(isStructuredError({ error: 'test_error', suggestion: 'Try this' })).toBe(true);
    });

    it('should return false for invalid structures', () => {
      expect(isStructuredError(null)).toBe(false);
      expect(isStructuredError(undefined)).toBe(false);
      expect(isStructuredError('string')).toBe(false);
      expect(isStructuredError({ message: 'missing error field' })).toBe(false);
      expect(isStructuredError({ error: 123 })).toBe(false); // error must be string
    });
  });

  describe('isInsufficientMemoryError', () => {
    it('should return true for insufficient memory errors', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: { error: 'insufficient_memory' } },
        status: 507,
        statusText: 'Insufficient Storage',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(isInsufficientMemoryError(error)).toBe(true);
    });

    it('should return false for other errors', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: { error: 'other_error' } },
        status: 507,
        statusText: 'Insufficient Storage',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(isInsufficientMemoryError(error)).toBe(false);
    });

    it('should return false for non-507 status', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: { error: 'insufficient_memory' } },
        status: 500,
        statusText: 'Internal Server Error',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(isInsufficientMemoryError(error)).toBe(false);
    });

    it('should return false for non-AxiosError', () => {
      expect(isInsufficientMemoryError(new Error('Regular error'))).toBe(false);
    });
  });

  describe('getInsufficientMemoryErrorDetails', () => {
    it('should return details for insufficient memory errors', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: {
          detail: {
            error: 'insufficient_memory',
            message: 'Not enough GPU memory',
            suggestion: 'Use a smaller model',
          },
        },
        status: 507,
        statusText: 'Insufficient Storage',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      const details = getInsufficientMemoryErrorDetails(error);
      expect(details).toEqual({
        message: 'Not enough GPU memory',
        suggestion: 'Use a smaller model',
      });
    });

    it('should provide default values when fields are missing', () => {
      const error = new AxiosError('Network error');
      error.response = {
        data: { detail: { error: 'insufficient_memory' } },
        status: 507,
        statusText: 'Insufficient Storage',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      const details = getInsufficientMemoryErrorDetails(error);
      expect(details).toEqual({
        message: 'Insufficient GPU memory for reranking',
        suggestion: 'Try using a smaller model or different quantization',
      });
    });

    it('should return null for non-insufficient memory errors', () => {
      const error = new Error('Regular error');
      expect(getInsufficientMemoryErrorDetails(error)).toBeNull();
    });
  });
});