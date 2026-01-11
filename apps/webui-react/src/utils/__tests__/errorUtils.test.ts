import { AxiosError, InternalAxiosRequestConfig } from 'axios';
import {
  isAxiosError,
  isError,
  getErrorMessage,
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

    it('should extract messages from AxiosError with validation detail array', () => {
      const error = new AxiosError('Validation error');
      error.response = {
        data: {
          detail: [
            { loc: ['body', 'username'], msg: 'Username must contain only alphanumeric characters and underscores' },
            { loc: ['body', 'password'], msg: 'Password must be at least 8 characters long' },
          ],
        },
        status: 422,
        statusText: 'Unprocessable Entity',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      expect(getErrorMessage(error)).toBe(
        'username: Username must contain only alphanumeric characters and underscores; password: Password must be at least 8 characters long'
      );
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

});
