import type { vi } from 'vitest';
import type { UseQueryResult } from '@tanstack/react-query';

// Common mock types for test files
export type MockedFunction = ReturnType<typeof vi.fn>;
export type MockedQuery<T = unknown> = UseQueryResult<T>;

// Helper type for mocking store hooks
export type MockedStore<T> = T & MockedFunction;

// Helper for typing axios error responses
export interface MockAxiosError {
  response?: {
    status?: number;
    statusText?: string;
    data?: unknown;
  };
  message?: string;
  code?: string;
}

// Helper for typing mock implementations with proper return types
export type MockImplementation<T extends (...args: never[]) => unknown> = 
  ((...args: Parameters<T>) => ReturnType<T>) & MockedFunction;