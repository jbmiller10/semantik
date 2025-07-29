import { vi } from 'vitest'
import { AxiosResponse, InternalAxiosRequestConfig } from 'axios'
import type { Collection, CollectionStatus, Operation, OperationType, OperationStatus } from '@/types/collection'
import type { Document } from '@/types/document'
import type { SearchResult } from '@/types/search'
import type { User } from '@/types/user'

// Vitest Mock Types
export type MockedFunction<T extends (...args: unknown[]) => unknown> = ReturnType<typeof vi.fn<T>>
export type MockedModule = Record<string, unknown>

// API Response Mock Types
export interface MockAxiosResponse<T = unknown> extends Partial<AxiosResponse<T>> {
  data: T
  status: number
  statusText: string
  headers: Record<string, string>
  config: InternalAxiosRequestConfig
}

// Common Mock Data Types
export interface MockCollection extends Collection {
  // Add any test-specific fields if needed
}

export interface MockOperation extends Operation {
  // Add any test-specific fields if needed
}

export interface MockDocument extends Document {
  // Add any test-specific fields if needed
}

export interface MockSearchResult extends SearchResult {
  // Add any test-specific fields if needed
}

export interface MockUser extends User {
  // Add any test-specific fields if needed
}

// Error Types
export interface MockError {
  message: string
  code?: string
  status?: number
}

export interface MockAxiosError {
  response?: {
    data: {
      detail?: string | { message?: string; error?: string; suggestion?: string }
    }
    status: number
    statusText: string
    headers: Record<string, string>
    config: InternalAxiosRequestConfig
  }
  message: string
  code?: string
}

// WebSocket Mock Types
export interface MockWebSocketMessage {
  type: string
  data: unknown
}

export interface MockWebSocketInstance {
  url: string
  readyState: number
  onopen: ((event: Event) => void) | null
  onclose: ((event: CloseEvent) => void) | null
  onerror: ((event: Event) => void) | null
  onmessage: ((event: MessageEvent) => void) | null
  send: (data: string | ArrayBuffer | Blob | ArrayBufferView) => void
  close: (code?: number, reason?: string) => void
  simulateOpen?: () => void
  simulateMessage?: (data: unknown) => void
  simulateError?: (error?: Error) => void
  simulateClose?: (code?: number, reason?: string) => void
}

// Store Mock Types
export interface MockUIStore {
  addToast: MockedFunction<(toast: { type: string; message: string }) => void>
  setShowCollectionDetailsModal: MockedFunction<(id: string | null) => void>
  setShowCreateCollectionModal: MockedFunction<(show: boolean) => void>
  setShowDeleteCollectionModal: MockedFunction<(id: string | null) => void>
  setShowRenameCollectionModal: MockedFunction<(id: string | null) => void>
  setShowReindexCollectionModal: MockedFunction<(id: string | null) => void>
  setShowAddDataModal: MockedFunction<(id: string | null) => void>
}

export interface MockAuthStore {
  user: User | null
  isAuthenticated: boolean
  login: MockedFunction<(credentials: { username: string; password: string }) => Promise<void>>
  logout: MockedFunction<() => void>
  initializeAuth: MockedFunction<() => Promise<void>>
}

// Test Helper Types
export interface TestWrapperProps {
  children: React.ReactNode
  initialEntries?: string[]
}

// Utility Types for Testing
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

export type MockApiFunction<TArgs extends unknown[], TReturn> = MockedFunction<(...args: TArgs) => Promise<TReturn>>

// Common Test Patterns
export interface MockApiModule {
  [key: string]: MockApiFunction<unknown[], unknown>
}

// Type Guards for Test Assertions
export function isMockAxiosError(error: unknown): error is MockAxiosError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'message' in error &&
    ('response' in error || 'code' in error)
  )
}

export function isMockError(error: unknown): error is MockError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'message' in error &&
    typeof (error as MockError).message === 'string'
  )
}

// Default Mock Data Factories
export function createMockCollection(overrides?: Partial<MockCollection>): MockCollection {
  return {
    id: 'test-collection-id',
    name: 'Test Collection',
    description: 'Test description',
    owner_id: 1,
    vector_store_name: 'test_vectors',
    embedding_model: 'test-model',
    quantization: 'float16',
    chunk_size: 512,
    chunk_overlap: 50,
    is_public: false,
    status: 'ready' as CollectionStatus,
    document_count: 0,
    vector_count: 0,
    total_size_bytes: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  }
}

export function createMockOperation(overrides?: Partial<MockOperation>): MockOperation {
  return {
    id: 'test-operation-id',
    collection_id: 'test-collection-id',
    type: 'index' as OperationType,
    status: 'pending' as OperationStatus,
    config: {},
    created_at: new Date().toISOString(),
    ...overrides,
  }
}

export function createMockUser(overrides?: Partial<MockUser>): MockUser {
  return {
    id: 1,
    username: 'testuser',
    is_active: true,
    is_admin: false,
    created_at: new Date().toISOString(),
    ...overrides,
  }
}

export function createMockAxiosResponse<T>(data: T, status = 200): MockAxiosResponse<T> {
  return {
    data,
    status,
    statusText: 'OK',
    headers: {},
    config: {} as InternalAxiosRequestConfig,
  }
}

export function createMockAxiosError(
  message: string,
  status?: number,
  detail?: string | { message?: string; error?: string; suggestion?: string }
): MockAxiosError {
  const error: MockAxiosError = {
    message,
  }

  if (status !== undefined) {
    error.response = {
      data: detail ? { detail } : {},
      status,
      statusText: 'Error',
      headers: {},
      config: {} as InternalAxiosRequestConfig,
    }
  }

  return error
}