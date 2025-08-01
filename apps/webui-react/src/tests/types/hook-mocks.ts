import type { UseQueryResult, UseMutationResult } from '@tanstack/react-query'
import type { Collection, Operation } from '@/types/collection'
import type { Document } from '@/types/document'
import type { SearchResult } from '@/types/search'
import { vi } from 'vitest'

// React Query Mock Types
export type MockUseQueryResult<TData = unknown, TError = Error> = Partial<UseQueryResult<TData, TError>>
export type MockUseMutationResult<TData = unknown, TError = Error, TVariables = unknown> = Partial<UseMutationResult<TData, TError, TVariables>>

// Collection Query Mock
export interface MockCollectionsQuery extends MockUseQueryResult<Collection[]> {
  data: Collection[]
  isLoading: boolean
  error: Error | null
  refetch: ReturnType<typeof vi.fn>
}

// Operation Query Mock
export interface MockOperationsQuery extends MockUseQueryResult<Operation[]> {
  data: Operation[]
  isLoading: boolean
  error: Error | null
  refetch: ReturnType<typeof vi.fn>
}

// Document Query Mock
export interface MockDocumentsQuery extends MockUseQueryResult<Document[]> {
  data: Document[]
  isLoading: boolean
  error: Error | null
  refetch: ReturnType<typeof vi.fn>
}

// Search Results Query Mock
export interface MockSearchQuery extends MockUseQueryResult<SearchResult[]> {
  data: SearchResult[]
  isLoading: boolean
  error: Error | null
  refetch: ReturnType<typeof vi.fn>
}

// Mutation Mocks
export interface MockCollectionMutation extends MockUseMutationResult {
  mutateAsync: ReturnType<typeof vi.fn>
  mutate: ReturnType<typeof vi.fn>
  isError: boolean
  isPending: boolean
  error: Error | null
}

// Helper functions to create properly typed mocks
export function createMockCollectionsQuery(overrides?: Partial<MockCollectionsQuery>): MockCollectionsQuery {
  return {
    data: [],
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  }
}

export function createMockOperationsQuery(overrides?: Partial<MockOperationsQuery>): MockOperationsQuery {
  return {
    data: [],
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  }
}

export function createMockDocumentsQuery(overrides?: Partial<MockDocumentsQuery>): MockDocumentsQuery {
  return {
    data: [],
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  }
}

export function createMockSearchQuery(overrides?: Partial<MockSearchQuery>): MockSearchQuery {
  return {
    data: [],
    isLoading: false,
    error: null,
    refetch: vi.fn(),
    ...overrides,
  }
}

export function createMockCollectionMutation(overrides?: Partial<MockCollectionMutation>): MockCollectionMutation {
  return {
    mutateAsync: vi.fn(),
    mutate: vi.fn(),
    isError: false,
    isPending: false,
    error: null,
    ...overrides,
  }
}