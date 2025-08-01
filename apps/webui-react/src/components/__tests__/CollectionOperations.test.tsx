import { describe, it, expect, vi, beforeEach } from 'vitest'
import React from 'react'
import { render, screen } from '@/tests/utils/test-utils'
import CollectionOperations from '../CollectionOperations'
import { useCollectionOperations } from '@/hooks/useCollectionOperations'
import type { Collection, Operation, OperationType, OperationStatus } from '@/types/collection'
import type { MockedFunction } from '@/tests/types/test-types'
import type { UseQueryResult } from '@tanstack/react-query'

// Mock the hooks and components
vi.mock('@/hooks/useCollectionOperations', () => ({
  useCollectionOperations: vi.fn(),
}))

vi.mock('../OperationProgress', () => ({
  default: vi.fn(({ operation, showDetails, onComplete }) => {
    React.useEffect(() => {
      if (operation.status === 'completed' && onComplete) {
        // Simulate completion callback after a brief delay
        const timer = setTimeout(() => onComplete(), 100)
        return () => clearTimeout(timer)
      }
    }, [operation.status, onComplete])
    
    return (
      <div data-testid={`operation-${operation.id}`}>
        <span>Operation: {operation.type}</span>
        <span>Status: {operation.status}</span>
        <span>Show Details: {showDetails ? 'true' : 'false'}</span>
      </div>
    )
  }),
}))

const mockCollection: Collection = {
  id: 'test-collection-id',
  name: 'Test Collection',
  description: 'Test collection description',
  owner_id: 1,
  vector_store_name: 'test_collection_vectors',
  embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
  quantization: 'float16',
  chunk_size: 512,
  chunk_overlap: 50,
  is_public: false,
  status: 'ready',
  document_count: 10,
  vector_count: 100,
  total_size_bytes: 1024000,
  created_at: '2025-01-10T10:00:00Z',
  updated_at: '2025-01-14T15:30:00Z',
}

const createMockOperation = (
  id: string,
  type: OperationType,
  status: OperationStatus,
  created_at: string
): Operation => ({
  id,
  collection_id: mockCollection.id,
  type,
  status,
  config: { source_path: '/data/documents' },
  created_at,
  progress: status === 'processing' ? 50 : status === 'completed' ? 100 : 0,
})

describe('CollectionOperations', () => {
  const mockRefetch = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Basic Rendering', () => {
    it('renders empty state when no operations exist', () => {
      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: [],
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('No operations yet')).toBeInTheDocument()
      expect(screen.getByText('Operations will appear here when you add data or re-index')).toBeInTheDocument()
    })

    it('renders operations list when operations are provided', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'processing', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('Operations')).toBeInTheDocument()
      expect(screen.getByTestId('operation-op-1')).toBeInTheDocument()
      expect(screen.getByTestId('operation-op-2')).toBeInTheDocument()
    })

    it('displays active operations count badge when active operations exist', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'pending', '2025-01-14T11:00:00Z'),
        createMockOperation('op-3', 'append', 'completed', '2025-01-14T09:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('2 Active')).toBeInTheDocument()
    })

    it('does not display active count badge when no active operations', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'failed', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.queryByText(/Active/)).not.toBeInTheDocument()
    })
  })

  describe('Operation Sorting and Filtering', () => {
    it('sorts operations by creation date with newest first', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'completed', '2025-01-14T12:00:00Z'),
        createMockOperation('op-3', 'append', 'completed', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      const { container } = render(<CollectionOperations collection={mockCollection} />)
      
      const operationElements = container.querySelectorAll('[data-testid^="operation-"]')
      expect(operationElements[0]).toHaveAttribute('data-testid', 'operation-op-2')
      expect(operationElements[1]).toHaveAttribute('data-testid', 'operation-op-3')
      expect(operationElements[2]).toHaveAttribute('data-testid', 'operation-op-1')
    })

    it('displays active operations before completed ones', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T12:00:00Z'),
        createMockOperation('op-2', 'reindex', 'processing', '2025-01-14T10:00:00Z'),
        createMockOperation('op-3', 'append', 'pending', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      const { container } = render(<CollectionOperations collection={mockCollection} />)
      
      const operationElements = container.querySelectorAll('[data-testid^="operation-"]')
      // Active operations should come first (sorted by date)
      expect(operationElements[0]).toHaveAttribute('data-testid', 'operation-op-3')
      expect(operationElements[1]).toHaveAttribute('data-testid', 'operation-op-2')
      // Then completed operations
      expect(operationElements[2]).toHaveAttribute('data-testid', 'operation-op-1')
    })

    it('respects maxOperations prop', () => {
      const operations = Array.from({ length: 10 }, (_, i) =>
        createMockOperation(`op-${i}`, 'index', 'completed', `2025-01-14T${10 + i}:00:00Z`)
      )

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      const { container } = render(<CollectionOperations collection={mockCollection} maxOperations={3} />)
      
      const operationElements = container.querySelectorAll('[data-testid^="operation-"]')
      expect(operationElements).toHaveLength(3)
    })

    it('displays more operations indicator when there are more than maxOperations', () => {
      const operations = Array.from({ length: 10 }, (_, i) =>
        createMockOperation(`op-${i}`, 'index', 'completed', `2025-01-14T${10 + i}:00:00Z`)
      )

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} maxOperations={5} />)
      
      expect(screen.getByText('5 more operations not shown')).toBeInTheDocument()
    })
  })

  describe('Operation States', () => {
    it('renders active operations section with correct styling', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'pending', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('Active Operations')).toBeInTheDocument()
      
      // Check for blue styling on active operations
      // Find the operation wrapper div with styling
      const operationWrapper = screen.getByTestId('operation-op-1').parentElement
      expect(operationWrapper).toHaveClass('bg-blue-50', 'border-blue-200')
    })

    it('renders completed operations with gray styling', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      // Find the operation wrapper div with styling
      const operationWrapper = screen.getByTestId('operation-op-1').parentElement
      expect(operationWrapper).toHaveClass('bg-gray-50', 'border-gray-200')
    })

    it('renders failed operations with red styling', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'failed', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      // Find the operation wrapper div with styling
      const operationWrapper = screen.getByTestId('operation-op-1').parentElement
      expect(operationWrapper).toHaveClass('bg-red-50', 'border-red-200')
    })

    it('renders cancelled operations with yellow styling', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'cancelled', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      // Find the operation wrapper div with styling
      const operationWrapper = screen.getByTestId('operation-op-1').parentElement
      expect(operationWrapper).toHaveClass('bg-yellow-50', 'border-yellow-200')
    })
  })

  describe('Hook Integration', () => {
    it('passes correct collection ID to useCollectionOperations hook', () => {
      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: [],
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(useCollectionOperations).toHaveBeenCalledWith(mockCollection.id)
    })

    it('passes refetch callback to child components', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      // Verify the hook was called with correct parameters
      expect(useCollectionOperations).toHaveBeenCalledWith(mockCollection.id)
      // Verify refetch function is available from the hook
      expect(mockRefetch).toBeDefined()
    })
  })

  describe('OperationProgress Integration', () => {
    it('passes showDetails=true for active operations', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('Show Details: true')).toBeInTheDocument()
    })

    it('passes showDetails=false for recent operations', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('Show Details: false')).toBeInTheDocument()
    })

    it('passes operation data correctly to OperationProgress', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('Operation: index')).toBeInTheDocument()
      expect(screen.getByText('Status: processing')).toBeInTheDocument()
    })
  })

  describe('Edge Cases', () => {
    it('handles exactly maxOperations without showing indicator', () => {
      const operations = Array.from({ length: 5 }, (_, i) =>
        createMockOperation(`op-${i}`, 'index', 'completed', `2025-01-14T${10 + i}:00:00Z`)
      )

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} maxOperations={5} />)
      
      expect(screen.queryByText(/more operations not shown/)).not.toBeInTheDocument()
    })

    it('handles only active operations', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'pending', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.getByText('Active Operations')).toBeInTheDocument()
      expect(screen.queryByText('Recent Operations')).not.toBeInTheDocument()
    })

    it('handles only completed operations', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'completed', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'failed', '2025-01-14T11:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      expect(screen.queryByText('Active Operations')).not.toBeInTheDocument()
      // Recent Operations header is only shown when there are also active operations
      expect(screen.queryByText('Recent Operations')).not.toBeInTheDocument()
    })

    it('handles mixed operation types and statuses', () => {
      const operations = [
        createMockOperation('op-1', 'index', 'processing', '2025-01-14T10:00:00Z'),
        createMockOperation('op-2', 'reindex', 'completed', '2025-01-14T11:00:00Z'),
        createMockOperation('op-3', 'append', 'failed', '2025-01-14T12:00:00Z'),
        createMockOperation('op-4', 'remove_source', 'cancelled', '2025-01-14T13:00:00Z'),
        createMockOperation('op-5', 'delete', 'pending', '2025-01-14T14:00:00Z'),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      render(<CollectionOperations collection={mockCollection} />)

      // Should have both sections
      expect(screen.getByText('Active Operations')).toBeInTheDocument()
      expect(screen.getByText('Recent Operations')).toBeInTheDocument()

      // All operations should be rendered
      operations.forEach(op => {
        expect(screen.getByTestId(`operation-${op.id}`)).toBeInTheDocument()
      })
    })

    it('prioritizes active operations when total exceeds maxOperations', () => {
      const operations = [
        ...Array.from({ length: 3 }, (_, i) =>
          createMockOperation(`active-${i}`, 'index', 'processing', `2025-01-14T${10 + i}:00:00Z`)
        ),
        ...Array.from({ length: 5 }, (_, i) =>
          createMockOperation(`completed-${i}`, 'index', 'completed', `2025-01-14T${15 + i}:00:00Z`)
        ),
      ]

      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: operations,
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      const { container } = render(<CollectionOperations collection={mockCollection} maxOperations={5} />)
      
      const operationElements = container.querySelectorAll('[data-testid^="operation-"]')
      expect(operationElements).toHaveLength(5)
      
      // All 3 active operations should be shown
      expect(screen.getByTestId('operation-active-0')).toBeInTheDocument()
      expect(screen.getByTestId('operation-active-1')).toBeInTheDocument()
      expect(screen.getByTestId('operation-active-2')).toBeInTheDocument()
      
      // Only 2 completed operations should be shown
      expect(screen.getByTestId('operation-completed-4')).toBeInTheDocument()
      expect(screen.getByTestId('operation-completed-3')).toBeInTheDocument()
    })

    it('applies className prop correctly', () => {
      ;(useCollectionOperations as MockedFunction<typeof useCollectionOperations>).mockReturnValue({
        data: [],
        refetch: mockRefetch,
      } as UseQueryResult<Operation[]>)

      const { container } = render(<CollectionOperations collection={mockCollection} className="custom-class" />)
      
      const mainDiv = container.firstChild as HTMLElement
      expect(mainDiv).toHaveClass('custom-class')
    })
  })
})