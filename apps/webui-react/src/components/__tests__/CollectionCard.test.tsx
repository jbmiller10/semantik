import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import CollectionCard from '../CollectionCard'
import { useUIStore } from '@/stores/uiStore'
import type { Collection, CollectionStatus } from '@/types/collection'
import type { MockedFunction } from '@/tests/types/test-types'

// Mock the UI store
vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(),
}))

const mockCollection: Collection = {
  id: 'test-id-123',
  name: 'test-collection',
  description: 'Test collection description',
  owner_id: 1,
  vector_store_name: 'test_collection_vectors',
  embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
  quantization: 'float16',
  chunk_size: 512,
  chunk_overlap: 50,
  is_public: false,
  status: 'ready',
  document_count: 1234,
  vector_count: 5678,
  total_size_bytes: 1024000,
  created_at: '2025-01-10T10:00:00Z',
  updated_at: '2025-01-14T15:30:00Z',
}

describe('CollectionCard', () => {
  const mockSetShowCollectionDetailsModal = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    ;(useUIStore as MockedFunction<typeof useUIStore>).mockReturnValue({
      setShowCollectionDetailsModal: mockSetShowCollectionDetailsModal,
    })
  })

  it('renders collection information correctly', () => {
    render(<CollectionCard collection={mockCollection} />)
    
    // Check collection name
    expect(screen.getByText('test-collection')).toBeInTheDocument()
    
    // Check description
    expect(screen.getByText('Test collection description')).toBeInTheDocument()
    
    // Check model name (should extract last part)
    expect(screen.getByText('Qwen3-Embedding-0.6B')).toBeInTheDocument()
    
    // Check status
    expect(screen.getByText('ready')).toBeInTheDocument()
    
    // Check formatted numbers
    expect(screen.getByText('1,234')).toBeInTheDocument() // document_count
    expect(screen.getByText('5,678')).toBeInTheDocument() // vector_count
    
    // Check labels
    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(screen.getByText('Vectors')).toBeInTheDocument()
    expect(screen.getByText('Last updated')).toBeInTheDocument()
    
    // Check formatted date
    expect(screen.getByText('Jan 14, 2025')).toBeInTheDocument()
  })

  it('renders different status colors correctly', () => {
    const statuses: CollectionStatus[] = ['ready', 'processing', 'error', 'degraded', 'pending']
    
    statuses.forEach(status => {
      const { rerender } = render(
        <CollectionCard collection={{ ...mockCollection, status }} />
      )
      
      expect(screen.getByText(status)).toBeInTheDocument()
      
      // Clean up for next iteration
      rerender(<div />)
    })
  })

  it('handles simple model names without path', () => {
    const simpleModelCollection = { ...mockCollection, embedding_model: 'bert-base' }
    render(<CollectionCard collection={simpleModelCollection} />)
    
    expect(screen.getByText('bert-base')).toBeInTheDocument()
  })

  it('shows status message for error state', () => {
    const errorCollection = { 
      ...mockCollection, 
      status: 'error' as CollectionStatus,
      status_message: 'Failed to connect to vector database'
    }
    render(<CollectionCard collection={errorCollection} />)
    
    expect(screen.getByText('error')).toBeInTheDocument()
    expect(screen.getByText('Failed to connect to vector database')).toBeInTheDocument()
  })

  it('shows status message for degraded state', () => {
    const degradedCollection = { 
      ...mockCollection, 
      status: 'degraded' as CollectionStatus,
      status_message: 'Some documents failed to index'
    }
    render(<CollectionCard collection={degradedCollection} />)
    
    expect(screen.getByText('degraded')).toBeInTheDocument()
    expect(screen.getByText('Some documents failed to index')).toBeInTheDocument()
  })

  it('handles long collection names with truncation', () => {
    const longNameCollection = {
      ...mockCollection,
      name: 'this-is-a-very-long-collection-name-that-should-be-truncated',
    }
    render(<CollectionCard collection={longNameCollection} />)
    
    const nameElement = screen.getByText('this-is-a-very-long-collection-name-that-should-be-truncated')
    expect(nameElement).toHaveClass('truncate')
    expect(nameElement).toHaveAttribute('title', 'this-is-a-very-long-collection-name-that-should-be-truncated')
  })

  it('opens collection details modal when Manage button is clicked', async () => {
    const user = userEvent.setup()
    render(<CollectionCard collection={mockCollection} />)
    
    const manageButton = screen.getByRole('button', { name: /manage/i })
    await user.click(manageButton)
    
    expect(mockSetShowCollectionDetailsModal).toHaveBeenCalledWith('test-id-123')
  })

  it('shows processing state with animation', () => {
    const processingCollection = { 
      ...mockCollection, 
      status: 'processing' as CollectionStatus,
      isProcessing: true,
      activeOperation: {
        id: 'op-123',
        collection_id: 'test-id-123',
        type: 'index' as const,
        status: 'processing' as const,
        config: {},
        created_at: '2025-01-14T15:00:00Z',
        progress: 65
      }
    }
    render(<CollectionCard collection={processingCollection} />)
    
    // Check processing status
    expect(screen.getByText('processing')).toBeInTheDocument()
    expect(screen.getByText('Indexing documents...')).toBeInTheDocument()
    
    // Check that the card has the processing styles
    const card = screen.getByText('test-collection').closest('div[class*="border-2"]')
    expect(card).toHaveClass('border-blue-500')
    expect(card).toHaveClass('bg-blue-50')
  })

  it('disables manage button during processing', () => {
    const processingCollection = { 
      ...mockCollection, 
      status: 'processing' as CollectionStatus,
      isProcessing: true
    }
    render(<CollectionCard collection={processingCollection} />)
    
    const manageButton = screen.getByRole('button', { name: /manage/i })
    expect(manageButton).toBeDisabled()
  })

  it('formats large numbers with commas', () => {
    const largeNumberCollection = {
      ...mockCollection,
      document_count: 1234567,
      vector_count: 8901234,
    }
    render(<CollectionCard collection={largeNumberCollection} />)
    
    expect(screen.getByText('1,234,567')).toBeInTheDocument()
    expect(screen.getByText('8,901,234')).toBeInTheDocument()
  })
})