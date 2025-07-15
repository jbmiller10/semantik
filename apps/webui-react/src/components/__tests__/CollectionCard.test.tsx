import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import CollectionCard from '../CollectionCard'
import { useUIStore } from '@/stores/uiStore'

// Mock the UI store
vi.mock('@/stores/uiStore', () => ({
  useUIStore: vi.fn(),
}))

const mockCollection = {
  name: 'test-collection',
  total_files: 1234,
  total_vectors: 5678,
  model_name: 'Qwen/Qwen3-Embedding-0.6B',
  created_at: '2025-01-10T10:00:00Z',
  updated_at: '2025-01-14T15:30:00Z',
  job_count: 3,
}

describe('CollectionCard', () => {
  const mockSetShowCollectionDetailsModal = vi.fn()
  
  beforeEach(() => {
    vi.clearAllMocks()
    ;(useUIStore as any).mockReturnValue({
      setShowCollectionDetailsModal: mockSetShowCollectionDetailsModal,
    })
  })

  it('renders collection information correctly', () => {
    render(<CollectionCard collection={mockCollection} />)
    
    // Check collection name
    expect(screen.getByText('test-collection')).toBeInTheDocument()
    
    // Check model name (should extract last part)
    expect(screen.getByText('Qwen3-Embedding-0.6B')).toBeInTheDocument()
    
    // Check job count
    expect(screen.getByText('3 jobs')).toBeInTheDocument()
    
    // Check formatted numbers
    expect(screen.getByText('1,234')).toBeInTheDocument() // total_files
    expect(screen.getByText('5,678')).toBeInTheDocument() // total_vectors
    
    // Check labels
    expect(screen.getByText('Documents')).toBeInTheDocument()
    expect(screen.getByText('Vectors')).toBeInTheDocument()
    expect(screen.getByText('Last updated')).toBeInTheDocument()
    
    // Check formatted date
    expect(screen.getByText('Jan 14, 2025')).toBeInTheDocument()
  })

  it('renders singular job text for single job', () => {
    const singleJobCollection = { ...mockCollection, job_count: 1 }
    render(<CollectionCard collection={singleJobCollection} />)
    
    expect(screen.getByText('1 job')).toBeInTheDocument()
  })

  it('handles simple model names without path', () => {
    const simpleModelCollection = { ...mockCollection, model_name: 'bert-base' }
    render(<CollectionCard collection={simpleModelCollection} />)
    
    expect(screen.getByText('bert-base')).toBeInTheDocument()
  })

  it('formats large numbers with commas', () => {
    const largeNumberCollection = {
      ...mockCollection,
      total_files: 1234567,
      total_vectors: 9876543,
    }
    render(<CollectionCard collection={largeNumberCollection} />)
    
    expect(screen.getByText('1,234,567')).toBeInTheDocument()
    expect(screen.getByText('9,876,543')).toBeInTheDocument()
  })

  it('shows manage button that opens details modal', async () => {
    const user = userEvent.setup()
    
    render(<CollectionCard collection={mockCollection} />)
    
    const manageButton = screen.getByRole('button', { name: /manage/i })
    expect(manageButton).toBeInTheDocument()
    
    await user.click(manageButton)
    
    expect(mockSetShowCollectionDetailsModal).toHaveBeenCalledWith('test-collection')
  })

  it('truncates long collection names with title attribute', () => {
    const longNameCollection = {
      ...mockCollection,
      name: 'this-is-a-very-long-collection-name-that-should-be-truncated-in-the-ui',
    }
    render(<CollectionCard collection={longNameCollection} />)
    
    const heading = screen.getByRole('heading', { level: 3 })
    expect(heading).toHaveTextContent(longNameCollection.name)
    expect(heading).toHaveAttribute('title', longNameCollection.name)
    expect(heading).toHaveClass('truncate')
  })

  it('formats dates correctly', () => {
    // Test that dates are formatted correctly
    render(<CollectionCard collection={mockCollection} />)
    
    // Check that a formatted date is shown (Jan 14, 2025)
    expect(screen.getByText('Jan 14, 2025')).toBeInTheDocument()
  })

  it('displays zero values correctly', () => {
    const emptyCollection = {
      ...mockCollection,
      total_files: 0,
      total_vectors: 0,
      job_count: 0,
    }
    render(<CollectionCard collection={emptyCollection} />)
    
    expect(screen.getByText('0 jobs')).toBeInTheDocument()
    const zeroElements = screen.getAllByText('0')
    expect(zeroElements.length).toBe(2) // total_files and total_vectors
  })
})