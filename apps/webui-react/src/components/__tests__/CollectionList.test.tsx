import { describe, it, expect, beforeEach, vi } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { http, HttpResponse } from 'msw'
import { server } from '../../tests/mocks/server'
import { render as renderWithProviders } from '@/tests/utils/test-utils'
import CollectionList from '../CollectionList'

const mockCollections = [
  {
    name: 'test-collection-1',
    total_files: 100,
    total_vectors: 500,
    model_name: 'Qwen/Qwen3-Embedding-0.6B',
    created_at: '2025-01-14T12:00:00Z',
    updated_at: '2025-01-14T13:00:00Z',
    job_count: 2,
  },
  {
    name: 'test-collection-2',
    total_files: 50,
    total_vectors: 250,
    model_name: 'Qwen/Qwen3-Embedding-0.6B',
    created_at: '2025-01-13T12:00:00Z',
    updated_at: '2025-01-13T15:00:00Z',
    job_count: 1,
  },
  {
    name: 'test-collection-3',
    total_files: 200,
    total_vectors: 1000,
    model_name: 'BAAI/bge-large-en-v1.5',
    created_at: '2025-01-12T10:00:00Z',
    updated_at: '2025-01-14T09:00:00Z',
    job_count: 5,
  },
]

describe('CollectionList', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading state initially', () => {
    server.use(
      http.get('/api/collections', async () => {
        // Delay to see loading state
        await new Promise(resolve => setTimeout(resolve, 100))
        return HttpResponse.json(mockCollections)
      })
    )

    renderWithProviders(<CollectionList />)

    // Check for loading spinner
    expect(screen.getByRole('status')).toHaveClass('animate-spin')
  })

  it('renders collections when loaded', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.json(mockCollections)
      })
    )

    renderWithProviders(<CollectionList />)

    // Wait for collections to load
    await waitFor(() => {
      expect(screen.getByText('Document Collections')).toBeInTheDocument()
    })

    // Check header
    expect(screen.getByText('Document Collections')).toBeInTheDocument()
    expect(screen.getByText(/Manage your indexed document collections/)).toBeInTheDocument()

    // Check that all collections are rendered
    mockCollections.forEach(collection => {
      expect(screen.getByText(collection.name)).toBeInTheDocument()
    })
  })

  it('renders empty state when no collections', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.json([])
      })
    )

    renderWithProviders(<CollectionList />)

    await waitFor(() => {
      expect(screen.getByText('No collections')).toBeInTheDocument()
    })

    expect(screen.getByText('Get started by creating a new job.')).toBeInTheDocument()
    
    // Check for empty state icon
    const svg = screen.getByRole('img', { hidden: true })
    expect(svg).toBeInTheDocument()
  })

  it('renders error state when API fails', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.error()
      })
    )

    renderWithProviders(<CollectionList />)

    await waitFor(() => {
      expect(screen.getByText('Failed to load collections')).toBeInTheDocument()
    })

    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })

  it('retries loading when retry button is clicked', async () => {
    let callCount = 0

    server.use(
      http.get('/api/collections', () => {
        callCount++
        if (callCount === 1) {
          return HttpResponse.error()
        }
        return HttpResponse.json(mockCollections)
      })
    )

    const user = userEvent.setup()
    renderWithProviders(<CollectionList />)

    // Wait for error state
    await waitFor(() => {
      expect(screen.getByText('Failed to load collections')).toBeInTheDocument()
    })

    // Click retry
    await user.click(screen.getByRole('button', { name: 'Retry' }))

    // Wait for successful load
    await waitFor(() => {
      expect(screen.getByText('Document Collections')).toBeInTheDocument()
    })

    // Check that collections are now displayed
    expect(screen.getByText('test-collection-1')).toBeInTheDocument()
  })

  it('renders collections in a responsive grid', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.json(mockCollections)
      })
    )

    renderWithProviders(<CollectionList />)

    await waitFor(() => {
      expect(screen.getByText('Document Collections')).toBeInTheDocument()
    })

    // Check for grid container with responsive classes
    const gridContainer = screen.getByText('test-collection-1').closest('.grid')
    expect(gridContainer).toHaveClass('grid-cols-1', 'sm:grid-cols-2', 'lg:grid-cols-3')
  })

  it('passes collection data to CollectionCard components', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.json(mockCollections)
      })
    )

    renderWithProviders(<CollectionList />)

    await waitFor(() => {
      expect(screen.getByText('Document Collections')).toBeInTheDocument()
    })

    // Check that CollectionCard components receive the data
    // Since CollectionCard is already tested, we just verify the data is passed
    mockCollections.forEach(collection => {
      expect(screen.getByText(collection.name)).toBeInTheDocument()
      // The CollectionCard component should render these details
      expect(screen.getByText(`${collection.total_files} files`)).toBeInTheDocument()
      expect(screen.getByText(`${collection.total_vectors} vectors`)).toBeInTheDocument()
    })
  })

  it('automatically refetches data periodically', async () => {
    vi.useFakeTimers()
    let callCount = 0

    server.use(
      http.get('/api/collections', () => {
        callCount++
        return HttpResponse.json(mockCollections)
      })
    )

    renderWithProviders(<CollectionList />)

    // Initial load
    await waitFor(() => {
      expect(screen.getByText('Document Collections')).toBeInTheDocument()
    })

    expect(callCount).toBe(1)

    // Fast forward 30 seconds (refetch interval)
    await vi.advanceTimersByTimeAsync(30000)

    // Should have made another call
    await waitFor(() => {
      expect(callCount).toBe(2)
    })

    vi.useRealTimers()
  })

  it('renders with proper accessibility', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.json(mockCollections)
      })
    )

    renderWithProviders(<CollectionList />)

    await waitFor(() => {
      expect(screen.getByText('Document Collections')).toBeInTheDocument()
    })

    // Check that heading has proper hierarchy
    const heading = screen.getByRole('heading', { name: 'Document Collections' })
    expect(heading).toBeInTheDocument()
    expect(heading.tagName).toBe('H2')
  })

  it('handles server error with specific message', async () => {
    server.use(
      http.get('/api/collections', () => {
        return HttpResponse.json(
          { detail: 'Database connection failed' },
          { status: 500 }
        )
      })
    )

    renderWithProviders(<CollectionList />)

    await waitFor(() => {
      expect(screen.getByText('Failed to load collections')).toBeInTheDocument()
    })

    // Error message is generic in the UI, specific error would be in console
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
  })
})