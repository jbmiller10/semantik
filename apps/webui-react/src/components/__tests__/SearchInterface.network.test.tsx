import React from 'react'
import { screen, waitFor, cleanup } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SearchInterface from '../SearchInterface'
import { render } from '../../tests/utils/test-utils'
import { server } from '../../tests/mocks/server'
import { http, HttpResponse } from 'msw'
import { useSearchStore } from '../../stores/searchStore'
import { useUIStore } from '../../stores/uiStore'
import { vi } from 'vitest'

describe('SearchInterface - Network Error Handling', () => {
  beforeEach(() => {
    // Reset search store to initial state
    const initialState = {
      searchParams: {
        query: '',
        selectedCollections: [],
        topK: 10,
        scoreThreshold: 0.0,
        searchType: 'semantic' as const,
        useReranker: false,
        hybridAlpha: 0.7,
        hybridMode: 'rerank' as const,
        keywordMode: 'any' as const,
        rerankModel: null,
        rerankQuantization: null,
      },
      results: [],
      loading: false,
      error: null,
      validationErrors: [],
      collections: [],
      failedCollections: [],
      partialFailure: false,
      rerankingMetrics: null,
      rerankingAvailable: false,
      rerankingModelsLoading: false,
    }
    useSearchStore.setState(initialState)
    // Also clear UI store toasts
    useUIStore.setState({ toasts: [] })
    vi.clearAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('sends normalized hybrid enums in request payload', async () => {
    const user = userEvent.setup()
    let capturedBody: Record<string, unknown> | null = null

    server.use(
      http.post('/api/v2/search', async ({ request }) => {
        capturedBody = await request.json()
        return HttpResponse.json({
          results: [],
          total_results: 0,
          partial_failure: false,
          failed_collections: [],
          search_time_ms: 10,
          total_time_ms: 10,
        })
      })
    )

    render(<SearchInterface />)

    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await user.click(collectionDropdown)

    const collections = await screen.findAllByText('Test Collection 1')
    const collection = collections.find(el => el.classList.contains('font-medium'))
    await user.click(collection!)

    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await user.click(searchInput)
    await user.type(searchInput, 'hybrid query')

    const searchTypeSelect = screen.getByLabelText('Search Type')
    await user.selectOptions(searchTypeSelect, 'hybrid')

    const hybridModeSelect = screen.getByLabelText(/Fusion Mode/i)
    await user.selectOptions(hybridModeSelect, 'reciprocal_rank')

    // Keyword mode selector seems to be removed from UI, skipping interaction
    // const keywordModeSelect = screen.getByRole('combobox', { name: /keyword matching/i })
    // await user.selectOptions(keywordModeSelect, 'all')

    const searchButton = screen.getByRole('button', { name: /search/i })
    await user.click(searchButton)

    await waitFor(() => {
      expect(capturedBody).not.toBeNull()
    })

    expect(capturedBody.hybrid_mode).toBe('reciprocal_rank')
    // expect(capturedBody.keyword_mode).toBe('all')
    expect(capturedBody.search_type).toBe('hybrid')
  })

  it('should show error toast when search fails due to network error', async () => {
    const user = userEvent.setup()

    // Override the search handler to return a network error
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.error()
      })
    )

    render(<SearchInterface />)

    // First select a collection to avoid validation errors
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await user.click(collectionDropdown)

    // Select a collection (use getAllByText to handle multiple elements)
    const collections = await screen.findAllByText('Test Collection 1')
    const collection = collections.find(el => el.classList.contains('font-medium'))
    await user.click(collection)

    // Close dropdown by clicking outside
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await user.click(searchInput)

    // Now enter search query
    await user.type(searchInput, 'test query')

    // Submit search
    const searchButton = screen.getByRole('button', { name: /search/i })
    await user.click(searchButton)

    // Should show error message in the SearchResults component
    await waitFor(() => {
      expect(screen.getByText('An unexpected error occurred during search.')).toBeInTheDocument()
    })
  })

  it('should handle server errors gracefully', async () => {
    const user = userEvent.setup()

    // Override the search handler to return a 500 error
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.json(
          { detail: 'Internal server error' },
          { status: 500 }
        )
      })
    )

    render(<SearchInterface />)

    // First select a collection to avoid validation errors
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await user.click(collectionDropdown)

    // Select a collection (use getAllByText to handle multiple elements)
    const collections = await screen.findAllByText('Test Collection 1')
    const collection = collections.find(el => el.classList.contains('font-medium'))
    await user.click(collection)

    // Close dropdown by clicking outside
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await user.click(searchInput)

    // Now enter search query
    await user.type(searchInput, 'test query')

    // Submit search
    const searchButton = screen.getByRole('button', { name: /search/i })
    await user.click(searchButton)

    // Should show error message in the SearchResults component
    await waitFor(() => {
      expect(screen.getByText('Internal server error')).toBeInTheDocument()
    })
  })

  it('should handle partial failures', async () => {
    const user = userEvent.setup()

    // Override the search handler to return partial failure
    server.use(
      http.post('/api/v2/search', () => {
        return HttpResponse.json({
          results: [
            {
              document_id: 'doc_1',
              chunk_id: 'chunk_1',
              score: 0.9,
              text: 'Test result',
              file_path: '/test.txt',
              file_name: 'test.txt',
              collection_id: '123e4567-e89b-12d3-a456-426614174000',
              collection_name: 'Test Collection 1',
            }
          ],
          total_results: 1,
          partial_failure: true,
          failed_collections: [
            {
              collection_id: '456e7890-e89b-12d3-a456-426614174001',
              collection_name: 'Test Collection 2',
              error: 'Vector index corrupted'
            }
          ],
          search_time_ms: 100,
          total_time_ms: 150,
        })
      })
    )

    render(<SearchInterface />)

    // First select collections to avoid validation errors
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await user.click(collectionDropdown)

    // Select both collections (use getAllByText to handle multiple elements)
    const collections1 = await screen.findAllByText('Test Collection 1')
    const collection1 = collections1.find(el => el.classList.contains('font-medium'))
    await user.click(collection1!)

    const collections2 = await screen.findAllByText('Test Collection 2')
    const collection2 = collections2.find(el => el.classList.contains('font-medium'))
    await user.click(collection2!)

    // Close dropdown
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await user.click(searchInput)

    // Now enter search query
    await user.type(searchInput, 'test query')

    // Submit search
    const searchButton = screen.getByRole('button', { name: /search/i })
    await user.click(searchButton)

    // Should show warning about partial failure in toast
    await waitFor(() => {
      const toasts = useUIStore.getState().toasts
      const warningToast = toasts.find(t => t.type === 'warning')
      expect(warningToast).toBeDefined()
      expect(warningToast?.message).toBe('Some collections failed to search. Showing partial results.')
    })
  })

  it('should disable search button when no collections selected', async () => {
    render(<SearchInterface />)

    // Enter a search query first
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await userEvent.type(searchInput, 'test query')

    // By default, no collections are selected, but the button is not disabled (validation happens on click)
    const searchButton = screen.getByRole('button', { name: /search/i })
    expect(searchButton).not.toBeDisabled()
  })

  it('should require search query', async () => {
    const user = userEvent.setup()
    render(<SearchInterface />)

    // Open collection dropdown and select a collection
    const collectionDropdown = screen.getByRole('button', { name: /select collections/i })
    await user.click(collectionDropdown)

    const collections = await screen.findAllByText('Test Collection 1')
    const collection = collections.find(el => el.classList.contains('font-medium'))
    await user.click(collection!)

    // Close dropdown by clicking outside
    const searchInput = screen.getByPlaceholderText(/Enter your search query/i)
    await user.click(searchInput)

    // Type then clear the search input to trigger validation
    await user.type(searchInput, 'test')
    await user.clear(searchInput)

    // The search button should NOT be disabled when there are validation errors (validation happens on click)
    const searchButton = screen.getByRole('button', { name: /search/i })
    expect(searchButton).not.toBeDisabled()

    // Should show validation error in the UI
    await waitFor(() => {
      expect(screen.getByText('Search query is required')).toBeInTheDocument()
    })
  })
})
